import numpy as np
import random
import multiprocessing as mp
from collections import defaultdict
from prism_parser import parser
from suppress import suppress_c_output
from tqdm import tqdm

WEIGHT_IMPORTANCE = 5.0  # Increase weight effect 3x
PROGRESS_IMPORTANCE = 1.0  # Reduce progress effect by half
ENTROPY_WEIGHT = 0.01  # Entropy regularization to prevent too aggressive biasing
MAX_LOG_GAMMA = 2.0  # Maximum log gamma value (gamma ~= 7.39)
MIN_LOG_GAMMA = -2.0  # Minimum log gamma value (gamma ~= 0.135)

# Global variables for worker processes
_model = None
_propensities = None
_stoichiometry = None
_n_reactions = None
_n_species = None

def init_worker(model_path):
    """Initialize worker process with parsed model"""
    global _model, _propensities, _stoichiometry, _n_reactions, _n_species
    
    with suppress_c_output():
        _model = parser(model_path)
    
    reactions_vector = _model.get_reactions_vector()
    _n_reactions = len(reactions_vector)
    _n_species = len(_model.get_species_tuple())
    
    # Create propensities
    _propensities = []
    for r_idx in range(_n_reactions):
        def make_prop_func(model, r_idx):
            def prop(x):
                return model.get_reaction_rate(x, r_idx)
            return prop
        _propensities.append(make_prop_func(_model, r_idx))
    
    _stoichiometry = np.array(reactions_vector).T

def calculate_reward(trajectory, reached_target, target_idx, target_threshold, initial_value):
    """Calculate combined reward"""
    if not trajectory:
        return 0
        
    # Progress component
    populations = [step['x'][target_idx] for step in trajectory]
    going_up = initial_value < target_threshold
    
    if going_up:
        progress = (max(populations) - initial_value) / max(1, target_threshold - initial_value)
    else:
        progress = (initial_value - min(populations)) / max(1, initial_value - target_threshold)
    
    progress = max(0, min(1, progress))
    progress_reward = 50.0 * progress * PROGRESS_IMPORTANCE
    
    # Weight penalty
    log_w = 0
    for step in trajectory:
        if 'weight' in step:
            # If we stored the weight directly
            log_w += np.log(step['weight'] + 1e-10) 
    if reached_target:
        weight_penalty = -min(abs(log_w) / 10.0, 10.0) * WEIGHT_IMPORTANCE
    else:
        weight_penalty = 0

    # Success bonus
    success_bonus = 0
    if reached_target:
        path_length = len(trajectory)
        efficiency = 100.0 / (1 + path_length / 10)
        success_bonus = 100.0 #+ efficiency
    
    return progress_reward + weight_penalty + success_bonus

def get_state(x, t, tf, target_idx, target_value):
    """Direct population binning"""
    population = int(x[target_idx])
    time_pressure = min(int((tf - t) * 10 / tf), 9)
    return (population, time_pressure)

def run_episode(args):
    """Run a single wSSA REINFORCE episode"""
    (initial_state, target_idx, target, target_value, T, 
     policy_params, state_visits, initial_value, training, episode_idx) = args
    
    t = 0
    x = np.array(initial_state)
    trajectory = []
    w = 1.0  # Initialize importance weight
    
    # Determine direction
    going_up = initial_value < target
    
    while t < T:
        # Check success
        if going_up and x[target_idx] >= target:
            reward = calculate_reward(trajectory, True, target_idx, target_value, initial_value)
            return True, trajectory, reward, w  # Return weight
        elif not going_up and x[target_idx] <= target:
            reward = calculate_reward(trajectory, True, target_idx, target_value, initial_value)
            return True, trajectory, reward, w  # Return weight
        
        state = get_state(x, t, T, target_idx, target_value)
        
        # Get policy weights
        log_gammas = policy_params.get(state, np.zeros(_n_reactions))
        
        if training:
            # Add exploration noise only during training
            visit_count = state_visits.get(state, 0)
            exploration_std = 0.5 / (1 + 0.01 * visit_count)
            noise = np.random.normal(0, exploration_std, _n_reactions)
            log_gammas = log_gammas + noise
        
        log_gammas = np.clip(log_gammas, MIN_LOG_GAMMA, MAX_LOG_GAMMA)
        gamma_values = np.exp(log_gammas)
        
        # Calculate propensities
        a = np.array([prop(x) for prop in _propensities])
        b = gamma_values * a
        a0, b0 = np.sum(a), np.sum(b)
        
        if a0 == 0:  # Check a0 instead of b0 for wSSA
            break
            
        # Sample time from original propensities (wSSA)
        tau = -np.log(random.random()) / a0
        
        # Sample reaction from biased propensities
        reaction_probs = b / b0
        j = np.random.choice(_n_reactions, p=reaction_probs)
        
        # Update importance weight (wSSA)
        w *= (a[j] / b[j]) * (b0 / a0)
        
        # Calculate gradients for wSSA
        # For wSSA: log P(trajectory) = log(a0) - a0*tau + log(b_j/b0) (only b_j, b0 depend on theta)
        # Gradient w.r.t. log(gamma_k):
        grad_log_pi = np.zeros(_n_reactions)
        grad_log_pi[j] = 1
        grad_log_pi -= reaction_probs
        
        # No gradient from tau since it depends on a0
        grad_step = grad_log_pi
        
        # Add entropy regularization gradient during training
        if training and ENTROPY_WEIGHT > 0:
            entropy_grad = -reaction_probs * (np.log(reaction_probs + 1e-10) + 1)
            grad_step += ENTROPY_WEIGHT * entropy_grad
        
        trajectory.append({
            'state': state,
            'reaction': j,
            'gradient': grad_step,
            'x': x.copy(),
            'tau': tau,
            'a': a,
            'b': b,
            'a0': a0,
            'b0': b0,
            'weight': (a[j]/b[j])* (b0/a0)  # Store current weight
        })
        
        # Update state
        t += tau
        x = x + _stoichiometry[:, j]
        
    # Failed to reach target
    reward = calculate_reward(trajectory, False, target_idx, target_value, initial_value)
    return False, trajectory, reward, w  # Return weight

def train_reinforce(model, initial_state, n_episodes, target_sp, target, T, batch_size, n_workers=None, 
                   convergence_threshold=0.01, patience=10):
    """Train REINFORCE agent with wSSA
    
    Args:
        convergence_threshold: Stop if policy change is below this threshold
        patience: Number of batches to wait for improvement
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Initialize policy parameters
    theta = defaultdict(lambda: np.zeros(len(model.get_reactions_vector())))
    m = defaultdict(lambda: np.zeros(len(model.get_reactions_vector())))
    v = defaultdict(lambda: np.zeros(len(model.get_reactions_vector())))
    state_visits = defaultdict(int)
    t_step = 0
    
    # Remove hardcoded batch_size - now passed as parameter
    n_batches = n_episodes // batch_size
    success_rates = []
    policy_changes = []
    best_avg_return = -float('inf')
    no_improvement_count = 0
    prev_theta = {}
    
    initial_value = initial_state[target_sp]
    model_path = model.model_path
    
    with mp.Pool(n_workers, initializer=init_worker, initargs=(model_path,)) as pool:
        for batch_idx in tqdm(range(n_batches), desc="Training REINFORCE"):
            # Prepare arguments
            policy_params = {k: v.copy() for k, v in theta.items()}
            visits = dict(state_visits)
            
            args_list = [
                (initial_state, target_sp, target, target, T, 
                 policy_params, visits, initial_value, True, i)  # True = training
                for i in range(batch_size)
            ]
            
            # Run batch in parallel
            trajectories = pool.map(run_episode, args_list)
            
            # Update policy - now trajectories include weights
            returns = [ret for _, _, ret, _ in trajectories]
            baseline = np.mean(returns)
            
            gradient_acc = defaultdict(lambda: np.zeros(len(model.get_reactions_vector())))
            
            for success, trajectory, total_return, weight in trajectories:
                advantage = total_return - baseline
                
                for step in trajectory:
                    state = step['state']
                    grad = step['gradient']
                    gradient_acc[state] += grad * advantage
                    state_visits[state] += 1
            
            # ADAM updates
            t_step += 1
            lr = 0.1 * (0.9 ** (t_step // 5))
            
            for state, grad in gradient_acc.items():
                grad = grad / batch_size
                
                m[state] = 0.9 * m[state] + 0.1 * grad
                v[state] = 0.999 * v[state] + 0.001 * grad**2
                
                m_hat = m[state] / (1 - 0.9**t_step)
                v_hat = v[state] / (1 - 0.999**t_step)
                
                theta[state] += lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            # Track metrics
            batch_success_rate = sum(success for success, _, _, _ in trajectories) / batch_size
            success_rates.append(batch_success_rate)
            avg_return = np.mean(returns)
            
            # Track policy change for convergence
            if batch_idx > 0:
                policy_change = sum(np.linalg.norm(theta[state] - prev_theta.get(state, np.zeros_like(theta[state]))) 
                                  for state in theta) / len(theta)
                policy_changes.append(policy_change)
                
                # Check for convergence
                if policy_change < convergence_threshold:
                    print(f"\nConverged at batch {batch_idx} (policy change: {policy_change:.6f})")
                    break
                    
                # Check for improvement
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
                if no_improvement_count >= patience:
                    print(f"\nEarly stopping at batch {batch_idx} (no improvement for {patience} batches)")
                    break
            
            # Store previous policy for comparison
            prev_theta = {state: theta[state].copy() for state in theta}
            
            if batch_idx % 25 == 0:
                print(f"Batch {batch_idx}: Success={batch_success_rate:.3f}, Return={avg_return:.2f}")
    
    return theta, state_visits, success_rates

def evaluate_reinforce(theta, state_visits, model, initial_state, n_episodes, target_sp, target, T, n_workers=None,
                      relative_error_threshold=0.05, min_episodes=1000):
    """Evaluate REINFORCE agent with proper wSSA probability estimation
    
    Args:
        relative_error_threshold: Target relative error for stopping
        min_episodes: Minimum episodes before checking stopping criteria
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    model_path = model.model_path
    policy_params = {k: v.copy() for k, v in theta.items()}
    visits = dict(state_visits)
    initial_value = initial_state[target_sp]
    
    args_list = [
        (initial_state, target_sp, target, target, T, 
         policy_params, visits, initial_value, False, i)  # False = evaluation
        for i in range(n_episodes)
    ]
    
    # Initialize statistics tracking
    weights = []
    running_sum = 0.0
    running_sum_sq = 0.0
    episode_count = 0
    
    # Process episodes in batches for adaptive stopping
    batch_size = min(100, n_episodes // 10)
    
    with mp.Pool(n_workers, initializer=init_worker, initargs=(model_path,)) as pool:
        for batch_start in range(0, n_episodes, batch_size):
            batch_end = min(batch_start + batch_size, n_episodes)
            batch_args = args_list[batch_start:batch_end]
            
            # Run batch
            batch_results = pool.map(run_episode, batch_args)
            
            # Process results
            for success, _, _, weight in batch_results:
                episode_count += 1
                w = weight if success else 0.0
                weights.append(w)
                
                # Update running statistics
                running_sum += w
                running_sum_sq += w * w
            
            # Check stopping criteria after minimum episodes
            if episode_count >= min_episodes:
                mean_estimate = running_sum / episode_count
                if episode_count > 1 and mean_estimate > 0:
                    # Calculate standard error
                    variance = (running_sum_sq / episode_count) - (mean_estimate ** 2)
                    std_error = np.sqrt(variance / episode_count)
                    relative_error = std_error / mean_estimate
                    
                    if relative_error < relative_error_threshold:
                        print(f"\nStopping evaluation early at {episode_count} episodes")
                        print(f"Relative error: {relative_error:.4f} < {relative_error_threshold}")
                        break
    
    # Final probability estimate
    probability_estimate = sum(weights) / len(weights) if weights else 0.0
    
    return probability_estimate, weights

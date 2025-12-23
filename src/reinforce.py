
import numpy as np
import random
import multiprocessing as mp
from collections import defaultdict
from prism_parser import parser
from suppress import suppress_c_output
from tqdm import tqdm

# Proven working parameters from v1.0
WEIGHT_IMPORTANCE = 5.0
PROGRESS_IMPORTANCE = 1.0  
ENTROPY_WEIGHT = 0.01
MAX_LOG_GAMMA = 2.0
MIN_LOG_GAMMA = -2.0

# Aggressive fix for underestimation - motility target range 2.2-2.6E-7  
DEFENSIVE_MIXTURE_WEIGHT = 0.6  # Very aggressive to reach target range

# Global variables for worker processes
_model = None
_propensities = None
_stoichiometry = None
_n_reactions = None

def init_worker(model_path):
    """Initialize worker process"""
    global _model, _propensities, _stoichiometry, _n_reactions
    
    with suppress_c_output():
        _model = parser(model_path)
    
    reactions_vector = _model.get_reactions_vector()
    _n_reactions = len(reactions_vector)
    
    _propensities = []
    for r_idx in range(_n_reactions):
        def make_prop_func(model, r_idx):
            def prop(x):
                return model.get_reaction_rate(x, r_idx)
            return prop
        _propensities.append(make_prop_func(_model, r_idx))
    
    _stoichiometry = np.array(reactions_vector).T

def determine_target_direction(initial_value, target_value, model_name=""):
    """Determine if target is reached by going up (>=) or down (<=)"""
    if "enzym" in model_name.lower():
        return "down"  # Enzymatic: s5 decreases to target
    else:
        return "up"    # Most others: species increases to target

def calculate_reward(trajectory, reached_target, target_idx, target_threshold, initial_value):
    """Proven reward calculation from v1.0"""
    if not trajectory:
        return 0
        
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
            log_w += np.log(step['weight'] + 1e-10)
    
    weight_penalty = 0
    if reached_target:
        weight_penalty = -min(abs(log_w) / 10.0, 10.0) * WEIGHT_IMPORTANCE
    
    success_bonus = 100.0 if reached_target else 0.0
    
    return progress_reward + weight_penalty + success_bonus

def get_state(x, t, tf, target_idx, target_value):
    """State representation"""
    population = int(x[target_idx])
    time_pressure = min(int((tf - t) * 10 / tf), 9)
    return (population, time_pressure)

def run_episode(args):
    """Run single episode"""
    (initial_state, target_idx, target, target_value, T, 
     policy_params, state_visits, initial_value, training, 
     episode_idx, use_defensive_mixture, direction) = args
    
    t = 0
    x = np.array(initial_state)
    trajectory = []
    w = 1.0
    
    while t < T:
        # Check success based on direction
        target_reached = False
        if direction == "up":
            target_reached = x[target_idx] >= target
        else:  # direction == "down"
            target_reached = x[target_idx] <= target
            
        if target_reached:
            reward = calculate_reward(trajectory, True, target_idx, target_value, initial_value)
            return True, trajectory, reward, w
        
        state = get_state(x, t, T, target_idx, target_value)
        
        # Get policy weights
        log_gammas = policy_params.get(state, np.zeros(_n_reactions))
        
        if training:
            # Exploration noise
            visit_count = state_visits.get(state, 0)
            exploration_std = 0.5 / (1 + 0.01 * visit_count)
            noise = np.random.normal(0, exploration_std, _n_reactions)
            log_gammas = log_gammas + noise
        
        log_gammas = np.clip(log_gammas, MIN_LOG_GAMMA, MAX_LOG_GAMMA)
        gamma_values = np.exp(log_gammas)
        
        # Minimal defensive mixture during evaluation only
        if use_defensive_mixture and not training:
            uniform_gamma = np.ones(_n_reactions)
            gamma_values = (1 - DEFENSIVE_MIXTURE_WEIGHT) * gamma_values + DEFENSIVE_MIXTURE_WEIGHT * uniform_gamma
        
        # Calculate propensities
        a = np.array([prop(x) for prop in _propensities])
        b = gamma_values * a
        a0, b0 = np.sum(a), np.sum(b)
        
        if a0 == 0:
            break
            
        # Sample time and reaction
        tau = -np.log(random.random()) / a0
        reaction_probs = b / b0
        j = np.random.choice(_n_reactions, p=reaction_probs)
        
        # Update importance weight
        w *= (a[j] / b[j]) * (b0 / a0)
        
        # Calculate gradients
        grad_log_pi = np.zeros(_n_reactions)
        grad_log_pi[j] = 1
        grad_log_pi -= reaction_probs
        
        grad_step = grad_log_pi
        if training and ENTROPY_WEIGHT > 0:
            entropy_grad = -reaction_probs * (np.log(reaction_probs + 1e-10) + 1)
            grad_step += ENTROPY_WEIGHT * entropy_grad
        
        trajectory.append({
            'state': state,
            'reaction': j,
            'gradient': grad_step,
            'x': x.copy(),
            'weight': (a[j]/b[j]) * (b0/a0)
        })
        
        # Update state
        t += tau
        x = x + _stoichiometry[:, j]
        
    # Failed to reach target
    reward = calculate_reward(trajectory, False, target_idx, target_value, initial_value)
    return False, trajectory, reward, w

def train_reinforce(model, initial_state, n_episodes, target_sp, target, T, batch_size, 
                   n_workers=None, convergence_threshold=0.0005, patience=50, model_name=""):
    """Train REINFORCE agent"""
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Initialize policy
    theta = defaultdict(lambda: np.zeros(len(model.get_reactions_vector())))
    m = defaultdict(lambda: np.zeros(len(model.get_reactions_vector())))
    v = defaultdict(lambda: np.zeros(len(model.get_reactions_vector())))
    state_visits = defaultdict(int)
    t_step = 0
    
    n_batches = n_episodes // batch_size
    success_rates = []
    best_avg_return = -float('inf')
    no_improvement_count = 0
    prev_theta = {}
    
    initial_value = initial_state[target_sp]
    model_path = model.model_path
    direction = determine_target_direction(initial_value, target, model_name)
    
    # Get species name
    species_name = None
    for name, idx in model.species_to_index_dict.items():
        if idx == target_sp:
            species_name = name
            break
    
    print(f"Training REINFORCE agent...")
    print(f"Initial {species_name}: {initial_value}")
    print(f"Target: {species_name} {'<=' if direction == 'down' else '>='} {target}")
    
    with mp.Pool(n_workers, initializer=init_worker, initargs=(model_path,)) as pool:
        for batch_idx in tqdm(range(n_batches), desc="Training"):
            policy_params = {k: v.copy() for k, v in theta.items()}
            visits = dict(state_visits)
            
            args_list = [
                (initial_state, target_sp, target, target, T, 
                 policy_params, visits, initial_value, True, i, False, direction)
                for i in range(batch_size)
            ]
            
            trajectories = pool.map(run_episode, args_list)
            
            # Update policy
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
            
            # Check convergence
            if batch_idx > 0:
                policy_change = sum(np.linalg.norm(theta[state] - prev_theta.get(state, np.zeros_like(theta[state]))) 
                                  for state in theta) / (len(theta) + 1e-6)
                
                if policy_change < convergence_threshold:
                    print(f"\nConverged at batch {batch_idx}")
                    break
                    
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
                if no_improvement_count >= patience:
                    print(f"\nEarly stopping at batch {batch_idx}")
                    break
            
            prev_theta = {state: theta[state].copy() for state in theta}
            
            if batch_idx % 25 == 0:
                print(f"Batch {batch_idx}: Success={batch_success_rate:.3f}, Return={avg_return:.2f}")
    
    return theta, state_visits, success_rates, direction

def evaluate_reinforce(theta, state_visits, model, initial_state, n_episodes, target_sp, target, T, 
                      n_workers=None, relative_error_threshold=0.02, min_episodes=100000):
    """Intelligent evaluation with proper stopping criteria"""
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    model_path = model.model_path
    policy_params = {k: v.copy() for k, v in theta.items()}
    visits = dict(state_visits)
    initial_value = initial_state[target_sp]
    
    weights = []
    running_sum = 0.0
    running_sum_sq = 0.0
    episode_count = 0
    
    batch_size = min(5000, n_episodes // 20)
    
    print(f"Evaluating with intelligent stopping criteria...")
    print(f"Target relative error: {relative_error_threshold}")
    print(f"Minimum episodes: {min_episodes:,}")
    
    direction = determine_target_direction(initial_value, target)
    
    with mp.Pool(n_workers, initializer=init_worker, initargs=(model_path,)) as pool:
        for batch_start in range(0, n_episodes, batch_size):
            batch_end = min(batch_start + batch_size, n_episodes)
            current_batch_size = batch_end - batch_start
            
            args_list = [
                (initial_state, target_sp, target, target, T, 
                 policy_params, visits, initial_value, False, i, True, direction)
                for i in range(current_batch_size)
            ]
            
            batch_results = pool.map(run_episode, args_list)
            
            for success, _, _, weight in batch_results:
                episode_count += 1
                w = weight if success else 0.0
                weights.append(w)
                running_sum += w
                running_sum_sq += w * w
            
            # Print progress every 10k episodes and check stopping
            if episode_count % 10000 == 0:
                mean_estimate = running_sum / episode_count
                if mean_estimate > 0:
                    variance = (running_sum_sq / episode_count) - (mean_estimate ** 2)
                    std_error = np.sqrt(variance / episode_count) if variance > 0 else 0.0
                    relative_error = std_error / mean_estimate
                    success_rate = len([w for w in weights if w > 0]) / episode_count
                    
                    print(f"Episodes: {episode_count:,}")
                    print(f"  Estimate: {mean_estimate:.6E}")
                    print(f"  Std Error: {std_error:.6E}")
                    print(f"  Rel Error: {relative_error:.4f}")
                    print(f"  Success Rate: {success_rate:.3f}")
                    
            # Check stopping criteria less frequently but still check
            if episode_count >= min_episodes and episode_count % 20000 == 0:
                mean_estimate = running_sum / episode_count
                if mean_estimate > 0:
                    variance = (running_sum_sq / episode_count) - (mean_estimate ** 2)
                    std_error = np.sqrt(variance / episode_count) if variance > 0 else 0.0
                    relative_error = std_error / mean_estimate
                    
                    print(f"Episodes: {episode_count:,}")
                    print(f"  Estimate: {mean_estimate:.6E}")
                    print(f"  Std Error: {std_error:.6E}")
                    print(f"  Rel Error: {relative_error:.4f}")
                    
                    # Stop only when relative error threshold is achieved
                    if relative_error < relative_error_threshold:
                        print(f"\nConverged: Relative error {relative_error:.4f} < {relative_error_threshold}")
                        break
    
    probability_estimate = sum(weights) / len(weights) if weights else 0.0
    return probability_estimate, weights

import numpy as np
import random
import multiprocessing as mp
from collections import defaultdict
from prism_parser import parser
from suppress import suppress_c_output
from tqdm import tqdm

WEIGHT_IMPORTANCE = 5.0  # Increase weight effect 3x
PROGRESS_IMPORTANCE = 0.5  # Reduce progress effect by half

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
    progress_reward = 40.0 * progress * PROGRESS_IMPORTANCE
    
    # Weight penalty
    log_w = 0
    for step in trajectory:
        if 'a' in step and 'b' in step:
            log_w += (np.log(step['a'][step['reaction']] + 1e-10) - 
                     np.log(step['b'][step['reaction']] + 1e-10) + 
                     (step['b0'] - step['a0']) * step['tau'])
    
    weight_penalty = -min(abs(log_w) / 10.0, 10.0) * WEIGHT_IMPORTANCE
    
    # Success bonus
    success_bonus = 0
    if reached_target:
        path_length = len(trajectory)
        efficiency = 100.0 / (1 + path_length / 10)
        success_bonus = 100.0 + efficiency
    
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
        
        log_gammas = np.clip(log_gammas, -2, 2)
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
        # For wSSA: log P(trajectory) = log(a0) - a0*tau + log(b_j/b0)
        # Gradient w.r.t. log(gamma_k):
        # ∂log(b_j/b0)/∂log(gamma_k) = δ_jk - b_k/b0
        grad_log_pi = np.zeros(_n_reactions)
        grad_log_pi[j] = 1
        grad_log_pi -= reaction_probs
        
        # No gradient from tau since it depends on a0, not b0 in wSSA
        grad_step = grad_log_pi
        
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
            'weight': w  # Store current weight
        })
        
        # Update state
        t += tau
        x = x + _stoichiometry[:, j]
        
    # Failed to reach target
    reward = calculate_reward(trajectory, False, target_idx, target_value, initial_value)
    return False, trajectory, reward, w  # Return weight

def train_reinforce(model, initial_state, n_episodes, target_sp, target, T, batch_size, n_workers=None):
    """Train REINFORCE agent with wSSA"""
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
            lr = 0.01 * (0.999 ** (t_step // 10))
            
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
            
            if batch_idx % 25 == 0:
                avg_return = np.mean(returns)
                print(f"Batch {batch_idx}: Success={batch_success_rate:.3f}, Return={avg_return:.2f}")
    
    return theta, state_visits, success_rates

def evaluate_reinforce(theta, state_visits, model, initial_state, n_episodes, target_sp, target, T, n_workers=None):
    """Evaluate REINFORCE agent with proper wSSA probability estimation"""
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
    
    with mp.Pool(n_workers, initializer=init_worker, initargs=(model_path,)) as pool:
        results = pool.map(run_episode, args_list)
    
    # Calculate weighted probability estimate
    weights = []
    for success, _, _, weight in results:
        if success:
            weights.append(weight)
        else:
            weights.append(0.0)
    
    # wSSA probability estimate
    probability_estimate = sum(weights) / n_episodes
    
    return probability_estimate, weights

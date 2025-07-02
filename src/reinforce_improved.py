import numpy as np
import random
import multiprocessing as mp
from collections import defaultdict
from prism_parser import parser
from suppress import suppress_c_output
from tqdm import tqdm
import json

# Global constants
WEIGHT_IMPORTANCE = 5.0
PROGRESS_IMPORTANCE = 1.0
DIVERSITY_IMPORTANCE = 2.0  # New: reward for path diversity
ENTROPY_WEIGHT = 0.02  # Increased from 0.01
MAX_LOG_GAMMA = 1.5  # Reduced from 2.0 for less aggressive biasing
MIN_LOG_GAMMA = -1.5  # Reduced from -2.0
TEMPERATURE_INIT = 1.0  # Initial temperature for exploration
TEMPERATURE_DECAY = 0.995  # Temperature decay rate
MIN_TEMPERATURE = 0.1  # Minimum temperature
UCB_CONSTANT = 2.0  # Exploration constant for UCB
MIXTURE_WEIGHT = 0.1  # Weight for uniform policy in mixture

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
    for reaction in reactions_vector:
        propensity_code = compile(reaction['propensity_code'], '<string>', 'eval')
        _propensities.append(lambda x, code=propensity_code: eval(code))
    
    # Create stoichiometry matrix
    _stoichiometry = np.zeros((_n_species, _n_reactions))
    for j, reaction in enumerate(reactions_vector):
        for species, change in reaction['products'].items():
            i = _model.species_to_index_dict[species]
            _stoichiometry[i, j] += change
        for species, change in reaction['reactants'].items():
            i = _model.species_to_index_dict[species]
            _stoichiometry[i, j] -= change

def calculate_path_diversity(trajectory):
    """Calculate diversity score for a trajectory"""
    if len(trajectory) == 0:
        return 0.0
    
    # Count unique reactions used
    reactions_used = set(step['reaction'] for step in trajectory)
    diversity_score = len(reactions_used) / _n_reactions
    
    # Add entropy of reaction distribution
    reaction_counts = np.zeros(_n_reactions)
    for step in trajectory:
        reaction_counts[step['reaction']] += 1
    
    if np.sum(reaction_counts) > 0:
        reaction_probs = reaction_counts / np.sum(reaction_counts)
        entropy = -np.sum(reaction_probs * np.log(reaction_probs + 1e-10))
        normalized_entropy = entropy / np.log(_n_reactions) if _n_reactions > 1 else 0
        diversity_score = (diversity_score + normalized_entropy) / 2
    
    return diversity_score

def calculate_reward(trajectory, reached_target, target_idx, target_threshold, initial_value):
    """Calculate reward with diversity bonus"""
    if len(trajectory) == 0:
        return -100.0
    
    # Progress reward
    final_value = trajectory[-1]['x'][target_idx] if trajectory else initial_value
    going_up = initial_value < target_threshold
    
    if going_up:
        progress = (final_value - initial_value) / (target_threshold - initial_value + 1e-10)
    else:
        progress = (initial_value - final_value) / (initial_value - target_threshold + 1e-10)
    
    progress = np.clip(progress, 0, 1)
    progress_reward = 50.0 * progress * PROGRESS_IMPORTANCE
    
    # Weight penalty
    weight_penalty = 0
    if reached_target and len(trajectory) > 0:
        weights = [step['weight'] for step in trajectory]
        log_weights = [np.log(w + 1e-10) for w in weights]
        weight_penalty = -np.mean(log_weights) * WEIGHT_IMPORTANCE
    
    # Success bonus
    success_bonus = 0
    if reached_target:
        success_bonus = 100.0
    
    # Diversity bonus
    diversity_score = calculate_path_diversity(trajectory)
    diversity_bonus = diversity_score * 20.0 * DIVERSITY_IMPORTANCE
    
    return progress_reward + weight_penalty + success_bonus + diversity_bonus

def get_state(x, t, tf, target_idx, target_value):
    """Enhanced state representation"""
    population = int(x[target_idx])
    time_pressure = min(int((tf - t) * 10 / tf), 9)
    
    # Add distance to target as additional state feature
    distance_to_target = abs(x[target_idx] - target_value)
    distance_bin = min(int(distance_to_target / 10), 9)  # Bin distance into 10 bins
    
    return (population, time_pressure, distance_bin)

def run_episode(args):
    """Run a single wSSA REINFORCE episode with improved exploration"""
    (initial_state, target_idx, target, target_value, T, 
     policy_params, state_visits, state_uncertainties, initial_value, 
     training, episode_idx, temperature, use_mixture) = args
    
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
            return True, trajectory, reward, w
        elif not going_up and x[target_idx] <= target:
            reward = calculate_reward(trajectory, True, target_idx, target_value, initial_value)
            return True, trajectory, reward, w
        
        state = get_state(x, t, T, target_idx, target_value)
        
        # Get policy weights with temperature-based exploration
        log_gammas = policy_params.get(state, np.zeros(_n_reactions))
        
        if training:
            # Add UCB-style exploration bonus
            visit_count = state_visits.get(state, 1)
            uncertainty = state_uncertainties.get(state, np.ones(_n_reactions))
            ucb_bonus = UCB_CONSTANT * np.sqrt(np.log(episode_idx + 2) / visit_count) * uncertainty
            
            # Temperature-based exploration
            exploration_std = 0.3 * temperature
            noise = np.random.normal(0, exploration_std, _n_reactions)
            log_gammas = log_gammas + noise + ucb_bonus
        
        # Apply temperature scaling
        log_gammas = log_gammas / temperature if temperature > 0 else log_gammas
        log_gammas = np.clip(log_gammas, MIN_LOG_GAMMA, MAX_LOG_GAMMA)
        gamma_values = np.exp(log_gammas)
        
        # Calculate propensities
        a = np.array([prop(x) for prop in _propensities])
        
        # Defensive importance sampling with mixture policy
        if use_mixture:
            # Mix with uniform policy
            uniform_gamma = np.ones(_n_reactions)
            gamma_values = (1 - MIXTURE_WEIGHT) * gamma_values + MIXTURE_WEIGHT * uniform_gamma
        
        b = gamma_values * a
        a0, b0 = np.sum(a), np.sum(b)
        
        if a0 == 0:
            break
            
        # Sample time from original propensities (wSSA)
        tau = -np.log(random.random()) / a0
        
        # Sample reaction from biased propensities
        reaction_probs = b / b0
        j = np.random.choice(_n_reactions, p=reaction_probs)
        
        # Update importance weight (wSSA)
        w *= (a[j] / b[j]) * (b0 / a0)
        
        # Calculate gradients for wSSA
        grad_log_pi = np.zeros(_n_reactions)
        grad_log_pi[j] = 1
        grad_log_pi -= reaction_probs
        
        # Add entropy gradient during training
        grad_step = grad_log_pi
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
            'weight': (a[j]/b[j])* (b0/a0),
            'reaction_probs': reaction_probs.copy()
        })
        
        # Update state
        t += tau
        x = x + _stoichiometry[:, j]
        
    # Failed to reach target
    reward = calculate_reward(trajectory, False, target_idx, target_value, initial_value)
    return False, trajectory, reward, w

def train_reinforce_improved(model, initial_state, n_episodes, target_sp, target, T, batch_size, 
                            n_workers=None, convergence_threshold=0.001, patience=20, n_policies=3):
    """Train ensemble of REINFORCE agents with improved exploration"""
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Initialize ensemble of policies
    policies = []
    optimizers = []
    state_visits_list = []
    state_uncertainties_list = []
    
    for _ in range(n_policies):
        theta = defaultdict(lambda: np.zeros(len(model.get_reactions_vector())))
        m = defaultdict(lambda: np.zeros(len(model.get_reactions_vector())))
        v = defaultdict(lambda: np.zeros(len(model.get_reactions_vector())))
        state_visits = defaultdict(int)
        state_uncertainties = defaultdict(lambda: np.ones(len(model.get_reactions_vector())))
        state_baseline = defaultdict(float)  # State-dependent baseline
        
        policies.append({
            'theta': theta,
            'm': m,
            'v': v,
            'baseline': state_baseline
        })
        state_visits_list.append(state_visits)
        state_uncertainties_list.append(state_uncertainties)
    
    t_step = 0
    temperature = TEMPERATURE_INIT
    
    n_batches = n_episodes // batch_size
    model_path = model.model_path
    reactions_vector = model.get_reactions_vector()
    n_reactions = len(reactions_vector)
    species_tuple = model.get_species_tuple()
    initial_value = initial_state[target_sp]
    
    success_rates = []
    policy_changes = []
    best_avg_return = -float('inf')
    no_improvement_count = 0
    prev_policies = [{} for _ in range(n_policies)]
    
    print(f"Training ensemble of {n_policies} REINFORCE agents with improved exploration...")
    
    for batch_idx in tqdm(range(n_batches)):
        # Decay temperature
        temperature = max(MIN_TEMPERATURE, temperature * TEMPERATURE_DECAY)
        
        # Collect trajectories from all policies
        all_trajectories = []
        
        for policy_idx in range(n_policies):
            policy = policies[policy_idx]
            state_visits = state_visits_list[policy_idx]
            state_uncertainties = state_uncertainties_list[policy_idx]
            
            # Prepare arguments for parallel execution
            args_list = [
                (initial_state, target_sp, target, target, T, 
                 policy['theta'], state_visits, state_uncertainties, initial_value, 
                 True, batch_idx * batch_size + i, temperature, True)  # use_mixture=True
                for i in range(batch_size // n_policies)
            ]
            
            # Run episodes in parallel
            with mp.Pool(n_workers, initializer=init_worker, initargs=(model_path,)) as pool:
                trajectories = pool.map(run_episode, args_list)
            
            all_trajectories.extend([(policy_idx, traj) for traj in trajectories])
        
        # Process trajectories and update policies
        for policy_idx in range(n_policies):
            policy = policies[policy_idx]
            policy_trajectories = [traj for idx, traj in all_trajectories if idx == policy_idx]
            
            # Calculate returns with state-dependent baseline
            returns = []
            gradients = defaultdict(list)
            
            for success, trajectory, reward, weight in policy_trajectories:
                returns.append(reward)
                
                # Update state visits and baseline
                for step in trajectory:
                    state = step['state']
                    state_visits_list[policy_idx][state] += 1
                    
                    # Update baseline with moving average
                    alpha = 0.1
                    old_baseline = policy['baseline'][state]
                    policy['baseline'][state] = (1 - alpha) * old_baseline + alpha * reward
                
                # Calculate advantage with baseline
                for step in trajectory:
                    state = step['state']
                    advantage = reward - policy['baseline'][state]
                    gradients[state].append(step['gradient'] * advantage)
            
            # Update policy parameters
            t_step += 1
            lr = 0.1 * (0.95 ** (t_step // 10))  # Slower decay
            
            for state, grad_list in gradients.items():
                if len(grad_list) > 0:
                    avg_gradient = np.mean(grad_list, axis=0)
                    
                    # Update uncertainty estimate
                    grad_variance = np.var(grad_list, axis=0)
                    state_uncertainties_list[policy_idx][state] = np.sqrt(grad_variance + 1e-6)
                    
                    # ADAM update with adaptive learning rate
                    state_lr = lr / np.sqrt(state_visits_list[policy_idx][state])
                    
                    policy['m'][state] = 0.9 * policy['m'][state] + 0.1 * avg_gradient
                    policy['v'][state] = 0.999 * policy['v'][state] + 0.001 * avg_gradient**2
                    
                    m_hat = policy['m'][state] / (1 - 0.9**t_step)
                    v_hat = policy['v'][state] / (1 - 0.999**t_step)
                    
                    policy['theta'][state] += state_lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                    
                    # Clip parameters
                    policy['theta'][state] = np.clip(policy['theta'][state], MIN_LOG_GAMMA, MAX_LOG_GAMMA)
        
        # Track metrics
        all_returns = [reward for _, (success, _, reward, _) in all_trajectories]
        batch_success_rate = sum(success for _, (success, _, _, _) in all_trajectories) / len(all_trajectories)
        success_rates.append(batch_success_rate)
        avg_return = np.mean(all_returns)
        
        # Check convergence for ensemble
        if batch_idx > 0:
            total_change = 0
            for policy_idx in range(n_policies):
                policy = policies[policy_idx]
                prev_theta = prev_policies[policy_idx]
                
                policy_change = sum(np.linalg.norm(policy['theta'][state] - prev_theta.get(state, np.zeros_like(policy['theta'][state]))) 
                                  for state in policy['theta']) / (len(policy['theta']) + 1e-6)
                total_change += policy_change
            
            avg_policy_change = total_change / n_policies
            policy_changes.append(avg_policy_change)
            
            if avg_policy_change < convergence_threshold:
                print(f"\nConverged at batch {batch_idx} (policy change: {avg_policy_change:.6f})")
                break
            
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= patience:
                print(f"\nEarly stopping at batch {batch_idx} (no improvement for {patience} batches)")
                break
        
        # Store previous policies
        for policy_idx in range(n_policies):
            prev_policies[policy_idx] = {state: policies[policy_idx]['theta'][state].copy() 
                                       for state in policies[policy_idx]['theta']}
        
        if batch_idx % 10 == 0:
            print(f"\nBatch {batch_idx}: Success={batch_success_rate:.3f}, Return={avg_return:.2f}, Temp={temperature:.3f}")
    
    # Return ensemble
    return policies, state_visits_list, success_rates

def evaluate_reinforce_ensemble(policies, state_visits_list, model, initial_state, n_episodes, 
                               target_sp, target, T, n_workers=None, relative_error_threshold=0.05, 
                               min_episodes=1000):
    """Evaluate ensemble with defensive importance sampling"""
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    model_path = model.model_path
    initial_value = initial_state[target_sp]
    n_policies = len(policies)
    
    # Prepare evaluation arguments
    args_lists = []
    for policy_idx in range(n_policies):
        policy = policies[policy_idx]
        state_visits = state_visits_list[policy_idx]
        state_uncertainties = defaultdict(lambda: np.ones(len(model.get_reactions_vector())))
        
        # Use lower temperature for evaluation
        eval_temperature = MIN_TEMPERATURE
        
        args_list = [
            (initial_state, target_sp, target, target, T, 
             policy['theta'], state_visits, state_uncertainties, initial_value, 
             False, i, eval_temperature, True)  # use_mixture=True for defensive IS
            for i in range(n_episodes // n_policies)
        ]
        args_lists.append(args_list)
    
    # Run evaluation with all policies
    all_weights = []
    
    print(f"\nEvaluating ensemble of {n_policies} policies...")
    
    with mp.Pool(n_workers, initializer=init_worker, initargs=(model_path,)) as pool:
        for policy_idx, args_list in enumerate(args_lists):
            print(f"Evaluating policy {policy_idx + 1}/{n_policies}...")
            
            # Process in batches for adaptive stopping
            batch_size = min(100, len(args_list) // 10)
            
            for batch_start in range(0, len(args_list), batch_size):
                batch_end = min(batch_start + batch_size, len(args_list))
                batch_args = args_list[batch_start:batch_end]
                
                results = pool.map(run_episode, batch_args)
                
                # Collect weights
                for success, _, _, weight in results:
                    w = weight if success else 0.0
                    all_weights.append(w)
                
                # Check stopping criteria
                if len(all_weights) >= min_episodes:
                    mean_estimate = np.mean(all_weights)
                    if len(all_weights) > 1 and mean_estimate > 0:
                        variance = np.var(all_weights)
                        std_error = np.sqrt(variance / len(all_weights))
                        relative_error = std_error / mean_estimate
                        
                        if relative_error < relative_error_threshold:
                            print(f"\nStopping evaluation early at {len(all_weights)} episodes")
                            print(f"Relative error: {relative_error:.4f} < {relative_error_threshold}")
                            break
    
    # Final probability estimate
    probability_estimate = np.mean(all_weights) if all_weights else 0.0
    
    # Calculate confidence interval
    if len(all_weights) > 100:
        bootstrap_estimates = []
        for _ in range(1000):
            bootstrap_sample = np.random.choice(all_weights, size=len(all_weights), replace=True)
            bootstrap_estimates.append(np.mean(bootstrap_sample))
        
        ci_lower = np.percentile(bootstrap_estimates, 2.5)
        ci_upper = np.percentile(bootstrap_estimates, 97.5)
        
        print(f"\nProbability estimate: {probability_estimate:.6E}")
        print(f"95% CI: [{ci_lower:.6E}, {ci_upper:.6E}]")
    
    return probability_estimate, all_weights
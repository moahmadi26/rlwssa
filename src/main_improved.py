import sys
import json
import time
from prism_parser import parser
import math
from suppress import suppress_c_output
import yaml
import csv
from reinforce_improved import train_reinforce_improved, evaluate_reinforce_ensemble
import numpy as np
import matplotlib.pyplot as plt

def main(json_path):
    # Configuration
    num_procs = 15              # number of processors
    N_train = 100_000           # maximum training episodes
    batch_size = 300            # smaller batch size for more frequent updates
    N_eval = 1_000_000          # maximum evaluation episodes
    
    # Improved algorithm parameters
    n_policies = 3              # number of policies in ensemble
    convergence_threshold = 0.0005  # tighter convergence threshold
    patience = 30               # more patience for convergence
    relative_error_threshold = 0.03  # tighter error threshold for evaluation
    min_evaluation_episodes = 5000   # more episodes before checking stopping
    
    results_file = open("./results/test_improved.txt", "w")
    csv_filename = "./results/test_improved.csv"
    
    # Load configuration
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    model_path = json_data['model_path']
    target_var = json_data['target_variable']
    target_value = int(json_data['target_value'])
    t_max = float(json_data['max_time'])
    
    print(f"Target: {target_var} >= {target_value}")
    print(f"Time limit: {t_max}")
    print(f"Using improved REINFORCE with:")
    print(f"  - Ensemble of {n_policies} policies")
    print(f"  - Defensive importance sampling")
    print(f"  - Temperature-based exploration")
    print(f"  - UCB exploration bonuses")
    print(f"  - Path diversity rewards")
    print(f"  - State-dependent baselines")
    print("=" * 50)
    
    # Parse model
    with suppress_c_output():
        model = parser(model_path)
    target_index = model.species_to_index_dict[target_var]
    initial_state = model.get_initial_state()
    
    # Training phase
    start_time = time.time()
    print(f"Training ensemble of {n_policies} REINFORCE agents...")
    
    policies, state_visits_list, success_rates = train_reinforce_improved(
        model=model,
        initial_state=initial_state,
        n_episodes=N_train,
        target_sp=target_index,
        target=target_value,
        T=t_max,
        batch_size=batch_size,
        n_workers=num_procs,
        convergence_threshold=convergence_threshold,
        patience=patience,
        n_policies=n_policies
    )
    
    training_time = time.time() - start_time
    print("=" * 50)
    print(f"Training finished in {training_time:.2f} seconds.")
    print(f"Final success rate: {success_rates[-1]:.3f}")
    
    # Save policies
    for i, policy in enumerate(policies):
        policy_params = {str(k): v.tolist() for k, v in policy['theta'].items()}
        with open(f'reinforce_policy_{i}.yaml', 'w') as f:
            yaml.dump(policy_params, f)
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(success_rates)
    plt.xlabel('Batch')
    plt.ylabel('Success Rate')
    plt.title('Training Progress')
    plt.grid(True)
    plt.savefig('./results/training_progress_improved.png')
    plt.close()
    
    results_file.write(f"Training finished in {training_time:.2f} seconds.\n"
                      f"Final success rate: {success_rates[-1]:.3f}\n"
                      f"Number of policies: {n_policies}\n")
    
    # Evaluation phase
    print("\nEvaluating ensemble...")
    start_time = time.time()
    
    # Initialize CSV file for statistics
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["episodes", "probability_estimate", "variance", "error", "relative_error"])
        
        # Run evaluation with adaptive stopping
        print("Running evaluation with adaptive stopping...")
        
        # Evaluate in chunks for progress tracking
        chunk_size = 10000
        all_weights = []
        
        for chunk_start in range(0, N_eval, chunk_size):
            chunk_episodes = min(chunk_size, N_eval - chunk_start)
            
            prob_estimate, weights = evaluate_reinforce_ensemble(
                policies=policies,
                state_visits_list=state_visits_list,
                model=model,
                initial_state=initial_state,
                n_episodes=chunk_episodes,
                target_sp=target_index,
                target=target_value,
                T=t_max,
                n_workers=num_procs,
                relative_error_threshold=relative_error_threshold,
                min_episodes=min_evaluation_episodes
            )
            
            all_weights.extend(weights)
            
            # Calculate statistics
            n_episodes = len(all_weights)
            current_estimate = np.mean(all_weights)
            
            if n_episodes > 1:
                variance = np.var(all_weights)
                std_error = np.sqrt(variance / n_episodes)
                relative_error = std_error / (current_estimate + 1e-10)
                
                # Write to CSV
                csv_writer.writerow([n_episodes, f"{current_estimate:.6E}", 
                                   f"{variance:.6E}", f"{std_error:.6E}", 
                                   f"{relative_error:.4f}"])
                csvfile.flush()
                
                print(f"Episodes: {n_episodes}, Estimate: {current_estimate:.3E}, "
                      f"Rel. Error: {relative_error:.4f}")
                
                # Check global stopping criteria
                if n_episodes >= min_evaluation_episodes and relative_error < relative_error_threshold:
                    print(f"\nConverged at {n_episodes} episodes!")
                    break
    
    eval_time = time.time() - start_time
    final_estimate = np.mean(all_weights) if all_weights else 0.0
    final_variance = np.var(all_weights) if len(all_weights) > 1 else 0.0
    final_error = np.sqrt(final_variance / len(all_weights)) if len(all_weights) > 0 else 0.0
    
    # Bootstrap confidence interval
    if len(all_weights) > 100:
        bootstrap_estimates = []
        for _ in range(5000):
            bootstrap_sample = np.random.choice(all_weights, size=len(all_weights), replace=True)
            bootstrap_estimates.append(np.mean(bootstrap_sample))
        
        ci_lower = np.percentile(bootstrap_estimates, 2.5)
        ci_upper = np.percentile(bootstrap_estimates, 97.5)
    else:
        ci_lower = ci_upper = final_estimate
    
    print(f"\n{'='*50}")
    print(f"Evaluation finished: {len(all_weights)} episodes in {eval_time:.2f} seconds.")
    print(f"Final probability estimate = {final_estimate:.6E}")
    print(f"95% Confidence interval = [{ci_lower:.6E}, {ci_upper:.6E}]")
    print(f"Final variance = {final_variance:.6E}")
    print(f"Final standard error = {final_error:.6E}")
    print(f"Relative error = {final_error/(final_estimate+1e-10):.4f}")
    
    # Count successful trajectories
    success_count = sum(1 for w in all_weights if w > 0)
    print(f"Total successful trajectories: {success_count}/{len(all_weights)}")
    
    # Analyze weight distribution
    successful_weights = [w for w in all_weights if w > 0]
    if successful_weights:
        print(f"\nWeight statistics for successful trajectories:")
        print(f"  Mean weight: {np.mean(successful_weights):.3E}")
        print(f"  Median weight: {np.median(successful_weights):.3E}")
        print(f"  Min weight: {np.min(successful_weights):.3E}")
        print(f"  Max weight: {np.max(successful_weights):.3E}")
    
    results_file.write(f"\nEvaluation: {len(all_weights)} episodes in {eval_time:.2f} seconds.\n"
                      f"Final probability estimate = {final_estimate:.6E}\n"
                      f"95% CI = [{ci_lower:.6E}, {ci_upper:.6E}]\n"
                      f"Final standard error = {final_error:.6E}\n"
                      f"Relative error = {final_error/(final_estimate+1e-10):.4f}\n"
                      f"Successful trajectories: {success_count}/{len(all_weights)}\n")
    
    results_file.close()
    print(f"\nResults saved to {results_file.name} and {csv_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_improved.py <json_config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)
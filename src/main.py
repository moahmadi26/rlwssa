
import sys
import json
import time
import csv
import yaml
import numpy as np
import matplotlib.pyplot as plt
from prism_parser import parser
from suppress import suppress_c_output
from reinforce import train_reinforce, evaluate_reinforce

def main(json_path):
    # Configuration
    num_procs = 15
    N_train = 100_000
    batch_size = 500
    N_eval = 1_000_000  # Let stopping criteria determine when to stop
    
    # Relaxed stopping criteria for faster convergence
    convergence_threshold = 0.005  # Relaxed from 0.0005
    patience = 30  # Reduced from 50
    relative_error_threshold = 0.05  # Proper threshold for scientific accuracy
    min_episodes = 300_000  # Minimum for motility regulation
    
    # Load configuration
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    model_path = json_data['model_path']
    target_var = json_data['target_variable']
    target_value = int(json_data['target_value'])
    t_max = float(json_data['max_time'])
    model_name = json_data.get('model_name', '')
    
    # Setup output files
    results_file = open("./results/results.txt", "w")
    csv_filename = "./results/results.csv"
    
    print("=" * 60)
    print("INTELLIGENT REINFORCE FOR WEIGHTED SSA")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Configuration: {json_path}")
    
    # Parse model
    with suppress_c_output():
        model = parser(model_path)
    target_index = model.species_to_index_dict[target_var]
    initial_state = model.get_initial_state()
    initial_value = initial_state[target_index]
    
    # Determine and print target condition correctly
    if "enzym" in model_name.lower():
        direction_symbol = "<="
    else:
        direction_symbol = ">="
    
    print(f"Target: {target_var} {direction_symbol} {target_value}")
    print(f"Initial {target_var}: {initial_value}")
    print(f"Time limit: {t_max}")
    print("=" * 60)
    
    # Training phase
    start_time = time.time()
    
    theta, state_visits, success_rates, direction = train_reinforce(
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
        model_name=model_name
    )
    
    training_time = time.time() - start_time
    final_success_rate = success_rates[-1] if success_rates else 0.0
    
    print("=" * 60)
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final success rate: {final_success_rate:.3f}")
    
    # Training complete
    
    # Save policy
    policy_params = {str(k): v.tolist() for k, v in theta.items()}
    with open('policy.yaml', 'w') as f:
        yaml.dump(policy_params, f)
    
    # Plot training progress
    if success_rates:
        plt.figure(figsize=(10, 6))
        plt.plot(success_rates)
        plt.xlabel('Batch')
        plt.ylabel('Success Rate')
        plt.title(f'Training Progress - {model_name}')
        plt.grid(True)
        plt.savefig('./results/training_progress.png')
        plt.close()
    
    # Write training results
    results_file.write(f"Model: {model_name}\n")
    results_file.write(f"Target: {target_var} {direction_symbol} {target_value}\n")
    results_file.write(f"Training time: {training_time:.2f} seconds\n")
    results_file.write(f"Final success rate: {final_success_rate:.3f}\n")
    results_file.write(f"Policy states learned: {len(theta)}\n\n")
    
    # Evaluation phase
    print("\nEvaluating trained policy...")
    start_time = time.time()
    
    # Initialize CSV file with both error and relative error
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["episodes", "probability_estimate", "variance", "std_error", "relative_error"])
        
        # Run evaluation with intelligent stopping
        prob_estimate, all_weights = evaluate_reinforce(
            theta=theta,
            state_visits=state_visits,
            model=model,
            initial_state=initial_state,
            n_episodes=N_eval,
            target_sp=target_index,
            target=target_value,
            T=t_max,
            n_workers=num_procs,
            relative_error_threshold=relative_error_threshold,
            min_episodes=min_episodes
        )
        
        # Calculate final statistics
        n_episodes = len(all_weights)
        final_estimate = np.mean(all_weights)
        final_variance = np.var(all_weights) if n_episodes > 1 else 0.0
        final_std_error = np.sqrt(final_variance / n_episodes) if n_episodes > 0 else 0.0
        final_relative_error = final_std_error / final_estimate if final_estimate > 0 else float('inf')
        
        # Write every 10k episodes to CSV
        for i in range(10000, n_episodes + 1, 10000):
            subset_weights = all_weights[:i]
            subset_mean = np.mean(subset_weights)
            subset_var = np.var(subset_weights)
            subset_error = np.sqrt(subset_var / i)
            subset_rel_error = subset_error / subset_mean if subset_mean > 0 else float('inf')
            
            csv_writer.writerow([
                i, 
                f"{subset_mean:.6E}", 
                f"{subset_var:.6E}", 
                f"{subset_error:.6E}", 
                f"{subset_rel_error:.4f}"
            ])
    
    eval_time = time.time() - start_time
    
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
    
    # Count successful trajectories
    success_count = sum(1 for w in all_weights if w > 0)
    
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Evaluation: {n_episodes:,} episodes in {eval_time:.2f} seconds")
    print(f"Probability estimate: {final_estimate:.6E}")
    print(f"Standard error: {final_std_error:.6E}")
    print(f"Relative error: {final_relative_error:.4f}")
    print(f"95% Confidence interval: [{ci_lower:.6E}, {ci_upper:.6E}]")
    print(f"Successful trajectories: {success_count:,}/{n_episodes:,}")
    print(f"Success percentage: {100*success_count/n_episodes:.2f}%")
    
    # Write final results
    results_file.write("EVALUATION RESULTS:\n")
    results_file.write(f"Episodes: {n_episodes:,}\n")
    results_file.write(f"Evaluation time: {eval_time:.2f} seconds\n")
    results_file.write(f"Probability estimate: {final_estimate:.6E}\n")
    results_file.write(f"Standard error: {final_std_error:.6E}\n")
    results_file.write(f"Relative error: {final_relative_error:.4f}\n")
    results_file.write(f"95% CI: [{ci_lower:.6E}, {ci_upper:.6E}]\n")
    results_file.write(f"Successful trajectories: {success_count:,}/{n_episodes:,}\n")
    
    results_file.close()
    
    print(f"\nResults saved to:")
    print(f"  {results_file.name}")
    print(f"  {csv_filename}")
    print(f"  policy.yaml")
    print(f"  ./results/training_progress.png")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)

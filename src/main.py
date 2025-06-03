import sys
import json
import time
from prism_parser import parser
import math
from suppress import suppress_c_output
import yaml
import csv
from reinforce import train_reinforce, evaluate_reinforce

def main(json_path):
    # Configuration
    num_procs = 15              # number of processors
    N_train = 100_000           # training episodes
    batch_size = 500           # batch size for policy updates
    N = 1_000_000                 # total evaluation episodes
    
    results_file = open("./results/circuit.txt", "w")
    csv_filename = "./results/circuit.csv"
    
    # Load configuration
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    model_path = json_data['model_path']
    target_var = json_data['target_variable']
    target_value = int(json_data['target_value'])
    t_max = float(json_data['max_time'])
    
    print(f"Target: {target_var} >= {target_value}")
    print(f"Time limit: {t_max}")
    print("=" * 50)
    
    # Parse model
    with suppress_c_output():
        model = parser(model_path)
    target_index = model.species_to_index_dict[target_var]
    initial_state = model.get_initial_state()
    
    # Training phase
    start_time = time.time()
    print(f"Training REINFORCE agent with {N_train} episodes...")
    
    theta, state_visits, success_rates = train_reinforce(
        model=model,
        initial_state=initial_state,
        n_episodes=N_train,
        target_sp=target_index,
        target=target_value,
        T=t_max,
        batch_size=batch_size,  # Add this parameter
        n_workers=num_procs
    )
    
    print("=" * 50)
    print(f"Training finished. {N_train} episodes simulated.")
    print(f"Time spent training: {time.time() - start_time:.2f} seconds.")
    print(f"Final success rate: {success_rates[-1]:.3f}")
    
    # Save policy
    policy_params = {k: v.tolist() for k, v in theta.items()}
    with open('reinforce_policy.yaml', 'w') as f:
        yaml.dump(policy_params, f)
    
    results_file.write(f"Training finished. {N_train} episodes simulated.\n"
                      f"Time spent training: {time.time() - start_time:.2f} seconds.\n"
                      f"Final success rate: {success_rates[-1]:.3f}\n")
    
    # Evaluation phase
    print("Evaluating agent...")
    start_time = time.time()
    
    # Run N simulations and collect statistics every 1000 simulations
    total_N = N  # Total simulations to run
    checkpoint_interval = 1000
    
    # Initialize CSV file for statistics
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["simulations", "probability_estimate", "variance", "error"])
        
        all_weights = []
        
        print("Running evaluation with checkpoints every 1000 simulations...")
        
        # Run simulations in chunks
        for checkpoint in range(checkpoint_interval, total_N + 1, checkpoint_interval):
            # Determine how many more simulations we need
            simulations_needed = checkpoint - len(all_weights)
            
            if simulations_needed > 0:
                prob_estimate, weights = evaluate_reinforce(
                    theta=theta,
                    state_visits=state_visits,
                    model=model,
                    initial_state=initial_state,
                    n_episodes=simulations_needed,
                    target_sp=target_index,
                    target=target_value,
                    T=t_max,
                    n_workers=num_procs
                )
                all_weights.extend(weights)
            
            # Calculate statistics for current checkpoint
            n_sims = len(all_weights)
            
            # Probability estimate = sum of weights / number of simulations
            prob_est = sum(all_weights) / n_sims
            
            # Variance = E[X²] - (E[X])²
            # Where X is the weight, E[X] = prob_est, E[X²] = sum(weights²) / n_sims
            sum_weights_squared = sum(w**2 for w in all_weights)
            second_moment = sum_weights_squared / n_sims
            variance = second_moment - prob_est**2
            
            # Error = sqrt(variance / n_sims)
            error = math.sqrt(variance / n_sims) if variance > 0 else 0.0
            
            # Print to console
            print(f"Checkpoint {n_sims}: Prob={prob_est:.3E}, Var={variance:.3E}, Error={error:.3E}")
            
            # Write to CSV with scientific notation
            csv_writer.writerow([n_sims, f"{prob_est:.6E}", f"{variance:.6E}", f"{error:.6E}"])
            csvfile.flush()  # Ensure data is written immediately
    
    eval_time = time.time() - start_time
    final_prob_estimate = sum(all_weights) / len(all_weights)
    final_variance = (sum(w**2 for w in all_weights) / len(all_weights)) - final_prob_estimate**2
    final_error = math.sqrt(final_variance / len(all_weights)) if final_variance > 0 else 0.0
    
    print(f"\nEvaluation finished. {len(all_weights)} episodes simulated in {eval_time:.2f} seconds.")
    print(f"Final probability estimate = {final_prob_estimate:.6E}")
    print(f"Final variance = {final_variance:.6E}")
    print(f"Final standard error = {final_error:.6E}")
    print(f"Statistics saved to {csv_filename}")
    
    results_file.write(f"Evaluation: {len(all_weights)} episodes in {eval_time:.2f} seconds.\n"
                      f"Final probability estimate = {final_prob_estimate:.6E}\n"
                      f"Final standard error = {final_error:.6E}\n"
                      f"Statistics saved to {csv_filename}\n")
    
    # Count successful trajectories
    success_count = sum(1 for w in all_weights if w > 0)
    print(f"Total successful trajectories: {success_count}")
    
    results_file.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <json_config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)

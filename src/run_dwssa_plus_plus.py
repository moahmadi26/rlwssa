"""
Main script for dwSSA++ algorithm
Integrates with existing model format and configuration files
"""

import sys
import json
import time
import math
from dwssa_plus_plus import dwSSAPlusPlus
from prism_parser import parser
from utils import suppress_c_output


def main(json_path):
    """
    Main function to run dwSSA++ with configuration from JSON file.
    Uses the same format as your existing dwSSA implementation.
    """
    print("="*60)
    print("dwSSA++ (Doubly weighted SSA with Polynomial Leaping)")
    print("="*60)
    
    # Load configuration
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    model_path = json_data['model_path']
    target_var = json_data['target_variable']
    target_value = int(json_data['target_value'])
    t_max = float(json_data['max_time'])
    model_name = json_data.get('model_name', 'Unknown')
    
    # dwSSA++ specific parameters (can be added to JSON or use defaults)
    ld = json_data.get('ld', 3)           # Number of past data points
    sigma = json_data.get('sigma', 5)     # Slow convergence threshold  
    smax = json_data.get('smax', 20)      # Maximum iterations
    
    # Standard parameters
    num_procs = json_data.get('num_procs', 16)
    K_train = json_data.get('K_train', 100000)    # Trajectories per training iteration
    K_estimate = json_data.get('K_estimate', 1000000)  # Trajectories per estimation ensemble
    rho = json_data.get('rho', 0.01)              # Elite trajectory percentage
    num_ensembles = json_data.get('num_ensembles', 10)  # Number of estimation ensembles
    
    print(f"Model: {model_name}")
    print(f"Configuration: {json_path}")
    print(f"dwSSA++ Parameters: ld={ld}, sigma={sigma}, smax={smax}")
    print()
    
    # Parse model to get target index
    with suppress_c_output():
        model = parser(model_path)
    
    target_index = model.species_to_index_dict[target_var]
    initial_state = model.get_initial_state()
    
    print(f"Target: {target_var} (index {target_index}) = {target_value}")
    print(f"Initial state: {initial_state}")
    print(f"Time limit: {t_max}")
    print()
    
    # Initialize dwSSA++
    dwssa_pp = dwSSAPlusPlus(ld=ld, sigma=sigma, smax=smax)
    
    # Phase 1: Parameter Learning with dwSSA++
    print("PHASE 1: Parameter Learning")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        optimal_gamma, converged = dwssa_pp.compute_optimal_parameters(
            model_path=model_path,
            x0=initial_state,
            E=target_value,
            tf=t_max,
            K=K_train,
            rho=rho,
            target_index=target_index,
            num_procs=num_procs
        )
        
        learning_time = time.time() - start_time
        
        print(f"\nLearning phase completed in {learning_time:.2f} seconds")
        print(f"Converged: {converged}")
        print(f"Iterations used: {dwssa_pp.iteration_count}")
        print(f"Total training trajectories: {dwssa_pp.iteration_count * K_train:,}")
        
        print(f"\nLearned biasing vector:")
        for i, gamma in enumerate(optimal_gamma):
            print(f"  Reaction {i}: {gamma:.6f}")
        
        if not converged:
            print("\nWARNING: Parameter learning did not converge within maximum iterations")
            print("The algorithm may still provide useful biasing parameters")
        
    except Exception as e:
        print(f"Error during parameter learning: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Phase 2: Probability Estimation
    print(f"\nPHASE 2: Probability Estimation")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        results = dwssa_pp.estimate_probability(
            model_path=model_path,
            gamma=optimal_gamma,
            target_index=target_index,
            target_value=target_value,
            tf=t_max,
            K=K_estimate,
            num_ensembles=num_ensembles,
            num_procs=num_procs
        )
        
        estimation_time = time.time() - start_time
        
        print(f"\nEstimation completed in {estimation_time:.2f} seconds")
        print(f"Total estimation trajectories: {results['num_trajectories']:,}")
        
    except Exception as e:
        print(f"Error during probability estimation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Results Summary
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    total_time = learning_time + estimation_time
    total_trajectories = dwssa_pp.iteration_count * K_train + results['num_trajectories']
    
    print(f"Model: {model_name}")
    print(f"Target Event: {target_var} = {target_value}")
    print()
    print(f"Learning Performance:")
    print(f"  Iterations: {dwssa_pp.iteration_count}")
    print(f"  Converged: {converged}")
    print(f"  Learning time: {learning_time:.2f} seconds")
    print(f"  Training trajectories: {dwssa_pp.iteration_count * K_train:,}")
    print()
    print(f"Estimation Results:")
    print(f"  Probability estimate: {results['probability_estimate']:.6e}")
    print(f"  Standard error: {results['standard_error']:.6e}")
    if results['probability_estimate'] > 0:
        print(f"  Relative error: {results['standard_error']/results['probability_estimate']:.4f}")
        print(f"  95% Confidence interval: [{results['probability_estimate'] - 1.96*results['standard_error']:.6e}, "
              f"{results['probability_estimate'] + 1.96*results['standard_error']:.6e}]")
    else:
        print(f"  Relative error: N/A (no successful trajectories)")
        print(f"  95% Confidence interval: N/A")
    print()
    print(f"Computational Performance:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Total trajectories: {total_trajectories:,}")
    print(f"  Trajectories per second: {total_trajectories/total_time:.0f}")
    
    # Individual ensemble results
    print(f"\nEnsemble Results:")
    for i, est in enumerate(results['individual_estimates']):
        print(f"  Ensemble {i+1}: {est:.6e}")
    
    # Save results to file
    output_file = f"dwssa_plus_plus_results_{model_name.lower().replace(' ', '_')}.txt"
    with open(output_file, 'w') as f:
        f.write(f"dwSSA++ Results for {model_name}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Configuration: {json_path}\n")
        f.write(f"Target: {target_var} = {target_value}\n")
        f.write(f"Time limit: {t_max}\n\n")
        
        f.write(f"dwSSA++ Parameters:\n")
        f.write(f"  ld (history length): {ld}\n")
        f.write(f"  sigma (convergence threshold): {sigma}\n")
        f.write(f"  smax (max iterations): {smax}\n\n")
        
        f.write(f"Learning Results:\n")
        f.write(f"  Iterations: {dwssa_pp.iteration_count}\n")
        f.write(f"  Converged: {converged}\n")
        f.write(f"  Learning time: {learning_time:.2f} seconds\n")
        f.write(f"  Training trajectories: {dwssa_pp.iteration_count * K_train:,}\n\n")
        
        f.write(f"Optimal Biasing Vector:\n")
        for i, gamma in enumerate(optimal_gamma):
            f.write(f"  Reaction {i}: {gamma:.6f}\n")
        f.write("\n")
        
        f.write(f"Probability Estimation:\n")
        f.write(f"  Estimate: {results['probability_estimate']:.6e}\n")
        f.write(f"  Standard error: {results['standard_error']:.6e}\n")
        if results['probability_estimate'] > 0:
            f.write(f"  Relative error: {results['standard_error']/results['probability_estimate']:.4f}\n")
        else:
            f.write(f"  Relative error: N/A (no successful trajectories)\n")
        f.write(f"  Estimation time: {estimation_time:.2f} seconds\n")
        f.write(f"  Estimation trajectories: {results['num_trajectories']:,}\n\n")
        
        f.write(f"Total Performance:\n")
        f.write(f"  Total time: {total_time:.2f} seconds\n")
        f.write(f"  Total trajectories: {total_trajectories:,}\n")
        f.write(f"  Efficiency: {total_trajectories/total_time:.0f} trajectories/second\n")
    
    print(f"\nResults saved to: {output_file}")
    print("="*60)


def create_example_config():
    """
    Create an example configuration file for dwSSA++.
    """
    config = {
        "model_path": "../crns/yeast/yeast_polarization.json",
        "target_variable": "Cdc42I",
        "target_value": 30,
        "max_time": 100.0,
        "model_name": "Yeast Polarization",
        
        "ld": 3,
        "sigma": 5,
        "smax": 20,
        
        "num_procs": 16,
        "K_train": 100000,
        "K_estimate": 1000000,
        "rho": 0.01,
        "num_ensembles": 10
    }
    
    with open("config_dwssa_plus_plus.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Example configuration created: config_dwssa_plus_plus.json")
    print("Modify the parameters as needed for your specific model.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main_dwssa_plus_plus.py <config.json>")
        print("  python main_dwssa_plus_plus.py --create-example")
        print()
        print("For creating an example configuration:")
        print("  python main_dwssa_plus_plus.py --create-example")
        sys.exit(1)
    
    if sys.argv[1] == "--create-example":
        create_example_config()
    else:
        config_path = sys.argv[1]
        main(config_path)
import sys
import json
import time
import multiprocessing
from dwssa import dwssa_train, dwssa_continuous_worker
from prism_parser import parser
import numpy as np
from utils import suppress_c_output
from biasing import find_biasing
import math

def main(json_path):
    #############################################################################################
    num_procs = 16       # number of processors used for parallel execution

    # Hyperparameters
    N_train = 100_000     # total number of trajectories used to learn the q-table
    rho = 0.01           # the percentage of trajectories from a batch selected as the current event
    K = 4                # K ensebles of size N_train are used to learn biasing parameters
    N_batch = 10_000     # batch size for continuous simulation
    threshold = 0.01     # relative error threshold for stopping
    #############################################################################################
   
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    model_path = json_data['model_path']
    target_var = json_data['target_variable']
    target_value = int(json_data['target_value'])
    t_max = float(json_data['max_time'])
    with suppress_c_output():
        model = parser(model_path) 
    target_index = model.species_to_index_dict[target_var]
    
    start_time = time.time()
    biasing_vector_prev = [1.0] * len(model.get_reactions_vector())

    iteration = 0
    flag = True
    while(flag):
        iteration += 1
        print(f"Iteration {iteration} \n++++")
        biasing_vector = [0.0] * len(model.get_reactions_vector()) 
        flag = True
        for i in range(K):
            print(f"trial number {i+1}")
            N_vec = [N_train // num_procs 
                if j != num_procs - 1 
                else N_train - ((num_procs - 1)*(N_train // num_procs)) 
                for j in range(num_procs)]
            
            tasks = [(model_path, N_vec_j, t_max, target_index, target_value, biasing_vector_prev) 
                     for N_vec_j in N_vec]
            
            with multiprocessing.Pool(processes = num_procs) as pool:
                results = pool.starmap(dwssa_train, tasks)
            
            trajectories = [trajectory for result in results for trajectory in result]
            
            flag_inner, biasing_vector_inner = find_biasing(model, trajectories, rho, len(biasing_vector)
                                                            , target_index, target_value)
            
            flag = flag and flag_inner
            biasing_vector = [biasing_vector[j] + biasing_vector_inner[j]
                              for j in range(len(biasing_vector))]
            print(f"biasing : {biasing_vector_inner}")
            print("-----")
        
        biasing_vector = [biasing_vector[j]/ K for j in range(len(biasing_vector))]
        print(f"biaisng vector after iteration = {biasing_vector}")
        biasing_vector_prev = biasing_vector
        print("_"*50) 
   
    print(f"Learning phase finished. {K * N_train} trajectories were simulated.")
    print(f"Time spent learning: {time.time() - start_time} seconds.") 
    print(f"Running the dwSSA continuously with the biasing vector: \n{biasing_vector}\n...")

    start_time = time.time()

    # Continuous simulation phase using Welford's online algorithm
    n_simulations = 0
    mean_estimate = 0.0
    M2 = 0.0  # Sum of squared differences from mean
    relative_error = float('inf')
    
    print(f"Starting continuous simulation with relative error threshold = {threshold}")
    
    while relative_error > threshold:
        # Distribute batch across processors
        batch_per_proc = N_batch // num_procs
        N_vec = [batch_per_proc if j != num_procs - 1 
                else N_batch - ((num_procs - 1) * batch_per_proc) 
                for j in range(num_procs)]
        
        tasks = [(model_path, N_vec_j, t_max, target_index, target_value, biasing_vector) 
                 for N_vec_j in N_vec]
        
        with multiprocessing.Pool(processes=num_procs) as pool:
            results = pool.starmap(dwssa_continuous_worker, tasks)
        
        # Collect all weights from this batch and update statistics online
        batch_weights = [w for result in results for w in result]
        
        for w in batch_weights:
            n_simulations += 1
            delta = w - mean_estimate
            mean_estimate += delta / n_simulations
            delta2 = w - mean_estimate
            M2 += delta * delta2
        
        # Calculate variance and relative error
        if n_simulations > 1 and mean_estimate > 0:
            variance = M2 / (n_simulations - 1)
            std_error = math.sqrt(variance / n_simulations)
            relative_error = std_error / mean_estimate
        else:
            variance = 0.0
            std_error = 0.0
            relative_error = float('inf')
        
        # Print progress every 10 batches
        if (n_simulations // N_batch) % 10 == 0:
            print(f"Simulations: {n_simulations}, Estimate: {mean_estimate:.6e}, " +
                  f"Relative Error: {relative_error:.4f}")
    
    variance = M2 / (n_simulations - 1) if n_simulations > 1 else 0.0
    std_error = math.sqrt(variance / n_simulations) if n_simulations > 0 else 0.0
    
    print(f"\nSimulation completed in {time.time() - start_time} seconds.")
    print(f"Total trajectories simulated: {n_simulations}")
    print(f"Probability estimate: {mean_estimate}")
    print(f"Variance: {variance}")
    print(f"Standard error: {std_error}")
    print(f"Relative error: {relative_error}")

if __name__ == "__main__":
    config_path = sys.argv[1]
    main(config_path)

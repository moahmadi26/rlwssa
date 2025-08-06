import sys
import json
import time
import multiprocessing
from dwssa import dwssa_train, dwssa
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
    K = 4                # K ensebles of size N are used to estimate the probability of event
    N = 10_000_000          # number of trajectories used in each ensemble 
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
    print(f"Final biasing vector: {biasing_vector}")
    
    # Probability estimation phase commented out
    # print(f"Running the dwSSA with the biasing vector: \n{biasing_vector}\n...")

    # start_time = time.time()

    # p_vector = [None] * K

    # # run K ensembles of size N. Keep the probablity estimates in a vector
    # for i in range(K):
    #     N_vec = [N // num_procs 
    #             if j != num_procs - 1 
    #             else N - ((num_procs - 1)*(N // num_procs)) 
    #             for j in range(num_procs)]

    #     tasks = [(model_path, N_vec_j, t_max, target_index, target_value, biasing_vector) 
    #                       for N_vec_j in N_vec]
            
    #     with multiprocessing.Pool(processes = num_procs) as pool:
    #             results = pool.starmap(dwssa, tasks)
        
    #     m_1 = 0.0
    #     for result in results:
    #         m_1 += result

    #     p_vector[i] = m_1 / N
    # p_hat = sum(p_vector) / K
    # s_2 = [(p_vector[i] - p_hat)**2 for i in range(len(p_vector))]
    # s_2 = sum(s_2) / (K-1)
    # error = math.sqrt(s_2) / math.sqrt(K)

    # print(f"simulating {N} trajectories took {time.time() - start_time} seconds.") 
    # print(f"probability estimate = {p_hat}")
    # print(f"standard error = {error}")

if __name__ == "__main__":
    config_path = sys.argv[1]
    main(config_path)

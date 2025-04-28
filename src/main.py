import sys
import json
import time
import multiprocessing
from dwssa_q import dwssa_q_train, dwssa_q
from prism_parser import parser
import numpy as np

def main(json_path):
    #############################################################################################
    num_procs = 4       # number of processors used for parallel execution

    # Hyperparameters
    N_train = 10_000     # total number of trajectories used to learn the q-table
    batch_size = 1000    # the number of trajectories simulated before q-table is updated
    rho = 0.05           # the percentage of trajectories from a batch selected as the current event
    min_temp = 1         # minimum softmax temperature
    max_temp = 1         # maximum softmax temperature
    K = 4                # K ensebles of size N are used to estimate the probability of event
    N = 100_000          # number of trajectories used in each ensemble 
    #############################################################################################
   
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    model_path = json_data['model_path']
    target_var = json_data['target_variable']
    target_value = int(json_data['target_value'])
    t_max = float(json_data['max_time'])
    model = parser(model_path)
    target_index = model.species_to_index_dict[target_var]
    
    # Q-table is stored as a dictionary : (state, action) -> Q-value
    q_table = {}
    
    simulated_trajectories = 0
    start_time = time.time()
    
    while(simulated_trajectories <= N_train):
        N_vec = [batch_size // num_procs 
            if j != num_procs - 1 
            else N - ((num_procs - 1)*(bath_size // num_procs)) 
            for j in range(num_procs)]
        
        tasks = [(model_path, N_vec_j, t_max, min_temp, max_temp, target_index, target_value, q_table) 
                 for N_vec_j in N_vec]
        
        with multiprocessing.Pool(processes = num_procs) as pool:
            results = pool.starmap(dwssa_q_train, tasks)
        
        simulated_trajectories += batch_size

    
    print(f"Learning phase finished. {N_train} trajectories were simulated.")
    print(f"Time spent learning: {time.time() - start_time} seconds.") 
    print("Running the dwSSA with the learned q_table...")

    start_time = time.time()

    p_vector = [None] * K

    # run K ensembles of size N. Keep the probablity estimates in a vector
    for in in range(K):
        N_vec = [N // num_procs 
                if j != num_procs - 1 
                else N - ((num_procs - 1)*(N // num_procs)) 
                for j in range(num_procs)]

        tasks = [(model_path, N_vec_j, t_max, min_temp, max_temp, target_index, target_value, q_table) 
                          for N_vec_j in N_vec]
            
        with multiprocessing.Pool(processes = num_procs) as pool:
                results = pool.starmap(dwssa_q, tasks)
        
        m_1 = 0.0
        for result in results:
            m_1 += result[1]

        p_vector[i] = m_1 / N
    
    p_hat = sum(p_vector)
    s_2 = [(p_vector[i] - p_hat)**2 for i in range(len(p_vector))]
    s_2 = sum(s_2) / (K-1)
    error = math.sqrt(s_2) / math.sqrt(K)

    print(f"simulating {N} trajectories took {time.time() - start_time} seconds.") 
    print(f"probability estimate = {p-hat}")
    print(f"standard error = {error}")

if __name__ == "__main__":
    config_path = sys.argv[1]
    main(config_path)

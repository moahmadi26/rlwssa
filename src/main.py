import sys
import json
import time
import multiprocessing
from wssa_q import wssa_q_train, wssa_q
from mc_learn import update_q_table
from prism_parser import parser
import numpy as np
import math
from suppress import suppress_c_output
import yaml

def main(json_path):
    #############################################################################################
    num_procs = 15             # number of processors used for parallel execution

    # Hyperparameters
    N_train = 10_000        # total number of trajectories used to learn the q-table
    batch_size = 100          # the number of trajectories simulated before q-table is updated
    min_temp = 1.0             # minimum softmax temperature
    max_temp = 1.0             # maximum softmax temperature
    K = 4                      # K ensebles of size N are used to estimate the probability of event
    N = 10_000                 # number of trajectories used in each ensemble 
    epsilon = 0.0005             # epsilon value in epsilon greedy
    learning_rate = 0.01       # learning rate
    discount_factor = 1.0      # discount factor
    distance_multiplier = 1.0  # the multiplier in the distance term of the reward
    max_distance_reward = 1.0  # the maximum distance reward given to a trajectory
    #############################################################################################
   
    results_file = open("results.txt", "w")

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    model_path = json_data['model_path']
    target_var = json_data['target_variable']
    target_value = int(json_data['target_value'])
    t_max = float(json_data['max_time'])
    
    with suppress_c_output():
        model = parser(model_path)
    target_index = model.species_to_index_dict[target_var]
    
    # Q-table is stored as a dictionary : (state, action) -> Q-value
    q_table = {}
    
    simulated_trajectories = 0
    batch_number = 0
    start_time = time.time()
    
    while(simulated_trajectories < N_train):
        N_vec = [batch_size // num_procs 
            if j != num_procs - 1 
            else batch_size - ((num_procs - 1)*(batch_size // num_procs)) 
            for j in range(num_procs)]
        
        tasks = [(model_path, N_vec_j, t_max, min_temp, max_temp, target_index
                  , target_value, epsilon, max_distance_reward, q_table) 
                 for N_vec_j in N_vec]
       
        with multiprocessing.Pool(processes = num_procs) as pool:
            results = pool.starmap(wssa_q_train, tasks)

        trajectories = [item for sublist in results for item in sublist[0]] 
        q_table, sum_reward, average_distance = update_q_table(model, trajectories, q_table
                                                               , learning_rate, discount_factor
                                                               , distance_multiplier, target_index, target_value) 
        simulated_trajectories += batch_size
        batch_number += 1

        print(f"batch: {batch_number}")
        print(f"average terminal state distance : {average_distance}")
        print(f"sum rewards : {sum_reward}")
        print("-" * 50)

    
    print("=" * 50)
    print(f"Learning phase finished. {N_train} trajectories were simulated.")
    print(f"Time spent learning: {time.time() - start_time} seconds.") 
    print(f"Length of q-table = {len(q_table)}")
    print("Running the wSSA_q with the learned q_table...")
    results_file.write(f"Learning phase finished. {N_train} trajectories were simulated.\n"
                       f"Time spent learning: {time.time() - start_time} seconds. \n"
                       f"Length of q-table = {len(q_table)}"
                       f"Running the wSSA_q with the learned q_table... \n"
                          )
    
    with open('q_table.yaml', 'w') as f:
        yaml.dump(q_table, f)

    start_time = time.time()

    p_vector = [None] * K
    count = [None] * K

    # run K ensembles of size N. Keep the probablity estimates in a vector
    for i in range(K):
        N_vec = [N // num_procs 
                if j != num_procs - 1 
                else N - ((num_procs - 1)*(N // num_procs)) 
                for j in range(num_procs)]
        
        tasks = [(model_path, N_vec_j, t_max, min_temp, max_temp, target_index, target_value, q_table) 
                          for N_vec_j in N_vec]
            
        with multiprocessing.Pool(processes = num_procs) as pool:
                results = pool.starmap(wssa_q, tasks)
        
        m_1 = 0.0
        count_ = 0
        for result in results:
            m_1 += result[0]
            count_ += result[1]
        
        count[i] = count_ 

        p_vector[i] = m_1 / N
    
    p_hat = sum(p_vector)
    p_hat = p_hat / K
    s_2 = [(p_vector[i] - p_hat)**2 for i in range(K)]
    s_2 = sum(s_2) / (K-1)
    error = math.sqrt(s_2) / math.sqrt(K)

    print(f"simulating {K * N} trajectories took {time.time() - start_time} seconds.") 
    print(f"probability estimate = {p_hat}")
    print(f"standard error = {error}")
    results_file.write(f"simulating {N} trajectories took {time.time() - start_time} seconds. \n"
                       f"probability estimate = {p_hat} \n"
                       f"standard error = {error}"
                       )
    print(p_vector)
    print(count)
if __name__ == "__main__":
    config_path = sys.argv[1]
    main(config_path)

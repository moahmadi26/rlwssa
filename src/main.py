import sys
import json
import time
import multiprocessing
from wssa_q import wssa_q
from mc_learn import update_q_table
from prism_parser import parser
import numpy as np
import math
from suppress import suppress_c_output
import csv 

def main(json_path):
    #############################################################################################
    num_procs = 15                   # number of processors used for parallel execution

    # Hyperparameters
    batch_size = 1000                 # the number of trajectories simulated before q-table is updated
    N = 10_000_000                       # number of trajectories used in each ensemble 
    min_temp = 0.5                   # minimum softmax temperature
    max_temp = 1.5                   # maximum softmax temperature
    learning_rate = 0.10              # learning rate
    learning_rate_decay_rate = 1.0  # learning rate decay rate
    discount_factor = 0.9            # discount factor
    #############################################################################################
   
    csv_file = open("enzym.csv", "w", newline = "")
    writer = csv.writer(csv_file)

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
    moment_1 = 0
    moment_2 = 0
    count_observed_reactions = 0
    sum_biasing = [0.0] * len(model.get_reactions_vector())

    start_time = time.time()
    
    while(simulated_trajectories < N):
        N_vec = [batch_size // num_procs 
            if j != num_procs - 1 
            else batch_size - ((num_procs - 1)*(batch_size // num_procs)) 
            for j in range(num_procs)]
        
        tasks = [(model_path, N_vec_j, t_max, min_temp, max_temp, target_index
                  , target_value, q_table) 
                 for N_vec_j in N_vec]
       
        with multiprocessing.Pool(processes = num_procs) as pool:
            results = pool.starmap(wssa_q, tasks)

        trajectories = [item for sublist in results for item in sublist[0]] 
        sum_biasing = [sum_biasing[j] + sum([sublist[1][j] for sublist in results]) 
                       for j in range(len(sum_biasing))]
        count_observed_reactions += sum([sublist[2] for sublist in results])
        moment_1 += sum([sublist[3] for sublist in results])
        moment_2 += sum([sublist[4] for sublist in results])

        q_table, sum_reward, average_distance = update_q_table(model, trajectories, q_table
                                                               , learning_rate, discount_factor
                                                               , target_index, target_value) 
        simulated_trajectories += batch_size
        batch_number += 1
        
        p_est = moment_1 / simulated_trajectories
        var_est = (moment_2 / simulated_trajectories) - p_est**2
        err_est = np.sqrt(var_est / simulated_trajectories)
        average_biasing = [sum_biasing[j] / count_observed_reactions for j in range(len(sum_biasing))]
        print(f"batch: {batch_number}")
        print(f"learning_rate = {learning_rate}")
        print(f"simulated_trajectories = {simulated_trajectories}")
        print(f"probability estimate = {p_est}")
        print(f"variance = {var_est}")
        print(f"error = {err_est}")
        print(f"average biasing = {average_biasing}")
        print("-" * 50)

        learning_rate = max(learning_rate*learning_rate_decay_rate, 0.0001) 
        
        writer.writerow([batch_number, simulated_trajectories, p_est, var_est, err_est
                         , json.dumps(average_biasing)])

    csv_file.close()
if __name__ == "__main__":
    config_path = sys.argv[1]
    main(config_path)

import sys
import json
import time
import multiprocessing
from dwssa_q import dwssa_q_train
from prism_parser import parser
import numpy as np

def main(json_path):
    num_procs = 4       # number of processors used for parallel execution

    # Hyperparameters
    N = 10_000       # total episodes
    min_temp = 1      # minimum softmax temperature
    max_temp = 1      # maximum softmax temperature
    
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
    # For storing stats
    moment_1 = 0.0
    moment_2 = 0.0
    
    start_time = time.time()
        
    N_vec = [N // num_procs 
            if j != num_procs - 1 
            else N - ((num_procs - 1)*(N // num_procs)) 
            for j in range(num_procs)]

    tasks = [(model_path, N_vec_j, t_max, min_temp, max_temp, target_index, target_value, q_table) 
                      for N_vec_j in N_vec]
        
    # for task in tasks:
    #     for e in task:
    #         print(f"{type(e)}")

    # import inspect
    # 
    # print("Worker is:", dwssa_q_train)
    # print("  is it a bound method?", hasattr(dwssa_q_train, "__self__"))
    # print("  closure vars:", inspect.getclosurevars(dwssa_q_train))
    # quit()
    with multiprocessing.Pool(processes = num_procs) as pool:
            results = pool.starmap(dwssa_q_train, tasks)

    print(f"simulating {N} trajectories took {time.time() - start_time} seconds.") 

    for result in results:
        for j in result[0]:
            if j[2] == True:
                moment_1 += j[1]
                moment_2 += j[1]**2
    
    p_hat = moment_1 / N 
    var = (moment_2 / N) - (p_hat**2)
    error = np.sqrt(var / N)

    print(f"Done! Ran {N} episodes.")
    print(f"Probability estimate: {p_hat}, var={var}, error={error}")
    print(f"total time: {time.time() - start_time} seconds")





    quit()

        




if __name__ == "__main__":
    config_path = sys.argv[1]
    main(config_path)

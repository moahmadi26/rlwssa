import math
import random
from prism_parser import parser
from utils import get_reaction_rate, is_target
from reward import reward
from suppress import suppress_c_output

def get_propensities(model, state):
    reactions = model.get_reactions_vector()
    a = [get_reaction_rate(state, model, r_idx) 
             for r_idx in range(len(reactions))]
    a_0 = sum(a)
    return a, a_0

def get_bias(model, q_table, state):
    a, a_0 = get_propensities(model, state)
    nz_indices = [] 
    for i in range(len(a)):
        if a[i] > 0: nz_indices.append(i)
    
    q_values = []
    
    for action in range(len(a)):
        if (state, action) in q_table.keys():
            q_values.append(q_table[(state, action)])
        else:
            q_values.append(0.0)
    
    # q_max = max(q_values)
    # q_values = [q_values[i] - q_max for i in range(len(q_values))]
    exp_q = [math.exp(qv / temperature) if i in nz_indices else 0.0 for i, qv in enumerate(q_values)]
    sum_exp_q = sum(exp_q)
    return [e/sum_exp_q for e in exp_q]

def get_bias_train(model, q_table, state, temperature, epsilon = 0.0):
    a, a_0 = get_propensities(model, state)
    nz_indices = [] 
    for i in range(len(a)):
        if a[i] > 0: nz_indices.append(i)
    r = random.random()
    if r < epsilon:
        index = random.choice(nz_indices)
        return [1.0 if i == index else 0.0 for i in range(len(a))]
    
    q_values = []
    for action in range(len(a)):
        if (state, action) in q_table.keys():
            q_values.append(q_table[(state, action)])
        else:
            q_values.append(0.0)
    
    # q_max = max(q_values)
    # q_values = [q_values[i] - q_max for i in range(len(q_values))]
    
    exp_q = [math.exp(qv / temperature) if i in nz_indices else 0.0 for i, qv in enumerate(q_values)]
    sum_exp_q = sum(exp_q)
    return [e/sum_exp_q for e in exp_q]

def wssa_q_train (model_path, N, t_max, temperature, target_index
                  , target_value, epsilon, max_distance_reward, q_table):
    
    with suppress_c_output():
        model = parser(model_path)
    
    sum_reward = 0
    count_observed_reactions = 0
    sum_biasings = [0.0] * len(model.get_reactions_vector())
    trajectories = [None] * N

    for i in range(N):
        curr_traj = [] 
        x = model.get_initial_state()
        curr_traj.append(x)
        t = 0
        w = 1

        a, a_0 = get_propensities(model, x)
        
        # Q-table values to bias value
        bias = get_bias_train(model, q_table, x, temperature, epsilon)
        b = [a[i] * bias[i] for i in range(len(a))]
        b_0 = sum(b)

        flag = False
        while (t < t_max):
            if is_target(x, target_index, target_value):
                flag = True
                break
            
            r1 = random.random()
            r2 = random.random()
            tau = (1.0 / a_0) * math.log(1.0 / r1)
           
            temp_sum = 0
            mu = 0
            while temp_sum <= r2*b_0:
                temp_sum = temp_sum + b[mu]
                mu += 1
            mu -= 1
            
            curr_traj.append(mu)
            curr_traj.append(tau)
            count_observed_reactions += 1
            sum_biasings = [sum_biasings[i] + bias[i] for i in range(len(bias))]


            w = w * (a[mu] / b[mu]) * (b_0 / a_0)
            t += tau
            reaction_updates = model.get_reactions_vector()[mu]
            x_prev = x
            x = tuple(x[i] + reaction_updates[i] for i in range(len(x)))
            done = True if t >= t_max else False
            r = reward(model, x_prev, x, target_index, target_value, w, max_distance_reward, done)
            sum_reward += r
            a, a_0 = get_propensities(model, x)
        
            # Q-table values to bias value
            bias = get_bias_train(model, q_table, x, temperature, epsilon) 
            b = [a[i] * bias[i] for i in range(len(a))]
            b_0 = sum(b)

            curr_traj.append(r)
            curr_traj.append(x)
        
        if flag:
            trajectories[i] = (curr_traj, w, True)
        else:
            trajectories[i] = (curr_traj, 0, False)
    
    return trajectories, sum_biasings, count_observed_reactions, sum_reward 

def wssa_q (model_path, N, t_max, target_index, target_value, q_table):
    
    with suppress_c_output():
        model = parser(model_path)

    m_1 = 0
    count_success = 0
    
    for i in range(N):
        x = model.get_initial_state()
        t = 0
        w = 1

        a, a_0 = get_propensities(model, x)
        
        # Q-table values to bias value
        bias = get_bias(model, q_table, x, temperature)
        b = [a[i] * bias[i] for i in range(len(a))]
        b_0 = sum(b)

        while (t < t_max):
            if is_target(x, target_index, target_value):
                m_1 += w
                count_success += 1
                break
            
            r1 = random.random()
            r2 = random.random()
            tau = (1.0 / a_0) * math.log(1.0 / r1)
           
            temp_sum = 0
            mu = 0
            while temp_sum <= r2*b_0:
                temp_sum = temp_sum + b[mu]
                mu += 1
            mu -= 1
            
            w = w * (a[mu] / b[mu]) * (b_0 / a_0)
            t += tau
            reaction_updates = model.get_reactions_vector()[mu]
            x = tuple(x[i] + reaction_updates[i] for i in range(len(x)))
            a, a_0 = get_propensities(model, x)
        
            # Q-table values to bias value
            bias = get_bias(model, q_table, x) 
            b = [a[i] * bias[i] for i in range(len(a))]
            b_0 = sum(b)

    return m_1, count_success 

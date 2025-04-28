import math
import random
from prism_parser import parser
from utils import get_reaction_rate, is_target
from reward import reward

def get_propensities(model, state):
    reactions = model.get_reactions_vector()
    a = [get_reaction_rate(state, model, r_idx) 
             for r_idx in range(len(reactions))]
    a_0 = sum(a)
    return a, a_0

def get_bias(model, q_table, state, temperature):
    return [1] * len(model.get_reactions_vector())
    # reactions = model.get_reactions_vector()
    # q_values = []
    # for a in range(len(reactions)):
    #     if (state, a) in q_table.keys():
    #         q_values.append(q_table[(state, a)])
    #     else:
    #         q_values.append(0.0)
    # 
    # exp_q = [math.exp(qv / temperature) for qv in q_values]
    # sum_exp_q = sum(exp_q)
    # return [e/sum_exp_q for e in exp_q]

def dwssa_q_train (model_path, N, t_max, min_temp, max_temp, target_index, target_value, q_table):
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
        temperature = min_temp + ((max_temp-min_temp)* ((t_max - t)/t_max))
        bias = get_bias(model, q_table, x, temperature)
        b = [a[i] * bias[i] for i in range(len(a))]
        b_0 = sum(b)

        flag = False
        while (t < t_max):
            if is_target(x, target_index, target_value):
                flag = True
                break
            
            r1 = random.random()
            r2 = random.random()
            tau = (1.0 / b_0) * math.log(1.0 / r1)
           
            temp_sum = 0
            mu = 0
            while temp_sum <= r2*b_0:
                temp_sum = temp_sum + b[mu]
                mu += 1
            mu -= 1
            
            curr_traj.append(mu)
            count_observed_reactions += 1
            sum_biasings = [sum_biasings[i] + bias[i] for i in range(len(bias))]


            w = w * (1.0 / bias[mu]) * math.exp((b_0 - a_0) * tau)
            t += tau
            reaction_updates = model.get_reactions_vector()[mu]
            x = tuple(x[i] + reaction_updates[i] for i in range(len(x)))
            done = True if t >= t_max else False
            r = reward(x, w, done)
            sum_reward += r
            a, a_0 = get_propensities(model, x)
        
            # Q-table values to bias value
            temperature = min_temp + ((max_temp-min_temp)* ((t_max - t)/t_max))
            bias = get_bias(model, q_table, x, temperature) 
            b = [a[i] * bias[i] for i in range(len(a))]
            b_0 = sum(b)

            
            curr_traj.append(r)
            curr_traj.append(x)
        
        if flag:
            trajectories[i] = (curr_traj, w, True)
        else:
            trajectories[i] = (curr_traj, 0, False)
    
    return trajectories, sum_biasings, count_observed_reactions, sum_reward 

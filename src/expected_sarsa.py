from utils import is_target
import math
from wssa_q import get_bias_train  

def update_q_table(model, trajectories, q_table, learning_rate
                   , discount_factor, distance_multiplier, target_index, target_value, temperature):
    
    sum_reward = 0
    sum_distance = 0

    for element in trajectories:
        
        # episode is a list of [state_t, action_t, tau_t, reward_t, state_t+1, ...] from t=0..T-1
        episode = element[0]
        
        sum_distance += abs(episode[-1][target_index] - target_value)

        # Walk backward from the final time step and update the Q-table
        for i in range(0, len(episode) - 4, 4):
            # a trajectory has the format state_i, action_i, time_i, reward_i, state_(i+1)
            state, action, reward, next_state = episode[i], episode[i+1], episode[i+3], episode[i+4]
            sum_reward += reward
            
            if i == len(episode) - 5:
                  target = reward
            else:
                pi_next = get_bias_train(model, q_table, next_state, temperature)
                exp_q_next = sum([pi_next[r_idx] * q_table.get((next_state, r_idx), 0.0) for r_idx in range(len(pi_next))])
                entropy_next = -sum([p*math.log(p) for p in pi_next if p > 0.0])
                target = reward + discount_factor*(exp_q_next + temperature*entropy_next)

            old_q = q_table.get((state, action), 0.0)
            new_q = old_q + learning_rate * (target - old_q)
            
            q_table[(state, action)] = new_q
        
    
    return q_table, sum_reward, sum_distance/len(trajectories) 

def stopping_criteria(curr_reward, past_rewards, window_size, min_delta):
    mean_reward = sum(past_rewards) / len(past_rewards)
    delta = curr_reward - mean_reward
#     if delta / curr_reward < min_delta:
#         return true, [curr_reward if i == 0 else past_reward[i-1] for 
    

import math
from utils import is_target

def reward(model, prev_state, curr_state, target_index, target_value, weight, max_distance_reward, done):
    
    # terminal reward
    # if not (done or is_target(curr_state, target_index, target_value)):
    #     dist_reward = 0
    # else:
    #     terminal_dist = abs(curr_state[target_index] - target_value)
    #     dist_reward = max_distance_reward * math.exp(-terminal_dist)
    
    

    # step reward
    curr_dist = abs(target_value - curr_state[target_index])
    prev_dist = abs(target_value - prev_state[target_index])
    dist_reward = 0
    if curr_dist > prev_dist:
        dist_reward = -1
    elif curr_dist == prev_dist:
        dist_reward = 0
    else:
        dist_reward = prev_dist - curr_dist
    
    if not(is_target(curr_state, target_index, target_value)):
        return dist_reward
    return dist_reward + math.exp(weight)







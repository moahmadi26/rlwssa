import math
from utils import is_target

def reward(model, prev_state, curr_state, target_index, target_value, weight, done):
    
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







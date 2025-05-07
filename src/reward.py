import math
from utils import is_target

def shape_reward(prev_state, curr_state, target_index, target_value):
  
    # step reward
    curr_dist = abs(target_value - curr_state[target_index])
    prev_dist = abs(target_value - prev_state[target_index])
    
    if curr_dist > prev_dist:
        dist_reward = -1
    elif curr_dist == prev_dist:
        dist_reward = 0
    else:
        dist_reward = prev_dist - curr_dist
    
    return dist_reward 

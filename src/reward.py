import math

def reward(model, prev_state, curr_state, target_index, target_value, weight, done):
    initial_value = model.get_initial_state()[target_index]
    current_value = curr_state[target_index]
    previous_value = prev_state[target_index]
    sign = target_value - initial_value
    diff = current_value - previous_value
    if sign < 0:
        if diff == 0:
            dist_reward = 0
        elif diff < 0:
            dist_reward = +1
        else:
            dist_reward = -1
    else:
        if diff == 0:
            dist_reward = 0
        elif diff > 0:
            dist_reward = +1 
        else:
            dist_reward = -1

    return dist_reward #+ math.log(weight)

from utils import is_target
import math

def update_q_table(model, trajectories, q_table, learning_rate
                   , discount_factor, target_index, target_value
                   , q_values_range, max_q_table_size):
    """
    Perform a Monte Carlo update on each stored trajectory in trajectories 

    We'll do a "backward pass" from the end of each trajectory, accumulating returns.
    Then we do an MC update to Q(s,a).
    """
    sum_reward = 0
    sum_distance = 0

    for element in trajectories:
        
        # episode is a list of [state_t, action_t, tau_t, reward_t, state_t+1, ...] from t=0..T-1
        episode = element[0]
        
        sum_distance += abs(episode[-1][target_index] - target_value)

        # We'll accumulate G from the end.
        G = 0.0
        
        # Walk backward from the final time step and update the Q-table
        start = len(episode) - 5

        for i in (range(start, -1, -4)):
            state, action, reward = episode[i], episode[i+1], episode[i+3]
           
            sum_reward += reward
            
            G = discount_factor * G + reward  # accumulate discounted return


            # Now do the incremental MC update
            old_q = q_table.get((state, action), 0.0)
            # target = G, so:
            new_q = old_q + learning_rate * (G - old_q)
            
            if new_q > q_values_range[1]:
                new_q = q_values_range[1]
            elif new_q < q_values_range[0]:
                new_q = q_values_range[0]

            q_table[(state, action)] = new_q
    
    return q_table, sum_reward, sum_distance/len(trajectories) 


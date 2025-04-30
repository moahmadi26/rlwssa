from utils import is_target
import math

def update_q_table(model, trajectories, q_table, learning_rate
                   , discount_factor, distance_multiplier, target_index, target_value):
    """
    Perform a Monte Carlo update on each stored trajectory in trajectories 

    We'll do a "backward pass" from the end of each trajectory, accumulating returns.
    Then we do an MC update to Q(s,a).
    """
    sum_reward = 0
    
    for element in trajectories:
        
        # episode is a list of [state_t, action_t, tau_t, reward_t, state_t+1, ...] from t=0..T-1
        episode = element[0]
        
        # We'll accumulate G from the end.
        G = 0.0
        
        weight = 0
        var_multiplier = 0
        if element[2]: 
            var_multiplier = 0.01
            weight = element[1]
        
        # Walk backward from the final time step and update the Q-table
        start = len(episode) - 5
        flag = True

        for i in (range(start, -1, -4)):
            state, action, reward = episode[i], episode[i+1], episode[i+3]
            if flag:
                if weight > 0:
                    reward = reward # + (-math.log(weight))
                flag = False
            
            q_value = q_table.get((state, action), 0.0)
            num_reactions = len(model.get_reactions_vector())
            Z = sum([math.exp(q_table.get((state, i), 0.0)) for i in range(num_reactions)])

            reward = reward * distance_multiplier
            reward = reward # + (var_multiplier * q_value - math.log(Z))
            sum_reward += reward
            G = discount_factor * G + reward  # accumulate discounted return


            # Now do the incremental MC update
            old_q = q_table.get((state, action), 0.0)
            # target = G, so:
            new_q = old_q + learning_rate * (G - old_q)
            
            q_table[(state, action)] = new_q
    return q_table

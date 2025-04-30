
def update_q_table(trajectories, q_table, learning_rate, discount_factor):
    """
    Perform a Monte Carlo update on each stored trajectory in trajectories 

    We'll do a "backward pass" from the end of each trajectory, accumulating returns.
    Then we do an MC update to Q(s,a).
    """
    sum_reward = 0
    for element in trajectories:
        episode = element[0]
        # episode is a list of [state_t, action_t, tau_t, reward_t, state_t+1, ...] from t=0..T-1
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
            
            q_table[(state, action)] = new_q
    return q_table

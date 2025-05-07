from utils import is_target
import math
from wssa_q import softmax

def reinforce_update(model, trajectories, q_table, learning_rate
                   , discount_factor, target_index, target_value
                   , temperature, use_baseline = True):
   
    """
    Performs *one* REINFORCE (Monte-Carlo policy-gradient) pass over `batch`
    and updates the Q-table **in place**.

    Returns the updated Q for convenience.
    """

    # -------------------------------------------------
    # 1) optional baseline  b = weighted-mean return
    #    (good variance reduction; keeps scale of W*G_t reasonable)
    # -------------------------------------------------
    if use_baseline:
        total_weight = sum(traj[1] for traj in trajectories)
        if total_weight == 0:
            b = 0.0
        else:
            # each traj's undiscounted return   G_0 = sum(r_t)
            sum_rewards_per_traj = [sum([traj[0][i+3] for i in range(0, len(traj) - 4, 4)]) 
                                    for traj in trajectories]
            weighted_returns = sum(
                traj[1] * sum_rewards_per_traj[i]
                for i, traj in enumerate(trajectories) 
            )
            b = weighted_returns / total_weight
     
    else:
        b = 0.0
     
    # -------------------------------------------------
    # 2) loop over trajectories and apply gradient
    # -------------------------------------------------
    sum_distance = 0
    sum_reward = 0
    count = 0
    for traj in trajectories:
        count+=1
        start = 0 
        end = len(traj[0]) - 4

        W        = traj[1]              # importance weight
        rewards  = [traj[0][i+3] for i in range(start, end, 4)]
        states   = [traj[0][i] for i in range(start, end, 4)]
        actions  = [traj[0][i+1] for i in range(start, end, 4)]

        sum_distance += traj[0][-1][target_index] - target_value
        sum_reward += sum(rewards)
        # ---- 2.a  compute discounted returns  G_t  ----
        G_t_list = [0.0] * len(rewards)
        G = 0.0                                # running return
        for t in reversed(range(len(rewards))):
            G = rewards[t] + discount_factor * G
            G_t_list[t] = G - b                # baseline subtraction 
        
        # ---- 2.b  update every (state, action) seen ----
        for s, a, G_t in zip(states, actions, G_t_list):
            # ensure state exists in table (initialise lazily if needed)
            if (s, a)  not in q_table:
                q_table[(s, a)] = 0.0

            probs = softmax(model, q_table, s, temperature)
            for j in range(len(probs)):         # j = reaction index
                grad = (1.0 if j == a else 0.0) - probs[j]

                # REINFORCE update:  Q ← Q + α · W · G_t · grad
                if (s, j)  not in q_table:
                    q_table[(s, j)] = 0.0

                q_table[(s, j)] += learning_rate * W * G_t * grad



    return q_table, sum_reward, sum_distance/len(trajectories) 

def stopping_criteria(curr_reward, past_rewards, window_size, min_delta):
    mean_reward = sum(past_rewards) / len(past_rewards)
    delta = curr_reward - mean_reward
#     if delta / curr_reward < min_delta:
#         return true, [curr_reward if i == 0 else past_reward[i-1] for 
    

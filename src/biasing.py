from utils import get_reaction_rate
import math

def find_biasing(model, trajectories, rho, num_reactions):
    trajectories.sort(key = lambda inner : inner[2])
    threshold = math.ceil(rho * len(trajectories))
    trajectories = trajectories[:threshold]

    count_reactions = [0] * num_reactions
    expected_count_reactions = [0.0] * num_reactions
    weighted_sum_dwssa = [0.0] * num_reactions
    weighted_sum_ssa = [0.0] * num_reactions

    flag = True
    for trajectory in trajectories:
        curr_traj = trajectory[0]
        if trajectory[2] != 0:
            flag = False
        
        for i in range(len(curr_traj)-1):
            if i%3 == 0:
               state = curr_traj[i] 
               # if state == curr_traj[3]:
               #     break
               # else:
               #     print("here")
            if i%3 == 1:
               reaction = curr_traj[i]
               count_reactions[reaction] +=1 
            if i%3 == 2:
               time = curr_traj[i]
               for j in range(num_reactions):
                   expected_count_reactions[j] += time * get_reaction_rate(state, model, j)
        
        weighted_sum_dwssa = [weighted_sum_dwssa[i] + (trajectory[1] * count_reactions[i])
                              for i in range(num_reactions)]
        weighted_sum_ssa = [weighted_sum_ssa[i] + (trajectory[1] * expected_count_reactions[i])
                            for i in range(num_reactions)]
    biasing_vector = [weighted_sum_dwssa[i] / weighted_sum_ssa[i] for i in range(num_reactions)]
    return flag, biasing_vector

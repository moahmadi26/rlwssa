import math
import random
from prism_parser import parser
from utils import get_reaction_rate, is_target
from reward import reward
from utils import suppress_c_output
def get_propensities(model, state):
    reactions = model.get_reactions_vector()
    a = [get_reaction_rate(state, model, r_idx) 
             for r_idx in range(len(reactions))]
    a_0 = sum(a)
    return a, a_0

def dwssa_train (model_path, N, t_max, target_index, target_value, biasing_vector):
    with suppress_c_output():
        model = parser(model_path)

    trajectories = [None] * N

    for i in range(N):
        curr_traj = [] 
        x = model.get_initial_state()
        min_dist = abs(x[target_index] - target_value)
        min_dist_state = x
        curr_traj.append(x)
        t = 0
        w = 1

        a, a_0 = get_propensities(model, x)
        
        b = [a[i] * biasing_vector[i] for i in range(len(a))]
        b_0 = sum(b)

        while (t < t_max):
            if is_target(x, target_index, target_value):
                break
            
            r1 = random.random()
            r2 = random.random()
            tau = (1.0 / b_0) * math.log(1.0 / r1)
           
            temp_sum = 0
            mu = 0
            while temp_sum <= r2*b_0:
                temp_sum = temp_sum + b[mu]
                mu += 1
            mu -= 1
            
            curr_traj.append(mu)
            curr_traj.append(tau)

            w = w * (1.0 / biasing_vector[mu]) * math.exp((b_0 - a_0) * tau)
            t += tau
            reaction_updates = model.get_reactions_vector()[mu]
            x = tuple(x[i] + reaction_updates[i] for i in range(len(x)))
            if abs(x[target_index] - target_value) < min_dist:
                min_dist = x[target_index] - target_value
                min_dist_state = x
            a, a_0 = get_propensities(model, x)
        
            b = [a[i] * biasing_vector[i] for i in range(len(a))]
            b_0 = sum(b)

            curr_traj.append(x)
        
        trajectories[i] = (curr_traj, w, min_dist, min_dist_state)
    return trajectories

def dwssa (model_path, N, t_max, target_index, target_value, biasing_vector):
    with suppress_c_output():
        model = parser(model_path)

    for i in range(N):
        x = model.get_initial_state()
        t = 0
        w = 1
        m_1 = 0

        a, a_0 = get_propensities(model, x)
        
        b = [a[i] * biasing_vector[i] for i in range(len(a))]
        b_0 = sum(b)

        while (t < t_max):
            if is_target(x, target_index, target_value):
                m_1 += w
                break
            
            r1 = random.random()
            r2 = random.random()
            tau = (1.0 / b_0) * math.log(1.0 / r1)
           
            temp_sum = 0
            mu = 0
            while temp_sum <= r2*b_0:
                temp_sum = temp_sum + b[mu]
                mu += 1
            mu -= 1
            
            w = w * (1.0 / biasing_vector[mu]) * math.exp((b_0 - a_0) * tau)
            t += tau
            reaction_updates = model.get_reactions_vector()[mu]
            x = tuple(x[i] + reaction_updates[i] for i in range(len(x)))
            a, a_0 = get_propensities(model, x)
        
            b = [a[i] * biasing_vector[i] for i in range(len(a))]
            b_0 = sum(b)

    return m_1 

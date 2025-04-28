import math

def reward(state, weight, done):
    return 0 
# def get_reward(self, current_state, previous_state, weight):
#     initial_value = self.model.get_initial_state()[self.target_index]
#     current_value = current_state[self.target_index]
#     previous_value = previous_state[self.target_index]
#     sign = self.target_value - initial_value
#     if not is_target(current_state, self.target_index, self.target_value):
#         if sign < 0:
#             return current_value - previous_value
#         else:
#             return previous_value - current_value
#     else:
#         if weight > 1:
#             divergence = weight
#         else: 
#             divergence = 1.0/weight
#         return self.hyper_A * (math.exp(0 - divergence))
    
# def get_reward_distance_terminal_state(model
#                         , current_state
#                         , target_index
#                         , target_value
#                         , done
#                         ):
#     if not done:
#         return 0
    
#     initial_value = model.get_initial_state()[target_index]
#     current_value = current_state[target_index]
#     sign = target_value - initial_value
#     if sign < 0:
#         return 1.0 / math.exp(current_value - target_value)
#     else:
#         return 1.0 / math.exp(target_value - current_value)
    

    
# def get_cost_distance_target_state(model
#                                          , current_state
#                                          , target_index
#                                          , target_value
#                                          , done
#                                          , decay_rate
#                                          ):
#     if not done:
#         return 0
    
#     initial_value = model.get_initial_state()[target_index]
#     current_value = current_state[target_index]
#     sign = target_value - initial_value
#     if sign < 0:
#         return (current_value - target_value)
#     else:
#         return (target_value - current_value)
    

def get_cost_distance_immediate(model
                                , current_state
                                , previous_state
                                , target_index
                                , target_value):
    initial_value = model.get_initial_state()[target_index]
    current_value = current_state[target_index]
    previous_value = previous_state[target_index]
    sign = target_value - initial_value
    diff = current_value - previous_value
    if sign < 0:
        if diff == 0:
            return 0
        elif diff < 0:
            return +1
        else:
            return -1
    else:
        if diff == 0:
            return 0
        elif diff > 0:
            return +1
        else:
            return -1
        

# def get_cost_distance_exp_target_state_and_immediate(model
#                                          , current_state
#                                          , previous_state
#                                          , target_index
#                                          , target_value
#                                          , done
#                                          , decay_rate
#                                          ):
#     initial_value = model.get_initial_state()[target_index]
#     current_value = current_state[target_index]
#     previous_value = previous_state[target_index]
#     sign = target_value - initial_value
#     diff = current_value - previous_value

#     if not done:
#         if sign < 0:
#             if diff == 0:
#                 return 0
#             elif diff < 0:
#                 return +1
#             else:
#                 return -1
#         else:
#             if diff == 0:
#                 return 0
#             elif diff > 0:
#                 return +1
#             else:
#                 return -1
    
    
#     if sign < 0:
#         return 1 - math.exp(decay_rate * (current_value - target_value))
#     else:
#         return 1 - math.exp(decay_rate * (target_value - current_value))
    


# def get_terminal_state_values(model
#                                          , current_state
#                                          , target_index
#                                          , target_value
#                                          , done
#                                          , decay_rate
#                                          ):
#     if not done: # if the episode was not done (current state was not terminal)
#         return 0
    
#     initial_value = model.get_initial_state()[target_index]
#     current_value = current_state[target_index]
#     sign = target_value - initial_value
#     if sign < 0:
#         return 1 - math.exp(decay_rate * (current_value - target_value))
#     else:
#         return 1 - math.exp(decay_rate * (target_value - current_value))

def get_total_outgoing_rate(var_values, model):
    rate = 0
    for i in range(len(model.get_reactions_vector())):
        rate = rate + model.get_reaction_rate(var_values, i)
    return rate

def get_reaction_rate(var_values, model, reaction):
    return model.get_reaction_rate(var_values, reaction)

def is_target(var_values, target_index, target_value):
    if var_values[target_index] == target_value:
        return True
    return False


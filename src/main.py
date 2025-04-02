import sys
import json
import random
import numpy as np
import time
from prism_parser import parser

from agent import MCAgent
from environment import Environment

def main(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    model_path = json_data['model_path']
    target_var = json_data['target_variable']
    target_value = int(json_data['target_value'])
    t_max = float(json_data['max_time'])
    model = parser(model_path)
    target_index = model.species_to_index_dict[target_var]

    # Hyperparameters
    N = 100_000         # total episodes
    batch_size = 1000  # number of runs or episodes after which we do some update
    alpha = 0.1
    gamma = 0.95
    tau = 1.0
    decay_rate = 0.5

    # Create environment
    env = Environment(
        model=model,
        target_index=target_index,
        target_value=target_value,
        t_max=t_max,
        decay_rate = decay_rate,
    )

    # Create agent
    num_reactions = len(model.get_reactions_vector())
    agent = MCAgent(actions_size=num_reactions
                    , alpha=alpha
                    , gamma=gamma
                    , tau=tau
                    , batch_size=batch_size)

    # For storing stats
    moment_1 = 0.0
    moment_2 = 0.0
    overhead_time = 0.0
    count_target = 0
    total_reward = 0

    total_episodes = N
    episode = 0
    
    count_observed_reactions = 0
    average_policy = [0.0 for i in range(len(model.get_reactions_vector()))]

    start_time = time.time()

    while episode < total_episodes:
        # Start a new episode
        state, propensities = env.reset()
        weight = 1.0
        
        # Keep track if we reached the target in this episode
        done = False
        temperature = 1.5

        while not done:
            # Agent picks an action (reaction)
            action, w, average_policy = agent.select_action(state, propensities, average_policy, temperature)
            weight = weight * w

            # Step environment
            next_state, t, done, target, reward, propensities = env.step(action, weight)
            total_reward = total_reward + reward
            count_observed_reactions += 1
            rem_time = t_max - t
            if rem_time < 0:
                rem_time = 0
            temperature = 0.5 + (rem_time / t_max)
            # Store for agent
            agent.store_transition(state, action, reward)

            # Move to next state
            state = next_state

        agent.end_episode()
        if target:
            moment_1 += weight
            moment_2 += (weight**2)
            count_target += 1

        # End of episode
        episode += 1
        # Periodically print out progress, partial estimates
        if episode % batch_size == 0:
            p_est = moment_1 / episode
            var_est = (moment_2 / episode) - p_est**2
            err_est = np.sqrt(var_est / episode)
            print(f"Episode {episode}, p_est={p_est}, err_est={err_est}")
            print(f"Percentage of trajectories reaching target={count_target/episode}")
            # print(f"Average reward per trajectory={total_reward/episode}")
            print(f"Size of Q-table={len(agent.Q_table)}")
            print("=" * 50)

    end_time = time.time()

    # Final results
    p_hat = moment_1 / total_episodes
    var = (moment_2 / total_episodes) - (p_hat**2)
    error = np.sqrt(var / total_episodes)
    

    print(f"Done! Ran {total_episodes} episodes.")
    print(f"Probability estimate: {p_hat}, var={var}, error={error}")
    print(f"Total time: {end_time - start_time:.2f} sec")
    print(f"average policy: {[average_policy[i]/count_observed_reactions for i in range(len(average_policy))]}")

    ### temporary
    print("=" * 50)
    print("=" * 50)
    ls = list(env.state_count.values())
    import statistics
    mean = statistics.mean(ls)
    median = statistics.median(ls)
    print(f"average number of times each state is visited = {mean}")
    print(f"median of the number of times each state is visited = {median}")
    print("=" * 50)
    # print("number of times each state is visited:")
    # for key, value in env.state_count.items():
    #     print(f"{key} : {value}")
    # print("=" * 50)
    from collections import defaultdict
    q_table = defaultdict(float)
    for key,value in agent.Q_table.items():
        state = key[0]
        reaction = key[1]
        if state not in q_table:
            q_table[key[0]] = [None] * len(model.get_reactions_vector())
        q_table[key[0]][key[1]] = value
    
    # for key, value in q_table.items():
    #     print (f"{key} : {value}")
    # print ("=" * 50)
    print(f"number of observed states = {len(q_table.keys())}")
    ###
    print("\n" * 50)
    for key, value in agent.Q_table.items():
        print(f"{key} : {value}")

if __name__ == "__main__":
    config_path = sys.argv[1]
    main(config_path)

import random
from collections import defaultdict
import math

class MCAgent:
    def __init__(self, actions_size, alpha=0.01, gamma=0.99, tau=1, batch_size=1000):
       
        self.actions_size = actions_size
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.previous_batch_average_terminal_values = 0

        # Q_table: dict[(state, action)] -> float
        self.Q_table = defaultdict(float)

        # For batch updates:
        self.episodes_memory = []  # will hold multiple episodes
        self.current_episode = []  # will hold transitions for 1 episode
        

    def select_action(self, state, propensities, average_policy, temperature):
        q_values = []
        for a in range(self.actions_size):
            if (state, a) in self.Q_table.keys():
                q_values.append([a, self.Q_table[(state, a)]])
            else:
                if len(self.Q_table) < 100_000_000:
                    self.Q_table[(state, a)] = 0.0
                    q_values.append([a, 0.0])
    
        # Boltzmann with temperature 'temp'
        temp = temperature 
        exp_q = [math.exp(qv[1] / temp) for qv in q_values]
        exp_q_p = [eqv * propensities[i] for i, eqv in enumerate(exp_q)]
        sum_exp_q_p = sum(exp_q_p)
        probs = [v / sum_exp_q_p for v in exp_q_p]
        # exp_q_p = [eqv[1] * propensities[i] for i, eqv in enumerate(q_values)]
        # sum_exp_q_p = sum(exp_q_p)
        # probs = [v / sum_exp_q_p for v in exp_q_p]
        r = random.uniform(0, 1)
        j = 0
        temp_sum = 0
        while temp_sum <= r:
            temp_sum = temp_sum + (probs[j])
            j = j+1
        j = j-1
        
        return (q_values[j][0]
                , ((propensities[j] / exp_q_p[j]) * (sum_exp_q_p / sum(propensities)))
                , [average_policy[i] + (exp_q[i]/sum(exp_q)) for i in range(len(average_policy))]
                )
        

    def store_transition(self, state, action, reward):
        """
        We'll finalize the episode when we hit a terminal state.
        """
        self.current_episode.append((state, action, reward))

    def end_episode(self):
        """
        The episode has terminated. 
        We store the entire episode in episodes_memory.
        If we have reached batch_size episodes, do MC update.
        """
        self.episodes_memory.append(self.current_episode)
        self.current_episode = []  # reset for next episode

        if len(self.episodes_memory) >= self.batch_size:
            self.update_Q_table()
            self.episodes_memory = []  # clear after batch update

    def update_Q_table(self):
        """
        Perform a Monte Carlo update on each stored episode in episodes_memory.

        We'll do a "backward pass" from the end of each episode, accumulating returns.
        Then we do an MC update to Q(s,a).
        """
        sum_terminal_values = 0
        sum_reward = 0
        for episode in self.episodes_memory:
            # episode is a list of (state, action, reward) from t=0..T-1
            # We'll accumulate G from the end.
            G = 0.0
            # Walk backward from the final time step and update the Q-table
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                if reward != 0:
                    sum_terminal_values += reward
                    reward = reward - self.previous_batch_terminal_values
                    sum_reward += reward
                G = self.gamma * G + reward  # accumulate discounted return


                # Now do the incremental MC update
                old_q = self.Q_table[(state, action)]
                # target = G, so:
                new_q = old_q + self.alpha * (G - old_q)
                if new_q < -150:
                    new_q = -150
                elif new_q > 150:
                    new_q = 150
                self.Q_table[(state, action)] = new_q
            
        self.previous_batch_terminal_values = sum_terminal_values / self.batch_size
        print(f"average terminal state values (V_b) in the batch = {self.previous_batch_terminal_values}")
        print(f"average reward in the batch = {sum_reward / self.batch_size}")
        sum_q_values = 0
        for value in self.Q_table.values():
            sum_q_values += value
        print(f"average of Q-values in Q-table at the end of this batch = {sum_q_values/ len(self.Q_table)}")
import random
import math
from utils import get_reaction_rate, is_target
from reward import *

class Environment:
    def __init__(
        self,
        model,
        target_index,
        target_value,
        t_max,
        decay_rate
    ):
        
        self.model = model
        self.target_index = target_index
        self.target_value = target_value
        self.t_max = t_max
        self.decay_rate = decay_rate

        # temporary
        from collections import defaultdict
        self.state_count = defaultdict(int) 
        #

        # Internal state variables
        self.reset()

    def reset(self):
        """Reset the environment to start a new episode (trajectory)."""
        self.current_state = self.model.get_initial_state()
        #
        if self.current_state in self.state_count:
            self.state_count[self.current_state] = self.state_count[self.current_state] + 1
        else:
            self.state_count[self.current_state] = 1
        #
        self.current_time = 0.0
        self.done = False
        self.target = False
        return self.current_state, self.get_propensities()[0]

    def get_propensities(self):
        """
        Returns the actual reaction rates (a) and sum of rates (a_0)
        for the current state.
        """
        reactions = self.model.get_reactions_vector()
        a = [get_reaction_rate(self.current_state, self.model, r_idx) 
             for r_idx in range(len(reactions))]
        a_0 = sum(a)
        return a, a_0

    def step(self, action, weight):
        """
        Perform one reaction step:
         1) Sample the time until next reaction (tau).
         2) Update state, time, reward, done, target.

        Returns
        -------
        next_state : tuple
            The new state after the reaction.
        done : bool
            Whether we've reached target or t > t_max.
        reward : float
            The reward obtained this step.
        propeonsities : list
            List of reaction propensities in next_state
        """

        # 1) compute actual propensities
        a, a_0 = self.get_propensities()

        # sample the time to the next event
        r1 = random.random()
        tau = (1.0 / a_0) * math.log(1.0 / r1)
        new_time = self.current_time + tau

        # 2) update the state
        # Agent explicitly provides the reaction index:
        chosen_reaction = action

        reaction_updates = self.model.get_reactions_vector()[chosen_reaction]
        next_state = tuple(
            self.current_state[i] + reaction_updates[i] 
            for i in range(len(self.current_state))
        )
        
        self.previous_state = self.current_state
        self.current_state = next_state
        self.current_time = new_time

        #
        if self.current_state in self.state_count:
            self.state_count[self.current_state] = self.state_count[self.current_state] + 1
        else:
            self.state_count[self.current_state] = 1
        #
        
        reached_target = (is_target(self.current_state, self.target_index, self.target_value) 
                          and self.current_time <= self.t_max)
        if reached_target:
            self.done = True
            self.target = True
            to_return = (
                    self.current_state
                    , self.current_time
                    , self.done
                    , self.target
                    # , get_cost_distance_immediate (
                    #     self.model
                    #     , self.current_state
                    #     , self.previous_state
                    #     , self.target_index
                    #     , self.target_value        
                    # )
                    , get_cost_distance_exp_target_state(self.model
                                         , self.current_state
                                         , self.target_index
                                         , self.target_value
                                         , self.done
                                         , self.decay_rate
                                         )
                    # , get_cost_distance_target_state(self.model
                    #                      , self.current_state
                    #                      , self.target_index
                    #                      , self.target_value
                    #                      , self.done
                    #                      , self.decay_rate
                    #                      )
                    # , get_cost_distance_exp_target_state_and_immediate(self.model
                    #                      , self.current_state
                    #                      , self.previous_state
                    #                      , self.target_index
                    #                      , self.target_value
                    #                      , self.done
                    #                      , self.decay_rate
                    #                      )
                    , self.get_propensities()[0]
                    )
        elif self.current_time > self.t_max:
            self.done = True
            self.target = False
            to_return = (
                    self.current_state
                    , self.current_time
                    , self.done
                    , self.target
                    # , get_cost_distance_immediate (
                    #     self.model
                    #     , self.current_state
                    #     , self.previous_state
                    #     , self.target_index
                    #     , self.target_value 
                    # )
                    , get_cost_distance_exp_target_state(self.model
                                         , self.current_state
                                         , self.target_index
                                         , self.target_value
                                         , self.done
                                         , self.decay_rate
                                         )
                    # , get_cost_distance_target_state(self.model
                    #                      , self.current_state
                    #                      , self.target_index
                    #                      , self.target_value
                    #                      , self.done
                    #                      , self.decay_rate
                    #                      )
                    # , get_cost_distance_exp_target_state_and_immediate(self.model
                    #                      , self.current_state
                    #                      , self.previous_state
                    #                      , self.target_index
                    #                      , self.target_value
                    #                      , self.done
                    #                      , self.decay_rate
                    #                      )
                    , self.get_propensities()[0]
                    )
        else:
            self.done = False
            self.target = False
            to_return = (
                    self.current_state
                    , self.current_time
                    , self.done
                    , self.target
                    # , get_cost_distance_immediate (
                    #     self.model
                    #     , self.current_state
                    #     , self.previous_state
                    #     , self.target_index
                    #     , self.target_value
                    # )
                    , get_cost_distance_exp_target_state(self.model
                                         , self.current_state
                                         , self.target_index
                                         , self.target_value
                                         , self.done
                                         , self.decay_rate
                                         )
                    # , get_cost_distance_target_state(self.model
                    #                      , self.current_state
                    #                      , self.target_index
                    #                      , self.target_value
                    #                      , self.done
                    #                      , self.decay_rate
                    #                      )
                    # , get_cost_distance_exp_target_state_and_immediate(self.model
                    #                      , self.current_state
                    #                      , self.previous_state
                    #                      , self.target_index
                    #                      , self.target_value
                    #                      , self.done
                    #                      , self.decay_rate
                    #                      )
                    , self.get_propensities()[0]
                    )

        return to_return
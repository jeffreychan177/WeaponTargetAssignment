import cupy as cp
import random
import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete, Dict
import numpy as np

class FullyObservableSingleStageEnvironment(gym.Env):
    def __init__(self, world, reset_callback, reward_callback, observation_callback):
        self.world = world
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback

        # Define a single observation space that includes the entire state
        num_weapon_types = self.world["weapons"]["types"]
        num_target_types = self.world["targets"]["types"]

        self.observation_space = Dict({
            "weapons": Tuple([Discrete(101) for _ in range(num_weapon_types)]),  # Weapon quantities (0-100)
            "targets": Tuple([Discrete(101) for _ in range(num_target_types)]),  # Target quantities
            "probabilities": Tuple([
                Tuple([Discrete(101) for _ in range(num_target_types)])
                for _ in range(num_weapon_types)
            ]),  # Success probabilities
            "costs": Tuple([
                Tuple([Discrete(11) for _ in range(num_target_types)])
                for _ in range(num_weapon_types)
            ])  # Costs
        })

        # Action space remains the same
        self.n_agents = num_weapon_types
        self.action_space = Tuple([Discrete(num_target_types + 1) for _ in range(self.n_agents)])

    def _get_obs(self):
        """
        Returns the full state of the environment as an observation.
        """
        obs = {
            "weapons": [int(cp.clip(w, 0, 100)) for w in self.world["weapons"]["quantities"].get()],
            "targets": [int(cp.clip(t, 0, 100)) for t in self.world["targets"]["quantities"].get()],
            "probabilities": [
                [int(cp.clip(p * 100, 0, 100)) for p in self.world["probabilities"][i].get()]
                for i in range(self.n_agents)
            ],
            "costs": [
                [int(cp.clip(c, 0, 10)) for c in self.world["costs"][i].get()]
                for i in range(self.n_agents)
            ],
        }
        return obs

    def reset(self, seed=None, options=None):
        """
        Resets the environment and supports seeding.
        """
        super().reset(seed=seed)  # Ensure proper seeding behavior
        self.reset_callback(self.world)
        return self._get_obs(), {}

    def step(self, actions):
        """
        Applies actions for all agents and returns the next state, combined reward, terminated, truncated, and info.
        """
        individual_rewards = []
        target_state_update = cp.zeros_like(self.world["targets"]["quantities"], dtype=cp.float32)

        for i, action in enumerate(actions):
            reward = 0

            # If action is the last discrete value, it's "no action"
            if action < len(self.world["targets"]["quantities"]):
                target = action
                if self.world["weapons"]["quantities"][i] > 0 and self.world["targets"]["quantities"][target] > 0:
                    # Compute survival probability
                    prob_survive = cp.prod(1 - self.world["probabilities"][:, target])
                    destroyed = 1 - prob_survive
                    target_state_update[target] += destroyed

                    # Compute reward for the action
                    success_prob = self.world["probabilities"][i, target]
                    reward += self.world["targets"]["values"][target] * destroyed
                    reward -= self.world["costs"][i, target]

                    # Update weapon and target states
                    self.world["weapons"]["quantities"][i] -= 1
                    self.world["targets"]["quantities"][target] -= destroyed

            individual_rewards.append(reward)

        # Update the target states based on survival probabilities
        self.world["targets"]["quantities"] = cp.maximum(
            self.world["targets"]["quantities"] - target_state_update, 0
        )

        # Combine rewards (e.g., sum all agent rewards)
        combined_reward = sum(individual_rewards)

        # Determine if the episode is terminated
        terminated = (
            all(q == 0 for q in self.world["targets"]["quantities"].get())
            or all(q == 0 for q in self.world["weapons"]["quantities"].get())
        )
        truncated = False  # Add any truncation logic if applicable (e.g., max steps)

        # Return observation, combined reward, terminated, truncated, and info
        return self._get_obs(), combined_reward, terminated, truncated, {}

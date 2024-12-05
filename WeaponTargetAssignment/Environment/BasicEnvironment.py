import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np


class BasicEnvironment(gym.Env):
    def __init__(self, world, reset_callback, reward_callback, observation_callback):
        self.world = world
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback

        # Define action and observation spaces
        num_weapon_types = self.world["weapons"]["types"]
        num_target_types = self.world["targets"]["types"]
        self.action_space = Box(low=0, high=10, shape=(num_weapon_types, num_target_types), dtype=int)
        self.observation_space = Box(low=0, high=100, shape=(num_weapon_types + num_target_types, ), dtype=float)

    def reset(self):
        """
        Resets the environment.
        """
        self.reset_callback(self.world)
        return self._get_obs()

    def step(self, action):
        """
        Applies an action and returns the next state, reward, done, and info.
        """
        # Ensure the action is within valid bounds
        action = np.clip(action, 0, 10).astype(int)

        # Compute reward
        reward = self.reward_callback(self.world, action)

        # Update world states based on action
        for i, weapon_qty in enumerate(self.world["weapons"]["quantities"]):
            for j, target_qty in enumerate(self.world["targets"]["quantities"]):
                if action[i][j] > 0 and weapon_qty >= action[i][j]:
                    self.world["weapons"]["quantities"][i] -= action[i][j]
                    self.world["targets"]["quantities"][j] -= 1 if self.world["targets"]["quantities"][j] > 0 else 0

        # Determine if the episode is done (all targets eliminated or no weapons left)
        done = all(q == 0 for q in self.world["targets"]["quantities"]) or all(q == 0 for q in self.world["weapons"]["quantities"])

        # Return observation, reward, done, and info
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """
        Returns the current observation.
        """
        obs = self.observation_callback(self.world)
        return np.concatenate([
            np.array(obs["weapons"]),
            np.array(obs["targets"])
        ])

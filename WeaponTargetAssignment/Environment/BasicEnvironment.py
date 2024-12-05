import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete, Dict
import numpy as np

class BasicEnvironment(gym.Env):
    def __init__(self, world, reset_callback, reward_callback, observation_callback, max_steps=100):
        self.world = world
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback

        # Multi-agent setup: number of agents = number of weapon types
        self.n_agents = self.world["weapons"]["types"]

        # Define individual action spaces as discrete (number of target types + 1 for "no action")
        num_target_types = self.world["targets"]["types"]
        self.action_space = Tuple([Discrete(num_target_types + 1) for _ in range(self.n_agents)])

        # Define observation spaces for each agent
        self.observation_space = Tuple([
            Dict({
                "weapons": Discrete(101),  # Weapon quantity (0-100)
                "targets": Dict({j: Discrete(101) for j in range(num_target_types)}),  # Individual target quantities (0-100)
                "probabilities": Dict({j: Discrete(101) for j in range(num_target_types)}),  # Success probabilities (scaled 0-100)
                "costs": Dict({j: Discrete(11) for j in range(num_target_types)})  # Costs (scaled 0-10)
            }) for _ in range(self.n_agents)
        ])

        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """
        Resets the environment and supports seeding.
        """
        super().reset(seed=seed)  # Ensure proper seeding behavior
        self.current_step = 0
        self.reset_callback(self.world)
        return self._get_obs(), {}

    def step(self, actions):
        """
        Applies actions for all agents and returns the next state, combined reward, terminated, truncated, and info.
        """
        individual_rewards = []
        info = {"actions": [], "rewards": []}

        for i, action in enumerate(actions):
            reward = 0

            # If action is the last discrete value, it's "no action"
            if action < len(self.world["targets"]["quantities"]):
                target = action
                if self.world["weapons"]["quantities"][i] > 0 and self.world["targets"]["quantities"][target] > 0:
                    # Compute reward for the action
                    success_prob = self.world["probabilities"][i][target]
                    reward += self.world["targets"]["values"][target] * success_prob
                    reward -= self.world["costs"][i][target]

                    # Update weapon and target states
                    self.world["weapons"]["quantities"][i] -= 1
                    self.world["targets"]["quantities"][target] -= 1 if self.world["targets"]["quantities"][target] > 0 else 0

            individual_rewards.append(reward)
            info["actions"].append((i, action))
            info["rewards"].append(reward)

        # Combine rewards (e.g., sum all agent rewards)
        combined_reward = sum(individual_rewards)

        # Determine if the episode is terminated
        terminated = (
            all(q == 0 for q in self.world["targets"]["quantities"]) or
            all(q == 0 for q in self.world["weapons"]["quantities"])
        )
        truncated = self.current_step >= self.max_steps

        self.current_step += 1

        # Return observation, combined reward, terminated, truncated, and info
        return self._get_obs(), combined_reward, terminated, truncated, info

    def _get_obs(self):
        """
        Returns the current observation for all agents.
        """
        obs = self.observation_callback(self.world)
        return tuple({
            "weapons": int(obs["weapons"][i]),  # Ensure each weapon count is an integer
            "targets": {j: int(obs["targets"][j]) for j in range(len(obs["targets"]))},  # Ensure target counts are integers
            "probabilities": {j: int(obs["probabilities"][i][j] * 100) for j in range(len(obs["targets"]))},  # Scale to 0-100
            "costs": {j: int(obs["costs"][i][j]) for j in range(len(obs["targets"]))}
        } for i in range(self.n_agents))

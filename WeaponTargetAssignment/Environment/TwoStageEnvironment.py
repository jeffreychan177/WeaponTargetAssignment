import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete, Dict
import numpy as np


class TwoStageEnvironment(gym.Env):
    def __init__(self, world, reset_callback, reward_callback, observation_callback):
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
                "targets": Discrete(101),  # Target quantities
                "probabilities": Dict({j: Discrete(101) for j in range(num_target_types)}),  # Success probabilities
                "costs": Dict({j: Discrete(11) for j in range(num_target_types)})  # Costs
            }) for _ in range(self.n_agents)
        ])

        self.stage = 1  # Start with the first stage

    def reset(self, seed=None, options=None):
        """
        Resets the environment and supports seeding.
        """
        super().reset(seed=seed)  # Ensure proper seeding behavior
        self.reset_callback(self.world)
        self.stage = 1  # Reset to the first stage
        return self._get_obs(), {}

    def step(self, actions):
        """
        Applies actions for all agents and returns the next state, combined reward, terminated, truncated, and info.
        """
        individual_rewards = []
        target_state_update = np.zeros_like(self.world["targets"]["quantities"])

        for i, action in enumerate(actions):
            reward = 0

            # If action is the last discrete value, it's "no action"
            if action < len(self.world["targets"]["quantities"]):
                target = action
                if self.world["weapons"]["quantities"][i] > 0 and self.world["targets"]["quantities"][target] > 0:
                    # Compute survival probability
                    prob_survive = np.prod([
                        1 - self.world["probabilities"][i][target]
                        for i in range(self.n_agents)
                    ])
                    destroyed = 1 - prob_survive
                    target_state_update[target] = destroyed

                    # Compute reward for the action
                    success_prob = self.world["probabilities"][i][target]
                    reward += self.world["targets"]["values"][target] * destroyed
                    reward -= self.world["costs"][i][target]

                    # Update weapon and target states
                    self.world["weapons"]["quantities"][i] -= 1
                    self.world["targets"]["quantities"][target] -= destroyed

            individual_rewards.append(reward)

        # Update the target states based on survival probabilities
        self.world["targets"]["quantities"] = np.maximum(
            self.world["targets"]["quantities"] - target_state_update, 0
        )

        # Combine rewards (e.g., sum all agent rewards)
        combined_reward = sum(individual_rewards)

        # Determine if the episode is terminated
        terminated = (
            all(q == 0 for q in self.world["targets"]["quantities"])
            or all(q == 0 for q in self.world["weapons"]["quantities"])
        )
        truncated = False  # Add any truncation logic if applicable (e.g., max steps)

        # Progress to the next stage if the first stage is complete
        if self.stage == 1 and terminated:
            self.stage = 2
            terminated = False

        # Return observation, combined reward, terminated, truncated, and info
        return self._get_obs(), combined_reward, terminated, truncated, {}

    def _get_obs(self):
        """
        Returns the current observation for all agents.
        """
        obs = self.observation_callback(self.world)
        return tuple({
            "weapons": int(np.clip(obs["weapons"][i], 0, 100)),
            "targets": int(np.clip(sum(obs["targets"]), 0, 100)),  # Ensure within range
            "probabilities": {j: int(np.clip(obs["probabilities"][i][j] * 100, 0, 100)) for j in range(len(obs["targets"]))},
            "costs": {j: int(np.clip(obs["costs"][i][j], 0, 10)) for j in range(len(obs["targets"]))}
        } for i in range(self.n_agents))



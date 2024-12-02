import numpy as np
from gymnasium import spaces
from environment.environment import WeaponTargetAssignmentEnv

class multiagentEnv(WeaponTargetAssignmentEnv):
    def __init__(self, num_weapons=3, num_targets=5, max_distance=15, front_line=1, render_mode=None):
        super().__init__(num_weapons, num_targets, max_distance, front_line, render_mode)
        
        # Define action and observation spaces for each agent
        self.action_spaces = [spaces.Discrete(self.num_targets + 1) for _ in range(self.num_weapons)]
        self.observation_spaces = [
            spaces.Box(
                low=0,
                high=1,
                shape=(self.num_targets * 2,),
                dtype=np.float32,
            )
            for _ in range(self.num_weapons)
        ]
        self.shared_reward = True  # Shared reward for cooperative setting

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observations for all agents."""
        obs, _ = super().reset(seed, options)
        return self.get_obs(), {}

    def get_obs(self):
        """Get observations for all agents."""
        return [
            self._get_observation().flatten()
            for _ in range(self.num_weapons)
        ]

    def step(self, actions):
        """
        Take a step in the environment.
        Each agent independently decides its action.
        """
        if len(actions) != self.num_weapons:
            raise ValueError(f"Expected {self.num_weapons} actions, got {len(actions)}")

        # Perform the step using the parent class
        obs, reward, done, info = super().step(actions)

        # Return observations for all agents
        agent_obs = self.get_obs()

        return agent_obs, reward, done, info

    def render(self, mode="human"):
        """Render the current state of the environment."""
        super().render()


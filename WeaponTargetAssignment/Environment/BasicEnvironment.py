import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BasicEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_weapons=3, num_targets=5, max_distance=15, front_line=1, 
                 reset_callback=None, reward_callback=None, observation_callback=None, render_mode=None):
        super(BasicEnvironment, self).__init__()

        # Parameters
        self.num_weapons = num_weapons
        self.num_targets = num_targets
        self.max_distance = max_distance
        self.front_line = front_line

        # Callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback

        # Observation and action spaces
        agent_observation_space = spaces.Box(
            low=np.array([0, 0] * self.num_targets),
            high=np.array([self.max_distance, 1] * self.num_targets),
            dtype=np.float32,
        )
        self.observation_space = spaces.Tuple([agent_observation_space for _ in range(self.num_weapons)])
        self.action_space = spaces.Tuple([spaces.Discrete(self.num_targets + 1) for _ in range(self.num_weapons)])

        self.render_mode = render_mode
        self.targets = []  # Initialize an empty list of targets
        self.done = False

    def reset(self, **kwargs):
        """Reset the environment using the reset callback if provided."""
        if self.reset_callback:
            return self.reset_callback()
        else:
            return self._default_reset()

    def _default_reset(self):
        """Default reset logic."""
        self.targets = [
            {"distance": np.random.uniform(5, self.max_distance),
             "base_probability": np.random.uniform(0.4, 0.8),
             "id": i}
            for i in range(self.num_targets)
        ]
        self.done = False
        return self._default_get_obs(), {}

    def get_obs(self):
        """Get observations for all agents using the observation callback if provided."""
        if self.observation_callback:
            return self.observation_callback()
        return self._default_get_obs()

    def _default_get_obs(self):
        """Default observation logic."""
        return [
            np.array([t["distance"], t["base_probability"]] for t in self.targets).flatten()
            for _ in range(self.num_weapons)
        ]

    def step(self, actions):
        """Take a step in the environment using the reward callback if provided."""
        if self.reward_callback:
            return self.reward_callback(actions)

        # Default reward logic
        reward = 0
        remaining_targets = []
        valid_targets = {i: t for i, t in enumerate(self.targets)}

        for weapon_id, target_idx in enumerate(actions):
            if target_idx < len(valid_targets):
                target = valid_targets[target_idx]
                hit_prob = min(1.0, target["base_probability"] + (1 - target["base_probability"]) * (1 / (1 + target["distance"])))
                if np.random.rand() < hit_prob:
                    reward += 10
                else:
                    target["distance"] -= 1
                    if target["distance"] > self.front_line:
                        remaining_targets.append(target)
                    else:
                        reward -= 5

        self.targets = remaining_targets
        self.done = len(self.targets) == 0
        return self._default_get_obs(), reward, self.done, {}

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print("\nTargets:")
            for target in self.targets:
                print(f"  Target {target['id']}: Distance = {target['distance']:.2f}, Base Probability = {target['base_probability']:.2f}")
            print("\nWeapons ready for action!")

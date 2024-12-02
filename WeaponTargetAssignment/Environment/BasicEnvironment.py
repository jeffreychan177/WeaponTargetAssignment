import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BasicEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_weapons=3, num_targets=5, max_distance=15, front_line=1, render_mode=None):
        super(BasicEnvironment, self).__init__()

        # Store parameters
        self.num_weapons = num_weapons
        self.num_targets = num_targets
        self.max_distance = max_distance
        self.front_line = front_line

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0] * self.num_targets),  # Min distances and probabilities
            high=np.array([self.max_distance, 1] * self.num_targets),  # Max distances and probabilities
            dtype=np.float32,
        )
        self.action_space = spaces.Tuple([spaces.Discrete(self.num_targets + 1) for _ in range(self.num_weapons)])

        # Render mode
        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        self.targets = [
            {"distance": np.random.uniform(5, self.max_distance),
             "base_probability": np.random.uniform(0.4, 0.8),
             "id": i}
            for i in range(self.num_targets)
        ]
        self.done = False
        return self.get_obs(), {}

    def get_obs(self):
        """Generate observations for all agents."""
        return [
            np.array([t["distance"], t["base_probability"]] for t in self.targets).flatten()
            for _ in range(self.num_weapons)
        ]

    def step(self, actions):
        """Take a step in the environment."""
        if self.done:
            raise RuntimeError("The environment must be reset before further use.")

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
        return self.get_obs(), reward, self.done, {}

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print("\nTargets:")
            for target in self.targets:
                print(f"  Target {target['id']}: Distance = {target['distance']:.2f}, Base Probability = {target['base_probability']:.2f}")
            print("\nWeapons ready for action!")

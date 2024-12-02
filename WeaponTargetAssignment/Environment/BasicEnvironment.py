import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BasicEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, world=None, reset_callback=None, reward_callback=None, observation_callback=None, render_mode=None):
        super(BasicEnvironment, self).__init__()

        # Callbacks and world
        self.world = world
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback

        # Use world properties if provided, else default values
        if world:
            self.num_weapons = world.num_weapons
            self.num_targets = world.num_targets
            self.max_distance = world.max_distance
            self.front_line = world.front_line
            self.observation_space = world.observation_space
            self.action_space = spaces.Tuple(world.action_spaces)
        else:
            self.num_weapons = 3
            self.num_targets = 5
            self.max_distance = 15
            self.front_line = 1
            self.observation_space = spaces.Box(
                low=np.array([0, 0] * self.num_targets),
                high=np.array([self.max_distance, 1] * self.num_targets),
                dtype=np.float32,
            )
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_targets + 1) for _ in range(self.num_weapons)])

        self.render_mode = render_mode
        self.reset()

    def reset(self, **kwargs):
        """Reset the environment."""
        if self.reset_callback:
            return self.reset_callback()
        else:
            self.targets = [
                {"distance": np.random.uniform(5, self.max_distance), 
                 "base_probability": np.random.uniform(0.4, 0.8),
                 "id": i}
                for i in range(self.num_targets)
            ]
            self.done = False
            return self.get_obs(), {}

    def get_obs(self):
        """Get observations for all agents."""
        if self.observation_callback:
            return self.observation_callback()
        return [np.array([t["distance"], t["base_probability"]] for t in self.targets).flatten() for _ in range(self.num_weapons)]

    def step(self, actions):
        """Take a step in the environment."""
        if self.reward_callback:
            return self.reward_callback(actions)

        # Default step logic
        reward = 0
        remaining_targets = []
        valid_targets = {i: t for i, t in enumerate(self.targets)}

        for weapon_id, target_idx in enumerate(actions):
            if target_idx < len(valid_targets):  # Weapon assigned to a valid target
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
        """Render the current state."""
        if self.render_mode == "human" and self.world:
            self.world.render()

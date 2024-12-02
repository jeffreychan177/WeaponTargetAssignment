import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BasicEnvironment(gym.Env):
    def __init__(
        self, 
        num_weapons=3, 
        num_targets=5, 
        max_distance=15, 
        front_line=1, 
        reset_callback=None, 
        reward_callback=None, 
        observation_callback=None, 
        render_mode=None
    ):
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

        # Define shared observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0] * self.num_targets),  # Min distances and probabilities
            high=np.array([self.max_distance, 1] * self.num_targets),  # Max distances and probabilities
            dtype=np.float32,
        )

        # Define individual action spaces for each agent
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

        self.shared_reward = True
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

    def _get_observation(self):
        """Get the current state observation."""
        return np.array([t["distance"], t["base_probability"]] for t in self.targets).flatten()

    def get_obs(self):
        """Get observations for all agents."""
        return [self._get_observation() for _ in range(self.num_weapons)]

    def calculate_hit_probability(self, distance, base_probability):
        """Probability of hitting increases as distance decreases."""
        return min(1.0, base_probability + (1 - base_probability) * (1 / (1 + distance)))

    def step(self, actions):
        """
        Take a step in the environment.
        
        actions: List of target indices or "no target" (self.num_targets) for each weapon.
        """
        if self.done:
            raise RuntimeError("The environment must be reset before further use.")

        reward = 0
        remaining_targets = []

        valid_targets = {i: t for i, t in enumerate(self.targets)}

        for weapon_id, target_idx in enumerate(actions):
            if target_idx < len(valid_targets):  # If the weapon is assigned to a valid target
                target = valid_targets[target_idx]
                hit_prob = self.calculate_hit_probability(target["distance"], target["base_probability"])
                if np.random.rand() < hit_prob:  # Successful hit
                    reward += 10
                    if self.render_mode == "human":
                        print(f"Weapon {weapon_id} hit Target {target['id']}!")
                else:  # Miss
                    target["distance"] -= 1  # Move closer
                    if target["distance"] > self.front_line:
                        remaining_targets.append(target)
                    else:  # Escaped
                        reward -= 5
                        if self.render_mode == "human":
                            print(f"Target {target['id']} escaped!")
            elif target_idx == self.num_targets:  # No target action
                if self.render_mode == "human":
                    print(f"Weapon {weapon_id} did not fire.")

        self.targets = remaining_targets
        self.done = len(self.targets) == 0

        agent_obs = self.get_obs()

        return agent_obs, reward, self.done, {}

    def render(self):
        """Render the current state."""
        if self.render_mode == "human":
            print("\nTargets:")
            for target in self.targets:
                print(f"  Target {target['id']}: Distance = {target['distance']:.2f}, Base Probability = {target['base_probability']:.2f}")

            print("\nWeapons ready for action!")

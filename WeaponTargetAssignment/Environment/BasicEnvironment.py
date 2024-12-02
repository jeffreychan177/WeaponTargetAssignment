import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BasicEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, world, reset_callback=None, reward_callback=None, observation_callback=None, render_mode=None):
        super(BasicEnvironment, self).__init__()

        # World and callbacks
        self.world = world
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.render_mode = render_mode

        # Number of agents (derived from the world)
        self.n_agents = len(world.shooters)

        # Action and observation spaces
        num_actions = len(world.agents) + 1  # Assuming all agents + a "no action" option
        self.single_action_space = spaces.Discrete(num_actions)
        self.action_space = spaces.Tuple([self.single_action_space] * self.n_agents)

        obs_dim = (
            2 +                                # Shooter position
            1 +                                # Ammo count
            1 +                                # Cooldown
            (len(world.agents) * 5) +          # Agent info
            len(world.agents) +                # Hit probabilities
            ((self.n_agents - 1) * 4) +        # Teammate info
            (30 * 2)                           # Action history
        )
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Tuple([self.single_observation_space] * self.n_agents)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.reset_callback:
            self.reset_callback(self.world)
        for shooter in self.world.shooters:
            shooter.reset_state()
        return self._get_obs(), {}

    def step(self, actions):
        if not isinstance(actions, list):
            actions = [actions]

        num_targets = len(self.world.agents)

        # Process actions for each shooter
        for i, action in enumerate(actions):
            shooter = self.world.shooters[i]
            if num_targets > 0 and 1 <= action <= num_targets:
                target = self.world.agents[action - 1]
                shooter.aim_and_shoot(target, True)
            else:
                shooter.aim_and_shoot(None, False)

        # Update world state
        self.world.update()

        # Get new observations
        obs = self._get_obs()

        # Compute rewards
        reward = sum(float(self.reward_callback(shooter, self.world)) for shooter in self.world.shooters)

        # Check if episode is done
        done = self.world.drone_eliminations >= self.world.max_drone_eliminations

        # Gym expects `truncated` to be a boolean. Here we assume it's `False`.
        truncated = False

        return obs, reward, done, truncated, {}

    def _get_obs(self):
        obs = []
        for shooter in self.world.shooters:
            # Get the raw observation
            shooter_obs = self.observation_callback(shooter, self.world)

            # Break observation into components
            shooter_position = shooter_obs[:2]            # 2 values
            ammo_cooldown = shooter_obs[2:4]             # 1 ammo, 1 cooldown
            drone_info = shooter_obs[4:4 + (len(self.world.agents) * 5)]
            probabilities = shooter_obs[4 + (len(self.world.agents) * 5):4 + (len(self.world.agents) * 5) + len(self.world.agents)]
            teammate_info = shooter_obs[4 + (len(self.world.agents) * 5) + len(self.world.agents):4 + (len(self.world.agents) * 5) + len(self.world.agents) + ((self.n_agents - 1) * 4)]
            action_history = shooter_obs[-60:]           # Last 60 values (30 * 2)

            # Combine all components
            shooter_obs = np.concatenate([
                shooter_position,
                ammo_cooldown,
                drone_info,
                probabilities,
                teammate_info,
                action_history
            ])

            obs.append(shooter_obs)

        # Return as tuple to match observation space
        return tuple(obs)

    def render(self):
        print("Rendering environment state...")  # Replace with actual rendering if needed

    def close(self):
        print("Closing environment...")

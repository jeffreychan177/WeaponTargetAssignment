import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

# Define Colors
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
AGENT_COLORS = [(0, 0, 255), (255, 165, 0), (75, 0, 130)]  # Unique colors for agents

# Define the maximum number of drones
MAX_DRONES = 16

class Environment(gym.Env):
    def __init__(self, world, reset_callback=None, reward_callback=None, observation_callback=None, render_mode=None, show_probabilities=False):
        super().__init__()
        self.world = world
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.render_mode = render_mode
        self.show_probabilities = show_probabilities  # Feature toggle for probability visualization

        self.n_agents = len(world.shooters)
        num_actions = MAX_DRONES + 1
        self.single_action_space = spaces.Discrete(num_actions)
        self.action_space = spaces.Tuple([self.single_action_space] * self.n_agents)

        obs_dim = (
            2 +                                # Shooter position
            1 +                                # Ammo count
            1 +                                # Cooldown
            (MAX_DRONES * 5) +                 # Drone info
            MAX_DRONES +                       # Hit probabilities
            ((self.n_agents - 1) * 4) +        # Teammate info
            (30 * 2)                           # Action history
        )
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Tuple([self.single_observation_space] * self.n_agents)

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.background_image = pygame.Surface((800, 600))
        self.background_image.fill(WHITE)

    def step(self, actions):
        if not isinstance(actions, list):
            actions = [actions]

        num_targets = sum(1 for agent in self.world.agents if agent.agent_type == 'drone')

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


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_callback(self.world)
        for shooter in self.world.shooters:
            shooter.reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for shooter in self.world.shooters:
            # Get the raw observation
            shooter_obs = self.observation_callback(shooter, self.world)

            # Break observation into expected components
            shooter_position = shooter_obs[:2]            # 2 values
            ammo_cooldown = shooter_obs[2:4]             # 1 ammo, 1 cooldown
            drone_info = shooter_obs[4:4 + (MAX_DRONES * 5)]
            probabilities = shooter_obs[4 + (MAX_DRONES * 5):4 + (MAX_DRONES * 5) + MAX_DRONES]
            teammate_info = shooter_obs[4 + (MAX_DRONES * 5) + MAX_DRONES:4 + (MAX_DRONES * 5) + MAX_DRONES + ((self.n_agents - 1) * 4)]
            action_history = shooter_obs[-60:]           # Last 60 values (30 * 2)

            # Check and fix lengths
            if len(drone_info) < MAX_DRONES * 5:
                drone_info = np.pad(drone_info, (0, (MAX_DRONES * 5) - len(drone_info)), mode='constant')
            if len(probabilities) < MAX_DRONES:
                probabilities = np.pad(probabilities, (0, MAX_DRONES - len(probabilities)), mode='constant')
            if len(teammate_info) < (self.n_agents - 1) * 4:
                teammate_info = np.pad(teammate_info, (0, ((self.n_agents - 1) * 4) - len(teammate_info)), mode='constant')

            # Combine all components
            shooter_obs = np.concatenate([
                shooter_position,
                ammo_cooldown,
                drone_info,
                probabilities,
                teammate_info,
                action_history
            ])

            # Validate observation size
            if len(shooter_obs) != self.single_observation_space.shape[0]:
                raise ValueError(
                    f"Observation dimension mismatch: Expected {self.single_observation_space.shape[0]}, got {len(shooter_obs)}"
                )

            obs.append(shooter_obs)

        # Return as tuple to match observation space
        return tuple(obs)



    def render(self):
        self.screen.blit(self.background_image, (0, 0))
        for drone in self.world.agents:
            if drone.agent_type == 'drone' and drone.active:
                pygame.draw.circle(self.screen, GRAY, (int(drone.position[0]), int(drone.position[1])), 20)
                
                # Feature toggle for probability visualization
                if self.show_probabilities:
                    for i, shooter in enumerate(self.world.shooters):
                        probability = shooter.calculate_hit_probability(drone)
                        offset_x, offset_y = 10 * i, -15 - (i * 15)
                        prob_text = self.font.render(f"{probability:.2f}", True, AGENT_COLORS[i % len(AGENT_COLORS)])
                        self.screen.blit(prob_text, (drone.position[0] + offset_x, drone.position[1] + offset_y))

        for shooter in self.world.shooters:
            shooter.draw(self.screen, self.font)
        pygame.display.flip()

    def close(self):
        pygame.quit()

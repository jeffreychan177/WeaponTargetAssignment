import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging

# Set up logging configuration
def setup_logging():
    level = logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

# Define maximum numbers
MAX_WEAPONS = 16
MAX_TARGETS = 16

class WTAEnvironment(gym.Env):
    def __init__(self, world, reset_callback=None, reward_callback=None, observation_callback=None, render_mode=None):
        super().__init__()
        setup_logging()
        self.world = world
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.render_mode = render_mode

        self.n_agents = len(world.weapons)  # Agents correspond to weapons
        self.single_action_space = spaces.Discrete(MAX_TARGETS)  # Each weapon targets one target
        self.action_space = spaces.Tuple([self.single_action_space] * self.n_agents)

        obs_dim = (
            MAX_WEAPONS +                    # Weapon states
            MAX_TARGETS +                    # Target states
            (MAX_WEAPONS * MAX_TARGETS) +    # Assignment matrix
            1                                # Time step
        )
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Tuple([self.single_observation_space] * self.n_agents)

        logging.info("WTAEnvironment initialized with %d agents", self.n_agents)

    def step(self, actions):
        if not isinstance(actions, list):
            actions = [actions]

        logging.debug("Actions received: %s", actions)

        # Process actions for each agent
        rewards = []
        for i, action in enumerate(actions):
            weapon = self.world.weapons[i]
            if 0 <= action < len(self.world.targets):
                target = self.world.targets[action]
                reward = self._assign_weapon_to_target(weapon, target)
                rewards.append(reward)
                logging.info("Weapon %d assigned to target %d", i, action)
            else:
                rewards.append(0)  # No valid assignment
                logging.info("Weapon %d did not fire", i)

        # Update world state
        self.world.update()
        logging.debug("World state updated")

        # Get new observations
        obs = self._get_obs()

        # Compute cumulative reward
        total_reward = sum(rewards)
        logging.info("Total reward: %f", total_reward)

        # Check if episode is done
        done = self.world.all_targets_destroyed() or self.world.time_exceeded()
        logging.info("Episode done: %s", done)

        truncated = False  # Gym expects this as a boolean

        return obs, total_reward, done, truncated, {}

    def _assign_weapon_to_target(self, weapon, target):
        """Calculate reward for assigning a weapon to a target."""
        pij = weapon.get_hit_probability(target)
        vj = target.value
        cost = weapon.get_assignment_cost(target)

        # Reward incorporates maximizing damage and minimizing cost
        reward = vj * pij - cost
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_callback(self.world)
        logging.info("Environment reset")
        return self._get_obs(), {}

    def _get_obs(self):
        """Generate observations for all agents."""
        obs = []
        for weapon in self.world.weapons:
            weapon_state = weapon.get_state()
            target_states = self.world.get_target_states()
            assignment_matrix = self.world.get_assignment_matrix()
            time_step = np.array([self.world.time_step], dtype=np.float32)

            weapon_obs = np.concatenate([weapon_state, target_states, assignment_matrix.flatten(), time_step])
            obs.append(weapon_obs)

        return tuple(obs)

    def render(self):
        """Render the environment (if applicable)."""
        logging.info("Render function called")

    def close(self):
        """Clean up resources."""
        logging.info("Environment closed")

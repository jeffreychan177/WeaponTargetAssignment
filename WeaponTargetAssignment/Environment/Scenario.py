import numpy as np
import random
import logging

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

AGENT_COLORS = [(0, 0, 255), (255, 165, 0), (75, 0, 130)]

class Shooter:
    def __init__(self, name, position, color, visibility_radius=200):
        self.name = name
        self.position = position
        self.color = color
        self.visibility_radius = visibility_radius 
        self.cooldown = 0
        self.max_cooldown = 1
        self.action_history = []
        self.agent_type = 'shooter'
        self.hits = 0
        self.misses = 0
        self.shots_fired = 0
        self.reward = 0.0  # Track reward for the shooter

    def reset_state(self):
        """Reset the shooter to its initial state."""
        self.cooldown = 0
        self.action_history = []
        self.hits = 0
        self.misses = 0
        self.shots_fired = 0
        self.reward = 0.0
        logging.info("Shooter %s state reset", self.name)

    def aim_and_shoot(self, target, shoot):
        action = (self._get_target_index(target), shoot)
        self.update_action_history(action)

        if self.cooldown == 0 and shoot and target and target.active:
            self.shots_fired += 1
            self.cooldown = self.max_cooldown

            # Determine hit or miss based on probability
            hit_probability = self.calculate_hit_probability(target)
            shot_hit = random.random() < hit_probability
            target_position = np.array(target.position)

            if shot_hit:
                self.hits += 1
                self.reward += 10.0  # Reward for a successful hit
                logging.info("Shooter %s hit target %s", self.name, target.name)
                target.handle_hit()  # Delayed deactivation
            else:
                self.misses += 1
                logging.info("Shooter %s missed target %s", self.name, target.name)
        else:
            logging.info("Shooter %s did not shoot", self.name)

    def _get_target_index(self, target):
        if target is None:
            return -1
        return int(target.name.split('_')[1])

    def update_action_history(self, action):
        """Update the shooter's action history."""
        self.action_history.append(action)
        if len(self.action_history) > 30:
            self.action_history.pop(0)
        logging.debug("Shooter %s action history updated: %s", self.name, self.action_history)

    def cooldown_tick(self):
        """Handle cooldown timer."""
        if self.cooldown > 0:
            self.cooldown -= 1
        logging.debug("Shooter %s cooldown: %d", self.name, self.cooldown)

    def calculate_hit_probability(self, target):
        """Calculate the probability of hitting the target based on distance."""
        distance = np.linalg.norm(np.array(self.position) - np.array(target.position))
        probability = 1 / (1 + np.exp(-0.05 * (distance - 300)))
        logging.debug("Shooter %s calculated hit probability for target %s: %f", self.name, target.name, probability)
        return max(0.01, min(probability, 0.9))

class Drone:
    def __init__(self, name, position=None, velocity_range=None):
        self.name = name
        self.position = position or [0, 0]
        self.velocity = velocity_range or [0, 4]
        self.agent_type = 'drone'
        self.active = True
        self.hit = False  # Flag to indicate if the drone was hit

    def update_position(self):
        # Continue updating position only if the drone is active
        if self.active:
            self.position[1] += self.velocity[1]
            logging.debug("Drone %s updated position: %s", self.name, self.position)

            # Check if the drone has moved past the bottom of the screen
            if self.position[1] > 600:
                self.deactivate()  # Deactivate drone when it moves past the screen

    def deactivate(self):
        """Deactivate the drone when it reaches the bottom or is hit."""
        self.active = False
        self.hit = False  # Reset hit status when deactivating
        logging.info("Drone %s deactivated", self.name)

    def handle_hit(self):
        """Called when the drone is hit."""
        self.hit = True
        self.deactivate()  # Deactivate immediately when hit
        logging.info("Drone %s hit and deactivated", self.name)

class World:
    def __init__(self, num_shooters=3):
        self.drones = []
        self.shooters = []
        self.num_shooters = num_shooters
        self.drone_eliminations = 0  # Count the number of eliminated drones
        self.max_drones = 16
        self.max_drone_eliminations = self.max_drones  
        self.drones_spawned = 0  # Track the total number of drones spawned
        logging.info("World initialized with %d shooters", num_shooters)

    def add_agent(self, agent):
        if agent.agent_type == 'shooter':
            self.shooters.append(agent)
        else:
            self.drones.append(agent)
        logging.info("Agent %s added to the world", agent.name)

    def update(self):
        # Update shooters and drones
        for shooter in self.shooters:
            shooter.cooldown_tick()
        for drone in self.drones:
            drone.update_position()
            # Check if the drone has been eliminated
            if not drone.active and drone.hit:
                self.drone_eliminations += 1
                drone.hit = False  # Prevent double counting the same drone elimination
                logging.info("Drone eliminations updated: %d", self.drone_eliminations)

        spawn_chance = 0.05  # 5% chance of spawning a drone each step
        if random.random() < spawn_chance and self.drones_spawned < self.max_drones:
            self.spawn_drone()

        # Reward shooters if all drones are eliminated
        if self.drone_eliminations == self.max_drone_eliminations:
            for shooter in self.shooters:
                shooter.reward += 50.0  # Additional reward for shooting down all drones
                logging.info("Shooter %s rewarded for all drones eliminated", shooter.name)

    def spawn_drone(self):
        # Spawn a new drone and add it to the agents list
        position = [random.randint(0, 800), 0]
        velocity = [3, 5]
        drone = Drone(name=f"Drone_{len(self.drones)+1}", position=position, velocity_range=velocity)
        drone.active = True  # Ensure new drones are active
        self.add_agent(drone)
        self.drones_spawned += 1  # Increment the count of spawned drones
        logging.info("Drone %s spawned at position %s", drone.name, position)

class Scenario:
    def make_world(self, num_shooters=3):
        world = World(num_shooters=num_shooters)
        spacing = 800 // (num_shooters + 1)
        for i in range(num_shooters):
            position = [(i + 1) * spacing, 550]
            color = AGENT_COLORS[i % len(AGENT_COLORS)]  
            shooter = Shooter(name=f"Hero_{i}", position=position, color=color)
            world.add_agent(shooter)
        logging.info("World created with %d shooters", num_shooters)
        return world

    def reset_world(self, world):
        world.agents = [agent for agent in world.agents if agent.agent_type == 'shooter']
        world.drones_spawned = 0
        world.drone_eliminations = 0
        for shooter in world.shooters:
            shooter.reset_state()
        logging.info("World reset")

    def reward(self, shooter, world):
        """Calculate the reward for a shooter."""
        reward = shooter.reward
        shooter.reward = 0.0
        logging.debug("Shooter %s reward calculated: %f", shooter.name, reward)
        return reward

    def observation(self, agent, world):
        """Generate partially observable observations for a given agent."""
        if agent.agent_type == 'shooter':
            shooter_pos = agent.position
            shooter_cooldown = [agent.cooldown]

            # Filter visible drones based on visibility radius
            drone_info = []
            probabilities = []
            active_drones = [
                drone for drone in world.agents if drone.agent_type == 'drone' and drone.active
            ]
            for drone in active_drones:
                distance = np.linalg.norm(np.array(drone.position) - np.array(shooter_pos))
                if distance <= agent.visibility_radius:
                    drone_info.extend(drone.position + drone.velocity)
                    probabilities.append(agent.calculate_hit_probability(drone))
                else:
                    drone_info.extend([-1, -1, 0, 0])  # Mask non-visible drones
                    probabilities.append(-1)  # Mask probabilities

            # Filter visible teammates based on visibility radius
            teammate_info = []
            teammates = [teammate for teammate in world.shooters if teammate != agent]
            for teammate in teammates:
                distance = np.linalg.norm(np.array(teammate.position) - np.array(shooter_pos))
                if distance <= agent.visibility_radius:
                    teammate_info.extend(teammate.position)
                    teammate_info.append(teammate.cooldown)
                else:
                    teammate_info.extend([-1, -1, -1])  # Mask non-visible teammates

            # Action history remains unchanged
            action_history_flat = [
                item for action in agent.action_history for item in (action[0], int(action[1]))
            ]
            while len(action_history_flat) < 30 * 2:
                action_history_flat.extend([0, 0])

            obs = np.array(
                shooter_pos + shooter_cooldown +
                drone_info + probabilities + teammate_info + action_history_flat,
                dtype=np.float32
            )
            logging.debug("Observation generated for shooter %s", agent.name)
            return obs
        else:
            return np.zeros(1)

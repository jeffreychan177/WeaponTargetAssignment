import random
import numpy as np
import logging

# Define agent colors for visualization (optional)
AGENT_COLORS = ["red", "blue", "green", "yellow", "purple"]

class World:
    def __init__(self, num_weapons=3, num_targets=5, max_time_steps=100):
        self.targets = []
        self.weapons = []
        self.num_weapons = num_weapons
        self.num_targets = num_targets
        self.time_step = 0
        self.max_time_steps = max_time_steps

        self.target_destroyed_count = 0
        self.targets_spawned = 0
        logging.info("World initialized with %d weapons and %d targets", num_weapons, num_targets)

    def add_agent(self, agent):
        if agent.agent_type == 'weapon':
            self.weapons.append(agent)
        elif agent.agent_type == 'target':
            self.targets.append(agent)
        logging.info("Agent %s added to the world", agent.name)

    def update(self):
        self.time_step += 1

        # Update targets and check destruction
        for target in self.targets:
            target.update_state()
            if target.destroyed and not target.processed:
                self.target_destroyed_count += 1
                target.processed = True
                logging.info("Target %s destroyed", target.name)

        # Check if all targets are destroyed or time is exceeded
        all_targets_destroyed = all(t.destroyed for t in self.targets)
        if all_targets_destroyed or self.time_step >= self.max_time_steps:
            logging.info("World update complete: all targets destroyed or max time reached")

    def all_targets_destroyed(self):
        return self.target_destroyed_count == self.num_targets

    def time_exceeded(self):
        return self.time_step >= self.max_time_steps

class Scenario:
    def make_world(self, num_weapons=3, num_targets=5):
        world = World(num_weapons=num_weapons, num_targets=num_targets)

        # Initialize weapons
        weapon_spacing = 800 // (num_weapons + 1)
        for i in range(num_weapons):
            position = [(i + 1) * weapon_spacing, 550]
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            weapon = Weapon(name=f"Weapon_{i+1}", position=position, color=color)
            world.add_agent(weapon)

        # Initialize targets
        for j in range(num_targets):
            position = [random.randint(100, 700), random.randint(50, 200)]
            target = Target(name=f"Target_{j+1}", position=position, value=random.randint(50, 100))
            world.add_agent(target)

        logging.info("World created with %d weapons and %d targets", num_weapons, num_targets)
        return world

    def reset_world(self, world):
        world.targets = [
            Target(name=f"Target_{i+1}", position=[random.randint(100, 700), random.randint(50, 200)], value=random.randint(50, 100))
            for i in range(world.num_targets)
        ]
        world.weapons = [
            Weapon(name=f"Weapon_{i+1}", position=[(i + 1) * 800 // (world.num_weapons + 1), 550])
            for i in range(world.num_weapons)
        ]
        world.time_step = 0
        world.target_destroyed_count = 0
        for weapon in world.weapons:
            weapon.reset_state()
        logging.info("World reset")

    def reward(self, weapon, world):
        """Calculate the reward for a weapon."""
        reward = weapon.reward
        weapon.reward = 0.0
        logging.debug("Weapon %s reward calculated: %f", weapon.name, reward)
        return reward

    def observation(self, agent, world):
        """Generate observations for an agent (weapon or target)."""
        if agent.agent_type == 'weapon':
            weapon_pos = agent.position
            weapon_state = [agent.ammo, agent.cooldown]

            # Targets in view
            target_info = []
            for target in world.targets:
                if not target.destroyed:
                    distance = np.linalg.norm(np.array(target.position) - np.array(weapon_pos))
                    if distance <= agent.visibility_radius:
                        target_info.extend(target.position + [target.value, agent.calculate_hit_probability(target)])
                    else:
                        target_info.extend([-1, -1, -1, -1])  # Mask invisible targets
                else:
                    target_info.extend([-1, -1, -1, -1])  # Mask destroyed targets

            # Time step
            time_info = [world.time_step]

            obs = np.array(weapon_pos + weapon_state + target_info + time_info, dtype=np.float32)
            logging.debug("Observation generated for weapon %s", agent.name)
            return obs
        elif agent.agent_type == 'target':
            return np.zeros(1)

# Define Weapon and Target classes
class Weapon:
    def __init__(self, name, position, color=None):
        self.name = name
        self.position = position
        self.color = color
        self.agent_type = 'weapon'
        self.ammo = 10
        self.cooldown = 0
        self.reward = 0.0
        self.visibility_radius = 300

    def calculate_hit_probability(self, target):
        distance = np.linalg.norm(np.array(self.position) - np.array(target.position))
        return max(0.1, 1.0 - (distance / self.visibility_radius))  # Decreases with distance

    def reset_state(self):
        self.ammo = 10
        self.cooldown = 0
        self.reward = 0.0

class Target:
    def __init__(self, name, position, value):
        self.name = name
        self.position = position
        self.value = value
        self.agent_type = 'target'
        self.destroyed = False
        self.processed = False

    def update_state(self):
        # Placeholder for target behavior (e.g., moving)
        pass

from WeaponTargetAssignment.Environment.BasicEnvironment import BasicEnvironment


class Scenario:
    def __init__(self):
        self.world = None

    def make_world(self, num_weapons=3, num_targets=5, max_distance=15, front_line=1):
        """Create and initialize the world."""
        self.world = BasicEnvironment(
            num_weapons=num_weapons,
            num_targets=num_targets,
            max_distance=max_distance,
            front_line=front_line,
            reset_callback=self.reset_world,
            reward_callback=self.reward,
            observation_callback=self.observation,
        )
        self.world.reset()  # Explicitly reset the environment after creation
        return self.world

    def reset_world(self):
        """Reset the world."""
        return self.world.reset()

    def reward(self, actions):
        """Calculate the reward."""
        _, reward, _, _ = self.world.step(actions)
        return reward

    def observation(self):
        """Generate the current observation."""
        return self.world.get_obs()


def generate_scenario():
    """Generate a scenario and return the instance."""
    scenario = Scenario()
    scenario.make_world()
    return scenario

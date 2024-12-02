from WeaponTargetAssignment.Environment.BasicEnvironment import BasicEnvironment

class Scenario:
    def __init__(self):
        self.world = None

    def create_world(self, num_weapons=3, num_targets=5, max_distance=15, front_line=1):
        """
        Create and initialize the world (environment) for multi-agent settings.
        """
        self.world = BasicEnvironment(
            num_weapons=num_weapons,
            num_targets=num_targets,
            max_distance=max_distance,
            front_line=front_line,
        )
        return self.world

    def reset_world(self):
        """
        Reset the world (environment) to its initial state.
        """
        self.world.reset()

    def reward(self, actions):
        """
        Calculate the reward based on the current state and actions.
        """
        _, reward, _, _ = self.world.step(actions)
        return reward

    def observation(self):
        """
        Generate the current observation for all agents.
        """
        return self.world.get_obs()


def generate_scenario():
    """
    Generates a multi-agent scenario and assigns the required callbacks.
    """
    # Create a scenario
    scenario = Scenario()

    # Create and initialize the world
    world = scenario.create_world()

    # Assign callbacks
    return {
        "world": world,
        "reset_callback": scenario.reset_world,
        "reward_callback": scenario.reward,
        "observation_callback": scenario.observation,
    }

import random

class Scenario:
    def __init__(self, n_agents=3, num_targets=3):
        """
        Initializes the scenario parameters.
        """
        self.n_agents = n_agents
        self.num_targets = num_targets

    def make_world(self):
        """
        Creates the initial world with random weapons and targets.
        """
        weapon_types = self.n_agents
        target_types = self.num_targets

        world = {
            "weapons": {
                "types": weapon_types,
                "quantities": [random.randint(1, 10) for _ in range(weapon_types)]
            },
            "targets": {
                "types": target_types,
                "quantities": [random.randint(1, 10) for _ in range(target_types)],
                "values": [random.randint(10, 50) for _ in range(target_types)],
                "threat_levels": [random.uniform(0.5, 1.0) for _ in range(target_types)]
            },
            "probabilities": [
                [random.uniform(0.2, 1.0) for _ in range(target_types)] for _ in range(weapon_types)
            ],
            "costs": [
                [random.randint(1, 5) for _ in range(target_types)] for _ in range(weapon_types)
            ]
        }

        return world

    def reset_world(self, world):
        """
        Resets the world state with new random values.
        """
        for i in range(world["weapons"]["types"]):
            world["weapons"]["quantities"][i] = random.randint(1, 10)
        for j in range(world["targets"]["types"]):
            world["targets"]["quantities"][j] = random.randint(1, 10)

    def reward(self, world, actions):
        """
        Computes the reward based on weapon-target assignments.
        """
        reward = 0
        for i, weapon_qty in enumerate(world["weapons"]["quantities"]):
            for j, target_qty in enumerate(world["targets"]["quantities"]):
                if actions[i] == j and weapon_qty > 0 and target_qty > 0:
                    success_prob = world["probabilities"][i][j]
                    reward += world["targets"]["values"][j] * success_prob
                    reward -= world["costs"][i][j]
        return reward

    def observation(self, world):
        """
        Returns the current full state of the world.
        """
        return {
            "weapons": world["weapons"]["quantities"],
            "targets": world["targets"]["quantities"],
            "probabilities": world["probabilities"],
            "costs": world["costs"]
        }

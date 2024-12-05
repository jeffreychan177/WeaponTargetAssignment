import numpy as np

class Scenario:
    def make_world(self):
        """
        Creates the initial world with weapons and targets.
        """
        world = {
            "weapons": {
                "types": 3,  # Example: 3 weapon types
                "quantities": [5, 4, 3]  # Initial quantities of each type
            },
            "targets": {
                "types": 3,  # Example: 3 target types
                "quantities": [4, 3, 5],  # Initial quantities of each type
                "values": [10, 15, 20],  # Values for destroying each target type
                "threat_levels": [0.7, 0.9, 0.8]  # Threat levels of targets
            },
            "probabilities": [
                [0.8, 0.6, 0.3],  # Weapon type 1 probabilities against each target type
                [0.5, 0.7, 0.6],  # Weapon type 2
                [0.2, 0.4, 0.9]   # Weapon type 3
            ],
            "costs": [
                [1, 2, 3],  # Weapon type 1 costs against each target type
                [2, 1, 2],  # Weapon type 2
                [3, 2, 1]   # Weapon type 3
            ]
        }
        return world

    def reset_world(self, world):
        """
        Resets the world state.
        """
        for i in range(world["weapons"]["types"]):
            world["weapons"]["quantities"][i] = 5 - i  # Reset weapon quantities
        for j in range(world["targets"]["types"]):
            world["targets"]["quantities"][j] = 4 + j  # Reset target quantities

    def reward(self, world, actions):
        """
        Computes the reward based on weapon-target assignments.
        """
        reward = 0
        for i, weapon_qty in enumerate(world["weapons"]["quantities"]):
            for j, target_qty in enumerate(world["targets"]["quantities"]):
                if actions[i][j] > 0:
                    # Check if we have enough weapons of type i
                    if weapon_qty >= actions[i][j]:
                        success_prob = world["probabilities"][i][j]
                        reward += world["targets"]["values"][j] * (1 - (1 - success_prob) ** actions[i][j])
                        reward -= world["costs"][i][j] * actions[i][j]
        return reward

    def observation(self, world):
        """
        Returns the current state of the world.
        """
        return {
            "weapons": world["weapons"]["quantities"],
            "targets": world["targets"]["quantities"],
            "probabilities": world["probabilities"],
            "costs": world["costs"]
        }
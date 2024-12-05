class Scenario:
    def make_world(self):
        """
        Creates the initial world with weapons and targets.
        """
        world = {
            "weapons": {
                "types": 3,  # Example: 3 weapon types
                "quantities": np.random.randint(3, 7, size=3).tolist()  # Randomized quantities of each type
            },
            "targets": {
                "types": 3,  # Example: 3 target types
                "quantities": np.random.randint(3, 6, size=3).tolist(),  # Randomized quantities of each type
                "values": [10, 15, 20],  # Values for destroying each target type
                "threat_levels": np.random.uniform(0.5, 1.0, size=3).tolist()  # Random threat levels
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
            ],
            "environment_factor": np.random.uniform(0.8, 1.2)  # Environmental effect on probabilities
        }
        return world

    def reset_world(self, world):
        """
        Resets the world state with random fluctuations.
        """
        world["weapons"]["quantities"] = np.random.randint(3, 7, size=3).tolist()
        world["targets"]["quantities"] = np.random.randint(3, 6, size=3).tolist()
        world["targets"]["threat_levels"] = np.random.uniform(0.5, 1.0, size=3).tolist()
        world["environment_factor"] = np.random.uniform(0.8, 1.2)

    def reward(self, world, actions):
        """
        Computes the reward based on weapon-target assignments with random events.
        """
        reward = 0
        random_event = np.random.choice(["none", "bonus_target", "malfunction"], p=[0.8, 0.1, 0.1])

        # Handle random events
        if random_event == "bonus_target":
            new_target_value = np.random.randint(5, 20)
            reward += new_target_value
        elif random_event == "malfunction":
            malfunction_penalty = np.random.randint(1, 5)
            reward -= malfunction_penalty

        for i, weapon_qty in enumerate(world["weapons"]["quantities"]):
            for j, target_qty in enumerate(world["targets"]["quantities"]):
                if actions[i][j] > 0:
                    # Check if we have enough weapons of type i
                    if weapon_qty >= actions[i][j]:
                        success_prob = world["probabilities"][i][j] * world["environment_factor"]
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
            "threat_levels": world["targets"]["threat_levels"],
            "probabilities": world["probabilities"],
            "costs": world["costs"],
            "environment_factor": world["environment_factor"]
        }

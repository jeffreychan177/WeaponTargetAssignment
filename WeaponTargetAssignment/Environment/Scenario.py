import torch
import random
from gymnasium.spaces import Tuple, Discrete, Dict
import numpy as np
import gymnasium as gym

class Scenario:
    def make_world(self, device="cpu"):
        """
        Creates the initial world with random weapons and targets.
        """
        weapon_types = 3
        target_types = 3

        world = {
            "weapons": {
                "types": weapon_types,
                "quantities": torch.randint(1, 11, (weapon_types,), device=device)  # Random quantities between 1 and 10
            },
            "targets": {
                "types": target_types,
                "quantities": torch.randint(1, 11, (target_types,), device=device),  # Random quantities between 1 and 10
                "values": torch.randint(10, 51, (target_types,), device=device),  # Random values between 10 and 50
                "threat_levels": torch.rand(target_types, device=device) * 0.5 + 0.5  # Random threat levels between 0.5 and 1.0
            },
            "probabilities": torch.rand(weapon_types, target_types, device=device) * 0.8 + 0.2,  # Random probabilities
            "costs": torch.randint(1, 6, (weapon_types, target_types), device=device)  # Random costs between 1 and 5
        }
        return world

    def reset_world(self, world):
        """
        Resets the world state with new random values.
        """
        world["weapons"]["quantities"] = torch.randint(1, 11, (world["weapons"]["types"],), device=world["weapons"]["quantities"].device)
        world["targets"]["quantities"] = torch.randint(1, 11, (world["targets"]["types"],), device=world["targets"]["quantities"].device)

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
        Returns the current state of the world.
        """
        return {
            "weapons": world["weapons"]["quantities"].cpu().numpy(),
            "targets": world["targets"]["quantities"].cpu().numpy(),
            "probabilities": world["probabilities"].cpu().numpy(),
            "costs": world["costs"].cpu().numpy()
        }
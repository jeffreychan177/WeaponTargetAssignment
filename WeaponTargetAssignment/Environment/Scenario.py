import cupy as cp
import random
import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete, Dict
import numpy as np

class Scenario:
    def make_world(self):
        """
        Creates the initial world with random weapons and targets.
        """
        weapon_types = 3
        target_types = 3

        world = {
            "weapons": {
                "types": weapon_types,
                "quantities": cp.array([random.randint(1, 10) for _ in range(weapon_types)], dtype=cp.int32)
            },
            "targets": {
                "types": target_types,
                "quantities": cp.array([random.randint(1, 10) for _ in range(target_types)], dtype=cp.int32),
                "values": cp.array([random.randint(10, 50) for _ in range(target_types)], dtype=cp.float32),
                "threat_levels": cp.array([random.uniform(0.5, 1.0) for _ in range(target_types)], dtype=cp.float32)
            },
            "probabilities": cp.array([
                [random.uniform(0.2, 1.0) for _ in range(target_types)] for _ in range(weapon_types)
            ], dtype=cp.float32),
            "costs": cp.array([
                [random.randint(1, 5) for _ in range(target_types)] for _ in range(weapon_types)
            ], dtype=cp.float32)
        }
        return world

    def reset_world(self, world):
        """
        Resets the world state with new random values.
        """
        world["weapons"]["quantities"] = cp.array(
            [random.randint(1, 10) for _ in range(world["weapons"]["types"])], dtype=cp.int32
        )
        world["targets"]["quantities"] = cp.array(
            [random.randint(1, 10) for _ in range(world["targets"]["types"])], dtype=cp.int32
        )

    def reward(self, world, actions):
        """
        Computes the reward based on weapon-target assignments.
        """
        reward = 0
        for i, weapon_qty in enumerate(world["weapons"]["quantities"].get()):
            for j, target_qty in enumerate(world["targets"]["quantities"].get()):
                if actions[i] == j and weapon_qty > 0 and target_qty > 0:
                    success_prob = world["probabilities"][i, j]
                    reward += world["targets"]["values"][j] * success_prob
                    reward -= world["costs"][i, j]
        return reward

    def observation(self, world):
        """
        Returns the current state of the world.
        """
        return {
            "weapons": world["weapons"]["quantities"].get().tolist(),
            "targets": world["targets"]["quantities"].get().tolist(),
            "probabilities": world["probabilities"].get().tolist(),
            "costs": world["costs"].get().tolist()
        }
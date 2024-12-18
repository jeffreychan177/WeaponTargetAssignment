import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete, Dict
import numpy as np
import torch

class MultiStageEnvironment(gym.Env):
    def __init__(self, 
                 world, 
                 reset_callback, 
                 reward_callback, 
                 observation_callback, 
                 num_stages=3, 
                 partial_obs=False, 
                 n_agents=None, 
                 num_targets=None,
                 device='cpu'):
        """
        MultiStageEnvironment (PyTorch version)

        Parameters
        ----------
        world : dict
            The initial world state.
        reset_callback : function
            Function to reset the world.
        reward_callback : function
            Function to compute rewards.
        observation_callback : function
            Function to get the full state of the world.
        num_stages : int
            Number of stages in an episode.
        partial_obs : bool
            Whether to return partial observations.
        n_agents : int or None
            Number of agents. If None, derived from world.
        num_targets : int or None
            Number of targets. If None, derived from world.
        device : str
            'cpu' or 'cuda' device for torch tensors.
        """
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.num_stages = num_stages
        self.partial_obs = partial_obs
        self.device = device

        self._initialize_world(world)

        self.n_agents = n_agents if n_agents is not None else self.world["weapons"]["types"]
        self.n_targets = num_targets if num_targets is not None else self.world["targets"]["types"]

        # Define individual action spaces as discrete (number of target types + 1 for "no action")
        self.action_space = Tuple([Discrete(self.n_targets + 1) for _ in range(self.n_agents)])

        # Define observation spaces for each agent
        self.observation_space = Tuple([
            Dict({
                "weapons": Discrete(101),  # Weapon quantity (0-100)
                "targets": Discrete(101),  # Target quantities
                "probabilities": Dict({j: Discrete(101) for j in range(self.n_targets)}),
                "costs": Dict({j: Discrete(11) for j in range(self.n_targets)})
            }) for _ in range(self.n_agents)
        ])

        self.stage = 1

    def _initialize_world(self, world):
        """
        Convert the world dictionary values to PyTorch tensors for internal usage.
        """
        self.world = {
            "weapons": {
                "types": world["weapons"]["types"],
                "quantities": torch.tensor(world["weapons"]["quantities"], dtype=torch.float32).to(self.device)
            },
            "targets": {
                "types": world["targets"]["types"],
                "quantities": torch.tensor(world["targets"]["quantities"], dtype=torch.float32).to(self.device),
                "values": torch.tensor(world["targets"]["values"], dtype=torch.float32).to(self.device),
                "threat_levels": torch.tensor(world["targets"]["threat_levels"], dtype=torch.float32).to(self.device)
            },
            "probabilities": torch.tensor(world["probabilities"], dtype=torch.float32).to(self.device),
            "costs": torch.tensor(world["costs"], dtype=torch.float32).to(self.device)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_callback(self._convert_to_cpu_world())  # Callback expects CPU/Python objects
        # Re-initialize the tensors from the possibly updated python dictionary after reset
        # Because reset_callback modifies in-place, we must fetch the modified values.
        updated_world = self._convert_to_cpu_world()
        self._initialize_world(updated_world)
        self.stage = 1
        obs = self._get_obs()
        return obs, {}

    def step(self, actions):
        # Convert actions to torch if needed
        # actions are typically python ints, just use them directly in indexing
        individual_rewards = []
        target_state_update = torch.zeros_like(self.world["targets"]["quantities"], dtype=torch.float32)

        for i, action in enumerate(actions):
            reward = 0.0  # ensure this is a Python float
            if action < self.world["targets"]["quantities"].shape[0]:
                target = action
                if self.world["weapons"]["quantities"][i] > 0 and self.world["targets"]["quantities"][target] > 0:
                    probs = self.world["probabilities"][:, target]
                    prob_survive = torch.prod(1 - probs)
                    destroyed = 1 - prob_survive
                    success_prob = self.world["probabilities"][i, target]
                    
                    val = self.world["targets"]["values"][target]
                    cost = self.world["costs"][i, target]
                    
                    # Use .item() to convert torch tensors to Python floats
                    destroyed_val = destroyed.item()
                    val = val.item()
                    cost = cost.item()
                    
                    reward += val * destroyed_val
                    reward -= cost

                    self.world["weapons"]["quantities"][i] -= 1
                    self.world["targets"]["quantities"][target] -= destroyed

            individual_rewards.append(reward)

        combined_reward = sum(individual_rewards)  # This should now be a float sum of Python floats
        combined_reward = float(combined_reward)   # Ensure explicit conversion (just for safety)


        # Check if stage is done
        stage_terminated = self._check_stage_termination()
        terminated = False
        if stage_terminated:
            if self.stage < self.num_stages:
                self.stage += 1
                self._on_stage_transition()
            else:
                terminated = True

        truncated = False

        obs = self._get_obs()
        return obs, combined_reward, terminated, truncated, {}

    def _check_stage_termination(self):
        return (
            torch.all(self.world["targets"]["quantities"] == 0) or
            torch.all(self.world["weapons"]["quantities"] == 0)
        )

    def _on_stage_transition(self):
        # Random initialization is on CPU, so we do it in numpy and convert
        new_target_quantities = np.random.randint(1, 10, size=self.world["targets"]["types"])
        new_weapon_quantities = np.random.randint(1, 5, size=self.world["weapons"]["types"])

        self.world["targets"]["quantities"] = torch.tensor(new_target_quantities, dtype=torch.float32).to(self.device)
        self.world["weapons"]["quantities"] = torch.tensor(new_weapon_quantities, dtype=torch.float32).to(self.device)

    def _convert_to_cpu_world(self):
        """
        Convert the torch-based world back to python lists and dicts for callbacks if needed.
        """
        return {
            "weapons": {
                "types": self.world["weapons"]["types"],
                "quantities": self.world["weapons"]["quantities"].cpu().tolist()
            },
            "targets": {
                "types": self.world["targets"]["types"],
                "quantities": self.world["targets"]["quantities"].cpu().tolist(),
                "values": self.world["targets"]["values"].cpu().tolist(),
                "threat_levels": self.world["targets"]["threat_levels"].cpu().tolist()
            },
            "probabilities": self.world["probabilities"].cpu().tolist(),
            "costs": self.world["costs"].cpu().tolist()
        }

    def _get_obs(self):
        # Get the full observation from scenario (in CPU/numpy format)
        full_world_cpu = self._convert_to_cpu_world()
        full_obs = self.observation_callback(full_world_cpu)

        # Apply partial observability
        visible_targets_count = self.n_targets // 2 if self.partial_obs else self.n_targets

        obs = []
        for i in range(self.n_agents):
            agent_obs = {
                "weapons": int(np.clip(full_obs["weapons"][i], 0, 100)),
                "targets": int(np.clip(sum(full_obs["targets"][:visible_targets_count]), 0, 100)),
                "probabilities": {
                    j: int(np.clip(full_obs["probabilities"][i][j] * 100, 0, 100))
                    for j in range(visible_targets_count)
                },
                "costs": {
                    j: int(np.clip(full_obs["costs"][i][j], 0, 10))
                    for j in range(visible_targets_count)
                }
            }
            obs.append(agent_obs)

        # Gym expects the observations in a numpy or python format
        return tuple(obs)

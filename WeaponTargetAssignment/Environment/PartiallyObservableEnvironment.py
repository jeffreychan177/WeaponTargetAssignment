import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete, Dict
import numpy as np

class MultiStageEnvironment(gym.Env):
    def __init__(self, 
                 world, 
                 reset_callback, 
                 reward_callback, 
                 observation_callback, 
                 num_stages=3, 
                 partial_obs=False, 
                 n_agents=None, 
                 num_targets=None):
        """
        Initialize the MultiStageEnvironment with given parameters.
        
        Parameters:
        -----------
        world : dict
            The initial world state dictionary (weapons, targets, probabilities, etc.).
        reset_callback : function
            A callback function to reset the world.
        reward_callback : function
            A callback function to compute rewards.
        observation_callback : function
            A callback function to produce observations.
        num_stages : int, optional
            Number of stages in the episode.
        partial_obs : bool, optional
            Whether to use partial observability.
        n_agents : int, optional
            Number of agents (weapons) to simulate.
        num_targets : int, optional
            Number of targets to simulate.
        """
        self.world = world
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.num_stages = num_stages
        self.partial_obs = partial_obs

        # Use provided n_agents and num_targets if available, else derive from world
        self.n_agents = n_agents if n_agents is not None else self.world["weapons"]["types"]
        n_targets = num_targets if num_targets is not None else self.world["targets"]["types"]

        # Define individual action spaces as discrete (number of target types + 1 for "no action")
        self.action_space = Tuple([Discrete(n_targets + 1) for _ in range(self.n_agents)])

        # Define observation spaces for each agent
        # Adjust observation space if partial_obs is True (for simplicity we keep the same shape, but in practice
        # partial observations might reduce the dimension or show only a subset of the world).
        self.observation_space = Tuple([
            Dict({
                "weapons": Discrete(101),  # Weapon quantity (0-100)
                "targets": Discrete(101),  # Target quantities
                "probabilities": Dict({j: Discrete(101) for j in range(n_targets)}),  # Success probabilities
                "costs": Dict({j: Discrete(11) for j in range(n_targets)})  # Costs
            }) for _ in range(self.n_agents)
        ])

        self.stage = 1  # Start with the first stage

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Ensure proper seeding behavior
        self.reset_callback(self.world)
        self.stage = 1  # Reset to the first stage
        return self._get_obs(), {}

    def step(self, actions):
        individual_rewards = []
        target_state_update = np.zeros_like(self.world["targets"]["quantities"])

        for i, action in enumerate(actions):
            reward = 0
            # If action is the last discrete value, it's "no action"
            if action < len(self.world["targets"]["quantities"]):
                target = action
                if self.world["weapons"]["quantities"][i] > 0 and self.world["targets"]["quantities"][target] > 0:
                    # Compute survival probability
                    prob_survive = np.prod([
                        1 - self.world["probabilities"][i][target]
                        for i in range(self.n_agents)
                    ])
                    destroyed = 1 - prob_survive
                    target_state_update[target] = destroyed

                    # Compute reward for the action
                    success_prob = self.world["probabilities"][i][target]
                    reward += self.world["targets"]["values"][target] * destroyed
                    reward -= self.world["costs"][i][target]

                    # Update weapon and target states
                    self.world["weapons"]["quantities"][i] -= 1
                    self.world["targets"]["quantities"][target] -= destroyed

            individual_rewards.append(reward)

        # Update the target states
        self.world["targets"]["quantities"] = np.maximum(
            self.world["targets"]["quantities"] - target_state_update, 0
        )

        # Combine rewards (e.g., sum all agent rewards)
        combined_reward = sum(individual_rewards)

        # Determine if the current stage is complete
        stage_terminated = self._check_stage_termination()
        terminated = False

        if stage_terminated:
            if self.stage < self.num_stages:
                self.stage += 1
                self._on_stage_transition()  # Handle stage-specific logic
            else:
                terminated = True  # End the episode after the final stage

        truncated = False  # Modify if you have truncation conditions

        return self._get_obs(), combined_reward, terminated, truncated, {}

    def _check_stage_termination(self):
        return (
            all(q == 0 for q in self.world["targets"]["quantities"]) or
            all(q == 0 for q in self.world["weapons"]["quantities"])
        )

    def _on_stage_transition(self):
        # Example stage transition logic
        self.world["targets"]["quantities"] = np.random.randint(1, 10, size=self.world["targets"]["types"])
        self.world["weapons"]["quantities"] = np.random.randint(1, 5, size=self.n_agents)

    def _get_obs(self):
        obs = self.observation_callback(self.world)
        # If partial_obs is True, modify obs to return partial information.
        # For demonstration, we just return the same format, but you would slice or mask the observation.
        # E.g., show fewer targets or only local info for partial observations.
        
        # Convert probabilities and costs to discrete values as before.
        return tuple({
            "weapons": int(np.clip(obs["weapons"][i], 0, 100)),
            "targets": int(np.clip(sum(obs["targets"]), 0, 100)),
            "probabilities": {
                j: int(np.clip(obs["probabilities"][i][j] * 100, 0, 100))
                for j in range(len(obs["targets"]))
            },
            "costs": {
                j: int(np.clip(obs["costs"][i][j], 0, 10))
                for j in range(len(obs["targets"]))
            }
        } for i in range(self.n_agents))

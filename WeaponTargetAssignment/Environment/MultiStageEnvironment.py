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
        MultiStageEnvironment

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
        """
        self.world = world
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.num_stages = num_stages
        self.partial_obs = partial_obs

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_callback(self.world)
        self.stage = 1
        return self._get_obs(), {}

    def step(self, actions):
        individual_rewards = []
        target_state_update = np.zeros_like(self.world["targets"]["quantities"])

        for i, action in enumerate(actions):
            reward = 0
            # If action == number of targets (the last discrete value), it's "no action"
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

        # Update target states
        self.world["targets"]["quantities"] = np.maximum(
            self.world["targets"]["quantities"] - target_state_update, 0
        )

        # Combine rewards
        combined_reward = sum(individual_rewards)

        # Determine if stage is done
        stage_terminated = self._check_stage_termination()
        terminated = False

        if stage_terminated:
            if self.stage < self.num_stages:
                self.stage += 1
                self._on_stage_transition()
            else:
                terminated = True

        truncated = False

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
        full_obs = self.observation_callback(self.world)

        # Apply partial observability if enabled
        # For demonstration, if partial_obs is True, we only show half of the targets (rounded down).
        # In a real scenario, you'd implement a more meaningful partial observation scheme.
        visible_targets_count = self.n_targets // 2 if self.partial_obs else self.n_targets

        # Clip values to defined discrete ranges
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

        return tuple(obs)

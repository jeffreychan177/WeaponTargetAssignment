from environment.scenario import generate_multiagent_scenario
from gymnasium import register

scenario = generate_multiagent_scenario()
world = scenario.make_world()

register(
    id='Environment-v0',
    entry_point='weaponTargetAssignmentEnv.environment:multiagentEnv',
    kwargs={
        'world': world,
        'reset_callback': scenario.reset_world,
        'reward_callback': scenario.reward,
        'observation_callback': scenario.observation
    }
)

from wtae.scenario import generate_scenario
from gymnasium import register

scenario = generate_scenario()
world = scenario.make_world()

register(
    id='Environment-v0',
    entry_point='wtae.multiagent_env:MultiAgentWeaponTargetEnv',
    kwargs={
        'world': world,
        'reset_callback': scenario.reset_world,
        'reward_callback': scenario.reward,
        'observation_callback': scenario.observation
    }
)

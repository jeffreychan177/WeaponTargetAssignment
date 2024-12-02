from WeaponTargetAssignment.Environment.Scenario import generate_scenario
from gymnasium import register

scenario = generate_scenario()
world = scenario.make_world()

register(
    id='Environment-v0',
    entry_point='WeaponTargetAssignment.Environment:BasicEnvironment',
    kwargs={
        'world': world,
        'reset_callback': scenario.reset_world,
        'reward_callback': scenario.reward,
        'observation_callback': scenario.observation
    }
)


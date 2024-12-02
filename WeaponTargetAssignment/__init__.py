from WeaponTargetAssignment.Environment.Scenario import generate_scenario
from gymnasium import register


scenario = generate_scenario()

register(
    id='Environment-v0',
    entry_point='WeaponTargetAssignment.Environment:BasicEnvironment',
    kwargs={
        'world': scenario.world, 
        'reset_callback': scenario.reset_world,
        'reward_callback': scenario.reward,
        'observation_callback': scenario.observation,
    }
)

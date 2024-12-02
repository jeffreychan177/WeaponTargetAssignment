from WeaponTargetAssignment.Environment.Scenario import Scenario
from gymnasium import register

scenario = Scenario()
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
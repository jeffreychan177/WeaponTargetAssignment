from WeaponTargetAssignment.Environment.Scenario import Scenario
from gymnasium import register

scenario = Scenario()
world = scenario.make_world()

register(
    id='Environment-v0',
    entry_point='WeaponTargetAssignment.Environment:SingleStageEnvironment',
    kwargs={
        'world': world,
        'reset_callback': scenario.reset_world,
        'reward_callback': scenario.reward,
        'observation_callback': scenario.observation
    }
)

register(
    id='Environment-v1',
    entry_point='WeaponTargetAssignment.Environment:TwoStageEnvironment',
    kwargs={
        'world': world,
        'reset_callback': scenario.reset_world,
        'reward_callback': scenario.reward,
        'observation_callback': scenario.observation
    }
)

register(
    id='Environment-v2',
    entry_point='WeaponTargetAssignment.Environment:MultiStageEnvironment',
    kwargs={
        'world': world,
        'reset_callback': scenario.reset_world,
        'reward_callback': scenario.reward,
        'observation_callback': scenario.observation
    }
)

register(
    id='Environment-v3',
    entry_point='WeaponTargetAssignment.Environment:FullyObservableSingleStageEnvironment',
    kwargs={
        'world': world,
        'reset_callback': scenario.reset_world,
        'reward_callback': scenario.reward,
        'observation_callback': scenario.observation
    }
)
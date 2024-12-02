from WeaponTargetAssignment.Environment.Scenario import generate_scenario
from gymnasium import register

# Generate the scenario
scenario = generate_scenario()
world = scenario.world

# Register the environment
register(
    id='Environment-v0',
    entry_point='WeaponTargetAssignment.Environment.BasicEnvironment:BasicEnvironment',
    kwargs={
        'world': world,
        'reset_callback': scenario.reset_world,
        'reward_callback': scenario.reward,
        'observation_callback': scenario.observation,
    }
)

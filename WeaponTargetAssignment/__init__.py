from WeaponTargetAssignment.Environment.Scenario import generate_scenario
from gymnasium import register

# Get the Scenario instance
scenario = generate_scenario()

register(
    id='Environment-v0',
    entry_point='WeaponTargetAssignment.Environment:BasicEnvironment',
    kwargs={
        'world': scenario.world,  # Access the world directly from the Scenario
        'reset_callback': scenario.reset_world,
        'reward_callback': scenario.reward,
        'observation_callback': scenario.observation,
    }
)

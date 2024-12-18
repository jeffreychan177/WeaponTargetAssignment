from itertools import product
from gymnasium import register
from WeaponTargetAssignment.Environment.Scenario import Scenario

# Define parameter ranges
partial_obs_options = [True, False]
num_agents_options = [1, 3, 5]      
num_stages_options = [1,2,3]         
num_targets_options = [5, 10] 

# For each combination, create a scenario and register a unique environment ID
for po, na, ns, nt in product(partial_obs_options, num_agents_options, num_stages_options, num_targets_options):
    # Create a scenario with given parameters (assuming Scenario can handle them)
    scenario = Scenario(n_agents=na, num_targets=nt, partial_obs=po)
    world = scenario.make_world()

    env_id = f"WTA-Po{int(po)}-A{na}-Stg{ns}-T{nt}-v0"
    
    register(
        id=env_id,
        entry_point='WeaponTargetAssignment.Environment:MultiStageEnvironment',
        kwargs={
            'world': world,
            'reset_callback': scenario.reset_world,
            'reward_callback': scenario.reward,
            'observation_callback': scenario.observation,
            'num_stages': ns,
            'partial_obs': po,
            'n_agents': na,
            'num_targets': nt
        }
    )

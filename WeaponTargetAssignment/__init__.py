from itertools import product
from gymnasium import register
from Scenario import Scenario

# Define parameter ranges
partial_obs_options = [True, False]
num_agents_options = [1, 2, 5]      
num_stages_options = [1,2,3]         
num_targets_options = [5, 10]       

for po, na, ns, nt in product(partial_obs_options, num_agents_options, num_stages_options, num_targets_options):
    scenario = Scenario(n_agents=na, num_targets=nt)
    world = scenario.make_world()

    env_id = f"WTA-Po{int(po)}-A{na}-Stg{ns}-T{nt}-v0"
    register(
        id=env_id,
        entry_point='MultiStageEnvironment:MultiStageEnvironment',
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

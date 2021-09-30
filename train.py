import numpy as np
import gym
import os, sys
from rrc_example_package.her.arguments import get_args
from mpi4py import MPI
# from rrc_example_package.her.rl_modules.ddpg_agent import ddpg_agent
from rrc_example_package.her.rl_modules.ddpg_agent_rrc import ddpg_agent_rrc
import random
import torch

from rrc_example_package import cube_trajectory_env
from rrc_example_package.benchmark_rrc.python.residual_learning.residual_wrappers import RandomizedEnvWrapper
from rrc_example_package.cube_trajectory_env import ActionType

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def main():
    # get the params
    args = get_args()
    if MPI.COMM_WORLD.Get_rank() == 0: print('\n#########\nArgs for {}:\n{}\n'.format(args.exp_dir, args))
        
    if args.action_type == 'torque':
        action_type = ActionType.TORQUE
    else:
        raise NotImplementedError("Only torque actions are currently supported")
        
    # initialise environment
    env = cube_trajectory_env.SimtoRealEnv(visualization=False,
                                           max_steps=args.ep_len,
                                           xy_only=(args.xy_only==1),
                                           steps_per_goal=args.steps_per_goal,
                                           step_size=args.step_size,
                                           env_type='sim',
                                           obs_type=args.obs_type,
                                           env_wrapped=(args.domain_randomization==1),
                                           action_type=action_type,
                                           increase_fps=(args.increase_fps==1),
                                           disable_arm3=args.disable_arm3,
                                           distance_threshold=0.02
                                           )
    # wrap in domain randomisation environment
    if args.domain_randomization == 1:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('....with Domain Randomization')
        env = RandomizedEnvWrapper(env, flatten_obs=True)
    

    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('Env params:')
        print(env_params)
        
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent_rrc(args, env, env_params)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    main()
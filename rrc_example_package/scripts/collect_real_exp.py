#!/usr/bin/env python3
"""
Incomplete code for collecting data with the real robot

"""
import json
import sys

import torch
from rrc_example_package.her.rl_modules.models import actor
from rrc_example_package.her.arguments import get_args
import gym
import numpy as np
from rrc_example_package import cube_trajectory_env
import time


# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std):
    clip_obs = 200
    clip_range = 5
    o_clip = np.clip(o, -clip_obs, clip_obs)
    g_clip = np.clip(g, -clip_obs, clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -clip_range, clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -clip_range, clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

# this function will choose action for the agent and do the exploration
def select_actions(pi, args, env_params):
    action = pi.cpu().numpy().squeeze()
    # add the gaussian
    action += args.noise_eps * env_params['action_max'] * np.random.randn(*action.shape)
    action = np.clip(action, -env_params['action_max'], env_params['action_max'])
    # random actions...
    random_actions = np.random.uniform(low=-env_params['action_max'], high=env_params['action_max'], \
                                        size=env_params['action'])
    # choose if use the random actions
    action += np.random.binomial(1, args.random_eps, 1)[0] * (random_actions - action)
    return action

def main():
    # get the params
    args = get_args()
    
    goal = None
    load_actor=True
    step_size=50
    max_steps=int(2500/step_size)
    print('REVERT max steps!!!')
    steps_per_goal=30
    difficulty=3
    env_type='real'
    obs_type='full'
    model_path = '/userhome/real_acmodel_latest.pt'
    disable_arm3=1
        
    # Make sim environment
    sim_env = cube_trajectory_env.SimtoRealEnv(visualization=False, max_steps=max_steps, \
                                               xy_only=False, steps_per_goal=steps_per_goal, step_size=step_size,\
                                                   env_type='sim', obs_type=obs_type, env_wrapped=False,\
                                                       increase_fps=False, goal_trajectory=goal, disable_arm3=disable_arm3)
    # get the env param
    observation = sim_env.reset(difficulty=difficulty, init_state='normal')
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': sim_env.action_space.shape[0], 
                  'action_max': sim_env.action_space.high[0],
                  }
    # delete sim so not using memory
    del sim_env
    
    if load_actor:
        # load the model param
        # model_path = 'src/rrc_example_package/rrc_example_package/her/saved_models/16pushsimptrajload2/ac_model25.pt'
        print('Loading in model from: {}'.format(model_path))
        o_mean, o_std, g_mean, g_std, model, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
        actor_network = actor(env_params)
        actor_network.load_state_dict(model)
        actor_network.eval()
    
    if disable_arm3:
        print('WARNING: disabling 3rd robot arm!!')
    
    print('Env type: {}'.format(env_type))
    # Make real environment
    env = cube_trajectory_env.SimtoRealEnv(visualization=False, max_steps=max_steps, \
                                               xy_only=False, steps_per_goal=steps_per_goal, step_size=step_size,\
                                                   env_type=env_type, obs_type=obs_type, env_wrapped=False,\
                                                       increase_fps=False, goal_trajectory=goal, disable_arm3=disable_arm3)
    print('Beginning to collect experience...')
    assert steps_per_goal == 30
    t0 = time.time()
    done = False
    mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
    success = []
    
    # reset the environment
    observation = env.reset(difficulty=difficulty, init_state='normal')
    obs = observation['observation']
    ag = observation['achieved_goal']
    g = observation['desired_goal']
    # start to collect samples
    while not done:
        with torch.no_grad():
            input_tensor = process_inputs(obs, g, o_mean, o_std, g_mean, g_std)
            pi = actor_network(input_tensor)
            action = select_actions(pi, args, env_params)
        # feed the actions into the environment
        observation_new, _, _, info = env.step(action)
        obs_new = observation_new['observation']
        ag_new = observation_new['achieved_goal']
        g_new = observation_new['desired_goal']
        # append rollouts
        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())
        ep_g.append(g.copy())
        ep_actions.append(action.copy())
        success.append(info['is_success'])
        # re-assign the observation
        obs = obs_new
        ag = ag_new
        g = g_new
    ep_obs.append(obs.copy())
    ep_ag.append(ag.copy())
    ep_g.append(g.copy())
    # Into minibatch
    mb_obs.append(ep_obs)
    mb_ag.append(ep_ag)
    mb_g.append(ep_g)
    mb_actions.append(ep_actions)
    # convert them into arrays
    mb_obs = np.array(mb_obs)
    mb_ag = np.array(mb_ag)
    mb_g = np.array(mb_g)
    mb_actions = np.array(mb_actions)
    success_rate = np.sum(success) / len(success)
    
    t1 = time.time()
    print('Time taken: {:.2f} seconds'.format(t1-t0))
    print()
    print('mb_obs.shape: {}'.format(mb_obs.shape))
    print('mb_ag.shape: {}'.format(mb_ag.shape))
    print('mb_g.shape: {}'.format(mb_g.shape))
    print('mb_actions.shape: {}'.format(mb_actions.shape))
    print()
    print('success_rate: {}'.format(success_rate))
    print('Check goals are changing at correct time!!!')
    print('Sort out exploration vs exploitiation')
    
    buffer = {
        'obs': mb_obs,
        'ag': mb_ag,
        'g': mb_g,
        'actions': mb_actions,
        'success_rate': success_rate
        }
    np.save('/buffer.npy', buffer)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotCubeTrajectoryEnv environment and runs one episode
using a dummy policy.
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

def main():
    # goal=None
    # the goal is passed as JSON string
    goal_json = sys.argv[1]
    goal = json.loads(goal_json)
    print(goal)
    
    load_actor=True
    max_steps=1000000
    steps_per_goal=100
    env_type='real'
    
    # # PUSH SUBMISSION
    # ###########
    # step_size=10
    # difficulty=1
    # obs_type='default'
    # model_path = '/userhome/26wcollisions_acmodel49.pt'
    # disable_arm3=0
    # ###########
    
    # # 'CUPPING' SUBMISSION
    # #############
    # step_size=50
    # difficulty=3
    # obs_type='full'
    # model_path = '/userhome/3armzfullDR_acmodel64.pt'
    # disable_arm3=0
    # #############
    
    # 'PINCH' SUBMISSION
    #############
    step_size=50
    difficulty=3
    obs_type='default'
    model_path = '/userhome/25wcollision_acmodel35.pt'
    disable_arm3=0
    #############
    
        
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
    print('Beginning evaluate_stage1...')
    t0 = time.time()
    done = False
    
    xy_fails = 0
    rand_actions = 5
    fails_threshold = 50
    
    observation = env.reset(difficulty=difficulty, init_state='normal')
    while not done:
        obs = observation['observation']
        g = observation['desired_goal']
        if difficulty == 1:
            g[2] = 0.0325
        if xy_fails < fails_threshold:
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
        else:
            action = env.action_space.sample()
            print('Stuck - taking random action!!!')
            if xy_fails > fails_threshold + rand_actions:
                xy_fails = 0
        # put actions into the environment
        observation, reward, done, info = env.step(action)
        
        if info['xy_fail']:
            xy_fails += 1
        else:
            xy_fails = 0
            
        print('t_step={}, g={}, r={}, rrc={:.1f}, xy_fail={}'.format(info['time_index'], g, reward, info['rrc_reward'], info['xy_fail']))
    tf = time.time()
    print('Time taken: {:.2f} seconds'.format(tf-t0))
    
    print('\nRRC reward: {}'.format(info['rrc_reward']))

if __name__ == "__main__":
    main()
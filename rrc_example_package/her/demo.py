import torch
from rrc_example_package.her.rl_modules.models import actor, critic
from rrc_example_package.her.arguments import get_args
import gym
import numpy as np

from rrc_example_package import cube_trajectory_env
from rrc_example_package.benchmark_rrc.python.residual_learning.residual_wrappers import RandomizedEnvWrapper
import time

# from trifinger_simulation.tasks import move_cube
# import trifinger_simulation.tasks.move_cube_on_trajectory as task

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
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

def get_z_reward(obs, g, args, env):
    # g_z = g[2] - (move_cube._CUBE_WIDTH / 2)
    # obs_z = obs[29] - (move_cube._CUBE_WIDTH / 2)
    # print('Goal z: {:2f}, AG z: {:2f}'.format(10*g_z, 10*obs_z))
    obs = np.expand_dims(obs[...,env.z_pos], axis=-1)
    g = np.expand_dims(g[...,2], axis=-1)
    # print('get_z_reward obs: {}, g: {}'.format(obs, g))
    z_dist = np.abs(g - obs)
    # punish less if above goal
    scale = g > obs
    scale = (scale + 1) / 2
    # reward is negative of z distance
    return -args.z_scale * scale * z_dist # TODO: verify still works if changed obs space

def main():
    # task.move_cube._CUBE_WIDTH = 0.02
    
    args = get_args()
    # load the model param
    model_path = 'src/rrc_example_package/rrc_example_package/her/saved_models/report/25_wcollision/25wcollision_acmodel35.pt' #args.save_dir + 'rrc_run6/ac_model130.pt'
    # model_path = 'src/rrc_example_package/trained_models/25_50step_acmodel102.pt'
    o_mean, o_std, g_mean, g_std, model, crit = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    args.simtoreal=1
    args.domain_randomization=0
    random_scale=1.0
    max_steps=300
    step_size=50
    steps_per_goal=30
    difficulty=3
    noisy_resets=1
    noise_level=1
    obs_type='default'
    explore=0
    disable_arm3=0
    
    # create the environment
    print()
    if args.simtoreal == 1:
        print('sim-to-real env')
        env = cube_trajectory_env.SimtoRealEnv(visualization=True, max_steps=max_steps, \
                                               xy_only=0, steps_per_goal=steps_per_goal, step_size=step_size,\
                                                   env_type='sim', obs_type=obs_type, env_wrapped=(args.domain_randomization==1),
                                                   disable_arm3=disable_arm3)
        if args.domain_randomization == 1:
            print('Domain randomization')
            env = RandomizedEnvWrapper(env, flatten_obs=True, random_scale=random_scale)
    else:
        print('Ordinary env')
        env = cube_trajectory_env.CustomSimCubeEnv(visualization=True, max_steps=max_steps, \
                                                   xy_only=0, steps_per_goal=steps_per_goal, step_size=step_size,
                                                   obs_type=obs_type, disable_arm3=disable_arm3)
    print()
    # get the env param
    observation = env.reset(difficulty=difficulty, init_state='normal', noisy=noisy_resets, noise_level=noise_level)
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # print('Observation Space:')
    # print(env.observation_space)
    print('\nAction space:')
    print(env.action_space)
    print(env.action_space.high[0])
    print(env.action_space.low[0])
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    critic_network = critic(env_params)
    critic_network.load_state_dict(crit)
    critic_network.eval()
    t0 = time.time()
    
    input()
    xy_fails = 0
    rand_actions = 5
    xy_threshold = 10
    for i in range(1):
        # observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        is_done = False
        print('obs[ag]: {}'.format(observation['achieved_goal']))
        # print('ENV.Z_POS: {}'.format(env.z_pos))
        for t in range(env._max_episode_steps):
        # while not is_done:
            obs = observation['observation']
            g = observation['desired_goal']
            # env.render()
            # input()
            # print('r_z: {}'.format(get_z_reward(obs, g, args, env)))
            # print('obs z_pos: {}, ag z_pos: {}'.format(obs[32], observation['achieved_goal'][2]))
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
                if explore or xy_fails > xy_threshold:
                    action = select_actions(pi, args, env_params)
                    print('taking random action')
                    if xy_fails > xy_threshold + rand_actions:
                        xy_fails = 0
                else:
                    action = pi.detach().numpy().squeeze()
                q = critic_network(torch.unsqueeze(inputs,0), torch.unsqueeze(torch.tensor(action),0))
            # print('Action: {}'.format(action.mean()))
            # put actions into the environment
            # action = env.action_space.sample()
            observation, reward, is_done, info = env.step(action)
            # input()
            # print('t={}, g={}, r={}, q={:.3f}, rrc={:.1f}'.format(info["time_index"]/step_size, observation['desired_goal'], reward, q.numpy().squeeze(), info["rrc_reward"]))
            if info['xy_fail']:
                xy_fails += 1
            else:
                xy_fails = 0
            print('xy_fail: {}, count: {}'.format(info['xy_fail'], xy_fails))
            # print(info)
            # obs = observation['observation']
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
    t1 = time.time()
    print('Time taken: {:.2f} seconds'.format(t1-t0))

if __name__ == '__main__':
    main()
    

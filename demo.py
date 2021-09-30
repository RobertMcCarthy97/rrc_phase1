import torch
from rrc_example_package.her.rl_modules.models import actor, critic
from rrc_example_package.her.arguments import get_args
import gym
import numpy as np

from rrc_example_package import cube_trajectory_env
from rrc_example_package.benchmark_rrc.python.residual_learning.residual_wrappers import RandomizedEnvWrapper
import time

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


def main():    
    args = get_args()
    
    # Manually implemented arguments
    args.domain_randomization=False
    max_steps=300
    step_size=50
    steps_per_goal=30
    difficulty=3
    noisy_resets=True
    obs_type='default'
    explore=False
    model_path = 'rrc_example_package/her/saved_models/final_pinch_policy.pt'
    
    # load the model param
    o_mean, o_std, g_mean, g_std, model, crit = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    # create the environment
    env = cube_trajectory_env.SimtoRealEnv(visualization=True, max_steps=max_steps, \
                                           xy_only=0, steps_per_goal=steps_per_goal, step_size=step_size,\
                                               env_type='sim', obs_type=obs_type, env_wrapped=(args.domain_randomization==1),
                                               )
    if args.domain_randomization == 1:
        print('Using domain randomization')
        env = RandomizedEnvWrapper(env, flatten_obs=True)

    # get the env params
    observation = env.reset(difficulty=difficulty, init_state='normal', noisy=noisy_resets)
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    print('\nEnv params:')
    print(env_params)
    
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    # create critic network
    critic_network = critic(env_params)
    critic_network.load_state_dict(crit)
    critic_network.eval()
    
    t0 = time.time()
    for i in range(1):
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        is_done = False
        for t in range(env._max_episode_steps):
            obs = observation['observation']
            g = observation['desired_goal']
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
                if explore:
                    action = select_actions(pi, args, env_params)
                else:
                    action = pi.detach().numpy().squeeze()
                # q = critic_network(torch.unsqueeze(inputs,0), torch.unsqueeze(torch.tensor(action),0))
            observation, reward, is_done, info = env.step(action)
            
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
    t1 = time.time()
    print('Time taken: {:.3f} seconds'.format(t1-t0))

if __name__ == '__main__':
    main()
    

import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from rrc_example_package.her.mpi_utils.mpi_utils import sync_networks, sync_grads
from rrc_example_package.her.rl_modules.replay_buffer import replay_buffer
from rrc_example_package.her.rl_modules.models import actor, critic
from rrc_example_package.her.mpi_utils.normalizer import normalizer
from rrc_example_package.her.her_modules.her import her_sampler


"""
ddpg with HER (MPI-version)

"""
class ddpg_agent_rrc:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward, self.env.steps_per_goal, self.args.xy_only, self.args.trajectory_aware)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # self.delta_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.exp_dir)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        
        if args.real_training:
            self.train_on_real_exp()
        else:
            if args.load_pretrained == 1:
                self.load_pretrained_nets(rollouts=self.args.init_rollouts)

    def learn(self):
        """
        train the network

        """
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('WARNING: any reset noise is not applied to robot arms')
            print('\n[{}] Beginning RRC HER training, difficulty = {}\n'.format(datetime.now(), self.args.difficulty))
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            if self.args.increment_noise and MPI.COMM_WORLD.Get_rank() == 0:
                print('Epoch {}, noise: {}'.format(epoch, self.args.noise_level))
            self.check_weights_synched()
            actor_loss, critic_loss, explore_success = [], [], []
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset(difficulty=self.args.difficulty, init_state=self.sample_init_state_type(), noisy=self.args.noisy_resets, noise_level=self.args.noise_level)
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        g_new = observation_new['desired_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                        g = g_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    ep_g.append(g.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    explore_success.append(info['is_success']*1)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    a_loss, q_loss = self._update_network()
                    actor_loss += [a_loss]
                    critic_loss += [q_loss]
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            explore_success = MPI.COMM_WORLD.allreduce(np.mean(explore_success), op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
            success_rate = self._eval_agent()
            self.save_model(epoch)
            if self.args.increment_noise:
                self.args.noise_level += 0.025
                self.args.noise_level = np.clip(self.args.noise_level, 0, 1)
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch: {} eval_rate: {:.3f} explore_rate: {:.3f} a_loss: {:.3f} q_loss: {:.3f} rrc: {:.0f} z_mean: {:.3f} xy: {:.3f}'\
                      .format(datetime.now(), epoch, success_rate, explore_success, np.mean(actor_loss), np.mean(critic_loss), self.rrc, self.z, self.xy))
                
                
    def save_model(self, epoch):
        if MPI.COMM_WORLD.Get_rank() == 0:
            # Save actor critic
            torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
                        self.model_path + '/acmodel{}.pt'.format(epoch))
            # Save optimizers
            torch.save([self.actor_optim.state_dict(), self.critic_optim.state_dict()], self.model_path + '/ac_optimizers.pt')
            # Save target nets
            torch.save([self.actor_target_network.state_dict(), self.critic_target_network.state_dict()], \
                        self.model_path + '/ac_targets.pt')
            # Save normalizer data
            
            if self.args.real_training:
                print('SORT out model paths!!!')
                # Save latest actor
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
                        self.model_path + + '/real_acmodel_latest.pt')
                # Save experience
                buffer = {
                    'obs': self.buffer.buffers['obs'][:self.buffer.current_size],
                    'ag': self.buffer.buffers['ag'][:self.buffer.current_size],
                    'g': self.buffer.buffers['g'][:self.buffer.current_size],
                    'actions': self.buffer.buffers['actions'][:self.buffer.current_size]
                    }
                np.save(self.model_path + "/buffer_all.npy", buffer)
                # Save normalizer data
                self.o_norm.save(self.model_path + '/o_norm.npy')
                self.g_norm.save(self.model_path + '/g_norm.npy')
        

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        mb_g_next = mb_g[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1] # Only using one rollout????
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g[:, :-1, :], 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                        'g_next': mb_g_next
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # # delta calculation
        # mb_obs = np.clip(mb_obs, -self.args.clip_obs, self.args.clip_obs)
        # mb_delta = mb_obs[:,1:,:] - mb_obs[:,:-1,:]
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # self.delta_norm.update(mb_delta)
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        # self.delta_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        if self.args.z_reward == 1:
            transitions['r'] += self.get_z_reward(transitions['obs'], transitions['g'])
        # pre-process the observation and goal
        o, o_next, g, g_next = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['g_next']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g_next)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            if self.args.z_reward == 1:
                clip_return += 50 # TODO: calculate proper value!!!
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()
        return actor_loss.detach().numpy(), critic_loss.detach().numpy()
    
    def get_z_reward(self, obs, g):
        obs = np.expand_dims(obs[...,self.env.z_pos], axis=-1)
        g = np.expand_dims(g[...,2], axis=-1)
        z_dist = np.abs(g - obs)
        # punish less if above goal
        scale = g > obs
        scale = (scale + 1) / 2
        # reward is negative of z distance
        r_z = -self.args.z_scale * scale * z_dist # TODO: verify still works if changed obs space
        return r_z

    
    def _collect_init_exp(self, rollouts=100):
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('Collecting {} initial rollouts...'.format(rollouts))
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        success = []
        explore = np.zeros(rollouts)
        explore[:int(self.args.init_explore * rollouts)] = 1
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('explore rate: {}'.format(np.sum(explore)/rollouts))
        for rollout in range(rollouts):
            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environment
            observation = self.env.reset(difficulty=self.args.difficulty, init_state='normal', noisy=self.args.init_noisy, noise_level=self.args.noise_level)
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            # start to collect samples
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # No exploration!! Just choose best action
                    if explore[rollout]:
                        action = self._select_actions(pi)
                    else:
                        action = pi.detach().cpu().numpy().squeeze()
                # feed the actions into the environment
                observation_new, r, _, info = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                g_new = observation_new['desired_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
                g = g_new
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_g.append(g.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)
            success.append(info['is_success'])
        # convert them into arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)
        # store the episodes
        self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        # Dont update normaliser as throws things off??
        self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
        success_rate = MPI.COMM_WORLD.allreduce(np.mean(success), op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('[{}] success rate: {}'.format(datetime.now(), success_rate))
        
    
    def sample_init_state_type(self, normal_prob=1.0, grasp_prob=0.0, pick_prob=0.0, prev_state_prob=0.0):
        return np.random.choice(['normal', 'grasp', 'pick', 'prev_state'], p=[normal_prob,grasp_prob,pick_prob,prev_state_prob])
        
    def load_pretrained_nets(self, rollouts=100):
        # load in pretrained networks (trained in difficulty 1)
        model_path = self.args.pretrained_path
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('[{}] Loading in Actor Critic from {} ...'.format(datetime.now(), model_path))
        o_mean, o_std, g_mean, g_std, actor_state, critic_state = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.actor_network.load_state_dict(actor_state)
        self.critic_network.load_state_dict(critic_state)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # Load in pretrained norms
        self.o_norm.mean, self.o_norm.std = o_mean, o_std
        self.g_norm.mean, self.g_norm.std = g_mean, g_std
        
        # Fill buffer with some rollouts to help prevent catastrophic forgetting
        self._collect_init_exp(rollouts=rollouts)
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     print('WARNING: not artificially updating normalizer')
        # Add some weight to loaded in norms (slightly dodgy stuff going on here)
        #  Performed after collection to avoid slight discrepancies effecting performance
        epochs = 100
        fake_rollouts = epochs * self.args.n_cycles * self.args.num_rollouts_per_mpi
        # obs first
        fake_obs = np.zeros((fake_rollouts, o_mean.shape[0])) + o_mean
        fake_obs[0:int(fake_rollouts/2)] += o_std
        fake_obs[int(fake_rollouts/2):] -= o_std
        self.o_norm.update(fake_obs)
        self.o_norm.recompute_stats()
        # then goal
        fake_g = np.zeros((fake_rollouts, g_mean.shape[0])) + g_mean
        fake_g[0:int(fake_rollouts/2)] += g_std
        fake_g[int(fake_rollouts/2):] -= g_std
        self.g_norm.update(fake_g)
        self.g_norm.recompute_stats()
        
    def load_buffer(self, buffer):
        assert self.args.real_training, 'This method only applies to real training!!!'
        # Load in all previous experience
        buffer_all_path = self.model_path + "/buffer_all.npy"
        if os.path.isfile(buffer_all_path):
            print('Loading in all previous experience...')
            load_buff=np.load(buffer_all_path, allow_pickle=True)
            mb_obs = load_buff.item()['obs']
            mb_ag = load_buff.item()['ag']
            mb_g = load_buff.item()['g']
            mb_actions = load_buff.item()['actions']
            # store the episodes
            buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
            self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
            print('mb_obs.shape: {}'.format(mb_obs.shape))
            print('mb_ag.shape: {}'.format(mb_ag.shape))
            print('mb_g.shape: {}'.format(mb_g.shape))
            print('mb_actions.shape: {}'.format(mb_actions.shape))
        # Load in latest collected experience
        load_buff=np.load(self.model_path + "/buffer_latest.npy", allow_pickle=True)
        mb_obs = load_buff.item()['obs']
        mb_ag = load_buff.item()['ag']
        mb_g = load_buff.item()['g']
        mb_actions = load_buff.item()['actions']
        success_rate = load_buff.item()['success_rate']
        # store the episodes
        buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
        print('mb_obs.shape: {}'.format(mb_obs.shape))
        print('mb_ag.shape: {}'.format(mb_ag.shape))
        print('mb_g.shape: {}'.format(mb_g.shape))
        print('mb_actions.shape: {}'.format(mb_actions.shape))
        print(self.buffer.buffers['g'][0,0:120])
        print(self.buffer.buffers['g'][0,-120:])
        print('Latest success rate: {}'.format(success_rate))
        print()
        print('Agent Buffer_size: {}'.format(self.buffer.current_size))
        
        
    def check_weights_synched(self):
        models = [self.actor_network, self.critic_network, self.actor_target_network, self.critic_target_network]
        names = ['actor', 'critic', 'actor_target', 'critic_target']
        i = 0
        for model in models:
            weight = model.fc1.weight[0,0].detach().numpy()
            mean_weight = MPI.COMM_WORLD.allreduce(weight, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
            if weight != mean_weight and MPI.COMM_WORLD.Get_rank() == 0:
                print('WARNING: {} weights are not synched!!!!'.format(names[i]))
            i += 1
            
    def train_on_real_exp(self):
        assert self.args.steps_per_goal == 30
        # load in pretrained networks
        model_path = self.model_path + '/real_acmodel_latest.pt'
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('[{}] Loading in Actor Critic from {} ...'.format(datetime.now(), model_path))
        o_mean, o_std, g_mean, g_std, actor_state, critic_state = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.actor_network.load_state_dict(actor_state)
        self.critic_network.load_state_dict(critic_state)
        # Load in targets
        targets_path = self.model_path + '/ac_targets.pt'
        if os.path.isfile(targets_path):
            t_actor_state, t_critic_state = torch.load(targets_path, map_location=lambda storage, loc: storage)
            print('Loading in saved target networks')
            self.actor_target_network.load_state_dict(t_actor_state)
            self.critic_target_network.load_state_dict(t_critic_state)
        else:
            print('Defining targets to be same as actual networks')
            # load the weights into the target networks
            self.actor_target_network.load_state_dict(self.actor_network.state_dict())
            self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # Load in optimizers
        opt_path = self.model_path + '/ac_optimizers.pt'
        if os.path.isfile(opt_path):
            print('Loading in optimizers')
            actor_opt_state, critic_opt_state = torch.load(opt_path, map_location=lambda storage, loc: storage)
            self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
            self.critic_optim
        else:
            print('No optimizers found')
        # Load in normalizers
        o_norm_path = self.model_path + '/o_norm.npy'
        g_norm_path = self.model_path + '/g_norm.npy'
        if os.path.isfile(o_norm_path) and os.path.isfile(g_norm_path):
            self.o_norm.load(o_norm_path)
            self.g_norm.load(g_norm_path)
        else:
            print('No normalizer data found, doing dodgy fill...')
            # Load in pretrained norms
            self.o_norm.mean, self.o_norm.std = o_mean, o_std
            self.g_norm.mean, self.g_norm.std = g_mean, g_std
            # Add some weight to loaded in norms (slightly dodgy stuff going on here)
            #  Performed after collection to avoid slight discrepancies effecting performance
            epochs = 100
            fake_rollouts = epochs * self.args.n_cycles * self.args.num_rollouts_per_mpi
            # obs first
            fake_obs = np.zeros((fake_rollouts, o_mean.shape[0])) + o_mean
            fake_obs[0:int(fake_rollouts/2)] += o_std
            fake_obs[int(fake_rollouts/2):] -= o_std
            self.o_norm.update(fake_obs)
            self.o_norm.recompute_stats()
            # then goal
            fake_g = np.zeros((fake_rollouts, g_mean.shape[0])) + g_mean
            fake_g[0:int(fake_rollouts/2)] += g_std
            fake_g[int(fake_rollouts/2):] -= g_std
            self.g_norm.update(fake_g)
            self.g_norm.recompute_stats()
        # Load in experience
        self.load_buffer(self.buffer)
        # Train models
        print('Actor weight pre: {}'.format(self.actor_network.fc1.weight[0,0:3]))
        print('Critic weight pre: {}'.format(self.critic_network.fc1.weight[0,0:3]))
        actor_loss, critic_loss = [], []
        for _ in range(self.args.n_cycles):
            for _ in range(self.args.n_batches):
                # train the network
                a_loss, q_loss = self._update_network()
                actor_loss += [a_loss]
                critic_loss += [q_loss]
            # soft update
            self._soft_update_target_network(self.actor_target_network, self.actor_network)
            self._soft_update_target_network(self.critic_target_network, self.critic_network)
        print('Actor weight pre: {}'.format(self.actor_network.fc1.weight[0,0:3]))
        print('Critic weight pre: {}'.format(self.critic_network.fc1.weight[0,0:3]))
        # Save models, target_models, optimizers, buffers, 
        self.save_model(self.args.real_epoch)
        
        
    # do the evaluation (and store the eval episodes in buffer)
    def _eval_agent(self):
        total_success_rate = []
        r_z, xy, rrc = [], [], []
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for n in range(self.args.n_test_rollouts):
            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            per_success_rate = []
            observation = self.env.reset(difficulty=self.args.difficulty, init_state=self.sample_init_state_type(), noisy=self.args.noisy_resets, noise_level=self.args.noise_level)
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, r, _, info = self.env.step(actions)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                g_new = observation_new['desired_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(actions.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
                g = g_new
                # Append success rates
                per_success_rate.append(info['is_success'])
                r_z.append(self.get_z_reward(obs, g))
                xy.append(np.linalg.norm(ag[0:2] - g[0:2], axis=-1))
            # Append obs
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_g.append(g.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)
            # Append success rates
            total_success_rate.append(per_success_rate)
            rrc.append(info['rrc_reward'])
        # convert them into arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)
        # store the episodes
        self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
        # Calculate success rates
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        self.rrc = MPI.COMM_WORLD.allreduce(np.mean(rrc), op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        self.z = MPI.COMM_WORLD.allreduce(np.mean(r_z), op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        self.xy = MPI.COMM_WORLD.allreduce(10*np.mean(xy), op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        return global_success_rate / MPI.COMM_WORLD.Get_size()

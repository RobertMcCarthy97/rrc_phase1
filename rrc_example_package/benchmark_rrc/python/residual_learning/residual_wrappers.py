import pybullet as p
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import gym
from rrc_example_package.benchmark_rrc.python.env.viz import CuboidMarker
from .domain_randomization import TriFingerRandomizer, Parameter
import rrc_example_package.benchmark_rrc.python.mp.const as const

from rrc_example_package.cube_trajectory_env import ActionType
from pprint import pprint


class ResidualActionGenerator(object):
    def __init__(self, state_machine, frameskip):
        self.state_machine = state_machine
        self.frameskip = frameskip


class ResidualWrapper(gym.Wrapper):
    '''
    Residual Wrapper which uses a StateMachine as a base policy.

    Params:
        env: base environment. Must have the AdaptiveActionSpaceWrapper.
        state_machine: A StateMachine class to use as the base policy.
        frameskip: The frameskip that the residual policy acts at.
        max_torque: [-max_torque, max_torque] defines the range of residual
            torque actions.
        residual_state: The state of the state machine in which to do residual
            learning.
        max_length: The max number of steps in the residual state before
            terminating the episode.
    '''

    def __init__(self, env, state_machine, frameskip, max_torque, residual_state,
                 max_length=None):
        super().__init__(env)
        self.state_machine = state_machine(env)
        self.frameskip = frameskip
        self.max_torque = max_torque
        self.max_length = max_length
        self.residual_state = residual_state
        # The residual controller will use a fixed frameskip, so remove it
        # from the action_space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=self.observation_space['robot_torque'].low.shape
        )
        self.observation_space = gym.spaces.Dict({
            'obs': self.observation_space,
            'action': gym.spaces.Dict({
                'torque_enabled': gym.spaces.Discrete(n=2),
                'position_enabled': gym.spaces.Discrete(n=2),
                'torque': self.observation_space['robot_torque'],
                'position': self.observation_space['robot_position'],
                'tip_positions': self.observation_space['robot_tip_positions']
            })
        })

    def reset(self):
        obs = self.env.reset()
        self.state_machine.reset()
        self._t = 0
        self._active = False
        try:
            obs, done = self._step_state_machine(obs, False)
        except Exception as e:
            raise e
            return self.reset()
        if done:
            return self.reset()
        return self._add_action_to_obs(obs, self.base_action)

    def step(self, action):
        self.residual_action = {
            'torque': action, 'frameskip': self.frameskip
        }
        self._t += self.frameskip
        reward = 0.
        while self.residual_action['frameskip'] > 0:
            action = self._generate_action()
            obs, r, done, info = self.env.step(action)
            reward += r
            if self.base_action['frameskip'] <= 0:
                obs, done = self._step_state_machine(obs, done)
        if self.max_length is not None:
            done = done or self._t >= self.max_length
        return self._add_action_to_obs(obs, self.base_action), reward, done, info

    def _step_state_machine(self, obs, done):
        state = self.state_machine.state.__class__.__name__
        self.base_action = self.state_machine(obs)
        while state != self.residual_state and not done:
            if self._active:
                # We entered and then left the residual state.
                return obs, True
            obs, _, done, _ = self.env.step(self.base_action)
            state = self.state_machine.state.__class__.__name__
            self.base_action = self.state_machine(obs)
        self._active = True
        return obs, done

    def _add_action_to_obs(self, obs, ac):
        obs = {'obs': obs, 'action': {}}
        if ac['torque'] is None:
            obs['action']['torque_enabled'] = np.array([0])
            obs['action']['torque'] = np.zeros_like(self.action_space.low)
        else:
            obs['action']['torque_enabled'] = np.array([1])
            obs['action']['torque'] = ac['torque']
        if ac['position'] is None:
            obs['action']['position_enabled'] = np.array([0])
            obs['action']['position'] = np.zeros_like(self.action_space.low)
            obs['action']['tip_positions'] = obs['obs']['robot_tip_positions']
        else:
            obs['action']['position_enabled'] = np.array([1])
            obs['action']['position'] = ac['position']
            obs['action']['tip_positions'] = np.array(
                self.pinocchio_utils.forward_kinematics(ac['position'])
            )
        return obs

    def _generate_action(self):
        action = {}
        if self.base_action['torque'] is not None:
            action['torque'] = np.clip(
                self.base_action['torque'] + self.max_torque * self.residual_action['torque'],
                self.env.action_space['torque'].low,
                self.env.action_space['torque'].high
            )
        else:
            action['torque'] = None

        action['position'] = self.base_action['position']

        frameskip = min(self.base_action['frameskip'],
                        self.residual_action['frameskip'])
        action['frameskip'] = frameskip
        self.base_action['frameskip'] -= frameskip
        self.residual_action['frameskip'] -= frameskip
        return action


class RandomizedEnvWrapper(gym.Wrapper):
    def __init__(self, env, camera_fps=10.0, visualize=False, flatten_obs=False, time_step_s=0.001):
        super().__init__(env)
        
        self.first_run = True
        self.randomizer = TriFingerRandomizer(timestep_low=0.9*time_step_s, timestep_high=1.2*time_step_s) #TODO: implement timestep better
        self.steps_per_camera_frame = int((1.0 / camera_fps) / time_step_s)
        self.visualize = visualize
        self.marker = None
        self._max_episode_steps = env._max_episode_steps

        self.step_count = 0
        self.params = self.randomizer.sample_dynamics()
        spaces = env.observation_space.spaces.copy()
        spaces['clean'] = env.observation_space
        spaces['params'] = self.randomizer.get_parameter_space()
        self.observation_space = gym.spaces.Dict(spaces)
        
        self.cube_width_sampler = Parameter('cube_width_scale', 0.8, 1.1)

    def reset(self, difficulty=None, init_state='normal', noisy=False, noise_level=1):
        if self.marker:
            del self.marker
            self.marker = None
        
        cube_width_scale = self.randomizer.sample_cube_width()
        self.env.cube_scale = cube_width_scale
        
        obs = self.env.reset(difficulty=difficulty, init_state=init_state, noisy=noisy, noise_level=noise_level)

        if self.first_run:
            self.finger_id = self.env.platform.simfinger.finger_id
            self.joint_indices = self.env.platform.simfinger.pybullet_joint_indices
            self.link_indices = self.env.platform.simfinger.pybullet_link_indices
            self.client_id = self.env.platform.simfinger._pybullet_client_id
            # self.cube_id = self.env.platform.cube.block
            self.cube_id = self.env.platform.cube._object_id
            self.first_run = False

        self.randomize_param()
        self.step_count = 0
        self.noisy_cube_pose = self.sample_noisy_cube(obs)
        if self.visualize:
            self.marker = CuboidMarker(
                size=const.CUBOID_SIZE,
                position=self.noisy_cube_pose['position'],
                orientation=self.noisy_cube_pose['orientation'],
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )
        obs = self.randomize_obs(obs)
        # Use noise when calculating velocities
        obs = self.env._update_obj_vel(obs, initial=True)
        if self.flatten_obs:
            obs = self.env.flatten_obs(obs)
        return obs

    def step(self, action):
        action_noisy = self.randomize_action(action.copy())
        # TODO: only change at start of episode?
        p.setTimeStep(self.randomizer.sample_timestep(),
                      physicsClientId=self.client_id)
        obs, reward, is_done, info = self.env.step(action_noisy)
        # Ensure do not reveal noisy action to agent
        obs['action'] = action

        self.step_count += self.unwrapped.step_size
        if self.step_count >= self.steps_per_camera_frame:
            self.noisy_cube_pose = self.sample_noisy_cube(obs)
            self.step_count -= self.steps_per_camera_frame
        obs = self.randomize_obs(obs)
        obs = self.env._update_obj_vel(obs, initial=False)

        if self.visualize:
            self.marker.set_state(position=obs['object_observation']['position'],
                                  orientation=obs['object_observation']['orientation'])
        if self.flatten_obs:
            obs = self.env.flatten_obs(obs)
        return obs, reward, is_done, info

    def randomize_action(self, action):
        noise = self.randomizer.sample_action_noise()
        if self.env.action_type == ActionType.TORQUE:
            action = np.clip(
                action + noise['action_torque'],
                self.action_space.low,
                self.action_space.high
            )
        elif self.env.action_type == ActionType.POSITION:
            action = np.clip(
                action + noise['action_position'],
                self.action_space.low,
                self.action_space.high
            )
        else:
            raise NotImplementedError()
        return action

    def randomize_obs(self, obs):
        clean_obs = deepcopy(obs)

        noise = self.randomizer.sample_robot_noise()

        # add noise to robot_position
        ob_space = self.env.observation_space['robot_observation']['position']
        obs['robot_observation']['position'] = np.clip(obs['robot_observation']['position'] + noise['robot_position'],
                                        ob_space.low, ob_space.high)
        obs['robot_observation']['tip_position'] = np.array(self.unwrapped.platform.forward_kinematics(obs['robot_observation']['position']))
        # add noise to robot_velocity
        ob_space = self.env.observation_space['robot_observation']['velocity']
        obs['robot_observation']['velocity'] = np.clip(obs['robot_observation']['velocity'] + noise['robot_velocity'],
                                        ob_space.low, ob_space.high)
        
        # add noise to robot_torque
        ob_space = self.env.observation_space['robot_observation']['torque']
        obs['robot_observation']['torque'] = np.clip(obs['robot_observation']['torque'] + noise['robot_torque'],
                                      ob_space.low, ob_space.high)
        # add noise to tip_force
        ob_space = self.env.observation_space['robot_observation']['tip_force']
        obs['robot_observation']['tip_force'] = np.clip(obs['robot_observation']['tip_force'] + noise['tip_force'],
                                   ob_space.low, ob_space.high)

        # use saved noisy object observation
        obs['object_observation']['position'] = self.noisy_cube_pose['position']
        obs['object_observation']['orientation'] = self.noisy_cube_pose['orientation']

        obs['clean'] = clean_obs
        obs['params'] = np.concatenate(
            [v.flatten() for v in self.params.values()]
        )
        return obs

    def sample_noisy_cube(self, obs):
        noise = self.randomizer.sample_cube_noise()
        q_obj = R.from_quat(obs['object_observation']['orientation'])
        q_noise = R.from_euler('ZYX', noise['cube_ori'], degrees=False)
        return {
            'position': obs['object_observation']['position'] + noise['cube_pos'],
            'orientation': (q_obj * q_noise).as_quat()
        }

    def randomize_param(self):
        self.params = self.randomizer.sample_dynamics()
        p.changeDynamics(bodyUniqueId=self.cube_id, linkIndex=-1,
                         physicsClientId=self.client_id,
                         mass=self.params['cube_mass'])
        robot_params = {k: v for k, v in self.params.items() if 'cube' not in k}
        self.set_robot_params(**robot_params)

    def set_robot_params(self, **kwargs):
        # set params by passing kw dictionary
        # all values of dict should be list which length is 3 or 9 for different params or float/int for the same param
        self.check_robot_param_dict(kwargs)
        for i, link_id in enumerate(self.link_indices):
            joint_kwargs = self.get_robot_param_dict(kwargs, i)
            p.changeDynamics(bodyUniqueId=self.finger_id, linkIndex=link_id,
                             physicsClientId=self.client_id, **joint_kwargs)

    def check_robot_param_dict(self, dic):
        for v in dic.values():
            if len(v.shape) > 0:
                assert len(v) in [3, 9]

    def get_robot_param_dict(self, dic, i):
        ret_dic = {}
        for k in dic.keys():
            if len(dic[k].shape) == 0:
                ret_dic[k] = dic[k]
            elif len(dic[k]) == 3:
                ret_dic[k] = dic[k][i % 3]
            elif len(dic[k]) == 9:
                ret_dic[k] = dic[k][i]
            else:
                raise ValueError("Weird param shape.")

        return ret_dic


"""Example Gym environment for the RRC 2021 Phase 2."""
import enum
import typing

import gym
import numpy as np
import time
import robot_fingers

# import trifinger_simulation
# import trifinger_simulation.visual_objects
# from trifinger_simulation import trifingerpro_limits
# import trifinger_simulation.tasks.move_cube_on_trajectory as task
# from trifinger_simulation.tasks import move_cube

import rrc_example_package.trifinger_simulation.python.trifinger_simulation as trifinger_simulation
import rrc_example_package.trifinger_simulation.python.trifinger_simulation.visual_objects
from rrc_example_package.trifinger_simulation.python.trifinger_simulation import trifingerpro_limits
import rrc_example_package.trifinger_simulation.python.trifinger_simulation.tasks.move_cube_on_trajectory as task
from rrc_example_package.trifinger_simulation.python.trifinger_simulation.tasks import move_cube



class ActionType(enum.Enum):
    """Different action types that can be used to control the robot."""

    #: Use pure torque commands.  The action is a list of torques (one per
    #: joint) in this case.
    TORQUE = enum.auto()
    #: Use joint position commands.  The action is a list of angular joint
    #: positions (one per joint) in this case.  Internally a PD controller is
    #: executed for each action to determine the torques that are applied to
    #: the robot.
    POSITION = enum.auto()
    #: Use both torque and position commands.  In this case the action is a
    #: dictionary with keys "torque" and "position" which contain the
    #: corresponding lists of values (see above).  The torques resulting from
    #: the position controller are added to the torques in the action before
    #: applying them to the robot.
    TORQUE_AND_POSITION = enum.auto()


class BaseCubeTrajectoryEnv(gym.GoalEnv):
    """Gym environment for moving cubes with TriFingerPro."""

    def __init__(
        self,
        goal_trajectory: typing.Optional[task.Trajectory] = None,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 1,
        disable_arm3 = False
    ):
        """Initialize.

        Args:
            goal_trajectory: Goal trajectory for the cube.  If ``None`` a new
                random trajectory is sampled upon reset.
            action_type: Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size:  Number of actual control steps to be performed in one
                call of step().
        """
        # Basic initialization
        # ====================

        if goal_trajectory is not None:
            task.validate_goal(goal_trajectory)
        self.goal = goal_trajectory

        self.action_type = action_type
        self.disable_arm3 = disable_arm3

        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        self.step_size = step_size

        # will be initialized in reset()
        self.platform = None

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )
        robot_tip_force_space = gym.spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
        )

        object_state_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.object_position.low,
                    high=trifingerpro_limits.object_position.high,
                ),
                "orientation": gym.spaces.Box(
                    low=trifingerpro_limits.object_orientation.low,
                    high=trifingerpro_limits.object_orientation.high,
                ),
                "confidence": gym.spaces.Box(
                    low=np.array(0),
                    high=np.array(1),
                ),
            }
        )

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "robot_observation": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                        "tip_force": robot_tip_force_space,
                    }
                ),
                "object_observation": gym.spaces.Dict(
                    {
                        "position": object_state_space["position"],
                        "orientation": object_state_space["orientation"],
                        "confidence": object_state_space["confidence"],
                    }
                ),
                "action": self.action_space,
                "desired_goal": object_state_space["position"],
                "achieved_goal": object_state_space["position"],
            }
        )

    def compute_reward_rrc(
        self,
        achieved_goal: task.Position,
        desired_goal: task.Position,
        info: dict,
    ) -> float:
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal: Current position of the object.
            desired_goal: Goal position of the current trajectory step.
            info: An info dictionary containing a field "time_index" which
                contains the time index of the achieved_goal.

        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true::

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        # This is just some sanity check to verify that the given desired_goal
        # actually matches with the active goal in the trajectory.
        active_goal = np.asarray(
            task.get_active_goal(
                self.info["trajectory"], self.info["time_index"]
            )
        )
        assert np.all(active_goal == desired_goal), "{}: {} != {}".format(
            info["time_index"], active_goal, desired_goal
        )

        return -task.evaluate_state(
            info["trajectory"], info["time_index"], achieved_goal
        )

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.

        .. note::

           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.

        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        task.seed(seed)
        return [seed]

    def _create_observation(self, t, action, obs_type='default'):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        object_observation = camera_observation.filtered_object_pose

        active_goal = np.asarray(
            task.get_active_goal(self.info["trajectory"], t)
        )
        
        full_observation = {
            "robot_observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
                "tip_force": robot_observation.tip_force,
            },
            "object_observation": {
                "position": object_observation.position,
                "orientation": object_observation.orientation,
                "confidence": object_observation.confidence,
            },
            "action": action,
            "desired_goal": active_goal,
            "achieved_goal": object_observation.position,
        }
        
        if obs_type == 'default':
            observation = {
                "robot_observation": {
                    "position": robot_observation.position,
                    "velocity": robot_observation.velocity,
                    "torque": robot_observation.torque,
                },
                "object_observation": {
                    "position": object_observation.position,
                    "orientation": object_observation.orientation,
                },
                "action": action,
                "desired_goal": active_goal,
                "achieved_goal": object_observation.position,
                "full_observation": full_observation
            }
        elif obs_type  == 'full':
            observation = full_observation
        else: assert False
        
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action


class SimCubeTrajectoryEnv(BaseCubeTrajectoryEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        goal_trajectory: typing.Optional[task.Trajectory] = None,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 1,
        visualization: bool = False,
        disable_arm3=False
    ):
        """Initialize.

        Args:
            goal_trajectory: Goal trajectory for the cube.  If ``None`` a new
                random trajectory is sampled upon reset.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        super().__init__(
            goal_trajectory=goal_trajectory,
            action_type=action_type,
            step_size=step_size,
            disable_arm3=disable_arm3
        )
        self.visualization = visualization
        self.disable_arm3 = disable_arm3

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            print('Violating action: {}'.format(action))
            print('Action space: {}'.format(self.action_space))
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.step_count + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)
        
        action_to_apply = action.copy()
        if self.disable_arm3:
            assert self.action_type == ActionType.TORQUE, 'Disabling of arm 3 only implemented for torque control'
            action_to_apply[6:9] = np.array([-self.action_space.high[0],self.action_space.high[0],-self.action_space.high[0]])

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > task.EPISODE_LENGTH:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action_to_apply)
            t = self.platform.append_desired_action(robot_action)
            
            # update goal visualization
            if self.visualization:
                goal_position = task.get_active_goal(
                    self.info["trajectory"], t
                )
                self.goal_marker.set_state(goal_position, (0, 0, 0, 1))

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            self.info["time_index"] = t + 1
            

            observation = self._create_observation(
                self.info["time_index"], action
            )

            self.info["rrc_reward"] += self.compute_reward_rrc(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

        is_done = self.step_count >= task.EPISODE_LENGTH

        return observation, reward, is_done, self.info

    def reset(self, init_state='normal', noisy=False, noise_level=1):
        
        assert noise_level <= 1, "Noise level > 1"
        
        """Reset the environment."""
        # hard-reset simulation
        del self.platform
        
        # Initialise with cube in robots grasp
        if init_state == 'grasp':
            # [right far, near, left far]
            rob_position = np.array([0.45, 0.9, -1.9, 0.28, 0.8, -1.93, 0.35, 0.9, -1.9], dtype=np.float32)
            # initialize cube
            cube_pos = (0, 0, (move_cube._CUBE_WIDTH / 2) + 0.062)
            cube_orient = trifingerpro_limits.object_orientation.default
            if noisy:
                # Add small noise to prevent overfitting
                rob_position += np.random.normal(loc=0.0, scale=0.005, size=rob_position.shape)
                cube_pos += np.random.normal(loc=0.0, scale=0.005, size=len(cube_pos))
                cube_orient += np.random.normal(loc=0.0, scale=0.005, size=cube_orient.shape)
        # Initialise cube on ground in center, with arm ready to pick it up        
        elif init_state == 'pick':
            # [right far, near, left far]
            rob_position = np.array([0.2, 0.75, -1.45, 0.2, 0.75, -1.45, 0.2, 0.75, -1.45], dtype=np.float32)
            cube_pos = task.INITIAL_CUBE_POSITION
            cube_orient = trifingerpro_limits.object_orientation.default
            if noisy:
                rob_position += np.random.normal(loc=0.0, scale=0.01, size=rob_position.shape)
                x_noise = np.random.normal(loc=0.0, scale=0.01)
                y_noise = np.random.normal(loc=0.0, scale=0.01)
                cube_pos = (x_noise, y_noise, move_cube._CUBE_WIDTH / 2)
                cube_orient[2] += np.random.normal(loc=0.0, scale=1)
        # Initialise state off previously visited state        
        elif init_state == 'prev_state' and self.buffer.current_size > 0:
            rob_position, cube_pos, cube_orient = self.sample_buffer_state()
        # Norlmal initialise - cube on ground somewhere, arms near center    
        else:
            rob_position = trifingerpro_limits.robot_position.default # = np.array([0.0, 0.9, -1.7] * n_fingers, dtype=np.float32)
            cube_pos = task.INITIAL_CUBE_POSITION
            cube_orient = trifingerpro_limits.object_orientation.default
            if noisy:
                # TODO: why rob_noise messing things up?
                # rob_position += np.random.normal(loc=0.0, scale=0.1, size=rob_position.shape) * noise_level
                x_noise = np.clip(np.random.normal(loc=0.0, scale=0.1), -0.1, 0.1) * noise_level
                y_noise = np.clip(np.random.normal(loc=0.0, scale=0.1), -0.1, 0.1) * noise_level
                cube_pos = (x_noise, y_noise, move_cube._CUBE_WIDTH / 2)
                cube_orient[2] += np.random.normal(loc=0.0, scale=1) * noise_level
        
        object_pose = task.move_cube.Pose(
            position=cube_pos,
            orientation=cube_orient
        )

        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=rob_position,
            initial_object_pose=object_pose,
        )
        self.platform.camera_rate_fps = 26 # TODO: UNDO!!!!!!!!!!!!!!!

        # if no goal is given, sample one randomly
        if self.goal is None:
            trajectory = task.sample_goal()
            if task.GOAL_DIFFICULTY == 2:
                # Add some noise to standard goal to prevent overfitting
                # WARNING: ONLY APPLIES TO 1st GOAL of trajectory
                trajectory[0] = list(trajectory[0])
                # Set goal to be above cube
                trajectory[0][1][0:2] = cube_pos[0:2]
                if noisy:
                    noise = np.random.uniform(low=-0.05, high=0.05, size=3)
                    noise[-1] = np.random.uniform(low=-0.01, high=0.03)
                    trajectory[0][1] += noise
                trajectory[0] = tuple(trajectory[0])
        else:
            trajectory = self.goal

        # visualize the goal
        if self.visualization:
            self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                width=task.move_cube._CUBE_WIDTH,
                position=trajectory[0][1],
                orientation=(0, 0, 0, 1),
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )

        self.info = {"time_index": -1, "trajectory": trajectory, "rrc_reward": 0}

        self.step_count = 0
        
        observation = self._create_observation(0, self._initial_action)

        return observation
    
    def sample_buffer_state(self):
        # Sample from any random state or just end state?
        assert self.buffer != None
        rollout = np.random.randint(self.buffer.current_size)
        timestep = np.random.randint(self.buffer.T + 1)
        rob_position = self.buffer.buffers['obs'][rollout, timestep, 0:9]
        cube_position = self.buffer.buffers['obs'][rollout, timestep, 27:30]
        cube_orientation = self.buffer.buffers['obs'][rollout, timestep, 30:34]
        # print('WARNING: sample reset could be wrong!!!!')
        return rob_position, cube_position, cube_orientation


class CustomSimCubeEnv():
    # TODO: make subclass of SimCubeTrajectoryEnv
    # Is step_size too coarse?
    # Can velocities of cube be inferred properly??
    
    def __init__(self, difficulty=None, sparse_rewards=True, step_size=40, distance_threshold=0.02, max_steps=50, visualization=False,
                 goal_trajectory=None, steps_per_goal=50, xy_only=False, obs_type='default',
                 action_type: ActionType = ActionType.TORQUE, disable_arm3=False):
        
        assert (max_steps / steps_per_goal) % 1 == 0
        
        if difficulty != None:
            task.GOAL_DIFFICULTY = difficulty
        if steps_per_goal != None:
            #: Number of time steps for which the first goal in the trajectory is active.
            task.FIRST_GOAL_DURATION = steps_per_goal * step_size
            #: Number of time steps for which following goals in the trajectory are active.
            task.GOAL_DURATION = steps_per_goal * step_size
        # task.EPISODE_LENGTH = max_steps * step_size
        
        self.env = SimCubeTrajectoryEnv(
            goal_trajectory=goal_trajectory,  # passing None to sample a random trajectory
            action_type=action_type,
            visualization=visualization,
            step_size=step_size,
            disable_arm3=disable_arm3
        )
        self.sparse_rewards = sparse_rewards
        self.prev_object_obs = None
        self._max_episode_steps = max_steps # 50 * step_size simulator steps
        self.distance_threshold = distance_threshold
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.step_size = step_size
        self.steps_per_goal = steps_per_goal
        self.xy_only = xy_only
        self.obs_type = obs_type
        self.disable_arm3 = disable_arm3
        
        if self.obs_type == 'default':
            self.z_pos = 29
        elif self.obs_type == 'full':
            self.z_pos=32
        else: assert False
        # self.seed = self.env.seed
        
        if self.sparse_rewards:
            self.compute_reward = self.compute_sparse_reward
        else:
            self.compute_reward = self.env.compute_reward_rrc
        
    def step(self, action):
        observation, reward, is_done, info = self.env.step(action)
        
        observation = self.custom_obs(observation)
        
        reward = self.compute_reward(observation['achieved_goal'], self._active_goal, info)
        
        info["is_success"] = self.compute_sparse_reward(observation['achieved_goal'], self._active_goal, None, check_success=True) == 0
        
        if info["time_index"] >= self._max_episode_steps * self.env.step_size:
            is_done = True
        
        # Compute reward based on goal active when action taken.
        # Then update active goal
        self._active_goal = observation['desired_goal']
        
        return observation, reward, is_done, info
        
    def reset(self, difficulty=None, init_state='normal', noisy=False, noise_level=1):
        if difficulty != None:
            task.GOAL_DIFFICULTY = difficulty
        observation = self.env.reset(init_state=init_state, noisy=noisy, noise_level=noise_level)
        self._active_goal = observation['desired_goal']
        # Initial cube velocities are 0 (or should be)
        observation['object_observation']['lin_vel'] = np.zeros_like(observation["object_observation"]["position"])
        observation['object_observation']['ang_vel'] = np.zeros_like(observation["object_observation"]["orientation"])
        # Set previous object obs to current (i.e. stationary)
        self.prev_object_obs = {
            "position": observation["object_observation"]["position"],
            "orientation": observation["object_observation"]["orientation"]
            }
        return self.custom_obs(observation)
    
    # Concat robot obs and object obs into single array.
    # Also include object obs of prev time to infer velocity
    # TODO: Include last 4 frames/calculate velocity directly??
    # WARNING - changes will effect ddpg_agent_rrc.py code!!!!!!!!!!!!
    def custom_obs(self, observation):
        if self.obs_type == 'full':
            observation = observation['full_observation']
            
        # Just take velocity as diff in observations
        observation['object_observation']['lin_vel'] = observation['object_observation']['position'] - self.prev_object_obs['position']
        observation['object_observation']['ang_vel'] = observation['object_observation']['orientation'] - self.prev_object_obs['orientation']
        # Set new previous positions
        self.prev_object_obs['position'] = observation['object_observation']['position']
        self.prev_object_obs['orientation'] = observation['object_observation']['orientation']
        observation['object_observation']['steps_since_update'] = np.array([0])
        
        state_obs = None
        if self.obs_type == 'full':
            # Robot obs        
            state_obs = observation['robot_observation']['position']
            state_obs = np.concatenate((state_obs, observation['robot_observation']['velocity']))
            state_obs = np.concatenate((state_obs, observation['robot_observation']['torque']))
            state_obs = np.concatenate((state_obs, observation['robot_observation']['tip_force']))
            # state_obs = np.concatenate((state_obs, observation['robot_observation']['tip_positions'])) # TODO: Include?
            # Object obs
            state_obs = np.concatenate((state_obs, observation['object_observation']['position']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['orientation']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['lin_vel']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['ang_vel']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['steps_since_update']))
            # state_obs = np.concatenate((state_obs, observation['object_observation']['confidence'])) # TODO: Include? Need to generate proper values if so
            # # Action
            # state_obs = np.concatenate((state_obs, observation['action']))
            # # Params
            # state_obs = np.concatenate((state_obs, observation['params']))
        elif self.obs_type =='default':
            # Robot obs        
            state_obs = observation['robot_observation']['position']
            state_obs = np.concatenate((state_obs, observation['robot_observation']['velocity']))
            state_obs = np.concatenate((state_obs, observation['robot_observation']['torque']))
            # Object obs
            state_obs = np.concatenate((state_obs, observation['object_observation']['position']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['orientation']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['lin_vel']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['ang_vel']))          
        custom_obs = {
            "observation": state_obs,
            "desired_goal": observation["desired_goal"],
            "achieved_goal": observation["achieved_goal"],
            }
        return custom_obs
    
    def compute_sparse_reward(self, achieved_goal, desired_goal, info, check_success=False): # include info to match fetch gym environments (better for compatibility with HER code)
        if self.xy_only and not check_success:
            d = np.linalg.norm(achieved_goal[...,0:2] - desired_goal[...,0:2], axis=-1)
            return -(d > self.distance_threshold * 0.8).astype(np.float32) # TODO: scale distance properly to account for 2D vs 3D
        else:
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return -(d > self.distance_threshold).astype(np.float32)
    
    def seed(self, seed=None):
        return self.env.seed(seed)


class SimtoRealEnv(BaseCubeTrajectoryEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        action_type: ActionType = ActionType.TORQUE,
        difficulty=None, sparse_rewards=True, step_size=101, distance_threshold=0.02, max_steps=50, visualization=False,
        goal_trajectory=None, steps_per_goal=50, xy_only=False,
        env_type='sim', obs_type='default', env_wrapped=False, increase_fps=False, reset_noise_level=1,
        disable_arm3=False
    ):
        """Initialize.

        Args:
            goal_trajectory: Goal trajectory for the cube.  If ``None`` a new
                random trajectory is sampled upon reset.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        super().__init__(
            goal_trajectory=goal_trajectory,
            action_type=action_type,
            step_size=step_size,
            disable_arm3=disable_arm3
        )
        self.visualization = visualization
        self.goal_trajectory = goal_trajectory
        self.sparse_rewards = sparse_rewards
        self.prev_object_obs = None
        self._max_episode_steps = max_steps # 50 * step_size simulator steps
        self.distance_threshold = distance_threshold
        self.step_size = step_size
        self.steps_per_goal = steps_per_goal
        self.xy_only = xy_only
        self.env_type = env_type
        self.obs_type = obs_type
        self.env_wrapped = env_wrapped
        self.increase_fps = increase_fps
        self.reset_noise_level = reset_noise_level # only applies to 'normal' resets
        self.disable_arm3 = disable_arm3
        # self.seed = self.env.seed
        # self.delete_me = 0
        
        self.cube_scale = 1
        
        if self.obs_type == 'default':
            self.z_pos = 29
        elif self.obs_type == 'full':
            self.z_pos=32
        else: assert False
            
        if self.sparse_rewards:
            self.compute_reward = self.compute_sparse_reward
        else:
            self.compute_reward = self.compute_reward_rrc
            
        if goal_trajectory is None:
            if difficulty != None:
                task.GOAL_DIFFICULTY = difficulty
            if steps_per_goal != None:
                #: Number of time steps for which the first goal in the trajectory is active.
                task.FIRST_GOAL_DURATION = steps_per_goal * step_size
                #: Number of time steps for which following goals in the trajectory are active.
                task.GOAL_DURATION = steps_per_goal * step_size
            

    def step(self, action, initial=False):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.info["time_index"] + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)
        
        action_to_apply = action.copy()
        if self.disable_arm3:
            assert self.action_type == ActionType.TORQUE, 'Disabling of arm 3 only implemented for torque control'
            action_to_apply[6:9] = np.array([-self.action_space.high[0],self.action_space.high[0],-self.action_space.high[0]])        
        
        # print('\nTime to return: {}'.format(time.time()-self.delete_me))
        reward = 0.0
        for _ in range(num_steps):
            # TODO: Speed-up/ensure action is ready asap
            # TODO: Implement delays??
            # Figure out delay between receiving obs and applying step
            # send action to robot
            robot_action = self._gym_action_to_robot_action(action_to_apply)
            t = self.platform.append_desired_action(robot_action)
            
            if self.env_type == 'sim' and self.visualization:
                # update goal visualization
                goal_position = task.get_active_goal(
                    self.info["trajectory"], t
                )
                self.goal_marker.set_state(goal_position, (0, 0, 0, 1))
                    
            self.info["time_index"] = t
            #TODO: skip this as it blocks?? - No as still want to wait for observation
            observation = self._create_observation(
                self.info["time_index"], action, obs_type='full'
            )
            
            self.info["rrc_reward"] += self.compute_reward_rrc(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )
            
            if initial:
                self._active_goal = observation["desired_goal"]
                break
        # self.delete_me = time.time()
        is_done = self.info["time_index"] >= self._max_episode_steps * self.step_size or self.info["time_index"] >= task.EPISODE_LENGTH
        
        # Flatten and update obs here if env is not wrapped, else do it in wrapper env
        if not self.env_wrapped:
            observation = self._update_obj_vel(observation, initial)
            observation = self.flatten_obs(observation)
        
        # Compute reward based on 'active goal'
        reward = self.compute_reward(observation['achieved_goal'], self._active_goal, self.info)
        self.info["is_success"] = self.compute_sparse_reward(observation['achieved_goal'], self._active_goal, None, check_success=True) == 0
        self.info["xy_fail"] = self.compute_xy_fail(observation['achieved_goal'], self._active_goal)
        # Update active goal (lags 1 behind)
        self._active_goal = observation["desired_goal"]
        
        return observation, reward, is_done, self.info

    def reset(self, difficulty=None, init_state='normal', noisy=False, noise_level=1):
        """Reset the environment."""
        
        move_cube._CUBE_WIDTH = move_cube._CUBE_WIDTH * self.cube_scale
        # print('CUBE WIDTH = {}, scale = {}'.format(move_cube._CUBE_WIDTH, self.cube_scale))
        
        if self.goal_trajectory == None and difficulty != None:
            task.GOAL_DIFFICULTY = difficulty
            
        # hard-reset simulation
        del self.platform
        
        if self.env_type == 'sim':
            rob_position, cube_pos, cube_orient = self.sample_init_state(init_state, noisy, noise_level=noise_level)
            object_pose = task.move_cube.Pose(
                position=cube_pos,
                orientation=cube_orient
            )
            self.platform = trifinger_simulation.TriFingerPlatform(
                visualization=self.visualization,
                initial_robot_position=rob_position,
                initial_object_pose=object_pose,
                cube_scale=self.cube_scale
            )
            if self.increase_fps:
                self.platform.camera_rate_fps = 26
        elif self.env_type == 'real':
            self.platform = robot_fingers.TriFingerPlatformWithObjectFrontend()
        else:
            assert False, "Env type must be either sim or real"

        # if no goal is given, sample one randomly
        if self.goal is None:
            trajectory = task.sample_goal()
            if task.GOAL_DIFFICULTY == 2:
                # Add some noise to standard goal to prevent overfitting
                # WARNING: ONLY APPLIES TO 1st GOAL of trajectory
                trajectory[0] = list(trajectory[0])
                # Set goal to be above cube
                trajectory[0][1][0:2] = cube_pos[0:2]
                if noisy:
                    noise = np.random.uniform(low=-0.05, high=0.05, size=3)
                    noise[-1] = np.random.uniform(low=-0.01, high=0.03)
                    trajectory[0][1] += noise
                trajectory[0] = tuple(trajectory[0])
        else:
            trajectory = self.goal

        # visualize the goal
        if self.visualization and self.env_type == 'sim':
            self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                width=task.move_cube._CUBE_WIDTH,
                position=trajectory[0][1],
                orientation=(0, 0, 0, 1),
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )

        self.info = {"time_index": -1, "trajectory": trajectory, "rrc_reward": 0, "xy_fail": False}
        
        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        observation, _, _, _ = self.step(self._initial_action, initial=True)

        return observation
    
    # TODO: test this is working properly!!!
    def _update_obj_vel(self, observation, initial):
        if initial:
            # Initial cube velocities are 0 (or should be)
            observation['object_observation']['lin_vel'] = np.zeros_like(observation["object_observation"]["position"])
            observation['object_observation']['ang_vel'] = np.zeros_like(observation["object_observation"]["orientation"])
            self._last_obs_vel = {
                "position": observation['object_observation']['lin_vel'].copy(),
                "orientation": observation['object_observation']['ang_vel'].copy()
                }
            self._steps_since_obj_update = 0
        else:
            lin_vel = observation['object_observation']['position'] - self._prev_object_obs['position']
            ang_vel = observation['object_observation']['orientation'] - self._prev_object_obs['orientation']
            # If object pose is updated, update object velocities
            if self.check_obs_updated(lin_vel, ang_vel):
                # Just take velocity as diff in observations
                observation['object_observation']['lin_vel'] = lin_vel
                observation['object_observation']['ang_vel'] = ang_vel
                # Update last obs-diff
                self._last_obs_vel['position'] = lin_vel
                self._last_obs_vel['orientation'] = ang_vel
                self._steps_since_obj_update = 0
            # Else maintain previous velocities and increment counter
            else:
                observation['object_observation']['lin_vel'] = self._last_obs_vel['position']
                observation['object_observation']['ang_vel'] = self._last_obs_vel['orientation']
                self._steps_since_obj_update += 1
        # Add counter to obs
        observation['object_observation']['steps_since_update'] = np.array([self._steps_since_obj_update])
        # Set previous object obs to current
        self._prev_object_obs = {
            "position": observation["object_observation"]["position"],
            "orientation": observation["object_observation"]["orientation"]
            }
        return observation
    
    def check_obs_updated(self, lin_vel, ang_vel):
        # If all differences are 0.0 then obs has not been updated
        check = np.sum(lin_vel != 0.0) + np.sum(ang_vel != 0.0)
        return check != 0
        
    # Concat robot obs and object obs into single array.
    # Also include object obs of prev time to infer velocity
    # TODO: Include last 4 frames/calculate velocity directly??
    # WARNING - changes will effect ddpg_agent_rrc.py code!!!!!!!!!!!!
    def flatten_obs(self, observation):
        # Concat all current robot and object observations
        # for key in observation:
        #     print(key)
        #     if type(observation[key]) is dict:
        #         for key2 in observation[key]:
        #             print('...{}'.format(key2))
        state_obs = None
        if self.obs_type == 'full':
            # Robot obs        
            state_obs = observation['robot_observation']['position']
            state_obs = np.concatenate((state_obs, observation['robot_observation']['velocity']))
            state_obs = np.concatenate((state_obs, observation['robot_observation']['torque']))
            state_obs = np.concatenate((state_obs, observation['robot_observation']['tip_force']))
            # state_obs = np.concatenate((state_obs, observation['robot_observation']['tip_positions'])) # TODO: Include?
            # Object obs
            state_obs = np.concatenate((state_obs, observation['object_observation']['position']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['orientation']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['lin_vel']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['ang_vel']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['steps_since_update']))
            # state_obs = np.concatenate((state_obs, observation['object_observation']['confidence'])) # TODO: Include? Need to generate proper values if so
            # # Action
            # state_obs = np.concatenate((state_obs, observation['action']))
            # # Params
            # state_obs = np.concatenate((state_obs, observation['params']))
        elif self.obs_type =='default':
            # Robot obs        
            state_obs = observation['robot_observation']['position']
            state_obs = np.concatenate((state_obs, observation['robot_observation']['velocity']))
            state_obs = np.concatenate((state_obs, observation['robot_observation']['torque']))
            # Object obs
            state_obs = np.concatenate((state_obs, observation['object_observation']['position']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['orientation']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['lin_vel']))
            state_obs = np.concatenate((state_obs, observation['object_observation']['ang_vel']))
        
        custom_obs = {
            "observation": state_obs,
            "desired_goal": observation["desired_goal"],
            "achieved_goal": observation["achieved_goal"],
            }
        
        return custom_obs
    
    def sample_init_state(self, init_state, noisy, noise_level=1):
        
        assert noise_level <= 1, "Noise level > 1"
        
        # Initialise with cube in robots grasp
        if init_state == 'grasp':
            # [right far, near, left far]
            rob_position = np.array([0.45, 0.9, -1.9, 0.28, 0.8, -1.93, 0.35, 0.9, -1.9], dtype=np.float32)
            # initialize cube
            cube_pos = (0, 0, (move_cube._CUBE_WIDTH / 2) + 0.062)
            cube_orient = trifingerpro_limits.object_orientation.default
            if noisy:
                # Add small noise to prevent overfitting
                rob_position += np.random.normal(loc=0.0, scale=0.005, size=rob_position.shape)
                cube_pos += np.random.normal(loc=0.0, scale=0.005, size=len(cube_pos))
                cube_orient += np.random.normal(loc=0.0, scale=0.005, size=cube_orient.shape)
        # Initialise cube on ground in center, with arm ready to pick it up        
        elif init_state == 'pick':
            # [right far, near, left far]
            rob_position = np.array([0.2, 0.75, -1.45, 0.2, 0.75, -1.45, 0.2, 0.75, -1.45], dtype=np.float32)
            cube_pos = task.INITIAL_CUBE_POSITION
            cube_orient = trifingerpro_limits.object_orientation.default
            if noisy:
                rob_position += np.random.normal(loc=0.0, scale=0.01, size=rob_position.shape)
                x_noise = np.random.normal(loc=0.0, scale=0.01)
                y_noise = np.random.normal(loc=0.0, scale=0.01)
                cube_pos = (x_noise, y_noise, move_cube._CUBE_WIDTH / 2)
                cube_orient[2] += np.random.normal(loc=0.0, scale=1)
        # Initialise state off previously visited state        
        elif init_state == 'prev_state' and self.buffer.current_size > 0:
            rob_position, cube_pos, cube_orient = self.sample_buffer_state()
        # Normal initialise - cube on ground somewhere, arms near center    
        else:
            rob_position = trifingerpro_limits.robot_position.default
            cube_pos = task.INITIAL_CUBE_POSITION
            cube_orient = trifingerpro_limits.object_orientation.default
            if noisy:
                # pos_range = trifingerpro_limits.robot_position.high - trifingerpro_limits.robot_position.low
                # rob_noise = np.random.normal(loc=0.0, scale=0.1, size=rob_position.shape) * pos_range
                # rob_position += rob_noise * noise_level
                # rob_position = np.clip(rob_position, trifingerpro_limits.robot_position.low, trifingerpro_limits.robot_position.high)
                # rob_position += np.random.normal(loc=0.0, scale=0.1, size=rob_position.shape) * noise_level
                x_noise = np.clip(np.random.normal(loc=0.0, scale=0.1), -0.1, 0.1) * noise_level
                y_noise = np.clip(np.random.normal(loc=0.0, scale=0.1), -0.1, 0.1) * noise_level
                cube_pos = (x_noise, y_noise, move_cube._CUBE_WIDTH / 2)
                cube_orient[2] += np.random.normal(loc=0.0, scale=1) * noise_level
                
        return rob_position, cube_pos, cube_orient
    
    def compute_sparse_reward(self, achieved_goal, desired_goal, info, check_success=False): # include info to match fetch gym environments (better for compatibility with HER code)
        if self.xy_only and not check_success:
            d = np.linalg.norm(achieved_goal[...,0:2] - desired_goal[...,0:2], axis=-1)
            return -(d > self.distance_threshold * 0.8).astype(np.float32) # TODO: scale distance properly to account for 2D vs 3D
        else:
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return -(d > self.distance_threshold).astype(np.float32)
        
    def compute_xy_fail(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal[...,0:2] - desired_goal[...,0:2], axis=-1)
        return d > 0.04
        
    # def apply_action_noise(self, action, n_scale=0):
    #     # TODO: 2nd correlated noise component - over episodes??
    #     if n_scale != 0:
    #         raise NotImplementedError()
    #     action_range = self.action_space.high - self.action_space.low
    #     noise = np.random.normal(loc=0.0, scale=action_range*n_scale, size=action_range.shape)
    #     action += noise
    #     action = np.clip(action, self.action_space.low, self.action_space.high)
    
    # def apply_obs_noise(self, observation, n_scale=0):
    #     # TODO: 2nd correlated noise component - over episodes??
    #     # For now just apply uncorrelated gaussian noise
    #     # TODO: only apply noise to policy??
    #     # TODO: choose better stds for noise
    #     if n_scale != 0:
    #         raise NotImplementedError()
    #     # TODO: calculate confidence
    #     # TODO: account for object velocities!!!!
    #     for key1 in observation:
    #         if key1 == "action":
    #             break
    #         print(key1)
    #         for key2 in observation[key1]:
    #             # if object velocity then skip
    #             print('...{}'.format(key2))
    #             obs_range = self.observation_space[key1][key2].high - self.observation_space[key1][key2].low
    #             noise = np.random.normal(loc=0.0, scale=obs_range*n_scale, size=obs_range.shape)
    #             observation[key1][key2] += noise
    #             observation[key1][key2] = np.clip(observation[key1][key2], self.observation_space[key1][key2].low, self.observation_space[key1][key2].high)
    
    

class RealRobotCubeTrajectoryEnv(BaseCubeTrajectoryEnv):
    """Gym environment for moving cubes with real TriFingerPro."""

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        Important: ``reset()`` needs to be called before doing the first step.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.info["time_index"] + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            self.info["time_index"] = t

            observation = self._create_observation(t, action)

            reward += self.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

            # make sure to not exceed the episode length
            if t >= task.EPISODE_LENGTH - 1:
                break

        is_done = t >= task.EPISODE_LENGTH

        return observation, reward, is_done, self.info

    def reset(self):
        # cannot reset multiple times
        if self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

        self.platform = robot_fingers.TriFingerPlatformWithObjectFrontend()

        # if no goal is given, sample one randomly
        if self.goal is None:
            trajectory = task.sample_goal()
        else:
            trajectory = self.goal

        self.info = {"time_index": -1, "trajectory": trajectory}

        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        observation, _, _, _ = self.step(self._initial_action)

        return observation

# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick gait tracking for Digit-v3."""

from typing import Any, Dict, Optional, Union
import time
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np
from jax import debug
import transforms3d as tf3
import transformations as tf
import os

from mujoco_playground._src import collision
from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.digit_v3 import base as digit_base
from mujoco_playground._src.locomotion.digit_v3 import digit_constants as consts
from mujoco_playground._src.locomotion.digit_v3 import digit_reference_data_loader as ref_loader


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.005,
      sim_dt=0.001, # 0.004,
      episode_length=1960,
      early_termination=True,
      action_repeat=1,
      action_scale=1, #0.3,
      history_len=20,
      hist_interval=6,
      ref_ee_pos_future_len=20,
      ref_ee_pos_future_interval=6,
      obs_noise=config_dict.create(
          level=0, #0.6,
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Rewards.
              tracking_joint_pos=0.3*10,
              tracking_root_pos=0.3*10, #0.2*10,
              tracking_root_ori=0.3*10,# 0.2*10,
              tracking_root_lin_vel=0.3*10, # 0.15*10,
              tracking_root_ang_vel=0.3*10, # 0.15*10,
              tracking_endeffector_pos=0.5*10, # 0.15*10,
              tracking_torque = 0.1*10,
              # Costs.
              root_motion_penalty=-1.0,
              projected_gravity_penalty=-2.0,
              action_rate=-0.01,
          ),
          tracking_sigma=0.5,
      ),
  )

class DigitRefTracking_Loco(digit_base.DigitEnv):
  """
  A class for tracking joystick-controlled gait in a simulated environment.
  """

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.FEET_ONLY_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._config = config
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = self._mj_model.keyframe("home").qpos[7:]
    self.gear_ratios = jp.array(self._mj_model.actuator_gear[:, 0])

    self._torso_id = self._mj_model.geom("base-b").id
    self._arm_geom_id = np.array([self._mj_model.geom(name).id for name in consts.ARM_GEOMS])
    
    self.base_init_pos        = self._mj_model.keyframe("home").qpos[:3]
    self.base_init_quat       = self._mj_model.keyframe("home").qpos[3:7] # w,x,y,z
    self.base_init_ori        = tf3.euler.quat2euler(self.base_init_quat) 
    self.base_init_rot        = tf3.euler.euler2mat(0, 0, self.base_init_ori[2])
    self.base_local_rot       = tf3.euler.euler2mat(0, 0, -self.base_init_ori[2])
    self.base_local_init_pos  = self.base_local_rot.dot(self._mj_model.keyframe("home").qpos[0:3])

    self.a_pos_index = jp.array([
      7,  8,  9,  14, 18, 23, 30, 31, 32, 33,  # actuator joint pos "hip-roll, yaw, pitch, knee, toe-A, B, shoulder-row, pitch, yaw, elbow"
      34, 35, 36, 41, 45, 50, 57, 58, 59, 60
    ]) 
    self.a_vel_index = jp.array([
      6,  7,  8,  12, 16, 20, 26, 27, 28, 29,  # actuator joint vel
      30, 31, 32, 36, 40, 44, 50, 51, 52, 53
    ]) 
    self.kp = jp.array([
      90, 90, 120, 120, 30, 30, 60, 60, 60, 60,   
      90, 90, 120, 120, 30, 30, 60, 60, 60, 60
    ])
    self.kd = jp.array([
      5.0, 5.0, 8.0, 8.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
      5.0, 5.0, 8.0, 8.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
    ])*0.2
    self.damping = jp.array([
      66.849, 26.1129, 38.05, 38.05, 15.5532, 15.5532, 66.849, 66.849, 26.1129, 66.849,
      66.849, 26.1129, 38.05, 38.05, 15.5532, 15.5532, 66.849, 66.849, 26.1129, 66.849
    ])
    self.kd += self.damping
    self.gear_ratio = jp.array(self._mj_model.actuator_gear[:, 0])
    self.ee_index = jp.array([21, 41, 16, 36]) # left hand, right hand, left foot, right foot
    # self.base_local_pos = jp.zeros(3)

    # reference trajectory path
    dir_path, name = os.path.split(os.path.abspath(__file__))
    self.reference_dataset_path = os.path.join(dir_path, "walking_with_torque") # walking_with_torque # testing
    self.ref_loader = ref_loader.JaxReferenceLoader(ref_traj_dir=self.reference_dataset_path)

  
  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, noise_rng, ref_rng = (  # pylint: disable=redefined-outer-name
        jax.random.split(rng, 3)
    )
    data = mjx_env.init(
        self.mjx_model, qpos=self._init_q, qvel=jp.zeros(self.mjx_model.nv)
    )
    
    # Base state in initial local coordinate
    self.base_local_pos = jp.dot(self.base_local_rot, data.qpos[0:3])
    self.base_local_trans = self.base_local_pos - self.base_local_init_pos
    self.base_local_quat = self.quaternion_multiply(self.quaternion_inverse(self.base_init_quat), data.qpos[3:7])
    self.base_local_ori = self.quat2euler(self.base_local_quat, axes='rxyz')
    self.base_local_lin_vel = jp.dot(self.base_local_rot, data.qvel[:3])
    self.base_local_ang_vel = jp.dot(self.base_local_rot, data.qvel[3:6])

    # base states in robot coordinate
    self.base_robot_rot = self.quat2mat(data.qpos[3:7])
    self.base_robot_lin_vel = jp.dot(self.base_robot_rot.T, data.qvel[0:3])
    self.base_robot_ang_vel = data.qvel[3:6]

    # End effector world/local positio
    self.base_world_ee_pos = data.xpos[self.ee_index] - data.qpos[:3]
    self.base_local_ee_pos = jp.dot(self.base_local_rot, self.base_world_ee_pos.T).T

    # Initialize history buffers.
    qpos_history = jp.zeros([self._config.history_len, 20])
    ref_ee_pos_future = jp.zeros([self._config.ref_ee_pos_future_len,12])
    self.actuator_torque = jp.zeros(self.mjx_model.nu)

    info = {
        "rng": rng,
        "step": 0,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "qpos_history": qpos_history,
        "ref_idx": self.sample_reference(ref_rng),
        "ref_ee_pos_future": ref_ee_pos_future,
    }



    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info, noise_rng)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], noise_rng = jax.random.split(state.info["rng"], 2)

    # motor_targets = state.data.ctrl + action * self._config.action_scale
    # motor_targets = self._default_pose + action * self._config.action_scale
    # motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    # motor_targets = state.data.qpos[self.a_pos_index] + action * self._config.action_scale
    motor_targets = self._init_q[self.a_pos_index] + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, 
        state.data, 
        motor_targets,
        self.kp,
        self.kd,
        self.a_pos_index,
        self.a_vel_index, 
        self.gear_ratio,
        self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    self.base_local_pos = jp.dot(self.base_local_rot, data.qpos[0:3])
    self.base_local_trans = self.base_local_pos - self.base_local_init_pos
    self.base_local_quat = self.quaternion_multiply(self.quaternion_inverse(self.base_init_quat), data.qpos[3:7])
    self.base_local_ori = self.quat2euler(self.base_local_quat, axes='rxyz')
    self.base_local_lin_vel = jp.dot(self.base_local_rot, data.qvel[:3])
    self.base_local_ang_vel = jp.dot(self.base_local_rot, data.qvel[3:6])

    
    # base states in robot coordinate
    self.base_robot_rot = self.quat2mat(data.qpos[3:7])
    self.base_robot_lin_vel = jp.dot(self.base_robot_rot.T, data.qvel[0:3])
    self.base_robot_ang_vel = data.qvel[3:6]

    self.base_world_ee_pos = data.xpos[self.ee_index] - data.qpos[:3]
    self.base_local_ee_pos = jp.dot(self.base_local_rot, self.base_world_ee_pos.T).T


    # history and future buffers.
    state.info["qpos_history"] = jp.roll(
      state.info["qpos_history"], shift=-1, axis=0
    ).at[-1].set(data.qpos[self.a_pos_index])

    # max_step = state.info["ref_motion_lens"] - 1
    max_step = jp.array(len(self.ref_loader.preloaded_refs["ref_base_local_pos"][state.info["ref_idx"]])) - 1
    future_index = jp.minimum(state.info["step"] + self._config.ref_ee_pos_future_len, max_step)
    state.info["ref_ee_pos_future"] = jp.roll(
      state.info["ref_ee_pos_future"], shift=-1, axis=0
      ).at[-1].set(jp.ravel(
        self.ref_loader.preloaded_refs["ref_base_local_ee_pos"][state.info["ref_idx"]][future_index]
        ))
    
    
    self.actuator_torque = self.get_act_joint_torques(self.gear_ratio, data)

    obs = self._get_obs(data, state.info, noise_rng)
    done = self._get_termination(data)
    

    pos, neg = self._get_reward(
        data, action, state.info, state.metrics, done,
    )
    pos = {k: v * self._config.reward_config.scales[k] for k, v in pos.items()}
    neg = {k: v * self._config.reward_config.scales[k] for k, v in neg.items()}
    rewards = pos | neg
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["step"] += 1
    state.info["rng"], ref_rng = jax.random.split(state.info["rng"])


    state.info["step"] = jp.where(
        done | (state.info["step"] > jp.array(len(self.ref_loader.preloaded_refs["ref_base_local_pos"][state.info["ref_idx"]])) - 10),
        0,
        state.info["step"],
    )

    state.info["ref_idx"] = jp.where(
      done | (state.info["step"] > jp.array(len(self.ref_loader.preloaded_refs["ref_base_local_pos"][state.info["ref_idx"]])) - 10),
      self.sample_reference(ref_rng),
      state.info["ref_idx"]
    )


    # # # TODO 
    # # # if the policy directly outputs a target position rather than a residual term, 
    # # # the action could be initialized to the initial position of the robot
    # state.info["last_act"] = jp.where(
    #     done | (state.info["step"] > jp.array(len(self.ref_loader.preloaded_refs["ref_base_local_pos"][state.info["ref_idx"]])) - 10),
    #     jp.zeros(self.mjx_model.nu),
    #     state.info["last_act"]
    # )

    # state.info["last_last_act"] = jp.where(
    #     done | (state.info["step"] > jp.array(len(self.ref_loader.preloaded_refs["ref_base_local_pos"][state.info["ref_idx"]])) - 10),
    #     jp.zeros(self.mjx_model.nu),
    #     state.info["last_last_act"]
    # )

    # state.info["qpos_history"] = jp.where(
    #     done | (state.info["step"] > jp.array(len(self.ref_loader.preloaded_refs["ref_base_local_pos"][state.info["ref_idx"]])) - 10),
    #     jp.zeros_like(state.info["qpos_history"]),
    #     state.info["qpos_history"]
    # )

    # state.info["ref_ee_pos_future"] = jp.where(
    #     done | (state.info["step"] > jp.array(len(self.ref_loader.preloaded_refs["ref_base_local_pos"][state.info["ref_idx"]])) - 10),
    #     jp.zeros_like(state.info["ref_ee_pos_future"]),
    #     state.info["ref_ee_pos_future"]
    # )
    
    # # state.info["qpos_history"] = jp.where(
    # #     state.info["step"] == 0,
    # #     jp.tile(self.ref_loader.preloaded_refs["ref_motor_joint_pos"][state.info["ref_idx"]][:1], (self._config.history_len, 1)),  
    # #     state.info["qpos_history"]
    # # )


    # # state.info["ref_ee_pos_future"] = jp.where(
    # #     state.info["step"] == 0,
    # #     jp.vstack([jp.ravel(self.ref_loader.preloaded_refs["ref_base_local_ee_pos"][state.info["ref_idx"]][i]) 
    # #               for i in range(self._config.ref_ee_pos_future_len)]),
    # #     state.info["ref_ee_pos_future"]
    # # )


    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)

    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    # Terminates if joint limits are exceeded or the robot falls.
    fall_termination = self.get_gravity(data)[-1] < 0.85
    base_too_low = data.qpos[2] < 0.8
    base_vel_crazy_check = jp.any(data.qvel[:3] > 2)
    torso_arm_is_colliding = jp.any(jp.array([
        collision.geoms_colliding(data, geom_id, self._torso_id)
        for geom_id in self._arm_geom_id
    ]))

    # termination_condition = fall_termination | base_too_low | base_vel_crazy_check | torso_arm_is_colliding
    termination_condition = jp.logical_or(
      jp.logical_or(fall_termination, base_too_low),
      jp.logical_or(base_vel_crazy_check, torso_arm_is_colliding)
  )
    return jp.where(
        self._config.early_termination,
        termination_condition,
        base_vel_crazy_check,
    )

  def _get_obs(
      self,
      data: mjx.Data,
      info: dict[str, Any],
      rng: jax.Array,
  ) -> jax.Array:
    
    hist_interval = self._config.hist_interval
    history_length = self._config.history_len
    qpos_hist_obs = jp.concatenate([info["qpos_history"][i * hist_interval,:] 
                     for i in range(int(history_length / hist_interval))]
    )
    future_interval = self._config.ref_ee_pos_future_interval
    future_length = self._config.ref_ee_pos_future_len
    ref_ee_pos_future = jp.concatenate([info["ref_ee_pos_future"][i * future_interval,:] 
                     for i in range(int(future_length / future_interval))]
    )
    
    obs = jp.concatenate([
        self.base_local_trans,
        self.base_robot_lin_vel, # 3
        self.base_robot_ang_vel, # 3
        # self.get_gyro(data),  # 3
        self.get_gravity(data),  # 3
        data.qpos[self.a_pos_index],  # 20
        # self.base_local_pos, # 3
        # self.base_local_ori, 
        jp.ravel(self.base_local_ee_pos), 
        self.ref_loader.preloaded_refs["ref_base_local_pos"][info["ref_idx"]][info["step"]],
        # self.ref_loader.preloaded_refs["ref_base_local_ori"][info["ref_idx"]][info["step"]],
        # self.ref_loader.preloaded_refs["ref_base_robot_lin_vel"][info["ref_idx"]][info["step"]],
        # self.ref_loader.preloaded_refs["ref_base_robot_ang_vel"][info["ref_idx"]][info["step"]],
        # self.ref_loader.preloaded_refs["ref_motor_joint_pos"][info["ref_idx"]][info["step"]],
        jp.ravel(self.ref_loader.preloaded_refs["ref_base_local_ee_pos"][info["ref_idx"]][info["step"]]),
        info["last_act"],  # 20
        qpos_hist_obs, # 20
        ref_ee_pos_future,
        # self.actuator_torque,
    ])  

    privileged_obs = jp.concatenate([
        self.base_local_trans,
        self.base_robot_lin_vel, # 3
        self.base_robot_ang_vel, # 3
        # self.get_gyro(data),  # 3
        self.get_gravity(data),  # 3
        data.qpos[self.a_pos_index],  # 20
        data.qvel[self.a_vel_index],
        self.base_local_pos, # 3
        self.base_local_ori,
        jp.ravel(self.base_local_ee_pos), 
        self.ref_loader.preloaded_refs["ref_base_local_trans"][info["ref_idx"]][info["step"]],
        self.ref_loader.preloaded_refs["ref_base_local_pos"][info["ref_idx"]][info["step"]],
        self.ref_loader.preloaded_refs["ref_base_local_ori"][info["ref_idx"]][info["step"]],
        self.ref_loader.preloaded_refs["ref_base_robot_lin_vel"][info["ref_idx"]][info["step"]],
        self.ref_loader.preloaded_refs["ref_base_robot_ang_vel"][info["ref_idx"]][info["step"]],
        self.ref_loader.preloaded_refs["ref_motor_joint_pos"][info["ref_idx"]][info["step"]],
        self.ref_loader.preloaded_refs["ref_motor_joint_vel"][info["ref_idx"]][info["step"]],
        jp.ravel(self.ref_loader.preloaded_refs["ref_base_local_ee_pos"][info["ref_idx"]][info["step"]]),
        info["last_act"],  # 20
        qpos_hist_obs, # 20
        ref_ee_pos_future,
        self.kp*0.01,
        self.kd*0.01,
        # self.actuator_torque,
    ]) 


    # Add noise.
    noise_vec = jp.zeros_like(obs)
    noise_vec = noise_vec.at[:3].set(
        self._config.obs_noise.level * self._config.obs_noise.scales.gyro
    )
    noise_vec = noise_vec.at[3:6].set(
        self._config.obs_noise.level * self._config.obs_noise.scales.gravity
    )
    noise_vec = noise_vec.at[6:25].set(
        self._config.obs_noise.level * self._config.obs_noise.scales.joint_pos
    )
    noise_vec = noise_vec.at[25:44].set(
        self._config.obs_noise.level * self._config.obs_noise.scales.joint_vel
    )
    obs = obs + (2 * jax.random.uniform(rng, shape=obs.shape) - 1) * noise_vec

    # # Update history.
    # qpos_history = (
    #     jp.roll(info["qpos_history"], 20)
    #     .at[:20]
    #     .set(data.qpos[self.a_pos_index])
    # )
    # info["qpos_history"] = qpos_history


    # Concatenate final observation.
    # obs = jp.hstack(
    #     [
    #         obs,
    #         qpos_history,
    #     ],
    # )
    return {
        "state": obs,
        "privileged_state": privileged_obs,
    }
  

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    del done, metrics  # Unused.
    pos = {
       "tracking_joint_pos": self._reward_tracking_joint_pos(
          self.ref_loader.preloaded_refs["ref_motor_joint_pos"][info["ref_idx"]][info["step"]],
          data.qpos[self.a_pos_index]
       ),
       "tracking_root_pos": self._reward_tracking_root_pos(
          self.ref_loader.preloaded_refs["ref_base_local_pos"][info["ref_idx"]][info["step"]], 
          self.base_local_pos
       ),
       "tracking_root_ori": self._reward_tracking_root_ori(
          self.ref_loader.preloaded_refs["ref_base_local_ori"][info["ref_idx"]][info["step"]], 
          self.base_local_ori
       ),
        "tracking_root_lin_vel": self._reward_tracking_lin_vel(
            self.ref_loader.preloaded_refs["ref_base_robot_lin_vel"][info["ref_idx"]][info["step"]],  
            self.base_robot_lin_vel,
        ),
        "tracking_root_ang_vel": self._reward_tracking_ang_vel( 
            self.ref_loader.preloaded_refs["ref_base_robot_ang_vel"][info["ref_idx"]][info["step"]],  
            self.base_robot_ang_vel,
        ),
        "tracking_endeffector_pos": self._reward_tracking_endeffector_pos(
            self.ref_loader.preloaded_refs["ref_base_local_ee_pos"][info["ref_idx"]][info["step"]],
            self.base_local_ee_pos
        ),
        "tracking_torque": self._reward_tracking_torque(
            self.ref_loader.preloaded_refs["ref_torque"][info["ref_idx"]][info["step"]],
            self.actuator_torque,
        ),
    }
    neg = {
        "root_motion_penalty": self._cost_root_motion(
          self.base_robot_lin_vel, 
          self.base_robot_ang_vel
        ),
        "projected_gravity_penalty": self._cost_projected_gravity(
          self.get_gravity(data)
        ),
        "action_rate": self._cost_action_rate(
          info["last_act"], 
          info["last_last_act"], 
          action
        ),
    }
    return pos, neg


  
  def _reward_tracking_joint_pos(
      self,
      ref_actuator_targets: jax.Array,
      actuator_pos: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    error = jp.sum(jp.square(ref_actuator_targets - actuator_pos))
    reward = jp.exp(-5*error)
    return reward
  
  def _reward_tracking_root_pos(
      self,
      ref_base_local_pos: jax.Array,
      base_local_pos: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    root_pos_error = jp.sum(jp.square(ref_base_local_pos - base_local_pos))
    reward = jp.exp(-20*root_pos_error)
    return reward
  
  def _reward_tracking_root_ori(
      self,
      ref_base_local_ori: jax.Array,
      base_local_ori: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    diff_euler = ref_base_local_ori - base_local_ori
    diff_euler = jax.vmap(self.wrap_to_pi)(diff_euler)
    rp_error = jp.sum(jp.square(diff_euler[:2]))
    y_error = jp.sum(jp.square(diff_euler[2])) 
    reward = jp.exp(-100 * rp_error - 50 * y_error)
    return reward

  def wrap_to_pi(
    self,
    angle: jax.Array,
   ) -> jax.Array:
    return (angle + jp.pi) % (2 * jp.pi) - jp.pi
  
  
  def _reward_tracking_lin_vel(
      self,
      ref_base_robot_lin_vel: jax.Array,
      base_robot_lin_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(base_robot_lin_vel[:2] - ref_base_robot_lin_vel[:2]))
    reward = jp.exp(-10 * lin_vel_error)
    return reward
  
  def _reward_tracking_ang_vel(
      self,
      ref_base_robot_ang_vel: jax.Array,
      base_robot_ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    ang_vel_error = jp.sum(jp.square(base_robot_ang_vel[2] - ref_base_robot_ang_vel[2]))
    reward = jp.exp(-ang_vel_error)
    return reward
  

  def _reward_tracking_endeffector_pos(
      self,
      ref_base_local_ee_pos: jax.Array,
      base_local_ee_pos: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    base_local_ee_pos = jp.ravel(base_local_ee_pos)
    ref_base_local_ee_pos = jp.ravel(ref_base_local_ee_pos)
    endeffector_pos_error = jp.sum(jp.square(base_local_ee_pos - ref_base_local_ee_pos))
    reward = jp.exp(-50*endeffector_pos_error)
    return reward
  

  def _reward_tracking_torque(
      self,
      ref_torque: jax.Array,
      state_torque: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    actuator_torque_limit = jp.array([80*1.583530725, 50*1.583530720, 16*13.557993625, 16*14.457309375, 
                                  50*0.839518840, 50*0.839518840,
                                  80*1.583530725, 80*1.583530725, 50*1.5835307200, 80*1.5835307250,
                                  80*1.583530725, 50*1.583530720, 16*13.557993625, 16*14.457309375, 
                                  50*0.839518840, 50*0.839518840,
                                  80*1.583530725, 80*1.583530725, 50*1.5835307200, 80*1.5835307250])

    torque_error = jp.sum(jp.square((ref_torque - state_torque)/actuator_torque_limit))
    reward = jp.exp(-1e-3*torque_error)
    return reward
  
  

  

  def _cost_root_motion(
      self, 
      base_robot_lin_vel: jax.Array,
      base_robot_ang_vel: jax.Array,
      ) -> jax.Array:
    # Penalize deviation from the default pose for certain joints.
    lin_vel_error = jp.square(base_robot_lin_vel[2])
    ang_vel_error = jp.sum(jp.square(base_robot_ang_vel[:2]))
    # return lin_vel_error + 0.5 * ang_vel_error
    return lin_vel_error + 2*ang_vel_error
  
  def _cost_projected_gravity(
      self, 
      projected_gravity: jax.Array,
      ) -> jax.Array:
    # Penalize deviation from the default pose for certain joints.
    error = jp.sum(jp.square(projected_gravity[:2]))
    # return lin_vel_error + 0.5 * ang_vel_error
    return error
  


  def _cost_lin_vel_z(self, global_linvel, gait: jax.Array) -> jax.Array:  # pylint: disable=redefined-outer-name
    # Penalize z axis base linear velocity unless pronk or bound.
    cost = jp.square(global_linvel[2])
    return cost * (gait > 0)

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    # Penalize first and second derivative of actions.
    c1 = jp.sum(jp.square(act - last_act))
    c2 = jp.sum(jp.square(act - 2 * last_act + last_last_act))
    return c1 + c2

  
  def quaternion_inverse(self,q: jax.Array)-> jax.Array:
    """Computes the inverse of a quaternion."""
    w, x, y, z = q
    return jp.array([w, -x, -y, -z]) / jp.dot(q, q)

  def quaternion_multiply(self, q1: jax.Array, q2: jax.Array)-> jax.Array:
    """Computes the product of two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
  
  def quat2euler(self, q: jax.Array, axes='rxyz')-> jax.Array:
    """Converts a quaternion to Euler angles."""
    w, x, y, z = q
    if axes == 'rxyz':
        roll = jp.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = jp.arcsin(2 * (w * y - z * x))
        yaw = jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return jp.array([roll, pitch, yaw])
    
    elif axes == 'rzyx':
        # Extrinsic rotations (z -> y -> x)
        yaw = jp.arctan2(2 * (w * z + x * y), 1 - 2 * (z**2 + y**2))
        pitch = jp.arcsin(2 * (w * y - x * z))
        roll = jp.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + z**2))
        return jp.array([roll, pitch, yaw])
    else:
        raise NotImplementedError(f"Unsupported axes: {axes}")
    

  def quat2mat(
      self, 
      q: jax.Array)-> jax.Array:
    """Converts a quaternion to a rotation matrix."""
    w, x, y, z = q
    return jp.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
    ])
  
  
  
  def sample_reference(self, rng: jax.Array) -> jax.Array:
    return jax.random.randint(rng, shape=(), minval=0, maxval=self.ref_loader.preloaded_refs["ref_motor_joint_pos"].shape[0])
  
  def get_act_joint_torques(self, gear_ratios: jax.Array, data: mjx.Data) -> jax.Array:
        """
        Returns actuator force in joint space.
        """
        motor_torques = data.actuator_force
        return motor_torques * gear_ratios 
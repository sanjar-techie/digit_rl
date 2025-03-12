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
"""RL config for Locomotion envs."""

from ml_collections import config_dict

from mujoco_playground._src import locomotion


def brax_ppo_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  env_config = locomotion.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=100_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.97,
      learning_rate=3e-4,
      entropy_cost=1e-2,
      num_envs=8192,
      batch_size=256,
      max_grad_norm=1.0,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=(128, 128, 128, 128),
          value_hidden_layer_sizes=(256, 256, 256, 256, 256),
          policy_obs_key="state",
          value_obs_key="state",
      ),
  )

  if env_name in ("Go1JoystickFlatTerrain", "Go1JoystickRoughTerrain"):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 10
    rl_config.num_resets_per_eval = 1
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in ("Go1Handstand", "Go1Footstand"):
    rl_config.num_timesteps = 100_000_000
    rl_config.num_evals = 5
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name == "Go1Backflip":
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 10
    rl_config.discounting = 0.95
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name == "Go1Getup":
    rl_config.num_timesteps = 50_000_000
    rl_config.num_evals = 5
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in ("G1JoystickFlatTerrain", "G1JoystickRoughTerrain"):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 20
    rl_config.clipping_epsilon = 0.2
    rl_config.num_resets_per_eval = 1
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in (
      "BerkeleyHumanoidJoystickFlatTerrain",
      "BerkeleyHumanoidJoystickRoughTerrain",
  ):
    rl_config.num_timesteps = 150_000_000
    rl_config.num_evals = 15
    rl_config.clipping_epsilon = 0.2
    rl_config.num_resets_per_eval = 1
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in (
      "DigitRefTracking_Loco_JaxPPO",
      "DigitRefTracking_Jumping_JaxPPO",
      "DigitRefTracking_Loco_JaxPPO_fivepoints",
  ):
    rl_config.num_timesteps=30_000_000
    rl_config.num_evals=10
    rl_config.reward_scaling=1.0
    rl_config.episode_length=env_config.episode_length
    rl_config.normalize_observations=True
    rl_config.action_repeat=1
    rl_config.unroll_length=20
    rl_config.num_minibatches=32
    rl_config.num_updates_per_batch=4
    rl_config.discounting=0.97
    rl_config.learning_rate=5e-5 #3e-4
    rl_config.entropy_cost=1e-2
    rl_config.num_envs=4096
    rl_config.batch_size=128 #256
    rl_config.max_grad_norm=1.0
    rl_config.network_factory=config_dict.create(
        policy_hidden_layer_sizes=(512, 512, 256, 256),
        value_hidden_layer_sizes=(512, 512, 512, 256, 256),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

    # rl_config.num_timesteps=300_000_000
    # rl_config.num_evals=10
    # rl_config.reward_scaling=1.0
    # rl_config.episode_length=env_config.episode_length
    # rl_config.normalize_observations=True
    # rl_config.action_repeat=1
    # rl_config.unroll_length=1024  # Matches SB3 n_steps=2048, but adjusted for batch size
    # rl_config.num_minibatches=32  # Adjusted to match batch_size=64
    # rl_config.num_updates_per_batch=10  # Matches SB3 n_epochs=10
    # rl_config.discounting=0.99  # Matches SB3 gamma=0.99
    # rl_config.gae_lambda=0.95 # Matches SB3 gae_lambda=0.95
    # # rl_config.clip_range=0.2 # Matches SB3 clip_range
    # rl_config.entropy_cost=0.0 # Matches SB3 ent_coef
    # # rl_config.value_loss_coef=0.5  # Matches SB3 vf_coef
    # rl_config.max_grad_norm=0.5 # Matches SB3 max_grad_norm
    # rl_config.learning_rate=5e-5  # Matches SB3 learning rate
    # rl_config.num_envs=2048 # Reduced to match batch size
    # rl_config.batch_size=64 # Matches SB3 batch size
    # rl_config.network_factory=config_dict.create(
    #     policy_hidden_layer_sizes=(512, 512, 256, 256),  # Adjusted to match SB3
    #     value_hidden_layer_sizes=(512, 512, 512, 256, 256),
    #     policy_obs_key="state",
    #     value_obs_key="privileged_state",
    # )




  elif env_name in (
      "T1JoystickFlatTerrain",
      "T1JoystickRoughTerrain",
  ):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 20
    rl_config.clipping_epsilon = 0.2
    rl_config.num_resets_per_eval = 1
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in (
      "BarkourJoystick",
      "H1InplaceGaitTracking",
      "H1JoystickGaitTracking",
      "Op3Joystick",
      "SpotFlatTerrainJoystick",
      "SpotGetup",
      "SpotJoystickGaitTracking",
  ):
    pass  # use default config
  else:
    raise ValueError(f"Unsupported env: {env_name}")

  return rl_config


def rsl_rl_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned RSL-RL PPO config for the given environment."""

  rl_config = config_dict.create(
      seed=1,
      runner_class_name="OnPolicyRunner",
      policy=config_dict.create(
          init_noise_std=1.0,
          actor_hidden_dims=[512, 256, 128],
          critic_hidden_dims=[512, 256, 128],
          # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
          activation="elu",
          class_name="ActorCritic",
      ),
      algorithm=config_dict.create(
          class_name="PPO",
          value_loss_coef=1.0,
          use_clipped_value_loss=True,
          clip_param=0.2,
          entropy_coef=0.001,
          num_learning_epochs=5,
          # mini batch size = num_envs*nsteps / nminibatches
          num_mini_batches=4,
          learning_rate=3.0e-4,  # 5.e-4
          schedule="fixed",  # could be adaptive, fixed
          gamma=0.99,
          lam=0.95,
          desired_kl=0.01,
          max_grad_norm=1.0,
      ),
      num_steps_per_env=24,  # per iteration
      max_iterations=100000,  # number of policy updates
      empirical_normalization=True,
      # logging
      save_interval=50,  # check for potential saves every this many iterations
      experiment_name="test",
      run_name="",
      # load and resume
      resume=False,
      load_run="-1",  # -1 = last run
      checkpoint=-1,  # -1 = last saved model
      resume_path=None,  # updated from load_run and chkpt
  )

  if env_name in (
      "Go1Getup",
      "BerkeleyHumanoidJoystickFlatTerrain",
      "G1Joystick",
      "Go1JoystickFlatTerrain",
  ):
    rl_config.max_iterations = 1000
  if env_name == "Go1JoystickFlatTerrain":
    rl_config.algorithm.learning_rate = 3e-4
    rl_config.algorithm.schedule = "fixed"

  if env_name in {
    "DigitRefTracking_Loco_RSLRL",
    "DigitRefTracking_Loco_JaxPPO",
    }:

    rl_config.policy=config_dict.create(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 512, 256, 256],  # Keep similar MLP size
            critic_hidden_dims=[512, 512, 512, 256, 256],
            activation="elu",
            class_name="ActorCritic",
        )
    rl_config.algorithm=config_dict.create(
            class_name="PPO",
            value_loss_coef=0.5,  # Matches SB3 vf_coef
            use_clipped_value_loss=True,
            clip_param=0.2,  # Matches SB3 clip_range
            entropy_coef=0.0,  # Matches SB3 ent_coef
            num_learning_epochs=10,  # Matches SB3 n_epochs
            num_mini_batches=64,  # Adjusted for batch size=64
            learning_rate=5e-5,  # Matches SB3 learning rate
            schedule="fixed",
            gamma=0.99,  # Matches SB3 gamma
            lam=0.95,  # Matches SB3 gae_lambda
            max_grad_norm=0.5,  # Matches SB3 max_grad_norm
        )
    rl_config.num_envs=1024 # Adjusted to match SB3 batch size
    rl_config.num_steps_per_env=512 # Adjusted to match SB3 batch size
    rl_config.max_iterations=2000
    rl_config.empirical_normalization=True  # Normalize observations like SB3
    

  return rl_config

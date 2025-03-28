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

"""
#Training:
python train_jax_ppo.py   --env_name=DigitRefTracking_Loco_JaxPPO_fivepoints   --num_timesteps=5000000   --num_evals=10   --reward_scaling=1.0   --episode_length=1000   --normalize_observations=True   --action_repeat=2   --unroll_length=20   --num_minibatches=16   --num_updates_per_batch=8   --discounting=0.99   --learning_rate=3e-4   --entropy_cost=1e-2   --num_envs=2048   --num_eval_envs=256   --batch_size=512   --max_grad_norm=0.5   --clipping_epsilon=0.2   --policy_hidden_layer_sizes=256,256,128   --value_hidden_layer_sizes=256,256,128   --seed=42   --use_wandb=True   --use_tb=True
# Evaluation
python train_jax_ppo.py   --env_name=DigitRefTracking_Loco_JaxPPO_fivepoints   --load_checkpoint_path=logs/DigitRefTracking_Loco_JaxPPO_fivepoints-
20250314-221114/checkpoints/  --play_only=True
"""
from datetime import datetime
import functools
import json
import os
import time
import warnings
import pandas as pd

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter
import wandb

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Set logging verbosity to INFO for more output
logging.set_verbosity(logging.INFO)

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "DigitRefTracking_Loco_JaxPPO_fivepoints",  # Updated default environment
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb", False, "Use Weights & Biases for logging (ignored in play-only mode)"
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", 1_000_000, "Number of timesteps")
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer("num_minibatches", 8, "Number of minibatches")
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer("num_eval_envs", 128, "Number of evaluation environments")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float("clipping_epsilon", 0.2, "Clipping epsilon for PPO")
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes", [64, 64, 64], "Policy hidden layer sizes"
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes", [64, 64, 64], "Value hidden layer sizes"
)
_POLICY_OBS_KEY = flags.DEFINE_string("policy_obs_key", "state", "Policy obs key")
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    print(f"Fetching RL config for env: {env_name}")
    try:
        if env_name in mujoco_playground.manipulation._envs:
            if _VISION.value:
                return manipulation_params.brax_vision_ppo_config(env_name)
            return manipulation_params.brax_ppo_config(env_name)
        elif env_name in mujoco_playground.locomotion._envs:
            if _VISION.value:
                return locomotion_params.brax_vision_ppo_config(env_name)
            return locomotion_params.brax_ppo_config(env_name)
        elif env_name in mujoco_playground.dm_control_suite._envs:
            if _VISION.value:
                return dm_control_suite_params.brax_vision_ppo_config(env_name)
            return dm_control_suite_params.brax_ppo_config(env_name)
        else:
            # Fallback for custom environments
            print(f"Env {env_name} not found in standard sets, using default locomotion config")
            return locomotion_params.brax_ppo_config(env_name)
    except Exception as e:
        print(f"Error in get_rl_config: {e}")
        raise ValueError(f"Env {env_name} not found or misconfigured in {registry.ALL_ENVS}.")


def main(argv):
    """Run training and evaluation for the specified environment."""
    print("Starting main function...")
    del argv

    try:
        # Load environment configuration
        print(f"Loading env config for {_ENV_NAME.value}")
        env_cfg = registry.get_default_config(_ENV_NAME.value)
        print(f"Env config loaded: {env_cfg}")

        ppo_params = get_rl_config(_ENV_NAME.value)

        # Apply flag overrides
        if _NUM_TIMESTEPS.present:
            ppo_params.num_timesteps = _NUM_TIMESTEPS.value
        if _PLAY_ONLY.present:
            ppo_params.num_timesteps = 0
        if _NUM_EVALS.present:
            ppo_params.num_evals = _NUM_EVALS.value
        if _REWARD_SCALING.present:
            ppo_params.reward_scaling = _REWARD_SCALING.value
        if _EPISODE_LENGTH.present:
            ppo_params.episode_length = _EPISODE_LENGTH.value
        if _NORMALIZE_OBSERVATIONS.present:
            ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
        if _ACTION_REPEAT.present:
            ppo_params.action_repeat = _ACTION_REPEAT.value
        if _UNROLL_LENGTH.present:
            ppo_params.unroll_length = _UNROLL_LENGTH.value
        if _NUM_MINIBATCHES.present:
            ppo_params.num_minibatches = _NUM_MINIBATCHES.value
        if _NUM_UPDATES_PER_BATCH.present:
            ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
        if _DISCOUNTING.present:
            ppo_params.discounting = _DISCOUNTING.value
        if _LEARNING_RATE.present:
            ppo_params.learning_rate = _LEARNING_RATE.value
        if _ENTROPY_COST.present:
            ppo_params.entropy_cost = _ENTROPY_COST.value
        if _NUM_ENVS.present:
            ppo_params.num_envs = _NUM_ENVS.value
        if _NUM_EVAL_ENVS.present:
            ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
        if _BATCH_SIZE.present:
            ppo_params.batch_size = _BATCH_SIZE.value
        if _MAX_GRAD_NORM.present:
            ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
        if _CLIPPING_EPSILON.present:
            ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
        if _POLICY_HIDDEN_LAYER_SIZES.present:
            ppo_params.network_factory.policy_hidden_layer_sizes = list(
                map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
            )
        if _VALUE_HIDDEN_LAYER_SIZES.present:
            ppo_params.network_factory.value_hidden_layer_sizes = list(
                map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
            )
        if _POLICY_OBS_KEY.present:
            ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
        if _VALUE_OBS_KEY.present:
            ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value

        if _VISION.value:
            env_cfg.vision = True
            env_cfg.vision_config.render_batch_size = ppo_params.num_envs
        print(f"Loading environment: {_ENV_NAME.value}")
        env = registry.load(_ENV_NAME.value, config=env_cfg)

        print(f"Environment Config:\n{env_cfg}")
        print(f"PPO Training Parameters:\n{ppo_params}")

        # Generate unique experiment name
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        exp_name = f"{_ENV_NAME.value}-{timestamp}"
        if _SUFFIX.value is not None:
            exp_name += f"-{_SUFFIX.value}"
        print(f"Experiment name: {exp_name}")

        # Set up logging directory
        logdir = epath.Path("logs").resolve() / exp_name
        logdir.mkdir(parents=True, exist_ok=True)
        print(f"Logs are being stored in: {logdir}")

        # Initialize Weights & Biases if required
        if _USE_WANDB.value and not _PLAY_ONLY.value:
            wandb.init(project="mjxrl", entity="diffusion_humanoid", name=exp_name)
            wandb.config.update(env_cfg.to_dict())
            wandb.config.update({"env_name": _ENV_NAME.value})

        # Initialize TensorBoard if required
        if _USE_TB.value and not _PLAY_ONLY.value:
            writer = SummaryWriter(logdir)

        # Handle checkpoint loading
        if _LOAD_CHECKPOINT_PATH.value is not None:
            ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
            if ckpt_path.is_dir():
                latest_ckpts = list(ckpt_path.glob("*"))
                latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
                latest_ckpts.sort(key=lambda x: int(x.name))
                latest_ckpt = latest_ckpts[-1]
                restore_checkpoint_path = latest_ckpt
                print(f"Restoring from: {restore_checkpoint_path}")
            else:
                restore_checkpoint_path = ckpt_path
                print(f"Restoring from checkpoint: {restore_checkpoint_path}")
        else:
            print("No checkpoint path provided, not restoring from checkpoint")
            restore_checkpoint_path = None

        # Set up checkpoint directory
        ckpt_path = logdir / "checkpoints"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint path: {ckpt_path}")

        # Save environment configuration
        with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
            json.dump(env_cfg.to_dict(), fp, indent=4)

        # Define policy parameters function for saving checkpoints
        def policy_params_fn(current_step, make_policy, params):
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(params)
            path = ckpt_path / f"{current_step}"
            orbax_checkpointer.save(path, params, force=True, save_args=save_args)

        training_params = dict(ppo_params)
        if "network_factory" in training_params:
            del training_params["network_factory"]

        network_fn = (
            ppo_networks_vision.make_ppo_networks_vision
            if _VISION.value
            else ppo_networks.make_ppo_networks
        )
        if hasattr(ppo_params, "network_factory"):
            network_factory = functools.partial(network_fn, **ppo_params.network_factory)
        else:
            network_factory = network_fn

        if _DOMAIN_RANDOMIZATION.value:
            training_params["randomization_fn"] = registry.get_domain_randomizer(_ENV_NAME.value)

        if _VISION.value:
            env = wrapper.wrap_for_brax_training(
                env,
                vision=True,
                num_vision_envs=env_cfg.vision_config.render_batch_size,
                episode_length=ppo_params.episode_length,
                action_repeat=ppo_params.action_repeat,
                randomization_fn=training_params.get("randomization_fn"),
            )

        num_eval_envs = ppo_params.num_envs if _VISION.value else ppo_params.get("num_eval_envs", 128)

        if "num_eval_envs" in training_params:
            del training_params["num_eval_envs"]

        train_fn = functools.partial(
            ppo.train,
            **training_params,
            network_factory=network_factory,
            policy_params_fn=policy_params_fn,
            seed=_SEED.value,
            restore_checkpoint_path=restore_checkpoint_path,
            wrap_env_fn=None if _VISION.value else wrapper.wrap_for_brax_training,
            num_eval_envs=num_eval_envs,
        )

        times = [time.monotonic()]

        def progress(num_steps, metrics):
            times.append(time.monotonic())
            if _USE_WANDB.value and not _PLAY_ONLY.value:
                wandb.log(metrics, step=num_steps)
            if _USE_TB.value and not _PLAY_ONLY.value:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, num_steps)
                writer.flush()
            print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")

        eval_env = None if _VISION.value else registry.load(_ENV_NAME.value, config=env_cfg)

        print("Starting training...")
        make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress, eval_env=eval_env)

        print("Done training.")
        if len(times) > 1:
            print(f"Time to JIT compile: {times[1] - times[0]}")
            print(f"Time to train: {times[-1] - times[1]}")

        print("Starting inference...")
        inference_fn = make_inference_fn(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)

        print("Preparing for evaluation...")
        num_envs = 1
        if _VISION.value:
            eval_env = env
            num_envs = env_cfg.vision_config.render_batch_size

        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)

        rng = jax.random.PRNGKey(123)
        rng, reset_rng = jax.random.split(rng)
        if _VISION.value:
            reset_rng = jp.asarray(jax.random.split(reset_rng, num_envs))
        state = jit_reset(reset_rng)
        state0 = jax.tree_util.tree_map(lambda x: x[0], state) if _VISION.value else state
        rollout = [state0]

        log_data = []
        ee_index = jp.array([21, 41, 16, 36])

        for episode in range(1):
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)
            episode_reward = 0
            episode_log = []

            for step in range(env_cfg.episode_length):
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, ctrl)
                state0 = jax.tree_util.tree_map(lambda x: x[0], state) if _VISION.value else state

                episode_log.append({
                    "episode": episode,
                    "step": step,
                    "qpos": state0.data.qpos.tolist(),
                    "qvel": state0.data.qvel.tolist(),
                    "xpos": state0.data.xpos[ee_index].tolist(),
                    "action": state0.info["last_act"].tolist(),
                    "reward": state0.reward
                })

                episode_reward += state0.reward
                rollout.append(state0)
                if state0.done:
                    break

            log_data.append({
                "episode": episode,
                "total_reward": episode_reward,
                "data": episode_log
            })
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"evaluation_log_{timestamp}.csv"

        df = pd.DataFrame([
            {"episode": ep["episode"], "step": entry["step"],
             "qpos": entry["qpos"], "qvel": entry["qvel"], "xpos": entry["xpos"], "reward": entry["reward"]}
            for ep in log_data for entry in ep["data"]
        ])
        df.to_csv(csv_filename, index=False)
        print(f"Logged data saved as {csv_filename}")

        render_every = 2
        fps = 1.0 / eval_env.dt / render_every
        print(f"FPS for rendering: {fps}")

        traj = rollout[::render_every]

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

        frames = eval_env.render(
            traj, camera="track", scene_option=scene_option, width=640, height=480
        )

        mp4_filename = f"rollout_{timestamp}.mp4"
        media.write_video(mp4_filename, frames, fps=fps)
        print(f"Rollout video saved as {mp4_filename}")

    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    app.run(main)
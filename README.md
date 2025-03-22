# Digit Five-Point Tracking RL Controller

## Overview
This project trains a reinforcement learning (RL) controller using Proximal Policy Optimization (PPO) to enable the Digit robot (simulated in MuJoCo) to track five specific points: the root (torso), left foot, right foot, left hand, and right hand. The goal is to achieve precise locomotion and manipulation by aligning these keypoints with reference trajectories, leveraging a pretrained model and evaluating its performance.

## Environment
- **Name**: `DigitRefTracking_Loco_JaxPPO_fivepoints`
- **Simulator**: MuJoCo (via JAX with MJX)
- **Tracked Points**:
  - Root (torso position and orientation)
  - Left foot (end-effector position)
  - Right foot (end-effector position)
  - Left hand (end-effector position)
  - Right hand (end-effector position)
- **Reward Config**:
  - Positive rewards for tracking end-effector positions (`tracking_endeffector_pos`), joint positions (`tracking_joint_pos`), and root dynamics (`tracking_root_*`).
  - Penalties for excessive action rates (`action_rate`), root motion (`root_motion_penalty`), and tilting (`projected_gravity_penalty`).

## Training
- **Algorithm**: PPO (via Brax and JAX)
- **Pretrained Model**: Loaded from `logs/DigitRefTracking_Loco_JaxPPO_fivepoints-20250314-221114/checkpoints/1001226240`, trained for 50M timesteps.
- **Hardware**: NVIDIA RTX 4090 (24GB VRAM, 1 GPU)
- **Key Hyperparameters**:
  - `num_timesteps`: 50M
  - `num_envs`: 2048 (parallel environments)
  - `num_eval_envs`: 256
  - `learning_rate`: 3e-4
  - `episode_length`: 1000 steps
  - `reward_scaling`: 1.0
- **Command** (training):
  ```bash
  python train_jax_ppo.py --env_name=DigitRefTracking_Loco_JaxPPO_fivepoints --num_timesteps=5000000 --num_evals=10 --load_checkpoint_path=logs/DigitRefTracking_Loco_JaxPPO_fivepoints-20250314-221114/checkpoints/1001226240 [other args]
  ```

## Evaluation
- **Mode**: Evaluation-only ("play only") to assess the pretrained policy.
- **Command**:
  ```bash
  python train_jax_ppo.py --env_name=DigitRefTracking_Loco_JaxPPO_fivepoints --num_timesteps=0 --num_evals=1 --load_checkpoint_path=logs/DigitRefTracking_Loco_JaxPPO_fivepoints-20250314-221114/checkpoints
  ```
- **Output**: Reward, steps per second (SPS), episode length, and optionally a video of the Digit robot tracking the five points.

## Results
- **Pretrained Performance**: After 50M timesteps, the policy achieves good tracking of the root, feet, and hands (reward ~14.889 at 4.6M steps in resumed run).
- **Logs**: Stored in `logs/DigitRefTracking_Loco_JaxPPO_fivepoints-20250322-144002/` with checkpoints every ~655k steps.
- **Visualization**: 
To visualize the performance, you can view the video below:

![Digit Robot Tracking](learning/rollout_20250322_150912.mp4)

## Setup
1. **Environment**: Activate the Conda env:
   ```bash
   conda activate mujoco_playground
   ```
2. **Dependencies**: JAX, Brax, MuJoCo, Orbax (installed in `mujoco_playground` env).
3. **Run**: Use the training or evaluation commands above.

## Notes
- The pretrained model was extended from 1M to 5M timesteps, improving reward from 7.382 to 14.889.
- Future work could tune penalties (e.g., `root_motion_penalty`) to reduce early terminations (episode length < 1000).
#!/usr/bin/env python
"""
A simple script to load a trained model and render a video using train_jax_ppo.py's play_only mode.
Usage: python play_checkpoint.py --checkpoint_dir=/path/to/checkpoint_dir --output=video.mp4
"""

import os
import argparse
import subprocess
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Play a trained model and create a video')
parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to checkpoint directory')
parser.add_argument('--env_name', type=str, default="DigitRefTracking_Loco_JaxPPO_fivepoints", help='Environment name')
parser.add_argument('--output', type=str, default=None, help='Output video filename')
args = parser.parse_args()

# Set environment variables for rendering
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Find latest checkpoint step
checkpoint_dir = args.checkpoint_dir
steps = [int(d) for d in os.listdir(checkpoint_dir) if d.isdigit()]
if not steps:
    raise ValueError(f"No checkpoint steps found in {checkpoint_dir}")
latest_step = max(steps)
latest_checkpoint = os.path.join(checkpoint_dir, str(latest_step))
print(f"Using checkpoint from step {latest_step}")

# Get timestamp for output file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_prefix = f"rollout_{timestamp}"

# Construct the command to run train_jax_ppo.py in play-only mode
cmd = [
    "python", "train_jax_ppo.py",
    f"--env_name={args.env_name}",
    "--play_only=True",
    f"--load_checkpoint_path={latest_checkpoint}",
]

print(f"Running command: {' '.join(cmd)}")
result = subprocess.run(cmd, check=True)

# Find the generated MP4 file
mp4_files = [f for f in os.listdir() if f.startswith(output_prefix) and f.endswith(".mp4")]
if not mp4_files:
    print("Warning: No MP4 file was generated.")
    exit(1)

newest_mp4 = max(mp4_files, key=os.path.getctime)

# Rename if output name specified
if args.output:
    os.rename(newest_mp4, args.output)
    print(f"Video saved as {args.output}")
else:
    print(f"Video saved as {newest_mp4}")
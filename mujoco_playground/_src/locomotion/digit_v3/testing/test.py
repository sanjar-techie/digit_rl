import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Path to the dataset file (adjust if needed)
dataset_path = "digit_crocoddyl_walkingforward_353114_20250304103147723801.npz"

# Check if the file exists
if not os.path.exists(dataset_path):
    print(f"Error: File {dataset_path} not found. Please verify the path.")
    exit(1)

# Load the .npz file
data = np.load(dataset_path)
print("Keys in the .npz file:", list(data.keys()))

# Extract ref_data
ref_data = data["ref_data"]
steps, total_dims = ref_data.shape
print(f"Shape of ref_data: {ref_data.shape} ({steps} timesteps, {total_dims} dimensions)")

# Expected components based on JaxReferenceLoader
ref_qpos = ref_data[:, 0:61]          # Joint positions (61D)
ref_qvel = ref_data[:, 61:61+54]      # Joint velocities (54D)
ref_ee_pos = ref_data[:, 61+54:61+54+12]  # End-effector positions (12D)
ref_torque = ref_data[:, 61+54+12:61+54+12+20]  # Torques (20D)
extra_data = ref_data[:, 61+54+12+20:]  # Extra 3D (150 - 147)

# Break down qpos
root_pos = ref_qpos[:, :3]            # Root position (x, y, z)
root_quat = ref_qpos[:, 3:7]          # Root orientation (w, x, y, z)
joint_pos = ref_qpos[:, 7:]           # Actuated joint positions (54D)

# Reshape end-effector positions
ref_ee_pos_reshaped = ref_ee_pos.reshape(steps, 4, 3)  # 4 end-effectors × 3D

# Component summary
print("\n=== Component Breakdown ===")
print(f"Root position (qpos[:3]): {root_pos.shape}")
print(f"Root quaternion (qpos[3:7]): {root_quat.shape}")
print(f"Joint positions (qpos[7:]): {joint_pos.shape}")
print(f"Joint velocities (qvel): {ref_qvel.shape}")
print(f"End-effector positions (ee_pos): {ref_ee_pos_reshaped.shape}")
print(f"Torques: {ref_torque.shape}")
print(f"Extra data (beyond 147D): {extra_data.shape}")

# Five points analysis
print("\n=== Five Points Analysis ===")
print(f"Root position: {root_pos.shape} (3D)")
print(f"End-effectors (left hand, right hand, left foot, right foot): {ref_ee_pos_reshaped.shape} (4 × 3D)")
print(f"Total five points: 1 root + 4 end-effectors = 5 points")
print(f"Total dimensions: {root_pos.shape[1] + ref_ee_pos_reshaped.shape[1] * ref_ee_pos_reshaped.shape[2]}D")

# Sample data (first and last timesteps)
print("\n=== Sample Data ===")
print("First timestep:")
print(f"Root position: {root_pos[0]}")
print(f"End-effectors:\n{ref_ee_pos_reshaped[0]}")
print("Last timestep:")
print(f"Root position: {root_pos[-1]}")
print(f"End-effectors:\n{ref_ee_pos_reshaped[-1]}")

# Statistics
def print_stats(name, array):
    print(f"{name} stats:")
    print(f"  Min: {np.min(array, axis=0)}")
    print(f"  Max: {np.max(array, axis=0)}")
    print(f"  Mean: {np.mean(array, axis=0)}")
    print(f"  Std: {np.std(array, axis=0)}")

print("\n=== Statistics ===")
print_stats("Root position", root_pos)
print_stats("End-effector positions", ref_ee_pos_reshaped.reshape(steps, -1))  # Flatten for stats
print_stats("Joint velocities", ref_qvel)
print_stats("Torques", ref_torque)
if extra_data.size > 0:
    print_stats("Extra data", extra_data)

# Visualization
fig = plt.figure(figsize=(12, 8))

# 3D Plot of Root and End-Effectors
ax = fig.add_subplot(111, projection='3d')
ax.plot(root_pos[:, 0], root_pos[:, 1], root_pos[:, 2], label="Root", color='b', linewidth=2)
for i, label in enumerate(["Left Hand", "Right Hand", "Left Foot", "Right Foot"]):
    ax.plot(ref_ee_pos_reshaped[:, i, 0], ref_ee_pos_reshaped[:, i, 1], ref_ee_pos_reshaped[:, i, 2], 
            label=label, linestyle='--')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Trajectory of Five Points (Root + 4 End-Effectors)")
ax.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
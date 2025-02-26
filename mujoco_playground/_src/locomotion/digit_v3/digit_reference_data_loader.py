import os
import jax
import jax.numpy as jp
import numpy as np
import time
from jax import ShapeDtypeStruct
from functools import partial
from jax import jit


class JaxReferenceLoader:
    def __init__(self, ref_traj_dir, subsample_factor=1, device="cpu"):
        """
        JAX-based reference trajectory loader that loads one random trajectory per reset.
        """
        self.subsample_factor = subsample_factor
        self.ref_data = None
        self.device = device
        self.ref_traj_dir = ref_traj_dir

        self.a_pos_index = jp.array([
            7,  8,  9,  14, 18, 23, 30, 31, 32, 33,  # actuator joint pos "hip-roll, yaw, pitch, knee, toe-A, B, shoulder-row, pitch, yaw, elbow"
            34, 35, 36, 41, 45, 50, 57, 58, 59, 60
        ]) 
        self.a_vel_index = jp.array([
            6,  7,  8,  12, 16, 20, 26, 27, 28, 29,     # actuator joint vel
            30, 31, 32, 36, 40, 44, 50, 51, 52, 53
        ]) 

        self.preloaded_refs = self._preload_trajectories()


    def _preload_trajectories(self):
        """
        Preload all reference trajectories into memory.
        """
        preloaded_refs = {
            # "ref_motion_lens": [],
            "ref_motor_joint_pos": [],
            "ref_motor_joint_vel": [],
            "ref_base_local_pos": [],
            "ref_base_local_trans": [],
            "ref_base_local_ori": [],
            "ref_base_local_ee_pos": [],
            "ref_base_robot_lin_vel": [],
            "ref_base_robot_ang_vel": [],
            "ref_base_robot_ee_pos": [],
        }

        for file in os.listdir(self.ref_traj_dir):
            if file.endswith(".npz"):
                path = os.path.join(self.ref_traj_dir, file)
                data = np.load(path)
                ref_data = data["ref_data"]
                ref_qpos = jp.array(ref_data[:, 0:61])
                ref_qvel = jp.array(ref_data[:, 61:61+54])
                ref_ee_pos = jp.array(ref_data[:, 61+54:61+54+12].reshape(-1, 4, 3))
                ref_base_local_quat = jp.array(ref_qpos[:, [6, 3, 4, 5]])  # (w, x, y, z) Notice: Specify the index for your reference trajectory

                ref_base_local_ori = self.quat2euler(ref_base_local_quat)
                ref_base_local_pos = jp.array(ref_qpos[:, :3])
                ref_base_local_trans = ref_base_local_pos - ref_base_local_pos[0]
                ref_base_local_ee_pos = ref_ee_pos - ref_base_local_pos[:, None, :]

                ref_base_rot = jax.vmap(self.euler2mat)(ref_base_local_ori)
                ref_base_rot = ref_base_rot.squeeze()
                ref_base_robot_ee_pos = jp.einsum('bij,bjk->bik', ref_base_rot.transpose(0, 2, 1), ref_base_local_ee_pos.transpose(0, 2, 1)).transpose(0, 2, 1)
                ref_base_robot_lin_vel = jp.einsum('bij,bj->bi', ref_base_rot.transpose(0, 2, 1), ref_qvel[:, :3])
                ref_base_robot_ang_vel = jp.einsum('bij,bj->bi', ref_base_rot.transpose(0, 2, 1), ref_qvel[:, 3:6])

                # preloaded_refs["ref_motion_lens"].append(ref_motion_lens)
                preloaded_refs["ref_motor_joint_pos"].append(ref_qpos[:,self.a_pos_index])
                preloaded_refs["ref_motor_joint_vel"].append(ref_qvel[:,self.a_vel_index])
                preloaded_refs["ref_base_local_pos"].append(ref_base_local_pos)
                preloaded_refs["ref_base_local_trans"].append(ref_base_local_trans)
                preloaded_refs["ref_base_local_ori"].append(ref_base_local_ori)
                preloaded_refs["ref_base_local_ee_pos"].append(ref_base_local_ee_pos)
                preloaded_refs["ref_base_robot_lin_vel"].append(ref_base_robot_lin_vel)
                preloaded_refs["ref_base_robot_ang_vel"].append(ref_base_robot_ang_vel)
                preloaded_refs["ref_base_robot_ee_pos"].append(ref_base_robot_ee_pos)

        for key in preloaded_refs:
            preloaded_refs[key] = jp.array(preloaded_refs[key])

        print("Successfully loaded the reference!!")

        return preloaded_refs # jax.tree_util.tree_map(lambda x: jp.stack(x, axis=0), preloaded_refs)
        

        
    def quat2euler(self, quat):
        """
        Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = jp.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = jp.arcsin(jp.clip(2*(w*y - z*x), -1, 1))
        yaw = jp.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return jp.stack([roll, pitch, yaw], axis=-1)

    
    def euler2mat(self, euler):
        """
        Convert Euler angles (roll, pitch, yaw) to rotation matrices.
        Supports both single `(3,)` and batched `(T, 3)` inputs.
        """
        roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

        cx, sx = jp.cos(roll), jp.sin(roll)
        cy, sy = jp.cos(pitch), jp.sin(pitch)
        cz, sz = jp.cos(yaw), jp.sin(yaw)

        rot_x = jp.stack([
            jp.ones_like(cx), jp.zeros_like(cx), jp.zeros_like(cx),
            jp.zeros_like(cx), cx, -sx,
            jp.zeros_like(cx), sx, cx
        ], axis=-1).reshape(-1, 3, 3)

        rot_y = jp.stack([
            cy, jp.zeros_like(cy), sy,
            jp.zeros_like(cy), jp.ones_like(cy), jp.zeros_like(cy),
            -sy, jp.zeros_like(cy), cy
        ], axis=-1).reshape(-1, 3, 3)

        rot_z = jp.stack([
            cz, -sz, jp.zeros_like(cz),
            sz, cz, jp.zeros_like(cz),
            jp.zeros_like(cz), jp.zeros_like(cz), jp.ones_like(cz)
        ], axis=-1).reshape(-1, 3, 3)

        return jp.einsum('bij,bjk,bkl->bil', rot_z, rot_y, rot_x)
    


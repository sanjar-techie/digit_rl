import os
import jax
import jax.numpy as jp
import numpy as np
import time

class JaxReferenceLoader:
    def __init__(self, ref_traj_path, subsample_factor=1):
        """
        JAX-based reference trajectory loader that loads one random trajectory per reset.
        """
        self.subsample_factor = subsample_factor
        self.ref_data = None

        if os.path.isdir(ref_traj_path):
            # Training mode
            self.ref_traj_dir = ref_traj_path
            self.ref_files = [f for f in os.listdir(self.ref_traj_dir) if f.endswith(".npz")]
            if not self.ref_files:
                raise ValueError(f"No reference trajectories found in {self.ref_traj_dir}")
            self.ref_traj_file = None
        else:
            # Testing mode
            if not ref_traj_path.endswith(".npz"):
                raise ValueError(f"Expected an .npz file, got: {ref_traj_path}")
            self.ref_traj_dir = None
            self.ref_traj_file = ref_traj_path


        self.reset()

    def _preload_trajectory(self, file_path):
        """
        Load a single reference trajectory into JAX arrays.
        """
        data = np.load(file_path)
        ref_data = data["ref_data"]
        ref_motion_lens = ref_data.shape[0]

        ref_qpos = jp.array(ref_data[:, 0:61])
        ref_qvel = jp.array(ref_data[:, 61:61+54])
        ref_ee_pos = jp.array(ref_data[:, 61+54:61+54+12].reshape(-1, 4, 3))
        ref_base_local_quat = jp.array(ref_qpos[:, [6, 3, 4, 5]])  # (w, x, y, z)

        ref_base_local_ori = self.quat2euler(ref_base_local_quat)
        ref_base_local_pos = jp.array(ref_qpos[:, :3])
        ref_base_local_trans = ref_base_local_pos - ref_base_local_pos[0]
        ref_base_local_ee_pos = ref_ee_pos - ref_base_local_pos[:, None, :]

        ref_base_rot = jax.vmap(self.euler2mat)(ref_base_local_ori)
        ref_base_rot = ref_base_rot.squeeze()
        ref_base_robot_ee_pos = jp.einsum('bij,bjk->bik', ref_base_rot.transpose(0, 2, 1), ref_base_local_ee_pos.transpose(0, 2, 1)).transpose(0, 2, 1)
        ref_base_robot_lin_vel = jp.einsum('bij,bj->bi', ref_base_rot.transpose(0, 2, 1), ref_qvel[:, :3])
        ref_base_robot_ang_vel = jp.einsum('bij,bj->bi', ref_base_rot.transpose(0, 2, 1), ref_qvel[:, 3:6])

        self.ref_data = {
            "ref_motion_lens":ref_motion_lens,
            "ref_qpos": ref_qpos,
            "ref_qvel": ref_qvel,
            "ref_base_local_pos": ref_base_local_pos,
            "ref_base_local_trans": ref_base_local_trans,
            "ref_base_local_ori": ref_base_local_ori,
            "ref_base_local_ee_pos": ref_base_local_ee_pos,
            "ref_base_robot_lin_vel": ref_base_robot_lin_vel,
            "ref_base_robot_ang_vel": ref_base_robot_ang_vel,
            "ref_base_robot_ee_pos": ref_base_robot_ee_pos,
        }


    def reset(self):
        """
        Randomly select a new trajectory and load it.
        """
        if self.ref_traj_dir:
            # training mode
            random_file = np.random.choice(self.ref_files)
            file_path = os.path.join(self.ref_traj_dir, random_file)
        else:
            # testing mode
            file_path = self.ref_traj_file

        self._preload_trajectory(file_path)



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



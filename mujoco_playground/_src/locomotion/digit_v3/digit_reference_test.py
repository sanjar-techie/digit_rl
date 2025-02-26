import jax.numpy as jp
import jax

preloaded_refs = {
    "ref_qpos": [jp.array([[1, 2], [3, 4]]),  # Trajectory 0 (2 timesteps, 2 features)
                 jp.array([[5, 6], [7, 8]]),  # Trajectory 1
                 jp.array([[9, 10], [11, 12]])],  # Trajectory 2
    "ref_qvel": [jp.array([[0.1, 0.2], [0.3, 0.4]]),
                 jp.array([[0.5, 0.6], [0.7, 0.8]]),
                 jp.array([[0.9, 1.0], [1.1, 1.2]])]
}

idx = 1  # Selecting trajectory 1

selected_traj = jax.tree_util.tree_map(lambda x: jp.take(jp.array(x), idx, axis=1), preloaded_refs)

print(selected_traj)

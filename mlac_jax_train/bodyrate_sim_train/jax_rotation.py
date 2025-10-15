import jax
import jax.numpy as jnp

def rotation_matrix_to_euler_jax(R):
    """
    Converts a rotation matrix to ZYX Euler angles in a JIT-compatible way.
    Handles the gimbal lock singularity.
    Returns: [roll, pitch, yaw] (x, y, z)
    """
    # Check for gimbal lock (when the pitch is +/- 90 degrees)
    # R[2, 0] is -sin(pitch)
    is_gimbal_lock = jnp.abs(R[2, 0]) > 0.99999

    def gimbal_lock_true(_):
        # When in gimbal lock, yaw is 0 and roll is derived from atan2 of other elements
        roll = jnp.arctan2(-R[0, 1], -R[0, 2])
        pitch = -jnp.arcsin(R[2, 0])
        yaw = 0.0
        return jnp.array([roll, pitch, yaw])

    def gimbal_lock_false(_):
        # Normal case
        roll = jnp.arctan2(R[2, 1], R[2, 2])
        pitch = -jnp.arcsin(R[2, 0])
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        return jnp.array([roll, pitch, yaw])

    # Use jax.lax.cond for JIT-safe conditional logic
    rpy = jax.lax.cond(is_gimbal_lock, gimbal_lock_true, gimbal_lock_false, None)
    
    # The result from the paper is [yaw, pitch, roll], but our state is often [roll, pitch, yaw]
    # The above implementation returns [roll, pitch, yaw]
    return rpy
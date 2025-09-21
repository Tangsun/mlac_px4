import jax
import jax.numpy as jnp
import numpy as np

import IPython

from mlac_sim.utils import quaternion_to_rotation_matrix, hat, vee


class BodyRateConverter:
    """
    Converts desired attitude into bodyrate commands using a Proportional controller.
    """
    def __init__(self, kp: float):
        self.kp = kp

    def attitude_to_bodyrate(self, q_current: jnp.ndarray, q_desired: jnp.ndarray) -> jnp.ndarray:
        """
        Converts desired attitude into bodyrate commands using a Proportional controller.

        Args:
            q_current (jnp.ndarray): Current attitude as a quaternion [qw, qx, qy, qz].
            q_desired (jnp.ndarray): Desired attitude as a quaternion [qw, qx, qy, qz].

        Returns:
            jnp.ndarray: Desired body rates [p, q, r] in radians per second.
        """
        # Convert quaternions to rotation matrices
        R_current = quaternion_to_rotation_matrix(q_current)
        R_desired = quaternion_to_rotation_matrix(q_desired)

        # IPython.embed()

        # Compute the rotation error
        R_error = 1/2 * (R_desired.T @ R_current - R_current.T @ R_desired)
        R_error = vee(R_error)

        # # Convert rotation error to axis-angle representation
        # angle_error = jnp.arccos((jnp.trace(R_error) - 1) / 2.0)
        # if jnp.isclose(angle_error, 0.0):
        #     axis_error = jnp.zeros(3)
        # else:
        #     axis_error = vee((R_error - R_error.T) / (2 * jnp.sin(angle_error))) * angle_error

        # Proportional control to compute body rates
        body_rate_cmd = - self.kp * R_error

        return body_rate_cmd
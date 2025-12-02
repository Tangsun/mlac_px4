import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
for path in (PARENT_DIR, GRANDPARENT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from mlac_jax_train.utils import rk38_step, hat, vee  # noqa: E402
from mlac_jax_train.dynamics import prior  # noqa: E402


def npy_reference_func(t, ts_ref, r_ref, dr_ref, ddr_ref, yaw_ref, yaw_rate_ref):
    r = jnp.array([jnp.interp(t, ts_ref, r_ref[:, i]) for i in range(3)])
    dr = jnp.array([jnp.interp(t, ts_ref, dr_ref[:, i]) for i in range(3)])
    ddr = jnp.array([jnp.interp(t, ts_ref, ddr_ref[:, i]) for i in range(3)])
    yaw = jnp.interp(t, ts_ref, yaw_ref)
    yaw_rate = jnp.interp(t, ts_ref, yaw_rate_ref)
    return r, dr, ddr, yaw, yaw_rate


def calculate_smc_command(z_tree, t, k_R, K_mat, Lambda_mat, reference_func):
    x, R_flatten, _ = z_tree
    q, dq = x[:3], x[3:]
    R = R_flatten.reshape((3, 3))

    r, dr, ddr, yaw_d, yaw_rate_d = reference_func(t)

    e = q - r
    de = dq - dr
    s = de + Lambda_mat @ e
    v = dr - Lambda_mat @ e
    dv = ddr - Lambda_mat @ de
    H, C, g, B = prior(q, dq)
    tau = H @ dv + C @ v + g - K_mat @ s
    u_d = jnp.linalg.solve(B, tau)
    f_d = jnp.linalg.norm(u_d)

    b_3d = u_d / (f_d + 1e-6)
    b_1d_desired = jnp.array([jnp.cos(yaw_d), jnp.sin(yaw_d), 0.0])
    b_2d_temp = jnp.cross(b_3d, b_1d_desired)
    b_2d = b_2d_temp / (jnp.linalg.norm(b_2d_temp) + 1e-6)
    b_1d = jnp.cross(b_2d, b_3d)
    R_d = jnp.column_stack((b_1d, b_2d, b_3d))

    e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
    world_yaw_rate = jnp.array([0.0, 0.0, yaw_rate_d])
    Omega_ff = R.T @ world_yaw_rate
    Omega_cmd = -k_R * e_R + Omega_ff

    return f_d, Omega_cmd


def simulation_ode_zoh(z, commands, dt, attitude_time_constant=0.02):
    f_d, Omega_cmd = commands
    x, R_flatten, Omega_state = z
    q, dq = x[:3], x[3:]
    R = R_flatten.reshape((3, 3))

    dR = R @ hat(Omega_state)
    u_applied = f_d * R @ jnp.array([0.0, 0.0, 1.0])
    H, C, g, _ = prior(q, dq)
    ddq = jnp.linalg.solve(H, u_applied - C @ dq - g)
    dx = jnp.concatenate((dq, ddq))
    tau = max(attitude_time_constant, 1e-6)
    dOmega = (Omega_cmd - Omega_state) / tau
    return dx, dR.flatten(), dOmega


def run_smc_window(reference_data, initial_state, gains, attitude_time_constant=0.02):
    ts_ref, r_ref, dr_ref, ddr_ref, yaw_ref, yaw_rate_ref = reference_data
    ts_ref = np.asarray(ts_ref)
    if ts_ref.size < 2:
        raise ValueError("Reference window must contain at least two samples.")

    k_R, K_mat, Lambda_mat = gains
    k_R = jnp.asarray(k_R)
    K_mat = jnp.asarray(K_mat)
    Lambda_mat = jnp.asarray(Lambda_mat)

    r_ref = jnp.asarray(r_ref)
    dr_ref = jnp.asarray(dr_ref)
    ddr_ref = jnp.asarray(ddr_ref)
    yaw_ref = jnp.asarray(yaw_ref)
    yaw_rate_ref = jnp.asarray(yaw_rate_ref)

    ts_local = ts_ref - ts_ref[0]
    dt = float(ts_local[1] - ts_local[0])

    # Build initial pytrees
    pos0 = jnp.array(initial_state[0:3])
    vel0 = jnp.array(initial_state[3:6])
    rpy0 = initial_state[6:9]
    R0 = jnp.array(Rotation.from_euler('xyz', rpy0).as_matrix())
    x0 = jnp.concatenate([pos0, vel0])
    z0_tree = (x0, R0.flatten(), jnp.zeros(3))
    z0_flat, unravel = jax.flatten_util.ravel_pytree(z0_tree)

    ref_partial = partial(
        npy_reference_func,
        ts_ref=jnp.asarray(ts_local),
        r_ref=r_ref,
        dr_ref=dr_ref,
        ddr_ref=ddr_ref,
        yaw_ref=yaw_ref,
        yaw_rate_ref=yaw_rate_ref,
    )

    def flat_dynamics(z_flat_inner, t_inner):
        z_tree_inner = unravel(z_flat_inner)
        commands_inner = calculate_smc_command(
            z_tree_inner, t_inner, k_R, K_mat, Lambda_mat, ref_partial
        )
        dz_tree_inner = simulation_ode_zoh(
            z_tree_inner, commands_inner, dt, attitude_time_constant
        )
        return jnp.concatenate(jax.tree_util.tree_leaves(dz_tree_inner))

    history = [z0_flat]
    current = z0_flat
    for t in ts_local[:-1]:
        current = rk38_step(flat_dynamics, dt, current, t)
        history.append(current)

    history = jnp.stack(history)
    z_hist = jax.vmap(unravel)(history)
    x_hist, R_hist_flat, Omega_hist = z_hist
    pos = x_hist[:, :3]
    vel = x_hist[:, 3:6]
    R_mats = R_hist_flat.reshape(-1, 3, 3)
    euler = Rotation.from_matrix(np.asarray(R_mats)).as_euler('xyz', degrees=True)
    return (
        np.asarray(ts_local),
        np.asarray(pos),
        np.asarray(vel),
        euler,
        np.asarray(Omega_hist),
    )

#!/usr/bin/env python3
import os
import pickle
import argparse
from functools import partial
import matplotlib.pyplot as plt  # For visualization
from tqdm import tqdm  # For progress bars
import traceback  # For printing full tracebacks

# Configure JAX for 64-bit precision if needed, and to handle NaNs
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)  # This should raise an error on NaN

import jax
import jax.numpy as jnp
import numpy as np  # For initial data loading, final result handling, and plotting

# Attempt to import from your utils.py
try:
    from utils import quaternion_to_rotation_matrix as q2R_util  # Rename to avoid conflict
    from utils import hat, vee, rk38_step, odeint_fixed_step
except ImportError:
    print("ERROR: Could not import functions from utils.py.")
    print("Please ensure utils.py is in the current directory or accessible in PYTHONPATH.")
    # Fallback definitions (ensure these match your utils.py if used)
    def q2R_util(quat_wxyz):  # JAX version from utils
        w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
        R = jnp.array([
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w,     1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w,     2 * y * z + 2 * x * w,     1 - 2 * x**2 - 2 * y**2]
        ], dtype=jnp.float64)
        return R

    def hat(v):
        return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=jnp.float64)

    def vee(R_skew):
        return jnp.array([R_skew[2, 1], R_skew[0, 2], R_skew[1, 0]], dtype=jnp.float64)

    if 'rk38_step' not in globals():
        raise ImportError("rk38_step not found. Please ensure utils.py is accessible.")

# --- Configuration ---
INERTIA_MATRIX_J_JAX = jnp.diag(jnp.array([0.02167, 0.02167, 0.04000], dtype=jnp.float64))
INERTIA_MATRIX_J_INV_JAX = jnp.linalg.inv(INERTIA_MATRIX_J_JAX)

# --- JAX Attitude Dynamics ODE ---
# J and J_inv are now regular arguments, not static
@jax.jit
def attitude_ode_jax(state_sim, t, R_d_target, Omega_d_target, k_R_gain_jax, k_Omega_gain_jax, J, J_inv):
    R_sim_flatten, Omega_sim = state_sim
    R_sim = R_sim_flatten.reshape((3, 3))
    e_R = 0.5 * vee(R_d_target.T @ R_sim - R_sim.T @ R_d_target)
    e_Omega = Omega_sim - R_sim.T @ R_d_target @ Omega_d_target
    M_control = -jnp.diag(k_R_gain_jax) @ e_R - jnp.diag(k_Omega_gain_jax) @ e_Omega \
    + jnp.cross(Omega_sim, J @ Omega_sim) - J @ (hat(Omega_sim) @ R_sim.T @ R_d_target @ Omega_d_target)
    dOmega_sim = J_inv @ (M_control - jnp.cross(Omega_sim, J @ Omega_sim))
    dR_sim = R_sim @ hat(Omega_sim)
    return (dR_sim.flatten(), dOmega_sim)

# --- Flat wrapper around attitude_ode_jax for rk38_step ---
def flat_attitude_ode(state_flat,
                      t,
                      R_d_target,
                      Omega_d_target,
                      k_R_gain_jax,
                      k_Omega_gain_jax,
                      J,
                      J_inv):
    """
    state_flat: jnp.ndarray of shape (12,)
      - state_flat[:9]   = R_flat (flattened 3×3)
      - state_flat[9:12] = Omega (3,)
    Returns a single jnp.ndarray of shape (12,) = [dR_flat; dOmega]
    """
    # Unpack
    R_flat = state_flat[:9]
    Omega = state_flat[9:]
    # Call the existing attitude_ode_jax, which expects state_sim = (R_flat, Omega)
    dR_flat, dOmega = attitude_ode_jax(
        (R_flat, Omega),
        t,
        R_d_target,
        Omega_d_target,
        k_R_gain_jax,
        k_Omega_gain_jax,
        J,
        J_inv,
    )
    # Concatenate back into a single 12-vector
    return jnp.concatenate([dR_flat, dOmega], axis=0)

# --- JAX Simulation Function (updated for consistent dtypes) ---
@partial(jax.jit, static_argnames=('num_total_states_output',))
def simulate_trajectory_attitude_jax_single(
    R_initial_sim_jax,        # jnp.ndarray shape (3,3), dtype float64
    Omega_initial_sim_jax,    # jnp.ndarray shape (3,), dtype float64
    desired_R_input_traj_jax, # jnp.ndarray shape (N, 3, 3), dtype float64
    k_R_test_jax,             # jnp.ndarray shape (3,), dtype float64
    k_Omega_test_jax,         # jnp.ndarray shape (3,), dtype float64
    num_total_states_output,  # int (should be N+1)
    dt,                       # float (Python), we’ll cast to float64
    J,                        # jnp.ndarray shape (3,3), dtype float64
    J_inv                     # jnp.ndarray shape (3,3), dtype float64
):
    """
    Runs one JAX‐ified simulation of attitude dynamics over a desired‐R trajectory.
    We now carry a single 12-vector (R_flat + Omega) into rk38_step, split it,
    orthonormalize R, then re-concatenate at each step. All arrays forced to float64.
    """

    # Ensure everything is float64
    R_initial_sim_jax = R_initial_sim_jax.astype(jnp.float64)
    Omega_initial_sim_jax = Omega_initial_sim_jax.astype(jnp.float64)
    desired_R_input_traj_jax = desired_R_input_traj_jax.astype(jnp.float64)
    k_R_test_jax = k_R_test_jax.astype(jnp.float64)
    k_Omega_test_jax = k_Omega_test_jax.astype(jnp.float64)
    J = J.astype(jnp.float64)
    J_inv = J_inv.astype(jnp.float64)

    # Convert dt to a jnp.float64 scalar
    dt_jax = jnp.array(dt, dtype=jnp.float64)

    # Desired angular velocity is always zero here, dtype float64
    Omega_d_target_jax = jnp.zeros(3, dtype=jnp.float64)

    # Flatten the initial rotation matrix and concatenate with Omega_initial
    R0_flat = R_initial_sim_jax.reshape((9,))                   # shape (9,)
    state0_flat = jnp.concatenate([R0_flat, Omega_initial_sim_jax], axis=0)  # shape (12,)

    def body_func_for_scan(carry_state_flat, R_d_current_step):
        # Force R_d_current_step to float64
        R_d_current_step = R_d_current_step.astype(jnp.float64)

        # Build a partial of flat_attitude_ode with this step's R_d_target
        ode_for_this_step = partial(
            flat_attitude_ode,
            R_d_target=R_d_current_step,
            Omega_d_target=Omega_d_target_jax,
            k_R_gain_jax=k_R_test_jax,
            k_Omega_gain_jax=k_Omega_test_jax,
            J=J,
            J_inv=J_inv,
        )

        # One RK3/8 step: carry_state_flat is shape (12,)
        next_state_flat = rk38_step(ode_for_this_step, dt_jax, carry_state_flat, 0.0)
        # next_state_flat has shape (12,) and dtype float64

        # Split into R_flat (9) and Omega (3)
        R_next_flat = next_state_flat[:9]    # shape (9,)
        Omega_next    = next_state_flat[9:]  # shape (3,)

        # Orthonormalize R_next_flat → R_ortho
        R_next = R_next_flat.reshape((3, 3))
        U, _, Vh = jnp.linalg.svd(R_next)
        R_ortho = U @ Vh
        R_ortho_flat = R_ortho.reshape((9,))

        # Re-concatenate R_ortho_flat and Omega_next into new carry (12,)
        new_carry = jnp.concatenate([R_ortho_flat, Omega_next], axis=0)

        # Return the new carry (dtype float64), plus (R_ortho_flat, Omega_next) for history
        return new_carry, (R_ortho_flat, Omega_next)

    # Run lax.scan over all steps of desired_R_input_traj_jax (length N)
    carry_init = state0_flat  # shape (12,), dtype float64
    _, history = jax.lax.scan(body_func_for_scan, carry_init, desired_R_input_traj_jax)
    # history is a tuple:
    #   history[0] has shape (N, 9), dtype float64  = R_flat at each step after orthonormalization
    #   history[1] has shape (N, 3), dtype float64  = Omega at each step

    # Prepend the initial state
    R_flat_history = jnp.concatenate([R0_flat[jnp.newaxis, :], history[0]], axis=0)    # (N+1, 9)
    Omega_history  = jnp.concatenate(
        [Omega_initial_sim_jax[jnp.newaxis, :], history[1]],
        axis=0
    )  # (N+1, 3)

    # Reshape R_flat_history into (N+1, 3, 3)
    sim_R_history_reshaped = jax.vmap(lambda r_flat: r_flat.reshape((3, 3)))(R_flat_history)

    # Return only up to num_total_states_output (which should be N+1), all dtype float64
    return sim_R_history_reshaped[:num_total_states_output], Omega_history[:num_total_states_output]


# --- Error Calculation (JAX compatible) ---
@jax.jit
def calculate_errors_jax(sim_R_hist_jax, sim_Omega_hist_jax, actual_R_hist_jax, actual_Omega_hist_jax):
    if sim_R_hist_jax.shape[0] <= 1 or actual_R_hist_jax.shape[0] <= 1:
        return jnp.array(jnp.nan, dtype=jnp.float64), jnp.array(jnp.nan, dtype=jnp.float64)

    sim_R_eff = sim_R_hist_jax[1:]
    act_R_eff = actual_R_hist_jax[1:]

    @jax.vmap
    def single_R_error(R_s, R_a):
        trace_val = jnp.trace(R_s.T @ R_a)
        angle = jnp.arccos(jnp.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
        return angle**2

    R_errors_sq = single_R_error(sim_R_eff, act_R_eff)
    mean_R_error_sq = jnp.mean(R_errors_sq)

    sim_Omega_eff = sim_Omega_hist_jax[1:]
    act_Omega_eff = actual_Omega_hist_jax[1:]
    Omega_errors_sq = jnp.sum((sim_Omega_eff - act_Omega_eff)**2, axis=1)
    mean_Omega_error_sq = jnp.mean(Omega_errors_sq)

    return mean_R_error_sq, mean_Omega_error_sq


# --- Visualization Function ---
def R_to_euler_angles_np(R_matrix_np):
    from scipy.spatial.transform import Rotation
    if not np.all(np.isfinite(R_matrix_np)):
        return np.array([np.nan, np.nan, np.nan])
    try:
        r = Rotation.from_matrix(R_matrix_np)
        return r.as_euler('xyz', degrees=False)
    except ValueError:
        return np.array([np.nan, np.nan, np.nan])


def plot_attitude_comparison(times_np,
                             R_desired_hist_np,
                             R_sim_hist_np,
                             R_px4_hist_np,
                             Omega_sim_hist_np,
                             Omega_px4_hist_np,
                             k_R_best,
                             k_Omega_best,
                             plot_title_prefix="Attitude Tracking"):
    euler_desired_np = np.array([R_to_euler_angles_np(R) for R in R_desired_hist_np])
    euler_sim_np = np.array([R_to_euler_angles_np(R) for R in R_sim_hist_np])
    euler_px4_np = np.array([R_to_euler_angles_np(R) for R in R_px4_hist_np])

    plot_title = (f"{plot_title_prefix}\n"
                  f"Best k_R=[{k_R_best[0]:.2f},{k_R_best[1]:.2f},{k_R_best[2]:.2f}], "
                  f"k_Omega=[{k_Omega_best[0]:.2f},{k_Omega_best[1]:.2f},{k_Omega_best[2]:.2f}]")

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.suptitle(plot_title, fontsize=14)
    angle_names = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    rate_names = ['Omega_x (Roll Rate)', 'Omega_y (Pitch Rate)', 'Omega_z (Yaw Rate)']

    for i in range(3):
        axs[0, i].plot(times_np, np.rad2deg(euler_desired_np[:, i]),
                       label='Desired (from u)', linestyle=':', color='gray')
        axs[0, i].plot(times_np, np.rad2deg(euler_sim_np[:, i]),
                       label='Simulated (Geom. Ctrl)', color='blue')
        axs[0, i].plot(times_np, np.rad2deg(euler_px4_np[:, i]),
                       label='Actual PX4', linestyle='--', color='red')
        axs[0, i].set_title(f'{angle_names[i]}')
        axs[0, i].set_ylabel('Angle (deg)')
        axs[0, i].legend()
        axs[0, i].grid(True)

        axs[1, i].plot(times_np, np.rad2deg(Omega_sim_hist_np[:, i]),
                       label='Simulated Omega', color='blue')
        axs[1, i].plot(times_np, np.rad2deg(Omega_px4_hist_np[:, i]),
                       label='Actual PX4 Omega', linestyle='--', color='red')
        axs[1, i].set_title(f'{rate_names[i]}')
        axs[1, i].set_xlabel('Time (s)')
        axs[1, i].set_ylabel('Rate (deg/s)')
        axs[1, i].legend()
        axs[1, i].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_filename = (f"attitude_comparison_{plot_title_prefix.replace(' ', '_').lower()}_"
                     f"kR_{k_R_best[0]:.1f}_{k_R_best[2]:.1f}_kO_{k_Omega_best[0]:.2f}.png")
    plt.savefig(plot_filename)
    print(f"Saved visualization to {plot_filename}")
    plt.show()
    plt.close(fig)


def main(args):
    print("Using JAX for simulation (Anisotropic kR, Isotropic kOmega).")
    print(f"JAX backend: {jax.default_backend()}")

    print("\nLoading dataset...")
    if not os.path.exists(args.dataset_path):
        print(f"ERROR: Dataset not found at {args.dataset_path}")
        return
    with open(args.dataset_path, 'rb') as file:
        raw_data_np = pickle.load(file)
    print(f"Dataset loaded. Contains {raw_data_np['q'].shape[0]} trajectories.")

    time_vector_np = np.array(raw_data_np['t'])
    if time_vector_np.ndim == 1 and time_vector_np.shape[0] > 1:
        dt_from_data = float(np.mean(np.diff(time_vector_np)))
        print(f"Using determined DT from dataset (common 1D time vector): {dt_from_data:.4f}")
    elif time_vector_np.ndim == 2 and time_vector_np.shape[0] > 0 and time_vector_np.shape[1] > 1:
        dt_from_data = float(np.mean(np.diff(time_vector_np[0, :])))
        print(f"Using determined DT from dataset (2D time vector, using first trajectory): {dt_from_data:.4f}")
        if time_vector_np.ndim == 2:
            print("Warning: 't' field in dataset is 2D. Assuming common DT from first trajectory, "
                  "and common time vector for plotting later. This might be incorrect if times differ per trajectory.")
            time_vector_np = time_vector_np[0, :]
    else:
        dt_from_data = 0.02
        print(f"Warning: Could not determine DT from dataset 't' shape {time_vector_np.shape}, using default: {dt_from_data}")
    DT = dt_from_data

    print("\nProcessing dataset arrays (batch conversion with JAX)...")
    if raw_data_np['u'].shape[-1] < 4:
        print(f"FATAL ERROR: raw_data_np['u'] last dimension is {raw_data_np['u'].shape[-1]}, expected at least 4 for quaternion.")
        return
    desired_quats_all_np = np.array(raw_data_np['u'][:, :, -4:])
    actual_quats_all_np = np.array(raw_data_np['quat'])
    actual_omegas_all_np = np.array(raw_data_np['omega'])

    common_time_vector_np = np.array(raw_data_np['t'])
    if common_time_vector_np.ndim == 2:
        common_time_vector_np = common_time_vector_np[0, :]

    desired_quats_all_jax_in = jax.device_put(desired_quats_all_np.astype(np.float64))
    actual_quats_all_jax_in = jax.device_put(actual_quats_all_np.astype(np.float64))

    vmapped_q2R_time = jax.vmap(q2R_util, in_axes=0)
    vmapped_q2R_traj_time = jax.vmap(vmapped_q2R_time, in_axes=0)

    print("Converting desired quaternions to rotation matrices (JAX vmap)...")
    desired_R_all_jax = vmapped_q2R_traj_time(desired_quats_all_jax_in)
    print("Converting actual quaternions to rotation matrices (JAX vmap)...")
    actual_R_all_jax = vmapped_q2R_traj_time(actual_quats_all_jax_in)

    desired_R_all_np = np.array(desired_R_all_jax)
    actual_R_all_np = np.array(actual_R_all_jax)

    actual_omegas_all_jax = jax.device_put(actual_omegas_all_np.astype(np.float64))
    print("Dataset processing and JAX conversion complete.")
    print("JIT compilation of simulation functions will occur on first use and may take a moment...")

    kR_xy_sweep_values_np = np.array(args.kR_xy_values, dtype=np.float64)
    kR_z_sweep_values_np = np.array(args.kR_z_values, dtype=np.float64)
    kOmega_isotropic_sweep_values_np = np.array(args.kOmega_values, dtype=np.float64)

    best_avg_combined_error = np.inf
    best_k_R_np = None
    best_k_Omega_np = None
    results_log = []

    num_trajectories_to_use = min(args.num_test_traj, desired_R_all_jax.shape[0])

    total_gain_combinations = (
        len(kR_xy_sweep_values_np)
        * len(kR_z_sweep_values_np)
        * len(kOmega_isotropic_sweep_values_np)
    )
    print(f"\nStarting JAX-based gain sweep: {total_gain_combinations} combinations "
          f"over {num_trajectories_to_use} trajectories each...")

    gain_iter = 0
    with tqdm(total=total_gain_combinations, desc="Gain Combinations") as pbar_gains:
        for kr_xy_val_np in kR_xy_sweep_values_np:
            for kr_z_val_np in kR_z_sweep_values_np:
                k_R_test_np = np.array([kr_xy_val_np, kr_xy_val_np, kr_z_val_np], dtype=np.float64)
                # k_R_test_np = np.array([2.9436266,  2.9684231,  0.41584948], dtype=np.float64)
                # k_R_test_np = np.array([1.5, 1.5, 1.5], dtype=np.float64)
                k_R_test_jax = jnp.array(k_R_test_np, dtype=jnp.float64)
                for kom_iso_val_np in kOmega_isotropic_sweep_values_np:
                    k_Omega_test_np = np.array([kom_iso_val_np, kom_iso_val_np, kom_iso_val_np], dtype=np.float64)
                    # k_Omega_test_np = np.array([0.15371324, 0.17776576, 0.15], dtype=np.float64)
                    # k_Omega_test_np = np.array([0.1, 0.1, 0.1], dtype=np.float64)
                    k_Omega_test_jax = jnp.array(k_Omega_test_np, dtype=jnp.float64)
                    gain_iter += 1
                    pbar_gains.set_description(
                        f"Gain Combo {gain_iter}/{total_gain_combinations} "
                        f"(kRxy={kr_xy_val_np:.1f},kRz={kr_z_val_np:.1f},kO={kom_iso_val_np:.2f})"
                    )

                    current_gains_avg_R_error_sq = 0.0
                    current_gains_avg_Omega_error_sq = 0.0
                    num_valid_sims_for_gains = 0

                    for traj_idx in range(num_trajectories_to_use):
                        if traj_idx == 0 and gain_iter == 1:
                            print("    (First JAX simulation for this trajectory length, "
                                  "JIT compilation may occur for simulate_trajectory_attitude_jax_single...)")

                        R_initial_sim_jax = actual_R_all_jax[traj_idx, 0].astype(jnp.float64)
                        Omega_initial_sim_jax = actual_omegas_all_jax[traj_idx, 0].astype(jnp.float64)

                        num_actual_states_avail = actual_R_all_jax[traj_idx].shape[0]
                        if num_actual_states_avail <= 1:
                            if args.verbose:
                                print(f"    Skipping traj {traj_idx}: not enough actual states "
                                      f"({num_actual_states_avail}) for simulation.")
                            continue

                        num_integration_steps = num_actual_states_avail - 1
                        if num_integration_steps > desired_R_all_jax[traj_idx].shape[0]:
                            if args.verbose:
                                print(
                                    f"    Skipping traj {traj_idx}: not enough desired inputs "
                                    f"({desired_R_all_jax[traj_idx].shape[0]}) "
                                    f"to perform {num_integration_steps} integration steps."
                                )
                            continue

                        current_desired_R_input_traj_jax = desired_R_all_jax[traj_idx, :num_integration_steps, :, :].astype(jnp.float64)
                        num_total_states_output_sim = num_integration_steps + 1

                        current_actual_R_hist_for_comp_jax = actual_R_all_jax[traj_idx, :num_total_states_output_sim].astype(jnp.float64)
                        current_actual_Omega_hist_for_comp_jax = actual_omegas_all_jax[traj_idx, :num_total_states_output_sim].astype(jnp.float64)

                        try:
                            sim_R_hist_jax, sim_Omega_hist_jax = simulate_trajectory_attitude_jax_single(
                                R_initial_sim_jax,
                                Omega_initial_sim_jax,
                                current_desired_R_input_traj_jax,
                                k_R_test_jax,
                                k_Omega_test_jax,
                                num_total_states_output_sim,
                                DT,
                                INERTIA_MATRIX_J_JAX,
                                INERTIA_MATRIX_J_INV_JAX,
                            )

                            mean_R_err_sq, mean_Omega_err_sq = calculate_errors_jax(
                                sim_R_hist_jax,
                                sim_Omega_hist_jax,
                                current_actual_R_hist_for_comp_jax,
                                current_actual_Omega_hist_for_comp_jax,
                            )

                            mean_R_err_sq_py = float(mean_R_err_sq)
                            mean_Omega_err_sq_py = float(mean_Omega_err_sq)

                            if args.verbose:
                                print(f"      Traj {traj_idx}: Raw errors R_sq={mean_R_err_sq_py:.4e}, "
                                      f"Omega_sq={mean_Omega_err_sq_py:.4e}")

                            if not (
                                np.isinf(mean_R_err_sq_py)
                                or np.isinf(mean_Omega_err_sq_py)
                                or np.isnan(mean_R_err_sq_py)
                                or np.isnan(mean_Omega_err_sq_py)
                            ):
                                current_gains_avg_R_error_sq += mean_R_err_sq_py
                                current_gains_avg_Omega_error_sq += mean_Omega_err_sq_py
                                num_valid_sims_for_gains += 1
                            elif args.verbose:
                                print(f"    Warning: Sim resulted in NaN/Inf for traj {traj_idx} "
                                      f"with k_R={k_R_test_np}, k_Omega={k_Omega_test_np}")
                        except Exception:
                            print(f"    ERROR during JAX simulation for traj {traj_idx} with "
                                  f"k_R={k_R_test_np}, k_Omega={k_Omega_test_np}:")
                            traceback.print_exc()
                            if args.verbose:
                                pass

                    if num_valid_sims_for_gains > 0:
                        avg_R_err_sq_final = current_gains_avg_R_error_sq / num_valid_sims_for_gains
                        avg_Omega_err_sq_final = current_gains_avg_Omega_error_sq / num_valid_sims_for_gains
                        combined_error = avg_R_err_sq_final + args.omega_error_weight * avg_Omega_err_sq_final

                        if args.verbose or (combined_error < best_avg_combined_error * 0.95) or (
                            combined_error < 1e-3 and best_avg_combined_error > 1e-3
                        ):
                            if not args.verbose and pbar_gains.n > 0:
                                print()
                            print(
                                f"    k_R={k_R_test_np}, k_O={k_Omega_test_np} -> "
                                f"R_err_sq={avg_R_err_sq_final:.2e}, O_err_sq={avg_Omega_err_sq_final:.2e}, "
                                f"Comb={combined_error:.2e}"
                            )

                        results_log.append({
                            'k_R': k_R_test_np.tolist(),
                            'k_Omega': k_Omega_test_np.tolist(),
                            'avg_R_error_sq': avg_R_err_sq_final,
                            'avg_Omega_error_sq': avg_Omega_err_sq_final,
                            'combined_error': combined_error
                        })

                        if combined_error < best_avg_combined_error:
                            best_avg_combined_error = combined_error
                            best_k_R_np = k_R_test_np
                            best_k_Omega_np = k_Omega_test_np
                            if not args.verbose and pbar_gains.n > 0:
                                print()
                            print(
                                f"    NEW BEST! k_R={best_k_R_np}, k_O={best_k_Omega_np}, "
                                f"CombErr={best_avg_combined_error:.3e}"
                            )
                    else:
                        if args.verbose:
                            print(f"    No valid simulations for k_R={k_R_test_np}, k_Omega={k_Omega_test_np}")
                        results_log.append({
                            'k_R': k_R_test_np.tolist(),
                            'k_Omega': k_Omega_test_np.tolist(),
                            'avg_R_error_sq': np.nan,
                            'avg_Omega_error_sq': np.nan,
                            'combined_error': np.nan
                        })
                    pbar_gains.update(1)

    print("\n--- Tuning Complete (JAX Anisotropic kR, Isotropic kOmega) ---")
    if best_k_R_np is not None and best_k_Omega_np is not None:
        print(f"Best k_R (anisotropic): {best_k_R_np}")
        print(f"Best k_Omega (isotropic): {best_k_Omega_np}")
        print(f"Best average combined error (squared): {best_avg_combined_error:.4e}")

        if args.visualize and num_trajectories_to_use > 0:
            print("\nGenerating visualization for the first test trajectory with best gains...")
            traj_idx_for_viz = 0

            # Use the actual-R length to determine how many integration steps we can do
            actual_len = actual_R_all_jax[traj_idx_for_viz].shape[0]
            num_integration_steps_viz = actual_len - 1

            if num_integration_steps_viz <= 0:
                print(f"Cannot visualize traj {traj_idx_for_viz}: not enough actual states ({actual_len}).")
            else:
                # Make sure we have at least num_integration_steps_viz desired-R inputs
                desired_available = desired_R_all_jax[traj_idx_for_viz].shape[0]
                if num_integration_steps_viz > desired_available:
                    print(f"Cannot visualize traj {traj_idx_for_viz}: only {desired_available} desired inputs, "
                          f"but need {num_integration_steps_viz}.")
                else:
                    num_total_states_viz = actual_len
                    desired_R_input_traj_viz_jax = desired_R_all_jax[
                        traj_idx_for_viz, :num_integration_steps_viz, :, :
                    ].astype(jnp.float64)

                    sim_R_hist_viz_jax, sim_Omega_hist_viz_jax = simulate_trajectory_attitude_jax_single(
                        actual_R_all_jax[traj_idx_for_viz, 0].astype(jnp.float64),
                        actual_omegas_all_jax[traj_idx_for_viz, 0].astype(jnp.float64),
                        desired_R_input_traj_viz_jax,
                        jnp.array(best_k_R_np, dtype=jnp.float64),
                        jnp.array(best_k_Omega_np, dtype=jnp.float64),
                        num_total_states_viz,
                        DT,
                        INERTIA_MATRIX_J_JAX,
                        INERTIA_MATRIX_J_INV_JAX,
                    )

                    times_viz_np = common_time_vector_np[:num_total_states_viz]

                    plot_desired_R_viz_np = np.zeros((num_total_states_viz, 3, 3), dtype=np.float64)
                    plot_desired_R_viz_np[:-1] = desired_R_all_np[
                        traj_idx_for_viz, :num_integration_steps_viz
                    ]
                    if num_integration_steps_viz > 0:
                        plot_desired_R_viz_np[-1] = plot_desired_R_viz_np[-2]
                    elif num_total_states_viz == 1:
                        plot_desired_R_viz_np[0] = np.array(actual_R_all_np[traj_idx_for_viz, 0], dtype=np.float64)

                    sim_R_hist_viz_np = np.array(sim_R_hist_viz_jax, dtype=np.float64)
                    sim_Omega_hist_viz_np = np.array(sim_Omega_hist_viz_jax, dtype=np.float64)
                    actual_R_hist_viz_np = np.array(actual_R_all_np[traj_idx_for_viz, :num_total_states_viz], dtype=np.float64)
                    actual_Omega_hist_viz_np = np.array(actual_omegas_all_np[traj_idx_for_viz, :num_total_states_viz], dtype=np.float64)

                    if (
                        times_viz_np.shape[0] == plot_desired_R_viz_np.shape[0]
                        == sim_R_hist_viz_np.shape[0]
                        == actual_R_hist_viz_np.shape[0]
                        and sim_R_hist_viz_np.shape[0] > 0
                    ):
                        plot_attitude_comparison(
                            times_viz_np,
                            plot_desired_R_viz_np,
                            sim_R_hist_viz_np,
                            actual_R_hist_viz_np,
                            sim_Omega_hist_viz_np,
                            actual_Omega_hist_viz_np,
                            best_k_R_np, best_k_Omega_np,
                            plot_title_prefix=f"Attitude_Traj{traj_idx_for_viz}"
                        )
                    else:
                        print("Could not visualize due to shape mismatch or empty data for plotting.")
                        print(
                            f"Shapes - Times: {times_viz_np.shape}, DesiredR: {plot_desired_R_viz_np.shape}, "
                            f"SimR: {sim_R_hist_viz_np.shape}, ActualR: {actual_R_hist_viz_np.shape}"
                        )
    else:
        print("No suitable gains found or no valid simulations.")

    if results_log:
        valid_results = [r for r in results_log if not np.isnan(r['combined_error'])]
        sorted_results = sorted(valid_results, key=lambda x: x['combined_error'])

        print("\nTop 5 results (lower combined error is better):")
        for i, res in enumerate(sorted_results[:5]):
            print(
                f"  {i+1}. k_R: {res['k_R']}, k_Omega: {res['k_Omega']}, "
                f"CombErrSq: {res['combined_error']:.3e} "
                f"(R_sq: {res['avg_R_error_sq']:.2e}, "
                f"Omega_sq: {res['avg_Omega_error_sq']:.2e})"
            )

        output_filename = args.output_file
        with open(output_filename, 'wb') as f:
            pickle.dump(results_log, f)
        print(f"\nFull tuning results saved to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tune attitude controller gains (anisotropic k_R, isotropic k_Omega) "
                    "using JAX and PX4 log data."
    )
    parser.add_argument('--dataset_path', type=str, default='data/batch_trajectory_dataset.pkl')
    parser.add_argument('--num_test_traj', type=int, default=10)
    parser.add_argument('--kR_xy_values', nargs='+', type=float, default=[1.6])
    parser.add_argument('--kR_z_values', nargs='+', type=float, default=[0.4])
    parser.add_argument('--kOmega_values', nargs='+', type=float, default=[0.24])
    parser.add_argument('--omega_error_weight', type=float, default=0.1)
    parser.add_argument('--output_file',
                        type=str,
                        default='att_gain_tune_aniso_kr_iso_komega.pkl')
    parser.add_argument('--visualize',
                        action='store_true',
                        help="Plot results for the best gains on one trajectory.")
    parser.add_argument('--verbose', action='store_true', help="Print more detailed progress and warnings.")

    cli_args = parser.parse_args()
    main(cli_args)

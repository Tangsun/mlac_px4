#!/usr/bin/env python3
"""
Step-wise open-loop comparison script.

The existing `openloop_comparison_np.py` integrates the entire control
sequence forward using NumPy dynamics. This variant performs a sanity
check by re-initialising the simulator at every command sample using the
measured state from the rosbag, advancing the state by a single command
interval, and comparing the predicted state to the subsequent measured
state. This prevents error accumulation across the full trajectory and
helps isolate per-step model discrepancies.
"""

import argparse
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt

# Ensure we can import sibling modules as well as mlac_jax_train utilities.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in (SCRIPT_DIR, PROJECT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# Re-use the ROS bag extraction utilities and dynamics helpers.
from openloop_comparison_np import (  # pylint: disable=wrong-import-position
    extract_open_loop_data_att_only_same_timing,
    rk4_step_numpy,
    simulation_ode_euler,
)


def resample_states_to_command_timeline(gazebo_states, commanded_inputs):
    """
    Interpolate pose, velocity, and orientation measurements onto the command
    timeline so that each control input has an aligned measured state.
    """
    t_pose, q_pose, quat_pose, t_vel, vel_body = gazebo_states
    t_cmd, _, _ = commanded_inputs

    if t_cmd.size < 2:
        raise RuntimeError("Commanded inputs must contain at least two samples.")
    if t_pose.size < 2 or q_pose.shape[0] < 2 or quat_pose.shape[0] < 2:
        raise RuntimeError("Pose/attitude data is insufficient for interpolation.")

    t_pose_min = float(np.min(t_pose))
    t_pose_max = float(np.max(t_pose))
    if t_pose_max <= t_pose_min:
        raise RuntimeError("Pose timeline must span a non-zero interval.")

    if np.any((t_cmd < t_pose_min) | (t_cmd > t_pose_max)):
        print(
            "Warning: command timeline extends beyond pose measurements; "
            "states will be clamped to the available window."
        )

    t_cmd_clamped = np.clip(t_cmd, t_pose_min, t_pose_max)

    # Interpolate position measurements to the command timeline.
    pos_on_cmd = np.column_stack([
        np.interp(t_cmd, t_pose, q_pose[:, idx]) for idx in range(3)
    ])

    # Interpolate orientation using a Slerp to avoid quaternion drift.
    rotations = Rotation.from_quat(quat_pose)
    slerp_pose = Slerp(t_pose, rotations)
    rot_on_cmd = slerp_pose(t_cmd_clamped)
    # 'zyx' returns yaw, pitch, roll -> reorder to roll, pitch, yaw.
    rpy_on_cmd = rot_on_cmd.as_euler('zyx', degrees=False)[:, ::-1]

    # Convert body-frame velocity measurements to the world frame before
    # interpolating. Fall back to finite-difference of position if velocity
    # measurements are unavailable.
    if t_vel.size >= 2 and vel_body.shape[0] >= 2:
        t_vel_clamped = np.clip(t_vel, t_pose_min, t_pose_max)
        rot_at_vel = slerp_pose(t_vel_clamped)
        vel_world_samples = rot_at_vel.apply(vel_body)
        vel_on_cmd = np.column_stack([
            np.interp(t_cmd, t_vel, vel_world_samples[:, idx]) for idx in range(3)
        ])
    else:
        # Use central differences to approximate velocity in absence of data.
        vel_on_cmd = np.gradient(pos_on_cmd, t_cmd, axis=0, edge_order=2)

    if (
        np.isnan(pos_on_cmd).any()
        or np.isnan(rpy_on_cmd).any()
        or np.isnan(vel_on_cmd).any()
    ):
        raise RuntimeError(
            "Interpolation produced NaNs; ensure the rosbag window overlaps "
            "with the command timeline."
        )

    measured_states = np.column_stack([pos_on_cmd, vel_on_cmd, rpy_on_cmd])
    return measured_states


def stepwise_forward_simulation(measured_states, commanded_inputs, mass):
    """
    Perform a single RK4 integration for each command sample using the measured
    state as the initial condition for that interval. Returns predicted states
    (aligned with t_{k+1}) and the per-step state deltas.
    """
    t_cmd, thrust_cmd, w_cmd = commanded_inputs
    predicted_states, errors, step_meta = [], [], []

    for i in range(len(t_cmd) - 1):
        dt = t_cmd[i + 1] - t_cmd[i]
        if dt <= 0:
            continue

        measured_now = measured_states[i]
        measured_next = measured_states[i + 1]
        held_commands = (thrust_cmd[i], w_cmd[i])

        predicted_next, _ = rk4_step_numpy(
            simulation_ode_euler,
            t_cmd[i],
            measured_now,
            dt,
            held_commands,
            mass,
        )

        predicted_states.append(predicted_next)
        errors.append(predicted_next - measured_next)
        step_meta.append((t_cmd[i], dt))

    return np.array(predicted_states), np.array(errors), np.array(step_meta)


def report_error_statistics(errors):
    """
    Print basic RMS and max metrics for translation, velocity, and attitude.
    """
    if errors.size == 0:
        print("No step-wise errors were computed (insufficient data).")
        return

    pos_err = np.linalg.norm(errors[:, 0:3], axis=1)
    vel_err = np.linalg.norm(errors[:, 3:6], axis=1)
    rpy_err = np.linalg.norm(errors[:, 6:9], axis=1)

    def stats(label, arr):
        print(
            f"{label}: RMS={np.sqrt(np.mean(arr ** 2)):.4f}, "
            f"Mean={np.mean(arr):.4f}, Max={np.max(arr):.4f}"
        )

    stats("Position error (m)", pos_err)
    stats("Velocity error (m/s)", vel_err)
    stats("Attitude error (rad)", rpy_err)


def plot_stepwise_errors(step_meta, errors, output_dir=None, show_plot=True):
    """
    Plot per-axis step-wise errors for position, velocity, and attitude.
    """
    if errors.size == 0:
        print("Skipping plots: no errors available.")
        return

    times = step_meta[:, 0] + step_meta[:, 1]
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    field_slices = [
        (slice(0, 3), ["Δx", "Δy", "Δz"], "Position error (m)"),
        (slice(3, 6), ["Δvx", "Δvy", "Δvz"], "Velocity error (m/s)"),
        (slice(6, 9), ["Δroll", "Δpitch", "Δyaw"], "Attitude error (rad)"),
    ]

    for ax, (slc, labels, ylabel) in zip(axs, field_slices):
        for idx, label in enumerate(labels):
            ax.plot(times, errors[:, slc][..., idx], label=label)
        ax.plot(
            times,
            np.linalg.norm(errors[:, slc], axis=1),
            'k--',
            label='Magnitude',
        )
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle("Step-wise state errors")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, "stepwise_errors.png")
        fig.savefig(fig_path)
        print(f"Saved step-wise error plot to {fig_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step-wise open-loop comparison. Each interval is simulated from "
            "the measured state using the recorded control input."
        )
    )
    parser.add_argument('--rosbag', type=str, required=True,
                        help="Path to the rosbag directory.")
    parser.add_argument('--mass', type=float, default=2.0,
                        help="Vehicle mass (kg).")
    parser.add_argument('--pose_topic', type=str,
                        default="/mavros/local_position/pose")
    parser.add_argument('--velocity_topic', type=str,
                        default="/mavros/local_position/velocity_body")
    parser.add_argument('--control_log_topic', type=str,
                        default="/mlac_mission_node/control_log")
    parser.add_argument('--attitude_setpoint_topic', type=str,
                        default="/mavros/setpoint_raw/attitude")
    parser.add_argument('--print_steps', type=int, default=5,
                        help="Number of individual step errors to print.")
    parser.add_argument('--plot', action='store_true',
                        help="Display/save per-axis step-wise error plots.")
    parser.add_argument('--plot-dir', type=str, default=None,
                        help="Directory to save plots (implies --plot).")

    args = parser.parse_args()

    print("Extracting rosbag data...")
    gazebo_states, commanded_inputs, _, _ = \
        extract_open_loop_data_att_only_same_timing(
            args.rosbag,
            args.pose_topic,
            args.velocity_topic,
            args.control_log_topic,
            args.attitude_setpoint_topic,
        )

    if gazebo_states[0] is None or gazebo_states[0].size == 0:
        print("Failed to extract pose timeline; aborting.")
        return

    print("Interpolating measured states to the command timeline...")
    measured_states = resample_states_to_command_timeline(
        gazebo_states,
        commanded_inputs,
    )

    print("Running step-wise simulations...")
    predicted_states, errors, step_meta = stepwise_forward_simulation(
        measured_states,
        commanded_inputs,
        args.mass,
    )

    report_error_statistics(errors)

    if args.plot or args.plot_dir:
        plot_stepwise_errors(
            step_meta,
            errors,
            output_dir=args.plot_dir,
            show_plot=args.plot,
        )

    # Optionally print the first few step-wise error entries for manual inspection.
    num_samples = min(args.print_steps, errors.shape[0])
    print(f"\nFirst {num_samples} step comparisons:")
    for i in range(num_samples):
        start_time, dt = step_meta[i]
        pos_err = np.linalg.norm(errors[i, 0:3])
        vel_err = np.linalg.norm(errors[i, 3:6])
        rpy_err = np.linalg.norm(errors[i, 6:9])
        print(
            f"  Step {i:03d}: t={start_time:.3f}s, dt={dt:.3f}s, "
            f"|Δp|={pos_err:.4f} m, |Δv|={vel_err:.4f} m/s, |Δrpy|={rpy_err:.4f} rad"
        )

    # Save arrays for offline analysis.
    output_npz = os.path.join(
        os.path.dirname(args.rosbag.rstrip('/')),
        "stepwise_comparison_results.npz",
    )
    np.savez(
        output_npz,
        t_cmd=commanded_inputs[0],
        measured_states=measured_states,
        predicted_states=predicted_states,
        errors=errors,
        step_meta=step_meta,
    )
    print(f"\nSaved raw arrays to {output_npz}")


if __name__ == "__main__":
    main()

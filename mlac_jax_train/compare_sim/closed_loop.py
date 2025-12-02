#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from rosbag_utils import (
    extract_attitude_data,
    resample_state_to_times,
    plot_pose_vs_reference,
    plot_command_streams,
)  # noqa: E402
from window_utils import generate_time_windows  # noqa: E402
from closed_loop_core import run_smc_window  # noqa: E402


def load_reference(traj_file, recorded_duration, feedforward):
    traj = np.load(traj_file)
    ts = traj[:, 0] - traj[0, 0]
    r = traj[:, 1:4]
    dr = traj[:, 4:7]
    yaw = traj[:, 7]
    ddr = traj[:, 8:11]

    mask = ts <= recorded_duration + 1e-6
    if np.count_nonzero(mask) < 2:
        raise RuntimeError("Reference trajectory shorter than recorded window.")

    ts = ts[mask]
    r = r[mask]
    dr = dr[mask]
    ddr = ddr[mask]
    yaw = yaw[mask]

    if feedforward:
        yaw_rate = np.gradient(np.unwrap(yaw), ts)
    else:
        yaw_rate = np.zeros_like(ts)

    return ts, r, dr, ddr, yaw, yaw_rate


def plot_closed_loop_trajectories(
        ts_axis,
        measured_states,
        jax_pos,
        jax_vel,
        jax_euler_deg,
        jax_omega,
        reference_data,
        attitude_cmd_data,
        bodyrate_cmd_data,
        vel_data,
        output_dir,
):
    """
    Generate comparison plots for position, velocity, attitude, and angular velocity.
    """
    os.makedirs(output_dir, exist_ok=True)

    ts_axis = np.asarray(ts_axis)
    measured_states = np.asarray(measured_states)
    jax_pos = np.asarray(jax_pos)
    jax_vel = np.asarray(jax_vel)
    jax_euler_deg = np.asarray(jax_euler_deg)
    jax_omega = np.asarray(jax_omega)
    t_ref, ref_pos, ref_vel, _, ref_yaw, _ = reference_data

    min_len = min(
        ts_axis.shape[0],
        measured_states.shape[0],
        jax_pos.shape[0],
        ref_pos.shape[0],
    )
    ts_axis = ts_axis[:min_len]
    measured_states = measured_states[:min_len]
    jax_pos = jax_pos[:min_len]
    jax_vel = jax_vel[:min_len]
    jax_euler_deg = jax_euler_deg[:min_len]
    jax_omega = jax_omega[:min_len]
    ref_pos = ref_pos[:min_len]
    ref_vel = ref_vel[:min_len]

    measured_pos = measured_states[:, 0:3]
    measured_vel = measured_states[:, 3:6]
    measured_rpy_deg = np.rad2deg(measured_states[:, 6:9])

    def interp_series(times_src, data_src, default):
        if data_src.size == 0 or len(times_src) == 0:
            return np.full((ts_axis.shape[0], default), np.nan)
        return np.column_stack([
            np.interp(ts_axis, times_src, data_src[:, i]) for i in range(data_src.shape[1])
        ])

    t_att, euler_cmd = attitude_cmd_data
    att_cmd_deg = interp_series(t_att, np.rad2deg(euler_cmd) if euler_cmd.size else np.empty((0, 3)), 3)

    t_body, _, w_cmd = bodyrate_cmd_data
    w_cmd_deg = interp_series(t_body, np.rad2deg(w_cmd) if w_cmd.size else np.empty((0, 3)), 3)

    t_ang_meas, _, ang_body_meas = vel_data
    ang_meas_deg = interp_series(
        t_ang_meas,
        np.rad2deg(ang_body_meas) if ang_body_meas.size else np.empty((0, 3)),
        3,
    )

    # Position plot
    fig_pos, axs_pos = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs_pos[i].plot(ts_axis, measured_pos[:, i], label='ROS Measured')
        axs_pos[i].plot(ts_axis, jax_pos[:, i], label='JAX Sim')
        axs_pos[i].plot(ts_axis, ref_pos[:, i], '--', label='Ref Cmd')
        axs_pos[i].set_ylabel(f'{labels[i]} (m)')
        axs_pos[i].grid(True)
    axs_pos[2].set_xlabel('Time (s)')
    axs_pos[0].legend(loc='upper right')
    fig_pos.suptitle('Position Comparison')
    fig_pos.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_pos.savefig(os.path.join(output_dir, "comparison_positions.png"))
    plt.close(fig_pos)

    # Velocity plot
    fig_vel, axs_vel = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for i in range(3):
        axs_vel[i].plot(ts_axis, measured_vel[:, i], label='ROS Measured')
        axs_vel[i].plot(ts_axis, jax_vel[:, i], label='JAX Sim')
        axs_vel[i].plot(ts_axis, ref_vel[:, i], '--', label='Ref Cmd')
        axs_vel[i].set_ylabel(f'{labels[i]} dot (m/s)')
        axs_vel[i].grid(True)
    axs_vel[2].set_xlabel('Time (s)')
    axs_vel[0].legend(loc='upper right')
    fig_vel.suptitle('Velocity Comparison')
    fig_vel.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_vel.savefig(os.path.join(output_dir, "comparison_velocities.png"))
    plt.close(fig_vel)

    # Attitude plot
    fig_att, axs_att = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    att_labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        axs_att[i].plot(ts_axis, measured_rpy_deg[:, i], label='ROS Measured')
        axs_att[i].plot(ts_axis, jax_euler_deg[:, i], label='JAX Sim')
        axs_att[i].plot(ts_axis, att_cmd_deg[:, i], '--', label='Command')
        axs_att[i].set_ylabel(f'{att_labels[i]} (deg)')
        axs_att[i].grid(True)
    axs_att[2].set_xlabel('Time (s)')
    axs_att[0].legend(loc='upper right')
    fig_att.suptitle('Attitude Comparison')
    fig_att.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_att.savefig(os.path.join(output_dir, "comparison_attitude.png"))
    plt.close(fig_att)

    # Angular velocity plot
    fig_rate, axs_rate = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    rate_labels = ['p', 'q', 'r']
    for i in range(3):
        axs_rate[i].plot(ts_axis, ang_meas_deg[:, i], label='ROS body rate')
        axs_rate[i].plot(ts_axis, np.rad2deg(jax_omega[:, i]), label='JAX body rate')
        axs_rate[i].plot(ts_axis, w_cmd_deg[:, i], '--', label='Command')
        axs_rate[i].set_ylabel(f'{rate_labels[i]} (deg/s)')
        axs_rate[i].grid(True)
    axs_rate[2].set_xlabel('Time (s)')
    axs_rate[0].legend(loc='upper right')
    fig_rate.suptitle('Angular Velocity Comparison')
    fig_rate.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_rate.savefig(os.path.join(output_dir, "comparison_angular_rates.png"))
    plt.close(fig_rate)


def main():
    parser = argparse.ArgumentParser(description="Rolling-window closed-loop comparison (SMC).")
    parser.add_argument("--rosbag", required=True)
    parser.add_argument("--traj-file", help="Reference .npy trajectory (used when --reference-source=traj).")
    parser.add_argument("--window-duration", type=float, default=2.0)
    parser.add_argument("--window-step", type=float, default=None)
    parser.add_argument("--pose-topic", type=str, default="/mavros/local_position/pose")
    parser.add_argument("--velocity-topic", type=str, default="/mavros/local_position/velocity_body")
    parser.add_argument("--control-log-topic", type=str, default="/mlac_mission_node/control_log")
    parser.add_argument("--attitude-setpoint-topic", type=str, default="/mavros/setpoint_raw/attitude")
    parser.add_argument("--kR", type=float, nargs=3, default=[0.3, 0.3, 0.3])
    parser.add_argument("--Kdiag", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    parser.add_argument("--Ldiag", type=float, nargs=3, default=[0.25, 0.25, 0.25])
    parser.add_argument("--attitude-time-constant", type=float, default=0.02)
    parser.add_argument("--feedforward", action="store_true")
    parser.add_argument("--reference-source", choices=["bag", "traj"], default="bag",
                        help="Use recorded ControllerLog references (bag) or the .npy trajectory (traj).")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="If set, saves pose/reference and command plots into this directory.")
    args = parser.parse_args()

    bag_data = extract_attitude_data(
        args.rosbag,
        pose_topic=args.pose_topic,
        velocity_topic=args.velocity_topic,
        control_log_topic=args.control_log_topic,
        att_setpoint_topic=args.attitude_setpoint_topic,
    )

    pose_data = bag_data["pose"]
    vel_data = bag_data["velocity"]

    reference_tuple = bag_data.get("reference")
    if args.reference_source == "bag":
        reference_tuple = bag_data.get("reference")
        if not reference_tuple or reference_tuple[0].size < 2:
            raise RuntimeError("Bag does not contain sufficient ControllerLog reference data.")
        ts_ref, r_ref, dr_ref, ddr_ref, yaw_ref, yaw_rate_ref = reference_tuple
    else:
        if not args.traj_file:
            raise RuntimeError("Must provide --traj-file when --reference-source=traj.")
        recorded_duration = pose_data[0][-1]
        ts_ref, r_ref, dr_ref, ddr_ref, yaw_ref, yaw_rate_ref = load_reference(
            args.traj_file, recorded_duration, args.feedforward
        )

    measured_on_ref = resample_state_to_times(ts_ref, pose_data, vel_data)

    if args.plot_dir and reference_tuple and reference_tuple[0].size > 0:
        os.makedirs(args.plot_dir, exist_ok=True)
        plot_pose_vs_reference(
            pose_data,
            reference_tuple,
            os.path.join(args.plot_dir, "pose_vs_reference.png"),
        )
        plot_command_streams(
            bag_data["attitude_cmd"],
            bag_data["bodyrate_cmd"],
            os.path.join(args.plot_dir, "command_streams.png"),
        )

    gains = (
        np.asarray(args.kR),
        np.diag(np.asarray(args.Kdiag)),
        np.diag(np.asarray(args.Ldiag)),
    )

    total_samples = measured_on_ref.shape[0]
    pred_pos = np.zeros((total_samples, 3))
    pred_vel = np.zeros((total_samples, 3))
    pred_euler_deg = np.zeros((total_samples, 3))
    pred_omega = np.zeros((total_samples, 3))
    pred_valid = np.zeros(total_samples, dtype=bool)

    window_step = args.window_step if args.window_step is not None else args.window_duration
    results = []

    for w_idx, (start_t, end_t, start_i, end_i) in enumerate(
        generate_time_windows(ts_ref, args.window_duration, window_step)
    ):
        if args.max_windows is not None and w_idx >= args.max_windows:
            break

        ref_slice = (
            ts_ref[start_i:end_i],
            r_ref[start_i:end_i],
            dr_ref[start_i:end_i],
            ddr_ref[start_i:end_i],
            yaw_ref[start_i:end_i],
            yaw_rate_ref[start_i:end_i],
        )
        initial_state = measured_on_ref[start_i].copy()
        ts_sim, pos_sim, vel_sim, euler_sim_deg, omega_sim = run_smc_window(
            ref_slice,
            initial_state,
            gains,
            attitude_time_constant=args.attitude_time_constant,
        )
        measured_slice = measured_on_ref[start_i:end_i]

        if pos_sim.shape[0] != measured_slice.shape[0]:
            min_len = min(pos_sim.shape[0], measured_slice.shape[0])
            pos_sim = pos_sim[:min_len]
            vel_sim = vel_sim[:min_len]
            euler_sim_deg = euler_sim_deg[:min_len]
            omega_sim = omega_sim[:min_len]
            ts_sim = ts_sim[:min_len]
            measured_slice = measured_slice[:min_len]
            end_i = start_i + min_len

        pos_err = np.linalg.norm(pos_sim - measured_slice[:, 0:3], axis=1)
        vel_err = np.linalg.norm(vel_sim - measured_slice[:, 3:6], axis=1)
        att_err = np.linalg.norm(
            np.deg2rad(euler_sim_deg) - measured_slice[:, 6:9],
            axis=1,
        )

        results.append({
            "window_index": w_idx,
            "start_time": float(start_t),
            "end_time": float(end_t),
            "pos_rms": float(np.sqrt(np.mean(pos_err**2))),
            "vel_rms": float(np.sqrt(np.mean(vel_err**2))),
            "att_rms": float(np.sqrt(np.mean(att_err**2))),
        })

        pred_pos[start_i:end_i] = pos_sim
        pred_vel[start_i:end_i] = vel_sim
        pred_euler_deg[start_i:end_i] = euler_sim_deg
        pred_omega[start_i:end_i] = omega_sim
        pred_valid[start_i:end_i] = True

        print(
            f"[Window {w_idx:03d}] t=[{start_t:.2f}, {end_t:.2f}] "
            f"pos_rms={results[-1]['pos_rms']:.3f} m, "
            f"vel_rms={results[-1]['vel_rms']:.3f} m/s, "
            f"att_rms={results[-1]['att_rms']:.3f} rad"
        )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, results=results)
        print(f"Saved window metrics to {args.output}")

    if args.plot_dir and np.any(pred_valid):
        mask = pred_valid
        plot_closed_loop_trajectories(
            ts_ref[mask],
            measured_on_ref[mask],
            pred_pos[mask],
            pred_vel[mask],
            pred_euler_deg[mask],
            pred_omega[mask],
            reference_tuple,
            bag_data["attitude_cmd"],
            bag_data["bodyrate_cmd"],
            bag_data["velocity"],
            args.plot_dir,
        )

if __name__ == "__main__":
    main()

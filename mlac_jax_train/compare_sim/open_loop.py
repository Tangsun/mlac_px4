#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

BODYRATE_SIM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "bodyrate_sim_train"))
if BODYRATE_SIM_DIR not in sys.path:
    sys.path.append(BODYRATE_SIM_DIR)

from rosbag_utils import resample_state_to_times, extract_open_loop_measured  # noqa: E402
from window_utils import generate_time_windows  # noqa: E402
from dynamics_numpy import simulation_ode_euler, simulation_ode_euler_fixed_attitude, rk4_step  # noqa: E402
from openloop_comparison_np import extract_open_loop_data_att_only_same_timing  # noqa: E402


def simulate_window(t_cmd, thrust, body_rates, start_idx, end_idx,
                    initial_state, mass, dynamics_fn=simulation_ode_euler,
                    measured_rpy=None):
    """
    Run an open-loop forward simulation over [start_idx, end_idx).

    When measured_rpy is provided (Mode B), the RPY portion of the state is
    overwritten from rosbag data after each integration step so that only
    translational dynamics are tested.
    """
    state = initial_state.copy()
    history = [state.copy()]
    for i in range(start_idx, end_idx - 1):
        dt = t_cmd[i + 1] - t_cmd[i]
        if dt <= 0:
            continue
        command = (thrust[i], body_rates[i])
        dynamics = lambda s, cmd=command: dynamics_fn(s, cmd, mass)
        state = rk4_step(dynamics, state, dt)
        if measured_rpy is not None:
            state[6:9] = measured_rpy[i + 1]
        history.append(state.copy())
    return np.asarray(history)


def main():
    parser = argparse.ArgumentParser(description="Rolling-window open-loop comparison.")
    parser.add_argument("--rosbag", required=True, help="Path to rosbag directory.")
    parser.add_argument("--window-duration", type=float, default=2.0,
                        help="Length of each simulation window (seconds).")
    parser.add_argument("--window-step", type=float, default=None,
                        help="Step between windows. Defaults to duration (no overlap).")
    parser.add_argument("--mass", type=float, default=2.0)
    parser.add_argument("--pose-topic", type=str, default="/mavros/local_position/pose")
    parser.add_argument("--velocity-topic", type=str, default="/mavros/local_position/velocity_body")
    parser.add_argument("--control-log-topic", type=str, default="/mlac_mission_node/control_log")
    parser.add_argument("--attitude-setpoint-topic", type=str, default="/mavros/setpoint_raw/attitude")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Optional cap on number of windows to evaluate.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional npz output file to store per-window results.")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="If set, saves comparison plots to this directory.")
    parser.add_argument("--rotation-mode", type=str, default="commanded",
                        choices=["commanded", "measured-rates", "measured-attitude"],
                        help="Source of rotational input: "
                             "'commanded' = setpoint body rates (default), "
                             "'measured-rates' = measured angular velocity from velocity_body, "
                             "'measured-attitude' = measured quaternion from pose (translational-only test).")
    args = parser.parse_args()

    rotation_mode = args.rotation_mode

    if rotation_mode == "commanded":
        gazebo_states, commanded_inputs, init_pose, init_vel_msg = \
            extract_open_loop_data_att_only_same_timing(
                args.rosbag, args.pose_topic, args.velocity_topic,
                args.control_log_topic, args.attitude_setpoint_topic
            )
        if gazebo_states[0] is None or gazebo_states[0].size == 0:
            raise RuntimeError("No pose data extracted from rosbag.")
        t_cmd, thrust_cmd, w_cmd = commanded_inputs
        t_pose, q_pose, quat_pose, t_vel, vel_body = gazebo_states
        pose_data = (t_pose, q_pose, quat_pose)
        velocity_data = (t_vel, vel_body, np.zeros_like(vel_body))
        dynamics_fn = simulation_ode_euler
    else:
        mdata = extract_open_loop_measured(
            args.rosbag,
            pose_topic=args.pose_topic,
            velocity_topic=args.velocity_topic,
            control_log_topic=args.control_log_topic,
            att_setpoint_topic=args.attitude_setpoint_topic,
        )
        t_cmd = mdata["t"]
        thrust_cmd = mdata["thrust"]
        w_cmd = mdata["ang_vel"]
        pose_data = mdata["pose"]
        velocity_data = mdata["velocity"]
        if rotation_mode == "measured-rates":
            dynamics_fn = simulation_ode_euler
        else:
            dynamics_fn = simulation_ode_euler_fixed_attitude

    measured_states = resample_state_to_times(t_cmd, pose_data, velocity_data)
    measured_rpy_all = measured_states[:, 6:9] if rotation_mode == "measured-attitude" else None

    window_step = args.window_step if args.window_step is not None else args.window_duration
    total_samples = measured_states.shape[0]
    pred_states = np.zeros_like(measured_states)
    pred_valid = np.zeros(total_samples, dtype=bool)

    print(f"Rotation mode: {rotation_mode}")
    print(f"Timeline samples: {total_samples}, duration: {t_cmd[-1] - t_cmd[0]:.2f}s")

    results = []
    for w_idx, (start_t, end_t, start_i, end_i) in enumerate(
        generate_time_windows(t_cmd, args.window_duration, window_step)
    ):
        if args.max_windows is not None and w_idx >= args.max_windows:
            break

        initial_state = measured_states[start_i].copy()
        predicted_window = simulate_window(
            t_cmd, thrust_cmd, w_cmd, start_i, end_i, initial_state, args.mass,
            dynamics_fn=dynamics_fn,
            measured_rpy=measured_rpy_all,
        )
        measured_window = measured_states[start_i:end_i]

        if predicted_window.shape[0] != measured_window.shape[0]:
            min_len = min(predicted_window.shape[0], measured_window.shape[0])
            predicted_window = predicted_window[:min_len]
            measured_window = measured_window[:min_len]
            end_i = start_i + min_len

        pos_err = np.linalg.norm(predicted_window[:, 0:3] - measured_window[:, 0:3], axis=1)
        vel_err = np.linalg.norm(predicted_window[:, 3:6] - measured_window[:, 3:6], axis=1)
        att_err = np.linalg.norm(predicted_window[:, 6:9] - measured_window[:, 6:9], axis=1)

        results.append({
            "window_index": w_idx,
            "start_time": start_t,
            "end_time": end_t,
            "start_idx": start_i,
            "end_idx": end_i,
            "pos_rms": float(np.sqrt(np.mean(pos_err**2))),
            "vel_rms": float(np.sqrt(np.mean(vel_err**2))),
            "att_rms": float(np.sqrt(np.mean(att_err**2))),
        })

        pred_states[start_i:end_i] = predicted_window
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
        plot_open_loop_trajectories(
            t_cmd[mask],
            measured_states[mask],
            pred_states[mask],
            (t_cmd, thrust_cmd, w_cmd),
            args.plot_dir,
        )


def plot_open_loop_trajectories(
    ts_axis,
    measured_states,
    predicted_states,
    bodyrate_cmd_data,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    measured_states = np.asarray(measured_states)
    predicted_states = np.asarray(predicted_states)
    ts_axis = np.asarray(ts_axis)

    measured_pos = measured_states[:, 0:3]
    measured_vel = measured_states[:, 3:6]
    measured_rpy_deg = np.rad2deg(measured_states[:, 6:9])

    pred_pos = predicted_states[:, 0:3]
    pred_vel = predicted_states[:, 3:6]
    pred_rpy_deg = np.rad2deg(predicted_states[:, 6:9])

    t_body, thrust_cmd, w_cmd = bodyrate_cmd_data
    if w_cmd.size > 0:
        w_cmd_deg = np.column_stack([
            np.interp(ts_axis, t_body, np.rad2deg(w_cmd[:, i])) for i in range(3)
        ])
    else:
        w_cmd_deg = np.full((ts_axis.shape[0], 3), np.nan)

    # Position
    fig_pos, axs_pos = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs_pos[i].plot(ts_axis, measured_pos[:, i], label='ROS measured')
        axs_pos[i].plot(ts_axis, pred_pos[:, i], label='JAX open-loop')
        axs_pos[i].set_ylabel(f'{labels[i]} (m)')
        axs_pos[i].grid(True)
    axs_pos[0].legend(loc='upper right')
    axs_pos[2].set_xlabel('Time (s)')
    fig_pos.suptitle('Open-loop Position Comparison')
    fig_pos.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_pos.savefig(os.path.join(output_dir, "openloop_positions.png"))
    plt.close(fig_pos)

    # Velocity
    fig_vel, axs_vel = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for i in range(3):
        axs_vel[i].plot(ts_axis, measured_vel[:, i], label='ROS measured')
        axs_vel[i].plot(ts_axis, pred_vel[:, i], label='JAX open-loop')
        axs_vel[i].set_ylabel(f'{labels[i]} dot (m/s)')
        axs_vel[i].grid(True)
    axs_vel[0].legend(loc='upper right')
    axs_vel[2].set_xlabel('Time (s)')
    fig_vel.suptitle('Open-loop Velocity Comparison')
    fig_vel.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_vel.savefig(os.path.join(output_dir, "openloop_velocities.png"))
    plt.close(fig_vel)

    # Attitude
    fig_att, axs_att = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    att_labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        axs_att[i].plot(ts_axis, measured_rpy_deg[:, i], label='ROS measured')
        axs_att[i].plot(ts_axis, pred_rpy_deg[:, i], label='JAX open-loop')
        axs_att[i].set_ylabel(f'{att_labels[i]} (deg)')
        axs_att[i].grid(True)
    axs_att[0].legend(loc='upper right')
    axs_att[2].set_xlabel('Time (s)')
    fig_att.suptitle('Open-loop Attitude Comparison')
    fig_att.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_att.savefig(os.path.join(output_dir, "openloop_attitude.png"))
    plt.close(fig_att)

    # Angular velocity
    fig_rate, axs_rate = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    rate_labels = ['p', 'q', 'r']
    for i in range(3):
        axs_rate[i].plot(ts_axis, w_cmd_deg[:, i], label='Command body rate (deg/s)')
        axs_rate[i].set_ylabel(rate_labels[i])
        axs_rate[i].grid(True)
    axs_rate[0].legend(loc='upper right')
    axs_rate[2].set_xlabel('Time (s)')
    fig_rate.suptitle('Open-loop Commanded Rates')
    fig_rate.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_rate.savefig(os.path.join(output_dir, "openloop_rates.png"))
    plt.close(fig_rate)


if __name__ == "__main__":
    main()

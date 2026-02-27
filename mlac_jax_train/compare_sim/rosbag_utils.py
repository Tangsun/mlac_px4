#!/usr/bin/env python3

"""
Utilities for reading rosbag2 files and producing aligned pose/control data
for simulation comparisons.
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
except ImportError as exc:
    raise RuntimeError(
        "rosbag_utils requires ROS 2 python packages. Make sure the ROS 2 "
        "environment is sourced before running the comparison scripts."
    ) from exc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# AttitudeTarget type mask bits (MAVROS definition)
IGNORE_ROLL_RATE = 1
IGNORE_PITCH_RATE = 2
IGNORE_YAW_RATE = 4
IGNORE_THRUST = 64


def _get_rosbag_options(path, storage_id='sqlite3'):
    storage_options = StorageOptions(uri=path, storage_id=storage_id)
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )
    return storage_options, converter_options


def _lvh_forward(arr, invalid_mask):
    """
    last-value-held for rows masked by invalid_mask.
    """
    arr = np.asarray(arr, dtype=float)
    valid = ~np.asarray(invalid_mask, dtype=bool)
    for i in range(arr.shape[0]):
        if not valid[i]:
            if i == 0:
                j = np.argmax(valid)
                arr[i] = arr[j] if valid[j] else 0.0
            else:
                arr[i] = arr[i - 1]
    return arr


def extract_attitude_data(
    rosbag_path,
    pose_topic="/mavros/local_position/pose",
    velocity_topic="/mavros/local_position/velocity_body",
    control_log_topic="/mlac_mission_node/control_log",
    att_setpoint_topic="/mavros/setpoint_raw/attitude",
):
    """
    Extract pose, velocity, and control command timelines from a rosbag.
    All timestamps are shifted so that the first attitude setpoint occurs at t=0.
    """
    if not os.path.exists(rosbag_path):
        raise FileNotFoundError(f"Rosbag directory not found: {rosbag_path}")

    reader = SequentialReader()
    storage_options, converter_options = _get_rosbag_options(rosbag_path)
    reader.open(storage_options, converter_options)

    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    required_topics = [pose_topic, velocity_topic, control_log_topic, att_setpoint_topic]
    missing = [t for t in required_topics if t not in topic_types]
    if missing:
        raise RuntimeError(f"Topics missing from rosbag: {missing}")

    PoseMsg = get_message(topic_types[pose_topic])
    VelocityMsg = get_message(topic_types[velocity_topic])
    ControllerLogMsg = get_message(topic_types[control_log_topic])
    AttitudeTargetMsg = get_message(topic_types[att_setpoint_topic])

    def header_to_ns(header):
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    # Pass 1: find tracking window from control log.
    traj_exec_start_ns = -1
    traj_exec_end_ns = -1
    reader.set_filter(StorageFilter(topics=[control_log_topic]))
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, ControllerLogMsg)
        if traj_exec_start_ns == -1 and msg.trajectory_execution_start_ros_time.sec > 0:
            traj_exec_start_ns = (
                msg.trajectory_execution_start_ros_time.sec * 1e9
                + msg.trajectory_execution_start_ros_time.nanosec
            )
        if msg.trajectory_execution_end_ros_time.sec > 0:
            traj_exec_end_ns = (
                msg.trajectory_execution_end_ros_time.sec * 1e9
                + msg.trajectory_execution_end_ros_time.nanosec
            )
            break

    reader.set_filter(StorageFilter(topics=required_topics))
    reader.seek(0)

    t_pose_ns, pos_list, quat_list = [], [], []
    t_vel_ns, lin_vel_body, ang_vel_body = [], [], []
    t_cmd_ns, thrust, w_cmd, mask_cmd = [], [], [], []
    t_cmd_att_ns, euler_cmd = [], []
    t_ref_ns, ref_pos, ref_vel, ref_acc, ref_yaw, ref_yaw_rate = [], [], [], [], [], []
    initial_pose_msg = None
    initial_velocity_msg = None

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if traj_exec_start_ns != -1 and t_ns < traj_exec_start_ns:
            continue
        if traj_exec_end_ns != -1 and t_ns > traj_exec_end_ns:
            break

        if topic == pose_topic:
            msg = deserialize_message(data, PoseMsg)
            if initial_pose_msg is None:
                initial_pose_msg = msg
            t_pose_ns.append(header_to_ns(msg.header))
            pos_list.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            quat_list.append([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ])
        elif topic == velocity_topic:
            msg = deserialize_message(data, VelocityMsg)
            if initial_velocity_msg is None:
                initial_velocity_msg = msg
            t_vel_ns.append(header_to_ns(msg.header))
            lin_vel_body.append([
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.linear.z,
            ])
            ang_vel_body.append([
                msg.twist.angular.x,
                msg.twist.angular.y,
                msg.twist.angular.z,
            ])
        elif topic == att_setpoint_topic:
            msg = deserialize_message(data, AttitudeTargetMsg)
            t_cmd_ns.append(header_to_ns(msg.header))
            thrust.append(float(msg.thrust))
            w_cmd.append([
                float(msg.body_rate.x),
                float(msg.body_rate.y),
                float(msg.body_rate.z),
            ])
            mask_cmd.append(int(msg.type_mask))
        elif topic == control_log_topic:
            msg = deserialize_message(data, ControllerLogMsg)
            t_ros_ns = header_to_ns(msg.header)
            t_cmd_att_ns.append(t_ros_ns)
            euler_cmd.append([
                float(msg.reference_roll),
                float(msg.reference_pitch),
                float(msg.reference_yaw),
            ])

            t_ref_ns.append(t_ros_ns)
            ref_pos.append([
                msg.reference_position.x,
                msg.reference_position.y,
                msg.reference_position.z,
            ])
            ref_vel.append([
                msg.reference_velocity.x,
                msg.reference_velocity.y,
                msg.reference_velocity.z,
            ])
            ref_acc.append([
                msg.reference_acceleration.x,
                msg.reference_acceleration.y,
                msg.reference_acceleration.z,
            ])
            ref_yaw.append(float(msg.reference_yaw))
            ref_yaw_rate.append(float(msg.reference_yaw_rate))

    if not t_cmd_ns:
        raise RuntimeError("No attitude setpoint messages found inside tracking window.")

    order_cmd = np.argsort(t_cmd_ns)
    t_cmd_ns = np.asarray(t_cmd_ns)[order_cmd]
    thrust = np.asarray(thrust)[order_cmd]
    w_cmd = np.asarray(w_cmd)[order_cmd]
    mask_cmd = np.asarray(mask_cmd)[order_cmd]

    t0_ns = t_cmd_ns[0]
    def to_rel_seconds(ns_array):
        if len(ns_array) == 0:
            return np.array([])
        return (np.asarray(ns_array) - t0_ns) * 1e-9

    t_cmd = to_rel_seconds(t_cmd_ns)
    thrust_invalid = (mask_cmd & IGNORE_THRUST) != 0 if len(mask_cmd) else np.zeros_like(thrust, dtype=bool)
    thrust = np.clip(_lvh_forward(thrust, thrust_invalid), 0.0, 1.0)

    rx_invalid = (mask_cmd & IGNORE_ROLL_RATE) != 0
    ry_invalid = (mask_cmd & IGNORE_PITCH_RATE) != 0
    rz_invalid = (mask_cmd & IGNORE_YAW_RATE) != 0
    w_cmd[:, 0] = _lvh_forward(w_cmd[:, 0], rx_invalid)
    w_cmd[:, 1] = _lvh_forward(w_cmd[:, 1], ry_invalid)
    w_cmd[:, 2] = _lvh_forward(w_cmd[:, 2], rz_invalid)

    pose_data = (
        to_rel_seconds(t_pose_ns),
        np.asarray(pos_list),
        np.asarray(quat_list),
    )
    vel_data = (
        to_rel_seconds(t_vel_ns),
        np.asarray(lin_vel_body),
        np.asarray(ang_vel_body),
    )
    attitude_cmd_data = (
        to_rel_seconds(t_cmd_att_ns),
        np.asarray(euler_cmd) if euler_cmd else np.empty((0, 3)),
    )
    bodyrate_cmd_data = (t_cmd, thrust, w_cmd)

    reference_data = (
        to_rel_seconds(t_ref_ns),
        np.asarray(ref_pos) if ref_pos else np.empty((0, 3)),
        np.asarray(ref_vel) if ref_vel else np.empty((0, 3)),
        np.asarray(ref_acc) if ref_acc else np.empty((0, 3)),
        np.asarray(ref_yaw) if ref_yaw else np.empty((0,)),
        np.asarray(ref_yaw_rate) if ref_yaw_rate else np.empty((0,)),
    )

    return {
        "pose": pose_data,
        "velocity": vel_data,
        "attitude_cmd": attitude_cmd_data,
        "bodyrate_cmd": bodyrate_cmd_data,
        "reference": reference_data,
        "initial_pose": initial_pose_msg,
        "initial_velocity": initial_velocity_msg,
    }


def resample_state_to_times(target_times, pose_data, velocity_data):
    """
    Interpolate pose/orientation/velocity data onto arbitrary target times.
    Returns Nx9 array [pos(3), vel_world(3), rpy(rad)].
    """
    target_times = np.asarray(target_times)
    t_pose, positions, quats = pose_data
    t_vel, lin_vel_body, _ = velocity_data

    if positions.size == 0 or quats.size == 0:
        raise RuntimeError("Pose data is empty; cannot resample.")

    rot_pose = Rotation.from_quat(quats)
    slerp_pose = Slerp(t_pose, rot_pose)
    times_clamped = np.clip(target_times, t_pose[0], t_pose[-1])
    rot_interp = slerp_pose(times_clamped)

    pos_interp = np.column_stack([
        np.interp(target_times, t_pose, positions[:, i]) for i in range(3)
    ])
    rpy_interp = rot_interp.as_euler('zyx', degrees=False)[:, ::-1]

    if lin_vel_body.size >= 2:
        rot_vel = slerp_pose(np.clip(t_vel, t_pose[0], t_pose[-1]))
        vel_world_samples = rot_vel.apply(lin_vel_body)
        vel_interp = np.column_stack([
            np.interp(target_times, t_vel, vel_world_samples[:, i]) for i in range(3)
        ])
    else:
        vel_interp = np.gradient(pos_interp, target_times, axis=0, edge_order=2)

    return np.column_stack([pos_interp, vel_interp, rpy_interp])


def extract_open_loop_measured(rosbag_path, **topic_kwargs):
    """
    Extract data for open-loop comparison using measured body rates and attitude
    (for rosbags recorded in attitude control mode where setpoint body_rate fields
    are not meaningful). Returns data keyed to velocity_body timestamps.
    """
    data = extract_attitude_data(rosbag_path, **topic_kwargs)
    t_vel, lin_vel_body, ang_vel_body = data["velocity"]
    t_cmd, thrust_cmd, _ = data["bodyrate_cmd"]
    thrust_interp = np.interp(t_vel, t_cmd, thrust_cmd)
    return {
        "t": t_vel,
        "thrust": thrust_interp,
        "ang_vel": ang_vel_body,
        "pose": data["pose"],
        "velocity": data["velocity"],
    }


def plot_pose_vs_reference(pose_data, reference_data, output_path=None):
    """
    Plot measured position/yaw against recorded references for a quick sanity check.
    """
    t_pose, positions, quats = pose_data
    t_ref, ref_pos, _, _, ref_yaw, _ = reference_data

    if positions.size == 0 or quats.size == 0:
        raise ValueError("Pose data is empty.")
    if t_ref.size == 0 or ref_pos.size == 0:
        raise ValueError("Reference data is empty.")

    ref_pos_interp = np.column_stack([
        np.interp(t_pose, t_ref, ref_pos[:, i]) for i in range(3)
    ])
    yaw_interp = np.interp(t_pose, t_ref, ref_yaw)
    euler_pose = Rotation.from_quat(quats).as_euler('zyx', degrees=True)
    yaw_pose_deg = euler_pose[:, 0]

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    labels = ['X (m)', 'Y (m)', 'Z (m)']
    for i in range(3):
        axs[i].plot(t_pose, positions[:, i], label='Measured')
        axs[i].plot(t_pose, ref_pos_interp[:, i], '--', label='Reference')
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid(True)

    axs[3].plot(t_pose, yaw_pose_deg, label="Measured yaw (deg)")
    axs[3].plot(t_pose, np.rad2deg(yaw_interp), '--', label="Reference yaw (deg)")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Yaw (deg)")
    axs[3].legend()
    axs[3].grid(True)

    fig.suptitle("Pose vs Controller Reference")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


def plot_command_streams(attitude_cmd_data, bodyrate_cmd_data, output_path=None):
    """
    Plot logged attitude references and body-rate/thrust commands.
    """
    t_att, euler_cmd = attitude_cmd_data
    t_body, thrust_cmd, w_cmd = bodyrate_cmd_data

    fig, axs = plt.subplots(4, 1, figsize=(12, 12))

    if euler_cmd.size > 0:
        for idx, label in enumerate(["Roll", "Pitch", "Yaw"]):
            axs[idx].plot(t_att, np.rad2deg(euler_cmd[:, idx]), label=f"{label} cmd (deg)")
            axs[idx].set_ylabel(f"{label} (deg)")
            axs[idx].legend()
            axs[idx].grid(True)
    else:
        for idx in range(3):
            axs[idx].text(0.5, 0.5, "No attitude command data", ha='center', va='center')
            axs[idx].set_axis_off()

    axs[3].plot(t_body, thrust_cmd, label="Normalized thrust")
    axs_omega = axs[3].twinx()
    axs_omega.plot(t_body, np.rad2deg(w_cmd[:, 2]), color='r', alpha=0.6, label="Yaw rate (deg/s)")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Thrust (norm)")
    axs[3].legend(loc='upper left')
    axs[3].grid(True)
    axs_omega.set_ylabel("Yaw rate (deg/s)")

    fig.suptitle("Command streams")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()

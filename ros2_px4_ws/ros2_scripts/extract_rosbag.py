#!/usr/bin/env python3

import argparse
import os
import sys # For sys.exit
import numpy as np
import rclpy # For Time conversion
from rclpy.time import Time
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import matplotlib.pyplot as plt 
# set font size for all plots
plt.rcParams.update({'font.size': 24})

# Attempt to import rosbag2_py components
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
except ImportError:
    print("CRITICAL ERROR: Failed to import 'rosbag2_py' components.")
    print("Please ensure your ROS 2 environment (e.g., Humble) is correctly sourced,")
    print("and that 'rosbag2_py' is installed as part of your ROS 2 distribution.")
    print("If using a Python virtual environment, activate it BEFORE sourcing your ROS 2 workspace.")
    sys.exit(1) 

# Attempt to import message type definitions
try:
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from mlac_msgs.msg import ControllerLog
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import message types: {e}.")
    print("Make sure your ROS 2 workspace (especially mlac_msgs) is built and sourced.")
    print("If using a Python virtual environment, activate it BEFORE sourcing your ROS 2 workspace.")
    sys.exit(1)


def quaternion_to_rotation_matrix(q_np: np.ndarray) -> np.ndarray:
    """Converts a quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q_np
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8: 
        return np.eye(3)
    s = 2.0 / Nq
    X, Y, Z = x*s, y*s, z*s
    wX, wY, wZ = w*X, w*Y, w*Z
    xX, xY, xZ = x*X, x*Y, x*Z
    yY, yZ       = y*Y, y*Z
    zZ           = z*Z
    return np.array([
        [1.0-(yY+zZ), xY-wZ, xZ+wY],
        [xY+wZ, 1.0-(xX+zZ), yZ-wX],
        [xZ-wY, yZ+wX, 1.0-(xX+yY)]
    ])

def get_rosbag_options(path, storage_id='sqlite3'):
    storage_options = StorageOptions(uri=path, storage_id=storage_id)
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    return storage_options, converter_options

def plot_extracted_data(data, bag_file_name):
    """
    Plots the extracted actual vs. reference trajectory data.
    Now includes the original trajectory file reference if available in 'data'.
    """
    print("Generating plots...")
    has_original_ref = 'original_ref_time_sec_aligned' in data

    # --- Position Plot ---
    fig_pos, axs_pos = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig_pos.suptitle(f'Position Tracking vs. Time\nBag: {os.path.basename(bag_file_name)}', fontsize=16)
    
    pos_labels = ['X', 'Y', 'Z']
    for i in range(3):
        if data['actual_position_m'].shape[0] > 0:
            axs_pos[i].plot(data['actual_time_sec'], data['actual_position_m'][:, i], label=f'Actual {pos_labels[i]} (Bag State)')
        if data['reference_position_m'].shape[0] > 0:
            axs_pos[i].plot(data['reference_time_sec'], data['reference_position_m'][:, i], label=f'Ref {pos_labels[i]} (Control Log)', linestyle='--')
        if has_original_ref and data['original_ref_position_m_aligned'].shape[0] > 0:
            axs_pos[i].plot(data['original_ref_time_sec_aligned'], data['original_ref_position_m_aligned'][:, i], label=f'Orig. Ref {pos_labels[i]} (File)', linestyle=':')
        axs_pos[i].set_ylabel(f'{pos_labels[i]} Position (m)')
        axs_pos[i].legend()
        axs_pos[i].grid(True)
    axs_pos[2].set_xlabel('Time (s, relative to trajectory start)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Velocity Plot (World Frame) ---
    fig_vel, axs_vel = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig_vel.suptitle(f'Velocity Tracking vs. Time (World Frame)\nBag: {os.path.basename(bag_file_name)}', fontsize=16)
    
    vel_labels = ['VX', 'VY', 'VZ']
    plot_actual_vel = 'actual_velocity_world_mps' in data and \
                      data['actual_velocity_world_mps'].shape[0] > 0 and \
                      data['actual_velocity_world_mps'].shape[0] == data['actual_velocity_body_time_sec'].shape[0]

    for i in range(3):
        if plot_actual_vel:
            axs_vel[i].plot(data['actual_velocity_body_time_sec'], data['actual_velocity_world_mps'][:, i], label=f'Actual {vel_labels[i]} (World)')
        if data['reference_velocity_mps'].shape[0] > 0:
            axs_vel[i].plot(data['reference_time_sec'], data['reference_velocity_mps'][:, i], label=f'Ref {vel_labels[i]} (Control Log)', linestyle='--')
        if has_original_ref and data['original_ref_velocity_mps_aligned'].shape[0] > 0:
            axs_vel[i].plot(data['original_ref_time_sec_aligned'], data['original_ref_velocity_mps_aligned'][:, i], label=f'Orig. Ref {vel_labels[i]} (File)', linestyle=':')
        axs_vel[i].set_ylabel(f'{vel_labels[i]} Velocity (m/s)')
        axs_vel[i].legend()
        axs_vel[i].grid(True)
    
    if not plot_actual_vel and data['reference_velocity_mps'].shape[0] == 0 and (not has_original_ref or data['original_ref_velocity_mps_aligned'].shape[0] == 0):
         print("Warning: No actual or reference velocity data available for plotting.")
    elif not plot_actual_vel:
        print("Warning: Actual world velocity data not available or mismatched for plotting.")

    axs_vel[2].set_xlabel('Time (s, relative to trajectory start)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- 3D Trajectory Plot ---
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    if data['actual_position_m'].shape[0] > 0:
        ax_3d.plot(data['actual_position_m'][:, 0], data['actual_position_m'][:, 1], data['actual_position_m'][:, 2], label='Actual Trajectory', color='b')
        ax_3d.scatter(data['actual_position_m'][0, 0], data['actual_position_m'][0, 1], data['actual_position_m'][0, 2], c='blue', marker='o', s=80, label='Actual Start', depthshade=False)
    if data['reference_position_m'].shape[0] > 0:
        ax_3d.plot(data['reference_position_m'][:, 0], data['reference_position_m'][:, 1], data['reference_position_m'][:, 2], label='Reference Trajectory', color='r', linestyle='--')
        ax_3d.scatter(data['reference_position_m'][0, 0], data['reference_position_m'][0, 1], data['reference_position_m'][0, 2], c='red', marker='x', s=80, label='Reference Start', depthshade=False)
    if has_original_ref and data['original_ref_position_m_aligned'].shape[0] > 0:
        ax_3d.plot(data['original_ref_position_m_aligned'][:, 0], data['original_ref_position_m_aligned'][:, 1], data['original_ref_position_m_aligned'][:, 2], label='Original Reference ', color='g', linestyle=':')
        ax_3d.scatter(data['original_ref_position_m_aligned'][0, 0], data['original_ref_position_m_aligned'][0, 1], data['original_ref_position_m_aligned'][0, 2], c='green', marker='^', s=80, label='Original Ref Start (File)', depthshade=False)

    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    # ax_3d.set_title(f'3D Trajectory Comparison\nBag: {os.path.basename(bag_file_name)}', fontsize=16)
    ax_3d.legend()
    ax_3d.grid(True)

    all_x_list, all_y_list, all_z_list = [], [], []
    if data['actual_position_m'].shape[0] > 0:
        all_x_list.append(data['actual_position_m'][:, 0]); all_y_list.append(data['actual_position_m'][:, 1]); all_z_list.append(data['actual_position_m'][:, 2])
    if data['reference_position_m'].shape[0] > 0:
        all_x_list.append(data['reference_position_m'][:, 0]); all_y_list.append(data['reference_position_m'][:, 1]); all_z_list.append(data['reference_position_m'][:, 2])
    if has_original_ref and data['original_ref_position_m_aligned'].shape[0] > 0:
        all_x_list.append(data['original_ref_position_m_aligned'][:, 0]); all_y_list.append(data['original_ref_position_m_aligned'][:, 1]); all_z_list.append(data['original_ref_position_m_aligned'][:, 2])

    if all_x_list:
        all_x = np.concatenate(all_x_list); all_y = np.concatenate(all_y_list); all_z = np.concatenate(all_z_list)
        if len(all_x) > 0 and len(all_y) > 0 and len(all_z) > 0: 
            max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 1.8
            if max_range < 0.1 or np.isnan(max_range) or np.isinf(max_range): max_range = 1.0 
            mid_x = (all_x.max()+all_x.min())*0.5; mid_y = (all_y.max()+all_y.min())*0.5; mid_z = (all_z.max()+all_z.min())*0.5
            ax_3d.set_xlim(mid_x - max_range, mid_x + max_range); ax_3d.set_ylim(mid_y - max_range, mid_y + max_range); ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.tight_layout()
    plt.show()

    # save fig_3d
    output_3d_plot_path = os.path.join(os.path.dirname(bag_file_name), '3d_trajectory_plot.png')
    fig_3d.savefig(output_3d_plot_path, bbox_inches='tight')
    print(f"3D trajectory plot saved to: {output_3d_plot_path}")


def extract_trajectory_data(bag_file_path: str, 
                            control_log_topic: str,
                            pose_topic: str,
                            velocity_topic: str,
                            output_npz_path: str,
                            original_traj_file_path: str | None, # New argument
                            original_traj_index: int # New argument
                            ):
    """
    Extracts state and reference data during trajectory tracking from a rosbag.
    Optionally loads and aligns an original .npy trajectory file.
    """
    if not os.path.exists(bag_file_path):
        print(f"Error: Rosbag directory not found: {bag_file_path}")
        return

    print(f"Processing rosbag: {bag_file_path}")

    # --- Load Original Trajectory File Data (if provided) ---
    orig_ref_data_loaded = False
    orig_ref_time_from_file = np.array([])
    orig_ref_pos_from_file = np.array([]).reshape(0,3)
    orig_ref_vel_from_file = np.array([]).reshape(0,3)
    orig_ref_acc_from_file = np.array([]).reshape(0,3) # Assuming 11-col for now
    orig_ref_yaw_from_file = np.array([])

    if original_traj_file_path:
        abs_orig_traj_path = os.path.expanduser(original_traj_file_path)
        if os.path.exists(abs_orig_traj_path):
            try:
                all_trajectories_data = np.load(abs_orig_traj_path)
                print(f"Loaded original trajectory file: {abs_orig_traj_path}")
                
                selected_orig_traj = None
                if all_trajectories_data.ndim == 3: # Batch of trajectories
                    num_trajs_in_file = all_trajectories_data.shape[0]
                    if 0 <= original_traj_index < num_trajs_in_file:
                        selected_orig_traj = all_trajectories_data[original_traj_index]
                        print(f"  Selected trajectory index {original_traj_index} from batch.")
                    else:
                        print(f"  Error: Original trajectory index {original_traj_index} out of bounds for file with {num_trajs_in_file} trajectories.")
                elif all_trajectories_data.ndim == 2: # Single trajectory
                    if original_traj_index == 0:
                        selected_orig_traj = all_trajectories_data
                        print("  Loaded single trajectory from file (index 0 assumed).")
                    else:
                        print(f"  Error: Original trajectory index {original_traj_index} requested, but file contains a single trajectory.")
                else:
                    print(f"  Error: Original trajectory file has unexpected shape: {all_trajectories_data.shape}")

                if selected_orig_traj is not None:
                    if selected_orig_traj.ndim == 2 and selected_orig_traj.shape[1] >= 11: # Check for 11 columns
                        orig_ref_time_from_file = selected_orig_traj[:, 0]
                        orig_ref_pos_from_file = selected_orig_traj[:, 1:4]
                        orig_ref_vel_from_file = selected_orig_traj[:, 4:7]
                        orig_ref_yaw_from_file = selected_orig_traj[:, 7]
                        orig_ref_acc_from_file = selected_orig_traj[:, 8:11]
                        orig_ref_data_loaded = True
                        print(f"  Original reference trajectory (idx {original_traj_index}) data extracted successfully ({selected_orig_traj.shape[0]} points).")
                    else:
                        print(f"  Error: Selected original trajectory (idx {original_traj_index}) does not have at least 11 columns. Shape: {selected_orig_traj.shape}")
            except Exception as e:
                print(f"  Error loading or parsing original trajectory file '{abs_orig_traj_path}': {e}")
        else:
            print(f"Warning: Original trajectory file not found at: {abs_orig_traj_path}")
    else:
        print("No original trajectory file path provided for comparison.")


    # ... (rest of the rosbag reading and processing logic remains the same as before) ...
    reader = SequentialReader()
    storage_options, converter_options = get_rosbag_options(bag_file_path)
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening rosbag: {e}")
        return

    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    required_topics = {control_log_topic, pose_topic, velocity_topic}
    for req_topic in required_topics:
        if req_topic not in topic_types:
            print(f"Error: Required topic '{req_topic}' not found in the bag. Available topics: {list(topic_types.keys())}")
            return
            
    try:
        ControllerLogMsgClass = get_message(topic_types[control_log_topic])
        PoseStampedMsgClass = get_message(topic_types[pose_topic])
        TwistStampedMsgClass = get_message(topic_types[velocity_topic])
    except Exception as e:
        print(f"Error getting message type definitions: {e}. Is the workspace sourced?")
        return

    print("Scanning for trajectory execution time window from control log...")
    traj_exec_start_ros_time_abs_ns = -1
    traj_exec_end_ros_time_abs_ns = -1
    fsm_timestamps_available = False 

    temp_reader = SequentialReader() 
    temp_reader.open(storage_options, converter_options)
    log_topic_filter = StorageFilter(topics=[control_log_topic]) 
    temp_reader.set_filter(log_topic_filter)
    first_log_msg_checked = False
    if temp_reader.has_next():
        (topic, data, t_ns) = temp_reader.read_next()
        if topic == control_log_topic:
            msg_check = deserialize_message(data, ControllerLogMsgClass)
            if hasattr(msg_check, 'trajectory_execution_start_ros_time') and \
               hasattr(msg_check, 'trajectory_execution_end_ros_time'):
                fsm_timestamps_available = True
                print("  Detected ControllerLog messages with FSM timestamp fields.")
            else:
                print("  WARNING: ControllerLog messages in this bag DO NOT have FSM timestamp fields.")
                print("           Cannot determine trajectory window from FSM. Will extract entire bag duration for topics.")
            first_log_msg_checked = True
        temp_reader.seek(0) 
    if not first_log_msg_checked and temp_reader.has_next(): 
        print("  WARNING: Could not verify ControllerLog message structure for FSM timestamps (e.g., very short bag).")
        print("           Assuming FSM timestamps are NOT available. Will extract entire bag duration for topics.")
    if fsm_timestamps_available:
        while temp_reader.has_next():
            (topic, data, t_ns) = temp_reader.read_next()
            if topic == control_log_topic:
                msg = deserialize_message(data, ControllerLogMsgClass)
                start_time_msg = msg.trajectory_execution_start_ros_time
                end_time_msg = msg.trajectory_execution_end_ros_time
                if traj_exec_start_ros_time_abs_ns == -1 and (start_time_msg.sec != 0 or start_time_msg.nanosec != 0):
                    traj_exec_start_ros_time_abs_ns = start_time_msg.sec * 1e9 + start_time_msg.nanosec
                    print(f"  Found trajectory execution start time: {traj_exec_start_ros_time_abs_ns / 1e9:.3f} s (absolute ROS time)")
                if traj_exec_end_ros_time_abs_ns == -1 and (end_time_msg.sec != 0 or end_time_msg.nanosec != 0):
                    traj_exec_end_ros_time_abs_ns = end_time_msg.sec * 1e9 + end_time_msg.nanosec
                    print(f"  Found trajectory execution end time: {traj_exec_end_ros_time_abs_ns / 1e9:.3f} s (absolute ROS time)")
    del temp_reader 
    if fsm_timestamps_available:
        if traj_exec_start_ros_time_abs_ns == -1:
            print("Error: Trajectory execution start time was expected but not found in control log. Cannot extract specific window.")
            return
        if traj_exec_end_ros_time_abs_ns == -1:
            print("Warning: Trajectory execution end time was expected but not found in control log. Will extract data until end of bag from start time.")
        print(f"Extracting data between {traj_exec_start_ros_time_abs_ns/1e9:.3f}s and "
              f"{traj_exec_end_ros_time_abs_ns/1e9 if traj_exec_end_ros_time_abs_ns != -1 else 'End of Bag'}s...")
    else:
        metadata = reader.get_metadata()
        bag_start_time_ns = metadata.starting_time.timestamp()
        bag_duration_ns = metadata.duration.total_seconds() * 1e9
        bag_end_time_ns = bag_start_time_ns + bag_duration_ns
        traj_exec_start_ros_time_abs_ns = bag_start_time_ns
        traj_exec_end_ros_time_abs_ns = bag_end_time_ns 
        print(f"Extracting data for the entire bag duration: "
              f"{bag_start_time_ns/1e9:.3f}s to {bag_end_time_ns/1e9:.3f}s (absolute ROS time)")

    actual_times, actual_pos_x, actual_pos_y, actual_pos_z = [], [], [], []
    actual_q_w, actual_q_x, actual_q_y, actual_q_z = [], [], [], []
    actual_vel_body_x, actual_vel_body_y, actual_vel_body_z = [], [], []
    actual_vel_body_times = [] 
    ref_times, ref_pos_x, ref_pos_y, ref_pos_z = [], [], [], []
    ref_vel_x, ref_vel_y, ref_vel_z = [], [], []
    ref_acc_x, ref_acc_y, ref_acc_z = [], [], []
    ref_yaw, ref_yaw_rate = [], []
    ref_q_w, ref_q_x, ref_q_y, ref_q_z = [], [], [], []

    reader.seek(0) 
    all_topics_filter = StorageFilter(topics=[control_log_topic, pose_topic, velocity_topic]) 
    reader.set_filter(all_topics_filter)
    first_valid_timestamp_ns = traj_exec_start_ros_time_abs_ns 

    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()
        if t_ns < traj_exec_start_ros_time_abs_ns: continue
        if traj_exec_end_ros_time_abs_ns != -1 and t_ns > traj_exec_end_ros_time_abs_ns and fsm_timestamps_available: continue
        relative_time_sec = (t_ns - first_valid_timestamp_ns) / 1e9
        if topic == pose_topic:
            msg = deserialize_message(data, PoseStampedMsgClass)
            actual_times.append(relative_time_sec); actual_pos_x.append(msg.pose.position.x); actual_pos_y.append(msg.pose.position.y); actual_pos_z.append(msg.pose.position.z)
            actual_q_w.append(msg.pose.orientation.w); actual_q_x.append(msg.pose.orientation.x); actual_q_y.append(msg.pose.orientation.y); actual_q_z.append(msg.pose.orientation.z)
        elif topic == velocity_topic:
            msg = deserialize_message(data, TwistStampedMsgClass)
            actual_vel_body_times.append(relative_time_sec); actual_vel_body_x.append(msg.twist.linear.x); actual_vel_body_y.append(msg.twist.linear.y); actual_vel_body_z.append(msg.twist.linear.z)
        elif topic == control_log_topic:
            msg = deserialize_message(data, ControllerLogMsgClass)
            ref_times.append(relative_time_sec); ref_pos_x.append(msg.reference_position.x); ref_pos_y.append(msg.reference_position.y); ref_pos_z.append(msg.reference_position.z)
            ref_vel_x.append(msg.reference_velocity.x); ref_vel_y.append(msg.reference_velocity.y); ref_vel_z.append(msg.reference_velocity.z)
            ref_acc_x.append(msg.reference_acceleration.x); ref_acc_y.append(msg.reference_acceleration.y); ref_acc_z.append(msg.reference_acceleration.z)
            ref_yaw.append(msg.reference_yaw); ref_yaw_rate.append(msg.reference_yaw_rate)
            ref_q_w.append(msg.reference_orientation_desired.w); ref_q_x.append(msg.reference_orientation_desired.x); ref_q_y.append(msg.reference_orientation_desired.y); ref_q_z.append(msg.reference_orientation_desired.z)
    
    print(f"Extracted {len(actual_times)} pose messages, {len(actual_vel_body_times)} velocity messages, and {len(ref_times)} control log messages within the window.")
    if not actual_times and not ref_times: print("No actual or reference data extracted. Aborting plots and save."); return

    actual_times = np.array(actual_times); actual_pos = np.array([actual_pos_x, actual_pos_y, actual_pos_z]).T; actual_quat = np.array([actual_q_w, actual_q_x, actual_q_y, actual_q_z]).T
    actual_vel_body_times = np.array(actual_vel_body_times); actual_vel_body = np.array([actual_vel_body_x, actual_vel_body_y, actual_vel_body_z]).T
    ref_times = np.array(ref_times); ref_pos = np.array([ref_pos_x, ref_pos_y, ref_pos_z]).T; ref_vel = np.array([ref_vel_x, ref_vel_y, ref_vel_z]).T
    ref_acc = np.array([ref_acc_x, ref_acc_y, ref_acc_z]).T; ref_yaw = np.array(ref_yaw); ref_yaw_rate = np.array(ref_yaw_rate); ref_quat = np.array([ref_q_w, ref_q_x, ref_q_y, ref_q_z]).T

    actual_vel_world = np.zeros_like(actual_vel_body)
    if actual_vel_body.shape[0] > 0 and actual_pos.shape[0] > 0:
        print("Converting body velocities to world frame...")
        interp_qw = np.interp(actual_vel_body_times, actual_times, actual_quat[:, 0]); interp_qx = np.interp(actual_vel_body_times, actual_times, actual_quat[:, 1])
        interp_qy = np.interp(actual_vel_body_times, actual_times, actual_quat[:, 2]); interp_qz = np.interp(actual_vel_body_times, actual_times, actual_quat[:, 3])
        for i in range(len(actual_vel_body_times)):
            q_actual_for_vel = np.array([interp_qw[i], interp_qx[i], interp_qy[i], interp_qz[i]])
            q_norm = np.linalg.norm(q_actual_for_vel)
            if q_norm > 1e-6: q_actual_for_vel /= q_norm
            else: q_actual_for_vel = np.array([1.0, 0.0, 0.0, 0.0])
            R_body_to_world = quaternion_to_rotation_matrix(q_actual_for_vel)
            actual_vel_world[i, :] = R_body_to_world @ actual_vel_body[i, :]
    
    output_data = {
        'actual_time_sec': actual_times, 'actual_position_m': actual_pos, 'actual_orientation_quat_wxyz': actual_quat, 
        'actual_velocity_body_time_sec': actual_vel_body_times, 'actual_velocity_body_mps': actual_vel_body, 'actual_velocity_world_mps': actual_vel_world, 
        'reference_time_sec': ref_times, 'reference_position_m': ref_pos, 'reference_velocity_mps': ref_vel, 
        'reference_acceleration_mps2': ref_acc, 'reference_yaw_rad': ref_yaw, 'reference_yaw_rate_rps': ref_yaw_rate,
        'reference_orientation_desired_quat_wxyz': ref_quat
    }

    # --- Interpolate Original Trajectory File Data onto common time axis ---
    if orig_ref_data_loaded and ref_times.size > 0: # Use ref_times as the common axis
        print("Interpolating original trajectory file data...")
        common_time_axis_for_orig_ref = ref_times # This is relative to trajectory start in bag
        
        # The time in orig_ref_time_from_file usually starts from 0 (or its own offset)
        # We need to align it with the *elapsed time* of the executed trajectory segment.
        # The FSM uses `target_time_in_traj_file = self.trajectory_start_file_time_offset + elapsed_execution_time_sec`
        # And `elapsed_execution_time_sec` is what `ref_times` (our `common_time_axis_for_orig_ref`) represents.
        # So, we should interpolate `orig_ref_time_from_file` at points `common_time_axis_for_orig_ref + trajectory_start_file_time_offset_from_npy`
        # However, mlac_fsm already handles this when it calls _get_trajectory_goal_at_time_fsm.
        # The `ref_pos`, `ref_vel` from control_log are *already* the result of this sampling.
        # What we want is to plot the *original* .npy trajectory data over the same *duration*
        # as the executed segment. The simplest way is to use the `ref_times` (elapsed time)
        # to sample the original .npy file, assuming its time column also represents elapsed time from its own start.

        # If the .npy file's time column (orig_ref_time_from_file) also starts near 0 and represents elapsed time:
        interp_orig_ref_pos = np.array([np.interp(common_time_axis_for_orig_ref, orig_ref_time_from_file, orig_ref_pos_from_file[:, i]) for i in range(3)]).T
        interp_orig_ref_vel = np.array([np.interp(common_time_axis_for_orig_ref, orig_ref_time_from_file, orig_ref_vel_from_file[:, i]) for i in range(3)]).T
        interp_orig_ref_yaw = np.interp(common_time_axis_for_orig_ref, orig_ref_time_from_file, orig_ref_yaw_from_file)
        interp_orig_ref_acc = np.array([np.interp(common_time_axis_for_orig_ref, orig_ref_time_from_file, orig_ref_acc_from_file[:, i]) for i in range(3)]).T


        output_data['original_ref_time_sec_aligned'] = common_time_axis_for_orig_ref
        output_data['original_ref_position_m_aligned'] = interp_orig_ref_pos
        output_data['original_ref_velocity_mps_aligned'] = interp_orig_ref_vel
        output_data['original_ref_yaw_rad_aligned'] = interp_orig_ref_yaw
        output_data['original_ref_acceleration_mps2_aligned'] = interp_orig_ref_acc
        print(f"  Aligned original trajectory data to {len(common_time_axis_for_orig_ref)} points.")


    output_npz_dir = os.path.dirname(os.path.abspath(output_npz_path))
    if not os.path.exists(output_npz_dir):
        os.makedirs(output_npz_dir, exist_ok=True)
        print(f"Created directory for output NPZ: {output_npz_dir}")

    np.savez(output_npz_path, **output_data)
    print(f"Cleanly extracted trajectory data saved to: {output_npz_path}")

    if actual_times.size > 0 or ref_times.size > 0: 
        plot_extracted_data(output_data, bag_file_path)
    else:
        print("Skipping plots as no data was extracted into the arrays.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract trajectory execution data from a ROS 2 bag and plot, with original .npy reference.")
    parser.add_argument('bag_file', type=str, help="Path to the input rosbag directory.")
    parser.add_argument('--output_npz', type=str, default="extracted_trajectory_data.npz",
                        help="Path to save the output .npz file (can include directory).")
    parser.add_argument('--control_log_topic', type=str, default="/mlac_mission_node/control_log",
                        help="Topic for mlac_msgs/msg/ControllerLog.")
    parser.add_argument('--pose_topic', type=str, default="/mavros/local_position/pose",
                        help="Topic for actual vehicle pose (geometry_msgs/msg/PoseStamped).")
    parser.add_argument('--velocity_topic', type=str, default="/mavros/local_position/velocity_body",
                        help="Topic for actual vehicle velocity (geometry_msgs/msg/TwistStamped, body frame).")
    parser.add_argument('--original_traj_file', type=str, default=None,
                        help="Path to the original .npy trajectory file for comparison (e.g., src/mlac_sim/traj_data/your_traj.npy).")
    parser.add_argument('--original_traj_index', type=int, default=0,
                        help="Index of the trajectory within the .npy file (if it's a batch).")
    
    args = parser.parse_args()
 
    try:
        extract_trajectory_data(
            args.bag_file,
            args.control_log_topic,
            args.pose_topic,
            args.velocity_topic,
            args.output_npz,
            args.original_traj_file,
            args.original_traj_index
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pass

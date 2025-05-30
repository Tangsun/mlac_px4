#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import pickle
import re
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import datetime # For timestamp in error log

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
except ImportError:
    print("CRITICAL ERROR: Failed to import 'rosbag2_py' components.")
    sys.exit(1)

try:
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from mlac_msgs.msg import ControllerLog
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import message types: {e}.")
    sys.exit(1)

def quaternion_to_rotation_matrix(q_np: np.ndarray) -> np.ndarray:
    w, x, y, z = q_np
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8: return np.eye(3)
    s = 2.0 / Nq
    X, Y, Z = x*s, y*s, z*s
    wX, wY, wZ = w*X, w*Y, w*Z
    xX, xY, xZ = x*X, x*Y, x*Z
    yY, yZ = y*Y, y*Z
    zZ = z*Z
    return np.array([
        [1.0-(yY+zZ), xY-wZ, xZ+wY],
        [xY+wZ, 1.0-(xX+zZ), yZ-wX],
        [xZ-wY, yZ+wX, 1.0-(xX+yY)]
    ])

def get_rosbag_options(path, storage_id='sqlite3'):
    storage_options = StorageOptions(uri=path, storage_id=storage_id)
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    return storage_options, converter_options

def process_single_bag_data(bag_file_path: str, 
                            original_trajectory_data: np.ndarray, 
                            target_time_vector: np.ndarray, 
                            control_log_topic: str,
                            pose_topic: str,
                            velocity_topic: str):
    if not os.path.exists(bag_file_path):
        print(f"  Error: Rosbag for this trial not found: {bag_file_path}")
        return None
    print(f"  Processing rosbag: {os.path.basename(bag_file_path)}")
    S_fixed = len(target_time_vector)
    reader = SequentialReader()
    storage_options, converter_options = get_rosbag_options(bag_file_path)
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"  Error opening rosbag {bag_file_path}: {e}")
        return None
    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    try:
        ControllerLogMsgClass = get_message(topic_types[control_log_topic])
        PoseStampedMsgClass = get_message(topic_types[pose_topic])
        TwistStampedMsgClass = get_message(topic_types[velocity_topic])
    except KeyError as e:
        print(f"  Error: A required topic ({e}) was not found in the bag {bag_file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"  Error getting message type definitions for {bag_file_path}: {e}. Is the workspace sourced?")
        return None

    traj_exec_start_ros_time_abs_ns = -1
    traj_exec_end_ros_time_abs_ns = -1
    fsm_timestamps_available = False
    temp_reader = SequentialReader()
    temp_reader.open(storage_options, converter_options)
    log_topic_filter = StorageFilter(topics=[control_log_topic])
    temp_reader.set_filter(log_topic_filter)
    if temp_reader.has_next():
        (_, data, _) = temp_reader.read_next()
        msg_check = deserialize_message(data, ControllerLogMsgClass)
        if hasattr(msg_check, 'trajectory_execution_start_ros_time') and \
           hasattr(msg_check, 'trajectory_execution_end_ros_time'): # Check both for safety
            fsm_timestamps_available = True
    del temp_reader
    
    if not fsm_timestamps_available:
        print(f"  Warning: FSM timestamps not in ControllerLog for {bag_file_path}. Cannot determine precise window. Skipping bag.")
        return None

    temp_reader_for_scan = SequentialReader() # New reader for the scan
    temp_reader_for_scan.open(storage_options, converter_options)
    temp_reader_for_scan.set_filter(log_topic_filter)
    while temp_reader_for_scan.has_next():
        (topic, data, t_ns) = temp_reader_for_scan.read_next()
        if topic == control_log_topic:
            msg = deserialize_message(data, ControllerLogMsgClass)
            start_time_msg = msg.trajectory_execution_start_ros_time
            end_time_msg = msg.trajectory_execution_end_ros_time
            if traj_exec_start_ros_time_abs_ns == -1 and (start_time_msg.sec != 0 or start_time_msg.nanosec != 0):
                traj_exec_start_ros_time_abs_ns = start_time_msg.sec * 1e9 + start_time_msg.nanosec
            if traj_exec_end_ros_time_abs_ns == -1 and (end_time_msg.sec != 0 or end_time_msg.nanosec != 0):
                traj_exec_end_ros_time_abs_ns = end_time_msg.sec * 1e9 + end_time_msg.nanosec
    del temp_reader_for_scan


    if traj_exec_start_ros_time_abs_ns == -1:
        print(f"  Error: Traj exec start time not found in {bag_file_path}. Skipping.")
        return None
    
    # If end time wasn't found in FSM logs, use the bag's full duration from the start time
    if traj_exec_end_ros_time_abs_ns == -1:
        print(f"  Warning: Traj exec end time not found in {bag_file_path} via FSM logs. Using bag metadata for end.")
        # Re-open reader or use existing 'reader' to get metadata if it's not already closed
        # For simplicity, let's assume 'reader' is still open here from the initial open.
        # If it was closed, it would need to be reopened.
        try:
            metadata = reader.get_metadata()
            # The end time from FSM logs is an absolute ROS time.
            # If not found, we'll use the bag's starting_time + duration
            # starting_time is an rclpy.time.Time object
            bag_start_ns = metadata.starting_time.nanoseconds_since_epoch
            bag_duration_ns = metadata.duration.total_seconds() * 1e9
            
            # If the FSM start time is *after* the bag's recorded start time (which it should be)
            # and the FSM end time is missing, we can assume the trajectory ran until the bag ended *relative to FSM start*.
            # Or, more simply, just use bag_start + bag_duration as the absolute end if FSM end is missing.
            traj_exec_end_ros_time_abs_ns = bag_start_ns + bag_duration_ns
            print(f"    Using bag end time: {traj_exec_end_ros_time_abs_ns / 1e9:.3f} s (absolute ROS time)")

        except Exception as e_meta:
            print(f"  Error getting bag metadata for end time: {e_meta}. Skipping bag.")
            return None

    execution_duration_ns = traj_exec_end_ros_time_abs_ns - traj_exec_start_ros_time_abs_ns
    if execution_duration_ns <= 0:
        print(f"  Warning: Invalid or zero execution duration ({execution_duration_ns/1e9:.3f}s) for {bag_file_path} based on FSM/bag times. Skipping.")
        return None
    
    print(f"    Exec window: Start={traj_exec_start_ros_time_abs_ns / 1e9:.3f}s, End={traj_exec_end_ros_time_abs_ns / 1e9:.3f}s (Dur: {execution_duration_ns/1e9:.3f}s)")

    actual_times_list, actual_pos_list, actual_quat_list = [], [], []
    actual_vel_body_times_list, actual_vel_body_list, actual_omega_list = [], [], []
    cmd_ref_times_list, cmd_ref_pos_list, cmd_ref_vel_list, cmd_ref_acc_list, cmd_ref_yaw_list, cmd_ref_dpsi_list, cmd_ref_quat_list = [], [], [], [], [], [], []
    
    reader.seek(0) # Rewind the main reader
    all_topics_filter = StorageFilter(topics=[control_log_topic, pose_topic, velocity_topic])
    reader.set_filter(all_topics_filter)

    # Ensure we don't try to create a negative length target_time_vector if execution_duration_ns is too small
    # The target_time_vector should span the *intended* or *fixed* duration for resampling, not the bag's execution window.
    # The relative_time_sec will be clipped to this target_time_vector's range during interpolation.

    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()
        if t_ns < traj_exec_start_ros_time_abs_ns: continue
        if t_ns > traj_exec_end_ros_time_abs_ns: continue # Stop reading if past the determined end
        
        relative_time_sec = (t_ns - traj_exec_start_ros_time_abs_ns) / 1e9

        if topic == pose_topic:
            msg = deserialize_message(data, PoseStampedMsgClass)
            actual_times_list.append(relative_time_sec)
            actual_pos_list.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            actual_quat_list.append([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        elif topic == velocity_topic:
            msg = deserialize_message(data, TwistStampedMsgClass)
            actual_vel_body_times_list.append(relative_time_sec)
            actual_vel_body_list.append([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
            actual_omega_list.append([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])
        elif topic == control_log_topic:
            msg = deserialize_message(data, ControllerLogMsgClass)
            cmd_ref_times_list.append(relative_time_sec)
            cmd_ref_pos_list.append([msg.reference_position.x, msg.reference_position.y, msg.reference_position.z])
            cmd_ref_vel_list.append([msg.reference_velocity.x, msg.reference_velocity.y, msg.reference_velocity.z])
            cmd_ref_acc_list.append([msg.reference_acceleration.x, msg.reference_acceleration.y, msg.reference_acceleration.z])
            cmd_ref_yaw_list.append(msg.reference_yaw)
            cmd_ref_dpsi_list.append(msg.reference_yaw_rate)
            cmd_ref_quat_list.append([msg.reference_orientation_desired.w, msg.reference_orientation_desired.x, msg.reference_orientation_desired.y, msg.reference_orientation_desired.z])
    
    if not actual_times_list: # If no pose data, we can't do much
        print(f"  Warning: No actual pose data extracted in window for {bag_file_path}. Skipping.")
        return None
    if not cmd_ref_times_list: # If no control log data, we can't get commanded refs
        print(f"  Warning: No commanded reference data (control_log) extracted in window for {bag_file_path}. Skipping.")
        return None


    actual_times_bag = np.array(actual_times_list); actual_pos_bag = np.array(actual_pos_list); actual_quat_bag = np.array(actual_quat_list)
    actual_vel_body_times_bag = np.array(actual_vel_body_times_list); actual_vel_body_bag = np.array(actual_vel_body_list); actual_omega_bag = np.array(actual_omega_list)
    cmd_ref_times_bag = np.array(cmd_ref_times_list); cmd_ref_pos_bag = np.array(cmd_ref_pos_list); cmd_ref_vel_bag = np.array(cmd_ref_vel_list)
    cmd_ref_acc_bag = np.array(cmd_ref_acc_list); cmd_ref_yaw_bag = np.array(cmd_ref_yaw_list); cmd_ref_dpsi_bag = np.array(cmd_ref_dpsi_list); cmd_ref_quat_bag = np.array(cmd_ref_quat_list)

    # Ensure actual_times_bag is not empty before interpolation
    if actual_times_bag.size == 0:
        print(f"  Warning: actual_times_bag is empty for {bag_file_path} after filtering. Skipping.")
        return None

    q_actual_resampled = np.array([np.interp(target_time_vector, actual_times_bag, actual_pos_bag[:,i], left=actual_pos_bag[0,i], right=actual_pos_bag[-1,i]) for i in range(3)]).T
    quat_actual_resampled = np.array([np.interp(target_time_vector, actual_times_bag, actual_quat_bag[:,i], left=actual_quat_bag[0,i], right=actual_quat_bag[-1,i]) for i in range(4)]).T
    
    vel_body_resampled = np.zeros((S_fixed, 3))
    omega_actual_resampled = np.zeros((S_fixed, 3)) 
    if len(actual_vel_body_times_bag) > 1:
        for i in range(3):
            vel_body_resampled[:, i] = np.interp(target_time_vector, actual_vel_body_times_bag, actual_vel_body_bag[:, i], left=actual_vel_body_bag[0,i], right=actual_vel_body_bag[-1,i])
            omega_actual_resampled[:, i] = np.interp(target_time_vector, actual_vel_body_times_bag, actual_omega_bag[:, i], left=actual_omega_bag[0,i], right=actual_omega_bag[-1,i])
    elif len(actual_vel_body_times_bag) == 1: # If only one point, replicate it
        vel_body_resampled[:, :] = actual_vel_body_bag[0, :]
        omega_actual_resampled[:, :] = actual_omega_bag[0, :]


    dq_actual_world_resampled = np.zeros((S_fixed, 3))
    for i in range(S_fixed):
        R = quaternion_to_rotation_matrix(quat_actual_resampled[i, :])
        dq_actual_world_resampled[i, :] = R @ vel_body_resampled[i, :]
    
    # Ensure cmd_ref_times_bag is not empty before interpolation
    if cmd_ref_times_bag.size == 0:
        print(f"  Warning: cmd_ref_times_bag is empty for {bag_file_path} after filtering. Skipping bag.")
        return None

    u_cmd_ref_resampled_list = [
        np.array([np.interp(target_time_vector, cmd_ref_times_bag, cmd_ref_pos_bag[:,i], left=cmd_ref_pos_bag[0,i], right=cmd_ref_pos_bag[-1,i]) for i in range(3)]).T,
        np.array([np.interp(target_time_vector, cmd_ref_times_bag, cmd_ref_vel_bag[:,i], left=cmd_ref_vel_bag[0,i], right=cmd_ref_vel_bag[-1,i]) for i in range(3)]).T,
        np.array([np.interp(target_time_vector, cmd_ref_times_bag, cmd_ref_acc_bag[:,i], left=cmd_ref_acc_bag[0,i], right=cmd_ref_acc_bag[-1,i]) for i in range(3)]).T,
        np.interp(target_time_vector, cmd_ref_times_bag, cmd_ref_yaw_bag, left=cmd_ref_yaw_bag[0], right=cmd_ref_yaw_bag[-1]).reshape(-1,1),
        np.interp(target_time_vector, cmd_ref_times_bag, cmd_ref_dpsi_bag, left=cmd_ref_dpsi_bag[0], right=cmd_ref_dpsi_bag[-1]).reshape(-1,1)
    ]
    u_cmd_ref_resampled = np.hstack(u_cmd_ref_resampled_list)

    orig_ref_time_from_file = original_trajectory_data[:, 0]
    # Adjust original trajectory time to start from 0 for interpolation
    orig_ref_time_from_file_relative = orig_ref_time_from_file - orig_ref_time_from_file[0]

    orig_ref_pos_from_file = original_trajectory_data[:, 1:4]
    orig_ref_vel_from_file = original_trajectory_data[:, 4:7]
    orig_ref_yaw_from_file = original_trajectory_data[:, 7]
    
    orig_ref_pos_resampled = np.array([np.interp(target_time_vector, orig_ref_time_from_file_relative, orig_ref_pos_from_file[:,i], left=orig_ref_pos_from_file[0,i], right=orig_ref_pos_from_file[-1,i]) for i in range(3)]).T
    orig_ref_vel_resampled = np.array([np.interp(target_time_vector, orig_ref_time_from_file_relative, orig_ref_vel_from_file[:,i], left=orig_ref_vel_from_file[0,i], right=orig_ref_vel_from_file[-1,i]) for i in range(3)]).T
    orig_ref_yaw_resampled = np.interp(target_time_vector, orig_ref_time_from_file_relative, orig_ref_yaw_from_file, left=orig_ref_yaw_from_file[0], right=orig_ref_yaw_from_file[-1])

    position_errors = q_actual_resampled - orig_ref_pos_resampled
    norm_errors_per_timestep = np.linalg.norm(position_errors, axis=1)
    mean_tracking_error_pos = np.mean(norm_errors_per_timestep)
    print(f"    Mean Position Tracking Error (vs original .npy): {mean_tracking_error_pos:.4f} m")

    return {
        'q': q_actual_resampled, 'dq': dq_actual_world_resampled, 'u_cmd_ref': u_cmd_ref_resampled,
        'quat': quat_actual_resampled, 'omega': omega_actual_resampled,
        'tracking_error_pos_mean': mean_tracking_error_pos,
        'original_ref_pos_resampled': orig_ref_pos_resampled,
        'original_ref_vel_resampled': orig_ref_vel_resampled,
        'original_ref_yaw_resampled': orig_ref_yaw_resampled,
    }

def parse_launch_log(log_file_path):
    trial_to_bag_map = {}
    if not os.path.exists(log_file_path):
        print(f"Warning: Launch log file not found: {log_file_path}")
        return trial_to_bag_map
    log_pattern = re.compile(r"Mission: \"([^\"]*Trial_(\d+)[^\"]*)\" \| Bag Directory: (\S+)")
    with open(log_file_path, 'r') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                trial_index = int(match.group(2))
                bag_dir_name = match.group(3)
                trial_to_bag_map[trial_index] = bag_dir_name
    print(f"Parsed {len(trial_to_bag_map)} entries from launch log: {log_file_path}")
    return trial_to_bag_map

def log_tracking_error(log_file, trial_idx, bag_path, error):
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] Trial: {trial_idx}, Bag: {os.path.basename(bag_path)}, Mean Pos Error: {error:.4f} m\n")
    except Exception as e:
        print(f"Error writing to tracking error log {log_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch process ROS2 bags.")
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of simulation runs to process.")
    parser.add_argument("--start_trial_index", type=int, default=0, help="Starting trial index.")
    parser.add_argument("--rosbag_base_dir", type=str, default=os.path.expanduser("~/mlac_px4/ros2_px4_ws/raw_traj_data/"))
    parser.add_argument("--launch_log_file", type=str, default=os.path.expanduser("~/mlac_px4/ros2_px4_ws/raw_traj_data/windy_data_traj.txt"))
    parser.add_argument("--original_traj_file", type=str, default=os.path.expanduser("~/mlac_px4/ros2_px4_ws/src/mlac_sim/traj_data/N100_T30.0_spline_11col_zero_yaw.npy"))
    parser.add_argument("--output_pkl_file", type=str, default="~/mlac_px4/ros2_px4_ws/processed_data/batch_trajectory_dataset.pkl")
    parser.add_argument("--tracking_error_log_file", type=str, default="~/mlac_px4/ros2_px4_ws/processed_data/batch_tracking_errors.txt")
    parser.add_argument("--fixed_traj_duration_sec", type=float, default=30.0)
    parser.add_argument("--fixed_dt_sec", type=float, default=0.02)
    parser.add_argument('--control_log_topic', type=str, default="/mlac_mission_node/control_log")
    parser.add_argument('--pose_topic', type=str, default="/mavros/local_position/pose")
    parser.add_argument('--velocity_topic', type=str, default="/mavros/local_position/velocity_body")
    args = parser.parse_args()

    args.output_pkl_file = os.path.expanduser(args.output_pkl_file) 
    args.tracking_error_log_file = os.path.expanduser(args.tracking_error_log_file) 

    if not os.path.exists(args.original_traj_file):
        print(f"Error: Original trajectory file not found: {args.original_traj_file}"); sys.exit(1)
    all_original_trajectories = np.load(args.original_traj_file)
    if all_original_trajectories.ndim != 3 or all_original_trajectories.shape[0] < args.start_trial_index + args.num_simulations:
        print(f"Error: Original trajectory file '{args.original_traj_file}' issue."); sys.exit(1)
    if all_original_trajectories.shape[2] < 11: # Check for at least 11 columns for t,p,v,psi,a
        print(f"Error: Original trajectory file '{args.original_traj_file}' needs >= 11 columns."); sys.exit(1)
    print(f"Loaded original trajectories from: {args.original_traj_file} (Shape: {all_original_trajectories.shape})")

    trial_to_bag_dir_map = parse_launch_log(args.launch_log_file)
    if not trial_to_bag_dir_map and args.num_simulations > 0 : 
        print(f"Warning: Could not parse any bag directories from {args.launch_log_file}.")

    S_fixed = int(args.fixed_traj_duration_sec / args.fixed_dt_sec) + 1
    target_time_vector = np.linspace(0, args.fixed_traj_duration_sec, S_fixed, dtype=np.float32)
    
    data_lists = {k: [] for k in ['q', 'dq', 'u_cmd_ref', 'quat', 'omega', 
                                 'tracking_error_pos_mean', 'original_ref_pos_resampled', 
                                 'original_ref_vel_resampled', 'original_ref_yaw_resampled']}
    processed_bag_count = 0

    for trial_idx_to_process in range(args.start_trial_index, args.start_trial_index + args.num_simulations):
        print(f"\n--- Processing Trial Index: {trial_idx_to_process} ---")
        bag_dir_name = trial_to_bag_dir_map.get(trial_idx_to_process)
        if not bag_dir_name:
            print(f"  Warning: No bag directory found in log for trial index {trial_idx_to_process}. Skipping.")
            continue
        current_bag_path = os.path.join(args.rosbag_base_dir, bag_dir_name)
        if not os.path.isdir(current_bag_path):
            print(f"  Warning: Rosbag directory '{current_bag_path}' for trial {trial_idx_to_process} does not exist. Skipping.")
            continue
        original_traj_for_this_run = all_original_trajectories[trial_idx_to_process]
        processed_data = process_single_bag_data(
            current_bag_path, original_traj_for_this_run, target_time_vector,
            args.control_log_topic, args.pose_topic, args.velocity_topic
        )
        if processed_data:
            for key in data_lists:
                if key in processed_data: 
                    data_lists[key].append(processed_data[key])
            log_tracking_error(args.tracking_error_log_file, trial_idx_to_process, current_bag_path, processed_data['tracking_error_pos_mean'])
            processed_bag_count += 1
        else:
            print(f"  Skipping trial {trial_idx_to_process} due to processing errors.")

    if processed_bag_count == 0:
        print("No bags were successfully processed. Exiting.")
        sys.exit(1)

    final_dataset = {'t': target_time_vector.astype(np.float32)}
    for key, lst in data_lists.items():
        if key == 'tracking_error_pos_mean': 
            final_dataset[key] = np.array(lst).astype(np.float32)
        elif lst: 
            final_dataset[key] = np.stack(lst, axis=0).astype(np.float32)
        else: 
            print(f"Warning: No data collected for key '{key}'. It will be missing or empty in the PKL.")

    print(f"\n--- Final Dataset Shapes (Processed {processed_bag_count} trajectories) ---")
    for key, value in final_dataset.items():
        if hasattr(value, 'shape'): print(f"  '{key}': {value.shape}")
        else: print(f"  '{key}': (scalar list or similar, length {len(value)})") 

    output_pkl_dir = os.path.dirname(args.output_pkl_file)
    if not os.path.exists(output_pkl_dir):
        os.makedirs(output_pkl_dir, exist_ok=True)
        print(f"Created directory for output PKL: {output_pkl_dir}")
    with open(args.output_pkl_file, 'wb') as f:
        pickle.dump(final_dataset, f)
    print(f"Batch processed data saved to: {args.output_pkl_file}")
    print(f"Tracking errors logged to: {args.tracking_error_log_file}")

if __name__ == '__main__':
    main()


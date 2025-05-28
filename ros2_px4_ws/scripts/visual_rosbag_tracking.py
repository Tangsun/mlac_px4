#!/usr/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from mpl_toolkits.mplot3d import Axes3D 

# Conditional import for rosbag2_py
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
except ImportError:
    print("Failed to import rosbag2_py. Please ensure ROS 2 environment is sourced and rosbag2_py is installed.")
    class SequentialReader: pass
    class StorageOptions: pass
    class ConverterOptions: pass

def quaternion_to_rotation_matrix(q_np: np.ndarray) -> np.ndarray:
    w, x, y, z = q_np; Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8: return np.eye(3)
    s = 2.0/Nq; X = x*s; Y = y*s; Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z; xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array([[1.0-(yY+zZ), xY-wZ, xZ+wY],
                     [xY+wZ, 1.0-(xX+zZ), yZ-wX],
                     [xZ-wY, yZ+wX, 1.0-(xX+yY)]])

def plot_quaternion_comparison(actual_time_rel, actual_orientations_filt, 
                               cmd_ref_time_rel, cmd_ref_orientations_filt, bag_file_name, time_base_label): # Added time_base_label
    if len(actual_time_rel) == 0 and len(cmd_ref_time_rel) == 0:
        print("Skipping quaternion plot: no data for either actual or commanded.")
        return

    fig_quat, axs_quat = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig_quat.suptitle(f'Quaternion Comparison vs. Time\nBag: {os.path.basename(bag_file_name)}', fontsize=14)
    
    q_labels = ['q_w', 'q_x', 'q_y', 'q_z']
    
    for i in range(4): 
        if len(actual_time_rel) > 0 and actual_orientations_filt.shape[0] == len(actual_time_rel):
            axs_quat[i].plot(actual_time_rel, actual_orientations_filt[:, i], label=f'Actual {q_labels[i]} (Pose Topic)')
        if len(cmd_ref_time_rel) > 0 and cmd_ref_orientations_filt.shape[0] == len(cmd_ref_time_rel) and cmd_ref_orientations_filt.shape[1] == 4 :
            axs_quat[i].plot(cmd_ref_time_rel, cmd_ref_orientations_filt[:, i], label=f'Commanded Ref {q_labels[i]} (Log)', linestyle='--')
        
        axs_quat[i].set_ylabel(q_labels[i])
        axs_quat[i].legend()
        axs_quat[i].grid(True)
    
    axs_quat[-1].set_xlabel(time_base_label) # Use the passed time_base_label
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def main(args):
    # --- 1. Reference Trajectory from .npy file (COMMENTED OUT - Not used for time windowing anymore) ---
    # ... (all .npy loading and printing can be kept commented or removed if not needed for other comparisons later) ...

    # --- 2. Read Rosbag Data ---
    if not os.path.exists(args.bag_file): print(f"Error: Rosbag not found: {args.bag_file}"); return
    storage_options = StorageOptions(uri=args.bag_file, storage_id=args.storage_id)
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = SequentialReader();
    try: reader.open(storage_options, converter_options)
    except Exception as e: print(f"Error opening rosbag: {e}"); return
    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    
    all_topics_to_check = [args.pose_topic, args.velocity_topic, args.control_log_topic] 
                           # args.rosout_topic, args.fsm_status_topic] # These are less critical now
    for topic in all_topics_to_check:
        if topic not in topic_types:
            print(f"Error: Critical Topic '{topic}' not found in bag. Available: {list(topic_types.keys())}"); return
            
    try:
        PoseStampedMsg = get_message(topic_types[args.pose_topic])
        TwistStampedMsg = get_message(topic_types[args.velocity_topic])
        ControllerLogMsg = get_message(topic_types[args.control_log_topic])
        # LogMsg = get_message(topic_types[args.rosout_topic]) if args.rosout_topic in topic_types else None # Optional
        # BoolMsg = get_message(topic_types[args.fsm_status_topic]) if args.fsm_status_topic in topic_types else None # Optional
    except Exception as e: print(f"Error getting msg types: {e}"); return

    bag_pose_times, bag_px,bag_py,bag_pz, bag_qw,bag_qx,bag_qy,bag_qz = [[] for _ in range(8)]
    bag_vel_body_times, bag_vx_body,bag_vy_body,bag_vz_body = [[] for _ in range(4)]
    bag_log_times, bag_log_px_ref,bag_log_py_ref,bag_log_pz_ref = [[] for _ in range(4)]
    bag_log_vx_ref,bag_log_vy_ref,bag_log_vz_ref = [[] for _ in range(3)]
    bag_log_q_ref_w, bag_log_q_ref_x, bag_log_q_ref_y, bag_log_q_ref_z = [[] for _ in range(4)] 
    # rosout_log_messages = [] # Optional
    # fsm_status_changes = [] # Optional

    first_timestamp_ns = -1

    print("Reading rosbag data...")
    while reader.has_next():
        (topic,data,timestamp_ns)=reader.read_next()
        if first_timestamp_ns == -1:
            first_timestamp_ns = timestamp_ns # Capture the very first timestamp
        
        ros_time_sec = timestamp_ns/1e9 # Keep absolute ROS time for now, or make relative to first_timestamp_ns later

        if topic == args.pose_topic:
            msg=deserialize_message(data,PoseStampedMsg); bag_pose_times.append(ros_time_sec); bag_px.append(msg.pose.position.x); bag_py.append(msg.pose.position.y); bag_pz.append(msg.pose.position.z)
            bag_qw.append(msg.pose.orientation.w); bag_qx.append(msg.pose.orientation.x); bag_qy.append(msg.pose.orientation.y); bag_qz.append(msg.pose.orientation.z)
        elif topic == args.velocity_topic:
            msg=deserialize_message(data,TwistStampedMsg); bag_vel_body_times.append(ros_time_sec); bag_vx_body.append(msg.twist.linear.x); bag_vy_body.append(msg.twist.linear.y); bag_vz_body.append(msg.twist.linear.z)
        elif topic == args.control_log_topic:
            msg=deserialize_message(data,ControllerLogMsg); bag_log_times.append(ros_time_sec)
            bag_log_px_ref.append(msg.reference_position.x); bag_log_py_ref.append(msg.reference_position.y); bag_log_pz_ref.append(msg.reference_position.z)
            bag_log_vx_ref.append(msg.reference_velocity.x); bag_log_vy_ref.append(msg.reference_velocity.y); bag_log_vz_ref.append(msg.reference_velocity.z)
            bag_log_q_ref_w.append(msg.reference_orientation_desired.w); bag_log_q_ref_x.append(msg.reference_orientation_desired.x)
            bag_log_q_ref_y.append(msg.reference_orientation_desired.y); bag_log_q_ref_z.append(msg.reference_orientation_desired.z)
        # elif topic == args.rosout_topic and LogMsg: # Optional
        #     msg = deserialize_message(data, LogMsg); rosout_log_messages.append((ros_time_sec, msg.name, msg.msg))
        # elif topic == args.fsm_status_topic and BoolMsg: # Optional
        #     msg = deserialize_message(data, BoolMsg); fsm_status_changes.append((ros_time_sec, msg.data))

    if first_timestamp_ns == -1 and (len(bag_pose_times) > 0 or len(bag_log_times) > 0):
        # Fallback if first_timestamp_ns wasn't captured but we have data
        all_times = []
        if len(bag_pose_times) > 0: all_times.append(bag_pose_times[0] * 1e9)
        if len(bag_log_times) > 0: all_times.append(bag_log_times[0] * 1e9)
        if len(bag_vel_body_times) > 0: all_times.append(bag_vel_body_times[0] * 1e9)
        if all_times: first_timestamp_ns = min(all_times)
        else: first_timestamp_ns = 0 # Absolute fallback
    
    first_timestamp_sec = first_timestamp_ns / 1e9
    time_base_label = f'Time since Bag Start (s) (Bag Start ROS Time: {first_timestamp_sec:.2f}s)'
    print(f"Bag data read. {time_base_label}")


    bag_pose_times=np.array(bag_pose_times); bag_px=np.array(bag_px); bag_py=np.array(bag_py); bag_pz=np.array(bag_pz)
    bag_orientations=np.array([bag_qw,bag_qx,bag_qy,bag_qz]).T
    bag_vel_body_times=np.array(bag_vel_body_times); bag_vx_body=np.array(bag_vx_body); bag_vy_body=np.array(bag_vy_body); bag_vz_body=np.array(bag_vz_body)
    bag_log_times=np.array(bag_log_times); bag_log_px_ref=np.array(bag_log_px_ref); bag_log_py_ref=np.array(bag_log_py_ref); bag_log_pz_ref=np.array(bag_log_pz_ref)
    bag_log_vx_ref=np.array(bag_log_vx_ref); bag_log_vy_ref=np.array(bag_log_vy_ref); bag_log_vz_ref=np.array(bag_log_vz_ref)
    bag_log_q_ref_w = np.array(bag_log_q_ref_w); bag_log_q_ref_x = np.array(bag_log_q_ref_x)
    bag_log_q_ref_y = np.array(bag_log_q_ref_y); bag_log_q_ref_z = np.array(bag_log_q_ref_z)

    if len(bag_pose_times)==0: print("Warning: No pose msgs found."); # Not returning, might still plot logs
    if len(bag_log_times)==0: print("Warning: No control_log msgs found."); # Not returning

    # --- 3. Trajectory Execution Phase START Identification (REMOVED) ---
    # --- 4. Plot END Time Identification (REMOVED) ---
    # We will plot ALL available data relative to the bag start time.

    # --- 5. Align Data Relative to Bag Start Time ---
    # No filtering by time window needed, but we make times relative to bag start.
    
    actual_time_rel = bag_pose_times - first_timestamp_sec if len(bag_pose_times) > 0 else np.array([])
    actual_px_filt = bag_px; actual_py_filt = bag_py; actual_pz_filt = bag_pz # Use all data
    actual_orientations_filt = bag_orientations
    print(f"  Actual_pose data points: {len(actual_time_rel)}")

    bag_vx_world, bag_vy_world, bag_vz_world = [], [], []
    actual_vel_time_rel = np.array([]) # Initialize
    if len(bag_vel_body_times) > 0 and len(bag_pose_times) > 0:
        # Align vel_body_times to the same relative time base for plotting
        temp_vel_times_rel = bag_vel_body_times - first_timestamp_sec
        
        for i, t_vel_abs in enumerate(bag_vel_body_times): # Use absolute time for finding corresponding pose
            pose_idx_original_array = np.argmin(np.abs(bag_pose_times - t_vel_abs)) # Match with absolute pose times
            q_for_vel = bag_orientations[pose_idx_original_array, :]
            v_body = np.array([bag_vx_body[i], bag_vy_body[i], bag_vz_body[i]])
            R_body_to_world = quaternion_to_rotation_matrix(q_for_vel); v_world = R_body_to_world @ v_body
            bag_vx_world.append(v_world[0]); bag_vy_world.append(v_world[1]); bag_vz_world.append(v_world[2])
        actual_vel_time_rel = temp_vel_times_rel # These are already relative
        actual_vx_world_filt = np.array(bag_vx_world); actual_vy_world_filt = np.array(bag_vy_world); actual_vz_world_filt = np.array(bag_vz_world)
        print(f"  Actual_velocity data points: {len(actual_vel_time_rel)}")
    else: actual_vx_world_filt, actual_vy_world_filt, actual_vz_world_filt = [np.array([])]*3


    cmd_ref_time_rel = bag_log_times - first_timestamp_sec if len(bag_log_times) > 0 else np.array([])
    cmd_ref_px_filt = bag_log_px_ref; cmd_ref_py_filt = bag_log_py_ref; cmd_ref_pz_filt = bag_log_pz_ref
    cmd_ref_vx_filt = bag_log_vx_ref; cmd_ref_vy_filt = bag_log_vy_ref; cmd_ref_vz_filt = bag_log_vz_ref
    
    cmd_ref_orientations_filt = np.array([]) 
    if len(bag_log_q_ref_w) > 0: 
        cmd_ref_orientations_filt = np.array([bag_log_q_ref_w, bag_log_q_ref_x, bag_log_q_ref_y, bag_log_q_ref_z]).T
    print(f"  Control_log data points: {len(cmd_ref_time_rel)}")
    
    # --- 6. Plotting ---
    print("Plotting results (entire simulation)...")
    fig_3d = plt.figure(figsize=(12, 9)) 
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    if len(actual_px_filt)>0 : ax_3d.plot(actual_px_filt, actual_py_filt, actual_pz_filt, label='Actual Trajectory (Bag)', color='b', alpha=0.9, linewidth=1.5)
    if len(cmd_ref_px_filt)>0 : ax_3d.plot(cmd_ref_px_filt, cmd_ref_py_filt, cmd_ref_pz_filt, label='Commanded Ref (Log)', linestyle='--', color='g', alpha=0.9, linewidth=1.5)
    if len(actual_px_filt)>0 : ax_3d.scatter(actual_px_filt[0], actual_py_filt[0], actual_pz_filt[0], c='blue', marker='o', s=80, label='Actual Start', depthshade=False, zorder=5)
    if len(actual_px_filt)>0 : ax_3d.scatter(actual_px_filt[-1], actual_py_filt[-1], actual_pz_filt[-1], c='cyan', marker='s', s=80, label='Actual End', depthshade=False, zorder=5)
    ax_3d.set_xlabel('X Position (m)'); ax_3d.set_ylabel('Y Position (m)'); ax_3d.set_zlabel('Z Position (m)')
    ax_3d.set_title('3D Trajectory Comparison (Entire Simulation)', fontsize=16); ax_3d.legend(); ax_3d.grid(True)
    # Auto-scaling for 3D plot
    if len(actual_px_filt)>0 or len(cmd_ref_px_filt)>0: 
        all_x_plot = []; all_y_plot = []; all_z_plot = []
        if len(actual_px_filt) > 0: all_x_plot.append(actual_px_filt); all_y_plot.append(actual_py_filt); all_z_plot.append(actual_pz_filt)
        if len(cmd_ref_px_filt) > 0: all_x_plot.append(cmd_ref_px_filt); all_y_plot.append(cmd_ref_py_filt); all_z_plot.append(cmd_ref_pz_filt)
        if all_x_plot: 
            all_x = np.concatenate(all_x_plot); all_y = np.concatenate(all_y_plot); all_z = np.concatenate(all_z_plot)
            if len(all_x)>0 and len(all_y)>0 and len(all_z)>0: 
                max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 1.8
                if max_range < 0.1: max_range = 1.0 
                mid_x = (all_x.max()+all_x.min())*0.5; mid_y = (all_y.max()+all_y.min())*0.5; mid_z = (all_z.max()+all_z.min())*0.5
                ax_3d.set_xlim(mid_x - max_range, mid_x + max_range); ax_3d.set_ylim(mid_y - max_range, mid_y + max_range); ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.tight_layout()

    fig_pos, axs_pos = plt.subplots(3, 1, figsize=(15, 10), sharex=True) 
    fig_pos.suptitle(f'Position vs. Time (Entire Simulation)\nBag: {os.path.basename(args.bag_file)}', fontsize=14)
    axs_pos[0].plot(actual_time_rel, actual_px_filt, label='Actual X (Bag)'); axs_pos[0].plot(cmd_ref_time_rel, cmd_ref_px_filt, label='Cmd Ref X (Log)', linestyle='--'); 
    axs_pos[0].set_ylabel('X Pos (m)'); axs_pos[0].legend(); axs_pos[0].grid(True)
    axs_pos[1].plot(actual_time_rel, actual_py_filt, label='Actual Y (Bag)'); axs_pos[1].plot(cmd_ref_time_rel, cmd_ref_py_filt, label='Cmd Ref Y (Log)', linestyle='--'); 
    axs_pos[1].set_ylabel('Y Pos (m)'); axs_pos[1].legend(); axs_pos[1].grid(True)
    axs_pos[2].plot(actual_time_rel, actual_pz_filt, label='Actual Z (Bag)'); axs_pos[2].plot(cmd_ref_time_rel, cmd_ref_pz_filt, label='Cmd Ref Z (Log)', linestyle='--'); 
    axs_pos[2].set_ylabel('Z Pos (m)'); axs_pos[2].set_xlabel(time_base_label); axs_pos[2].legend(); axs_pos[2].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_vel, axs_vel = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig_vel.suptitle(f'Velocity vs. Time (World Frame, Entire Simulation)\nBag: {os.path.basename(args.bag_file)}', fontsize=14)
    if len(actual_vel_time_rel) > 0: axs_vel[0].plot(actual_vel_time_rel, actual_vx_world_filt, label='Actual VX (World)')
    axs_vel[0].plot(cmd_ref_time_rel, cmd_ref_vx_filt, label='Cmd Ref VX (Log)', linestyle='--'); 
    axs_vel[0].set_ylabel('VX Vel (m/s)'); axs_vel[0].legend(); axs_vel[0].grid(True)
    if len(actual_vel_time_rel) > 0: axs_vel[1].plot(actual_vel_time_rel, actual_vy_world_filt, label='Actual VY (World)')
    axs_vel[1].plot(cmd_ref_time_rel, cmd_ref_vy_filt, label='Cmd Ref VY (Log)', linestyle='--'); 
    axs_vel[1].set_ylabel('VY Vel (m/s)'); axs_vel[1].legend(); axs_vel[1].grid(True)
    if len(actual_vel_time_rel) > 0: axs_vel[2].plot(actual_vel_time_rel, actual_vz_world_filt, label='Actual VZ (World)')
    axs_vel[2].plot(cmd_ref_time_rel, cmd_ref_vz_filt, label='Cmd Ref VZ (Log)', linestyle='--'); 
    axs_vel[2].set_ylabel('VZ Vel (m/s)'); axs_vel[2].set_xlabel(time_base_label); axs_vel[2].legend(); axs_vel[2].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 7. Plot Quaternion Comparison ---
    print("Plotting Quaternion Comparison...")
    plot_quaternion_comparison(actual_time_rel, actual_orientations_filt, 
                               cmd_ref_time_rel, cmd_ref_orientations_filt, 
                               args.bag_file, time_base_label) # Pass time_base_label
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trajectory tracking performance from a rosbag for the entire simulation duration.")
    parser.add_argument('--bag_file', type=str, required=True, help='Path to the rosbag directory.')
    # parser.add_argument('--ref_traj_file', type=str, required=False, help='Path to the .npy reference trajectory file (no longer used for time windowing).') # Kept for future, but not used now
    parser.add_argument('--pose_topic', type=str, default='/mavros/local_position/pose', help='Topic for vehicle pose.')
    parser.add_argument('--velocity_topic', type=str, default='/mavros/local_position/velocity_body', help='Topic for vehicle velocity (body frame).')
    parser.add_argument('--control_log_topic', type=str, default='/mlac_mission_node/control_log', help='Topic for controller log.')
    # parser.add_argument('--rosout_topic', type=str, default='/rosout', help="Topic for ROS log messages (optional).") # Optional
    # parser.add_argument('--fsm_logger_name', type=str, default='mlac_mission_node', help="Node name of the FSM logger (optional).") # Optional
    # parser.add_argument('--fsm_status_topic',type=str,default='/mlac_mission_node/trajectory_complete_status', help='Topic for FSM status (optional).') # Optional
    parser.add_argument('--storage_id', type=str, default='sqlite3', help='Rosbag storage ID.')
    # parser.add_argument('--start_match_threshold', type=float, default=0.3, help='(No longer used) Pos error threshold.')
    # parser.add_argument('--plot_time_padding', type=float, default=1.0, help='(No longer used) Extra time padding.')
    cli_args = parser.parse_args()
    main(cli_args)
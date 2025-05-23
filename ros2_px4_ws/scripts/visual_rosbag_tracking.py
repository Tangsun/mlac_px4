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

def main(args):
    # --- 1. Load Reference Trajectory from .npy file (still needed for duration and start point matching) ---
    if not os.path.exists(args.ref_traj_file):
        print(f"Error: Ref trajectory file not found: {args.ref_traj_file}"); return
    ref_traj_npy = np.load(args.ref_traj_file)
    ref_npy_time = ref_traj_npy[:, 0]
    ref_npy_px = ref_traj_npy[:, 1]; ref_npy_py = ref_traj_npy[:, 2]; ref_npy_pz = ref_traj_npy[:, 3]
    # ref_npy_vx = ref_traj_npy[:, 4]; ref_npy_vy = ref_traj_npy[:, 5]; ref_npy_vz = ref_traj_npy[:, 6] # Not plotted but loaded for consistency
    print(f"Loaded reference trajectory '{os.path.basename(args.ref_traj_file)}' ({ref_traj_npy.shape[0]} pts). Duration: {ref_npy_time[-1] - ref_npy_time[0]:.2f}s")
    print(f"  First .npy pt (t,x,y,z): {ref_npy_time[0]:.3f}, {ref_npy_px[0]:.3f}, {ref_npy_py[0]:.3f}, {ref_npy_pz[0]:.3f}")
    if ref_traj_npy.shape[0] > 1:
        print(f"  Second .npy pt (t,x,y,z): {ref_npy_time[1]:.3f}, {ref_npy_px[1]:.3f}, {ref_npy_py[1]:.3f}, {ref_npy_pz[1]:.3f}")

    # --- 2. Read Rosbag Data ---
    if not os.path.exists(args.bag_file): print(f"Error: Rosbag not found: {args.bag_file}"); return
    storage_options = StorageOptions(uri=args.bag_file, storage_id=args.storage_id)
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = SequentialReader();
    try: reader.open(storage_options, converter_options)
    except Exception as e: print(f"Error opening rosbag: {e}"); return
    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    
    all_topics_to_check = [args.pose_topic, args.velocity_topic, args.control_log_topic, 
                           args.rosout_topic, args.fsm_status_topic]
    for topic in all_topics_to_check:
        is_critical = topic not in [args.rosout_topic, args.fsm_status_topic] 
        if topic not in topic_types:
            message = f"Topic '{topic}' not found in bag."
            if is_critical: print(f"Error: {message} Available: {list(topic_types.keys())}"); return
            else: print(f"Warning: {message} Some features might be limited.")
    try:
        PoseStampedMsg = get_message(topic_types[args.pose_topic])
        TwistStampedMsg = get_message(topic_types[args.velocity_topic])
        ControllerLogMsg = get_message(topic_types[args.control_log_topic])
        LogMsg = get_message(topic_types[args.rosout_topic]) if args.rosout_topic in topic_types else None
        BoolMsg = get_message(topic_types[args.fsm_status_topic]) if args.fsm_status_topic in topic_types else None
    except Exception as e: print(f"Error getting msg types: {e}"); return

    bag_pose_times, bag_px,bag_py,bag_pz, bag_qw,bag_qx,bag_qy,bag_qz = [[] for _ in range(8)]
    bag_vel_body_times, bag_vx_body,bag_vy_body,bag_vz_body = [[] for _ in range(4)]
    bag_log_times, bag_log_px_ref,bag_log_py_ref,bag_log_pz_ref = [[] for _ in range(4)]
    bag_log_vx_ref,bag_log_vy_ref,bag_log_vz_ref = [[] for _ in range(3)]
    rosout_log_messages = []
    fsm_status_changes = []

    print("Reading rosbag data...")
    while reader.has_next():
        (topic,data,timestamp_ns)=reader.read_next(); ros_time_sec = timestamp_ns/1e9
        if topic == args.pose_topic:
            msg=deserialize_message(data,PoseStampedMsg); bag_pose_times.append(ros_time_sec); bag_px.append(msg.pose.position.x); bag_py.append(msg.pose.position.y); bag_pz.append(msg.pose.position.z)
            bag_qw.append(msg.pose.orientation.w); bag_qx.append(msg.pose.orientation.x); bag_qy.append(msg.pose.orientation.y); bag_qz.append(msg.pose.orientation.z)
        elif topic == args.velocity_topic:
            msg=deserialize_message(data,TwistStampedMsg); bag_vel_body_times.append(ros_time_sec); bag_vx_body.append(msg.twist.linear.x); bag_vy_body.append(msg.twist.linear.y); bag_vz_body.append(msg.twist.linear.z)
        elif topic == args.control_log_topic:
            msg=deserialize_message(data,ControllerLogMsg); bag_log_times.append(ros_time_sec); bag_log_px_ref.append(msg.reference_position.x); bag_log_py_ref.append(msg.reference_position.y); bag_log_pz_ref.append(msg.reference_position.z)
            bag_log_vx_ref.append(msg.reference_velocity.x); bag_log_vy_ref.append(msg.reference_velocity.y); bag_log_vz_ref.append(msg.reference_velocity.z)
        elif topic == args.rosout_topic and LogMsg:
            msg = deserialize_message(data, LogMsg); rosout_log_messages.append((ros_time_sec, msg.name, msg.msg))
        elif topic == args.fsm_status_topic and BoolMsg:
            msg = deserialize_message(data, BoolMsg); fsm_status_changes.append((ros_time_sec, msg.data))

    bag_pose_times=np.array(bag_pose_times); bag_px=np.array(bag_px); bag_py=np.array(bag_py); bag_pz=np.array(bag_pz)
    bag_orientations=np.array([bag_qw,bag_qx,bag_qy,bag_qz]).T
    bag_vel_body_times=np.array(bag_vel_body_times); bag_vx_body=np.array(bag_vx_body); bag_vy_body=np.array(bag_vy_body); bag_vz_body=np.array(bag_vz_body)
    bag_log_times=np.array(bag_log_times); bag_log_px_ref=np.array(bag_log_px_ref); bag_log_py_ref=np.array(bag_log_py_ref); bag_log_pz_ref=np.array(bag_log_pz_ref)
    bag_log_vx_ref=np.array(bag_log_vx_ref); bag_log_vy_ref=np.array(bag_log_vy_ref); bag_log_vz_ref=np.array(bag_log_vz_ref)
    if len(bag_log_times)==0: print("Error: No control_log msgs."); return
    if len(bag_pose_times)==0: print("Error: No pose msgs."); return
    max_bag_time = max(bag_pose_times.max() if len(bag_pose_times)>0 else 0, bag_log_times.max() if len(bag_log_times)>0 else 0)
    print(f"  Bag time for poses: [{bag_pose_times.min():.2f}s, {bag_pose_times.max():.2f}s] (Dur: {bag_pose_times.max()-bag_pose_times.min():.2f}s)")
    print(f"  Bag time for control_log: [{bag_log_times.min():.2f}s, {bag_log_times.max():.2f}s] (Dur: {bag_log_times.max()-bag_log_times.min():.2f}s)")

    # --- 3. Trajectory Execution Phase START Identification ---
    print("Identifying trajectory execution START...")
    traj_exec_start_ros_time = -1.0
    fsm_start_exec_log_msg_text = "FSM: Transitioning from MOVING_TO_TRAJECTORY_START to EXECUTING_TRAJECTORY"
    if LogMsg:
        for t, name, msg_text in rosout_log_messages:
            if name == args.fsm_logger_name and msg_text == fsm_start_exec_log_msg_text:
                traj_exec_start_ros_time = t
                print(f"  Found FSM log for EXEC START: '{fsm_start_exec_log_msg_text}' at ROS time: {t:.3f}s.")
                break
    if traj_exec_start_ros_time < 0: 
        print("  Warning: FSM log for EXEC START not found. Using p_ref sequence fallback.")
        p_npy_start = np.array([ref_npy_px[0], ref_npy_py[0], ref_npy_pz[0]])
        p_npy_second = np.array([ref_npy_px[1], ref_npy_py[1], ref_npy_pz[1]]) if ref_traj_npy.shape[0] > 1 else p_npy_start
        dt_npy = (ref_npy_time[1] - ref_npy_time[0]) if ref_traj_npy.shape[0] > 1 else 0.02
        for i in range(len(bag_log_times) - 1):
            p_log_curr = np.array([bag_log_px_ref[i], bag_log_py_ref[i], bag_log_pz_ref[i]])
            p_log_next = np.array([bag_log_px_ref[i+1], bag_log_py_ref[i+1], bag_log_pz_ref[i+1]])
            if (np.linalg.norm(p_log_curr - p_npy_start) < args.start_match_threshold and
                np.linalg.norm(p_log_next - p_npy_second) < args.start_match_threshold):
                dt_log = bag_log_times[i+1] - bag_log_times[i]
                if abs(dt_log - dt_npy) < (dt_npy / 2.0 + 0.005):
                    traj_exec_start_ros_time = bag_log_times[i]; print(f"    Fallback: p_ref seq match. Start: {traj_exec_start_ros_time:.3f}s (idx {i})"); break
        if traj_exec_start_ros_time < 0:
            print("    Warning: p_ref seq match failed. Using first P_npy[0] match fallback.")
            for i in range(len(bag_log_times)):
                if np.linalg.norm(np.array([bag_log_px_ref[i],bag_log_py_ref[i],bag_log_pz_ref[i]]) - p_npy_start) < args.start_match_threshold:
                    traj_exec_start_ros_time = bag_log_times[i]; print(f"      Fallback-2: First P_npy[0] match. Start: {traj_exec_start_ros_time:.3f}s"); break
    if traj_exec_start_ros_time < 0: print("Error: CRITICAL - Could not identify EXEC START. Cannot plot."); return
    
    # Diagnostic print (still useful to see what p_ref does around the identified start)
    # ... (This diagnostic print can be kept or commented if too verbose for the user) ...

    # --- 4. Plot END Time Identification (End of Landing) ---
    print("Identifying plot END time (end of LANDING)...")
    plot_end_ros_time = -1.0
    fsm_landed_log_msg_text = "FSM: Transitioning from LANDING to LANDED"
    if LogMsg:
        for t, name, msg_text in sorted(rosout_log_messages, key=lambda x: x[0]): 
            if t < traj_exec_start_ros_time : continue 
            if name == args.fsm_logger_name and msg_text == fsm_landed_log_msg_text:
                plot_end_ros_time = t; print(f"  Found FSM log for LANDED: '{fsm_landed_log_msg_text}' at ROS time: {t:.3f}s."); break
    if plot_end_ros_time < 0 and BoolMsg and fsm_status_changes: 
        print("  Warning: FSM log for LANDED not found. Using FSM status topic fallback.")
        for t, status in sorted(fsm_status_changes, key=lambda x: x[0]):
            if t > traj_exec_start_ros_time and status == True: 
                plot_end_ros_time = t; print(f"    Fallback: FSM status topic True at ROS time: {t:.3f}s."); break
    if plot_end_ros_time < 0: 
        plot_end_ros_time = max_bag_time
        print(f"  Warning: FSM LANDED signal not found. Plotting until end of bag data: {plot_end_ros_time:.3f}s.")

    ref_file_time_offset = ref_npy_time[0]
    traj_npy_duration = ref_npy_time[-1] - ref_file_time_offset
    print(f"  .npy trajectory duration (from file): {traj_npy_duration:.3f}s")
    print(f"  Effective plotting window in ROS time for bag data: [{traj_exec_start_ros_time:.3f}s, {plot_end_ros_time:.3f}s] (Padding: {args.plot_time_padding}s)")

    # --- 5. Filter and Align Data for the new extended window ---
    filter_display_end_ros_time = plot_end_ros_time + args.plot_time_padding

    pose_indices = (bag_pose_times >= traj_exec_start_ros_time) & (bag_pose_times <= filter_display_end_ros_time)
    actual_time_rel = bag_pose_times[pose_indices] - traj_exec_start_ros_time
    actual_px_filt = bag_px[pose_indices]; actual_py_filt = bag_py[pose_indices]; actual_pz_filt = bag_pz[pose_indices]
    actual_orientations_filt = bag_orientations[pose_indices, :]
    print(f"  Filtered actual_pose data points for plot: {len(actual_time_rel)}")

    bag_vx_world, bag_vy_world, bag_vz_world = [], [], []
    valid_vel_indices_for_transform = []
    if len(bag_vel_body_times) > 0 and len(bag_pose_times) > 0:
        for i, t_vel in enumerate(bag_vel_body_times):
            if t_vel >= traj_exec_start_ros_time and t_vel <= filter_display_end_ros_time:
                pose_idx_original_array = np.argmin(np.abs(bag_pose_times - t_vel))
                q_for_vel = bag_orientations[pose_idx_original_array, :]
                v_body = np.array([bag_vx_body[i], bag_vy_body[i], bag_vz_body[i]])
                R_body_to_world = quaternion_to_rotation_matrix(q_for_vel); v_world = R_body_to_world @ v_body
                bag_vx_world.append(v_world[0]); bag_vy_world.append(v_world[1]); bag_vz_world.append(v_world[2])
                valid_vel_indices_for_transform.append(i)                    
        actual_vel_time_rel = bag_vel_body_times[valid_vel_indices_for_transform] - traj_exec_start_ros_time
        actual_vx_world_filt = np.array(bag_vx_world); actual_vy_world_filt = np.array(bag_vy_world); actual_vz_world_filt = np.array(bag_vz_world)
        print(f"  Filtered actual_velocity data points for plot: {len(actual_vel_time_rel)}")
    else: actual_vel_time_rel, actual_vx_world_filt, actual_vy_world_filt, actual_vz_world_filt = [np.array([])]*4

    log_indices = (bag_log_times >= traj_exec_start_ros_time) & (bag_log_times <= filter_display_end_ros_time)
    cmd_ref_time_rel = bag_log_times[log_indices] - traj_exec_start_ros_time
    cmd_ref_px_filt = bag_log_px_ref[log_indices]; cmd_ref_py_filt = bag_log_py_ref[log_indices]; cmd_ref_pz_filt = bag_log_pz_ref[log_indices]
    cmd_ref_vx_filt = bag_log_vx_ref[log_indices]; cmd_ref_vy_filt = bag_log_vy_ref[log_indices]; cmd_ref_vz_filt = bag_log_vz_ref[log_indices]
    print(f"  Filtered control_log data points for plot: {len(cmd_ref_time_rel)}")

    # .npy file data: still useful to know its properties, but not plotted
    # ref_file_time_rel = ref_npy_time - ref_file_time_offset
    # npy_plot_indices = ref_file_time_rel <= traj_npy_duration + 0.01 
    # ref_file_time_plot = ref_file_time_rel[npy_plot_indices]
    # ref_npy_px_plot = ref_npy_px[npy_plot_indices]; ref_npy_py_plot = ref_npy_py[npy_plot_indices]; ref_npy_pz_plot = ref_npy_pz[npy_plot_indices]
    # ref_npy_vx_plot = ref_npy_vx[npy_plot_indices]; ref_npy_vy_plot = ref_npy_vy[npy_plot_indices]; ref_npy_vz_plot = ref_npy_vz[npy_plot_indices]
    # print(f"  (.npy ref data points, not plotted: {len(ref_file_time_plot)})")
    
    # --- 6. Plotting ---
    print("Plotting results (EXECUTE_TRAJECTORY to LANDING, .npy ref commented out)...")
    fig_3d = plt.figure(figsize=(12, 9)) 
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    if len(actual_px_filt)>0 : ax_3d.plot(actual_px_filt, actual_py_filt, actual_pz_filt, label='Actual Trajectory (Bag)', color='b', alpha=0.9, linewidth=1.5)
    if len(cmd_ref_px_filt)>0 : ax_3d.plot(cmd_ref_px_filt, cmd_ref_py_filt, cmd_ref_pz_filt, label='Commanded Ref (Log)', linestyle='--', color='g', alpha=0.9, linewidth=1.5)
    # ax_3d.plot(ref_npy_px_plot, ref_npy_py_plot, ref_npy_pz_plot, label='.npy File Ref (Exec Phase)', linestyle=':', color='r', alpha=0.7, linewidth=1) # COMMENTED OUT
    if len(actual_px_filt)>0 : ax_3d.scatter(actual_px_filt[0], actual_py_filt[0], actual_pz_filt[0], c='blue', marker='o', s=80, label='Actual Start on Plot', depthshade=False, zorder=5)
    # ax_3d.scatter(ref_npy_px_plot[0], ref_npy_py_plot[0], ref_npy_pz_plot[0], c='red', marker='x', s=80, label='.npy Start on Plot', depthshade=False, zorder=5) # COMMENTED OUT
    if len(actual_px_filt)>0 : ax_3d.scatter(actual_px_filt[-1], actual_py_filt[-1], actual_pz_filt[-1], c='cyan', marker='s', s=80, label='Actual End on Plot', depthshade=False, zorder=5)
    ax_3d.set_xlabel('X Position (m)'); ax_3d.set_ylabel('Y Position (m)'); ax_3d.set_zlabel('Z Position (m)')
    ax_3d.set_title('3D Trajectory Comparison (Execute Phase to Landing)', fontsize=16); ax_3d.legend(); ax_3d.grid(True)
    if len(actual_px_filt)>0 or len(cmd_ref_px_filt)>0: # Adjusted for commented .npy
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
    fig_pos.suptitle(f'Position vs. Time (Execute Phase to Landing)\nBag: {os.path.basename(args.bag_file)}', fontsize=14) # Removed ref from title
    common_time_label = 'Time since Identified Trajectory Execution Start (s)'
    axs_pos[0].plot(actual_time_rel, actual_px_filt, label='Actual X (Bag)'); axs_pos[0].plot(cmd_ref_time_rel, cmd_ref_px_filt, label='Cmd Ref X (Log)', linestyle='--'); 
    # axs_pos[0].plot(ref_file_time_plot, ref_npy_px_plot, label='.npy Ref X (Exec Phase)', linestyle=':'); # COMMENTED OUT
    axs_pos[0].set_ylabel('X Pos (m)'); axs_pos[0].legend(); axs_pos[0].grid(True)
    axs_pos[1].plot(actual_time_rel, actual_py_filt, label='Actual Y (Bag)'); axs_pos[1].plot(cmd_ref_time_rel, cmd_ref_py_filt, label='Cmd Ref Y (Log)', linestyle='--'); 
    # axs_pos[1].plot(ref_file_time_plot, ref_npy_py_plot, label='.npy Ref Y (Exec Phase)', linestyle=':'); # COMMENTED OUT
    axs_pos[1].set_ylabel('Y Pos (m)'); axs_pos[1].legend(); axs_pos[1].grid(True)
    axs_pos[2].plot(actual_time_rel, actual_pz_filt, label='Actual Z (Bag)'); axs_pos[2].plot(cmd_ref_time_rel, cmd_ref_pz_filt, label='Cmd Ref Z (Log)', linestyle='--'); 
    # axs_pos[2].plot(ref_file_time_plot, ref_npy_pz_plot, label='.npy Ref Z (Exec Phase)', linestyle=':'); # COMMENTED OUT
    axs_pos[2].set_ylabel('Z Pos (m)'); axs_pos[2].set_xlabel(common_time_label); axs_pos[2].legend(); axs_pos[2].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_vel, axs_vel = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig_vel.suptitle(f'Velocity vs. Time (World Frame, Execute Phase to Landing)\nBag: {os.path.basename(args.bag_file)}', fontsize=14) # Removed ref from title
    if len(actual_vel_time_rel) > 0: axs_vel[0].plot(actual_vel_time_rel, actual_vx_world_filt, label='Actual VX (World)')
    axs_vel[0].plot(cmd_ref_time_rel, cmd_ref_vx_filt, label='Cmd Ref VX (Log)', linestyle='--'); 
    # axs_vel[0].plot(ref_file_time_plot, ref_npy_vx_plot, label='.npy Ref VX (Exec Phase)', linestyle=':'); # COMMENTED OUT
    axs_vel[0].set_ylabel('VX Vel (m/s)'); axs_vel[0].legend(); axs_vel[0].grid(True)
    if len(actual_vel_time_rel) > 0: axs_vel[1].plot(actual_vel_time_rel, actual_vy_world_filt, label='Actual VY (World)')
    axs_vel[1].plot(cmd_ref_time_rel, cmd_ref_vy_filt, label='Cmd Ref VY (Log)', linestyle='--'); 
    # axs_vel[1].plot(ref_file_time_plot, ref_npy_vy_plot, label='.npy Ref VY (Exec Phase)', linestyle=':'); # COMMENTED OUT
    axs_vel[1].set_ylabel('VY Vel (m/s)'); axs_vel[1].legend(); axs_vel[1].grid(True)
    if len(actual_vel_time_rel) > 0: axs_vel[2].plot(actual_vel_time_rel, actual_vz_world_filt, label='Actual VZ (World)')
    axs_vel[2].plot(cmd_ref_time_rel, cmd_ref_vz_filt, label='Cmd Ref VZ (Log)', linestyle='--'); 
    # axs_vel[2].plot(ref_file_time_plot, ref_npy_vz_plot, label='.npy Ref VZ (Exec Phase)', linestyle=':'); # COMMENTED OUT
    axs_vel[2].set_ylabel('VZ Vel (m/s)'); axs_vel[2].set_xlabel(common_time_label); axs_vel[2].legend(); axs_vel[2].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trajectory tracking performance from a rosbag, from trajectory execution through landing.")
    parser.add_argument('--bag_file', type=str, required=True, help='Path to the rosbag directory.')
    parser.add_argument('--ref_traj_file', type=str, required=True, help='Path to the .npy reference trajectory file (used for duration and start point matching).')
    parser.add_argument('--pose_topic', type=str, default='/mavros/local_position/pose', help='Topic for vehicle pose.')
    parser.add_argument('--velocity_topic', type=str, default='/mavros/local_position/velocity_body', help='Topic for vehicle velocity (body frame).')
    parser.add_argument('--control_log_topic', type=str, default='/mlac_mission_node/control_log', help='Topic for controller log.')
    parser.add_argument('--rosout_topic', type=str, default='/rosout', help="Topic for ROS log messages (e.g., /rosout).")
    parser.add_argument('--fsm_logger_name', type=str, default='mlac_mission_node', help="Node name of the FSM logger (for /rosout parsing).")
    parser.add_argument('--fsm_status_topic',type=str,default='/mlac_mission_node/trajectory_complete_status', help='Topic for FSM trajectory complete status (std_msgs/Bool).')
    parser.add_argument('--storage_id', type=str, default='sqlite3', help='Rosbag storage ID.')
    parser.add_argument('--start_match_threshold', type=float, default=0.3, help='Pos error threshold (m) to match .npy start with control_log.p_ref for fallback.')
    parser.add_argument('--plot_time_padding', type=float, default=1.0, help='Extra time padding (s) for plotting beyond identified end time.')
    cli_args = parser.parse_args()
    main(cli_args)
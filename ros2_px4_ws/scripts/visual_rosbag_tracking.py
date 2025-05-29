#!/usr/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from mpl_toolkits.mplot3d import Axes3D 
import math # For math.degrees and RPY conversion

# Conditional import for rosbag2_py
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
except ImportError:
    print("Failed to import rosbag2_py. Please ensure ROS 2 environment is sourced and rosbag2_py is installed.")
    # Define dummy classes if rosbag2_py is not available, to allow basic script structure checking
    class SequentialReader: pass
    class StorageOptions: pass
    class ConverterOptions: pass

# Message type imports
from geometry_msgs.msg import PoseStamped as PoseStampedMsg
from geometry_msgs.msg import TwistStamped as TwistStampedMsg
from mlac_msgs.msg import ControllerLog as ControllerLogMsg
# If mlac_sim.helpers is in your python path when running this script, you can use it:
# from mlac_sim.helpers import get_rpy # Assuming get_rpy returns a Vector3-like object or tuple (r,p,y) in radians
# Otherwise, we define a local version:

def quaternion_to_rotation_matrix(q_np: np.ndarray) -> np.ndarray:
    w, x, y, z = q_np
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8: return np.eye(3)
    s = 2.0/Nq
    X = x*s; Y = y*s; Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array([[1.0-(yY+zZ), xY-wZ, xZ+wY],
                     [xY+wZ, 1.0-(xX+zZ), yZ-wX],
                     [xZ-wY, yZ+wX, 1.0-(xX+yY)]])

def quaternion_to_rpy_degrees(q_w, q_x, q_y, q_z):
    """
    Convert a quaternion (w, x, y, z) into euler angles (roll, pitch, yaw) in degrees.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q_w * q_y - q_z * q_x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp) # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def plot_quaternion_comparison(actual_time_rel, actual_orientations_filt, 
                               cmd_ref_time_rel, cmd_ref_orientations_filt, bag_file_name, time_base_label):
    if not (actual_time_rel.size > 0 and actual_orientations_filt.ndim == 2 and actual_orientations_filt.shape[1] == 4) and \
       not (cmd_ref_time_rel.size > 0 and cmd_ref_orientations_filt.ndim == 2 and cmd_ref_orientations_filt.shape[1] == 4):
        print("Skipping quaternion plot: insufficient data for either actual or commanded.")
        return

    fig_quat, axs_quat = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig_quat.suptitle(f'Quaternion Comparison vs. Time\nBag: {os.path.basename(bag_file_name)}', fontsize=14)
    
    q_labels = ['q_w', 'q_x', 'q_y', 'q_z']
    
    for i in range(4): 
        if actual_time_rel.size > 0 and actual_orientations_filt.shape[0] == len(actual_time_rel):
            axs_quat[i].plot(actual_time_rel, actual_orientations_filt[:, i], label=f'Actual {q_labels[i]} (Pose Topic)')
        if cmd_ref_time_rel.size > 0 and cmd_ref_orientations_filt.shape[0] == len(cmd_ref_time_rel) and cmd_ref_orientations_filt.shape[1] == 4 :
            axs_quat[i].plot(cmd_ref_time_rel, cmd_ref_orientations_filt[:, i], label=f'Commanded Ref {q_labels[i]} (Log)', linestyle='--')
        
        axs_quat[i].set_ylabel(q_labels[i])
        axs_quat[i].legend()
        axs_quat[i].grid(True)
    
    axs_quat[-1].set_xlabel(time_base_label)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_rpy_comparison(actual_time_rel, actual_roll, actual_pitch, actual_yaw,
                        cmd_ref_time_rel, cmd_ref_roll, cmd_ref_pitch, cmd_ref_yaw,
                        bag_file_name, time_base_label):
    if not (actual_time_rel.size > 0) and not (cmd_ref_time_rel.size > 0):
        print("Skipping RPY plot: insufficient data for either actual or commanded RPY.")
        return

    fig_rpy, axs_rpy = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig_rpy.suptitle(f'RPY Comparison vs. Time (Degrees)\nBag: {os.path.basename(bag_file_name)}', fontsize=14)
    
    # Roll
    plot_actual_roll = actual_time_rel.size > 0 and actual_roll.size == actual_time_rel.size
    plot_cmd_roll = cmd_ref_time_rel.size > 0 and cmd_ref_roll.size == cmd_ref_time_rel.size
    if plot_actual_roll:
        axs_rpy[0].plot(actual_time_rel, actual_roll, label='Actual Roll (Pose Topic)')
    if plot_cmd_roll:
        axs_rpy[0].plot(cmd_ref_time_rel, cmd_ref_roll, label='Commanded Ref Roll (Log)', linestyle='--')
    if plot_actual_roll or plot_cmd_roll:
        axs_rpy[0].legend()
    axs_rpy[0].set_ylabel('Roll (degrees)')
    axs_rpy[0].grid(True)

    # Pitch
    plot_actual_pitch = actual_time_rel.size > 0 and actual_pitch.size == actual_time_rel.size
    plot_cmd_pitch = cmd_ref_time_rel.size > 0 and cmd_ref_pitch.size == cmd_ref_time_rel.size
    if plot_actual_pitch:
        axs_rpy[1].plot(actual_time_rel, actual_pitch, label='Actual Pitch (Pose Topic)')
    if plot_cmd_pitch:
        axs_rpy[1].plot(cmd_ref_time_rel, cmd_ref_pitch, label='Commanded Ref Pitch (Log)', linestyle='--')
    if plot_actual_pitch or plot_cmd_pitch:
        axs_rpy[1].legend()
    axs_rpy[1].set_ylabel('Pitch (degrees)')
    axs_rpy[1].grid(True)

    # Yaw
    plot_actual_yaw = actual_time_rel.size > 0 and actual_yaw.size == actual_time_rel.size
    plot_cmd_yaw = cmd_ref_time_rel.size > 0 and cmd_ref_yaw.size == cmd_ref_time_rel.size
    if plot_actual_yaw:
        axs_rpy[2].plot(actual_time_rel, actual_yaw, label='Actual Yaw (Pose Topic)')
    if plot_cmd_yaw:
        axs_rpy[2].plot(cmd_ref_time_rel, cmd_ref_yaw, label='Commanded Ref Yaw (Log)', linestyle='--')
    if plot_actual_yaw or plot_cmd_yaw:
        axs_rpy[2].legend()
    axs_rpy[2].set_ylabel('Yaw (degrees)')
    axs_rpy[2].grid(True)
    
    axs_rpy[-1].set_xlabel(time_base_label)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def main(args):
    if not os.path.exists(args.bag_file):
        print(f"Error: Rosbag not found: {args.bag_file}")
        return
    storage_options = StorageOptions(uri=args.bag_file, storage_id=args.storage_id)
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening rosbag: {e}")
        return
        
    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    
    critical_topics = [args.pose_topic, args.velocity_topic, args.control_log_topic]
    for topic in critical_topics:
        if topic not in topic_types:
            print(f"Error: Critical Topic '{topic}' not found in bag. Available: {list(topic_types.keys())}")
            return
            
    bag_pose_times, bag_px, bag_py, bag_pz, bag_qw, bag_qx, bag_qy, bag_qz = [[] for _ in range(8)]
    bag_vel_body_times, bag_vx_body, bag_vy_body, bag_vz_body = [[] for _ in range(4)]
    bag_log_times, bag_log_px_ref, bag_log_py_ref, bag_log_pz_ref = [[] for _ in range(4)]
    bag_log_vx_ref, bag_log_vy_ref, bag_log_vz_ref = [[] for _ in range(3)]
    bag_log_q_ref_w, bag_log_q_ref_x, bag_log_q_ref_y, bag_log_q_ref_z = [[] for _ in range(4)]
    
    # Lists for RPY data
    bag_actual_roll, bag_actual_pitch, bag_actual_yaw = [], [], []
    bag_log_roll_ref, bag_log_pitch_ref, bag_log_yaw_ref = [], [], []

    first_timestamp_ns = -1

    print("Reading rosbag data...")
    while reader.has_next():
        (topic, data, timestamp_ns) = reader.read_next()
        if first_timestamp_ns == -1:
            first_timestamp_ns = timestamp_ns
        
        ros_time_sec = timestamp_ns / 1e9

        if topic == args.pose_topic:
            msg = deserialize_message(data, PoseStampedMsg)
            bag_pose_times.append(ros_time_sec)
            bag_px.append(msg.pose.position.x)
            bag_py.append(msg.pose.position.y)
            bag_pz.append(msg.pose.position.z)
            bag_qw.append(msg.pose.orientation.w)
            bag_qx.append(msg.pose.orientation.x)
            bag_qy.append(msg.pose.orientation.y)
            bag_qz.append(msg.pose.orientation.z)
            # Convert actual orientation to RPY (degrees)
            roll, pitch, yaw = quaternion_to_rpy_degrees(
                msg.pose.orientation.w, 
                msg.pose.orientation.x, 
                msg.pose.orientation.y, 
                msg.pose.orientation.z
            )
            bag_actual_roll.append(roll)
            bag_actual_pitch.append(pitch)
            bag_actual_yaw.append(yaw)
        elif topic == args.velocity_topic:
            msg = deserialize_message(data, TwistStampedMsg)
            bag_vel_body_times.append(ros_time_sec)
            bag_vx_body.append(msg.twist.linear.x)
            bag_vy_body.append(msg.twist.linear.y)
            bag_vz_body.append(msg.twist.linear.z)
        elif topic == args.control_log_topic:
            msg = deserialize_message(data, ControllerLogMsg)
            bag_log_times.append(ros_time_sec)
            bag_log_px_ref.append(msg.reference_position.x)
            bag_log_py_ref.append(msg.reference_position.y)
            bag_log_pz_ref.append(msg.reference_position.z)
            bag_log_vx_ref.append(msg.reference_velocity.x)
            bag_log_vy_ref.append(msg.reference_velocity.y)
            bag_log_vz_ref.append(msg.reference_velocity.z)
            bag_log_q_ref_w.append(msg.reference_orientation_desired.w)
            bag_log_q_ref_x.append(msg.reference_orientation_desired.x)
            bag_log_q_ref_y.append(msg.reference_orientation_desired.y)
            bag_log_q_ref_z.append(msg.reference_orientation_desired.z)
            # Append new reference RPY fields (assuming they are in radians in the msg)
            bag_log_roll_ref.append(math.degrees(msg.reference_roll))
            bag_log_pitch_ref.append(math.degrees(msg.reference_pitch))
            bag_log_yaw_ref.append(math.degrees(msg.reference_yaw))

    if first_timestamp_ns == -1 and (len(bag_pose_times) > 0 or len(bag_log_times) > 0):
        all_times_ns_temp = []
        if len(bag_pose_times) > 0: all_times_ns_temp.append(bag_pose_times[0] * 1e9)
        if len(bag_log_times) > 0: all_times_ns_temp.append(bag_log_times[0] * 1e9)
        if len(bag_vel_body_times) > 0: all_times_ns_temp.append(bag_vel_body_times[0] * 1e9)
        if all_times_ns_temp: first_timestamp_ns = min(all_times_ns_temp)
        else: first_timestamp_ns = 0 
    
    first_timestamp_sec = first_timestamp_ns / 1e9
    time_base_label = f'Time since Bag Start (s) (Bag Start ROS Time: {first_timestamp_sec:.2f}s)'
    print(f"Bag data read. {time_base_label}")

    bag_pose_times = np.array(bag_pose_times); bag_px = np.array(bag_px); bag_py = np.array(bag_py); bag_pz = np.array(bag_pz)
    bag_orientations = np.array([bag_qw, bag_qx, bag_qy, bag_qz]).T if bag_qw else np.array([])
    
    bag_vel_body_times = np.array(bag_vel_body_times); bag_vx_body = np.array(bag_vx_body); bag_vy_body = np.array(bag_vy_body); bag_vz_body = np.array(bag_vz_body)
    bag_log_times = np.array(bag_log_times); bag_log_px_ref = np.array(bag_log_px_ref); bag_log_py_ref = np.array(bag_log_py_ref); bag_log_pz_ref = np.array(bag_log_pz_ref)
    bag_log_vx_ref = np.array(bag_log_vx_ref); bag_log_vy_ref = np.array(bag_log_vy_ref); bag_log_vz_ref = np.array(bag_log_vz_ref)
    
    bag_log_q_ref_w = np.array(bag_log_q_ref_w); bag_log_q_ref_x = np.array(bag_log_q_ref_x)
    bag_log_q_ref_y = np.array(bag_log_q_ref_y); bag_log_q_ref_z = np.array(bag_log_q_ref_z)

    bag_actual_roll = np.array(bag_actual_roll); bag_actual_pitch = np.array(bag_actual_pitch); bag_actual_yaw = np.array(bag_actual_yaw)
    bag_log_roll_ref = np.array(bag_log_roll_ref); bag_log_pitch_ref = np.array(bag_log_pitch_ref); bag_log_yaw_ref = np.array(bag_log_yaw_ref)

    if len(bag_pose_times)==0: print("Warning: No pose msgs found.")
    if len(bag_log_times)==0: print("Warning: No control_log msgs found.")

    actual_time_rel = bag_pose_times - first_timestamp_sec if len(bag_pose_times) > 0 else np.array([])
    actual_px_filt = bag_px; actual_py_filt = bag_py; actual_pz_filt = bag_pz
    actual_orientations_filt = bag_orientations
    print(f"  Actual_pose data points: {len(actual_time_rel)}")

    bag_vx_world, bag_vy_world, bag_vz_world = [], [], []
    actual_vel_time_rel = np.array([])
    if len(bag_vel_body_times) > 0 and len(bag_pose_times) > 0:
        temp_vel_times_rel = bag_vel_body_times - first_timestamp_sec
        for i, t_vel_abs in enumerate(bag_vel_body_times):
            pose_idx_original_array = np.argmin(np.abs(bag_pose_times - t_vel_abs))
            q_for_vel = bag_orientations[pose_idx_original_array, :]
            v_body = np.array([bag_vx_body[i], bag_vy_body[i], bag_vz_body[i]])
            R_body_to_world = quaternion_to_rotation_matrix(q_for_vel); v_world = R_body_to_world @ v_body
            bag_vx_world.append(v_world[0]); bag_vy_world.append(v_world[1]); bag_vz_world.append(v_world[2])
        actual_vel_time_rel = temp_vel_times_rel
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
    
    # --- Plotting ---
    print("Plotting results (entire simulation)...")
    fig_3d = plt.figure(figsize=(12, 9)) 
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    if len(actual_px_filt)>0 : ax_3d.plot(actual_px_filt, actual_py_filt, actual_pz_filt, label='Actual Trajectory (Bag)', color='b', alpha=0.9, linewidth=1.5)
    if len(cmd_ref_px_filt)>0 : ax_3d.plot(cmd_ref_px_filt, cmd_ref_py_filt, cmd_ref_pz_filt, label='Commanded Ref (Log)', linestyle='--', color='g', alpha=0.9, linewidth=1.5)
    if len(actual_px_filt)>0 : ax_3d.scatter(actual_px_filt[0], actual_py_filt[0], actual_pz_filt[0], c='blue', marker='o', s=80, label='Actual Start', depthshade=False, zorder=5)
    if len(actual_px_filt)>0 : ax_3d.scatter(actual_px_filt[-1], actual_py_filt[-1], actual_pz_filt[-1], c='cyan', marker='s', s=80, label='Actual End', depthshade=False, zorder=5)
    ax_3d.set_xlabel('X Position (m)'); ax_3d.set_ylabel('Y Position (m)'); ax_3d.set_zlabel('Z Position (m)')
    ax_3d.set_title('3D Trajectory Comparison (Entire Simulation)', fontsize=16); ax_3d.legend(); ax_3d.grid(True)
    if len(actual_px_filt)>0 or len(cmd_ref_px_filt)>0: 
        all_x_plot = []; all_y_plot = []; all_z_plot = []
        if len(actual_px_filt) > 0: all_x_plot.append(actual_px_filt); all_y_plot.append(actual_py_filt); all_z_plot.append(actual_pz_filt)
        if len(cmd_ref_px_filt) > 0: all_x_plot.append(cmd_ref_px_filt); all_y_plot.append(cmd_ref_py_filt); all_z_plot.append(cmd_ref_pz_filt)
        if all_x_plot: 
            all_x_concat = np.concatenate(all_x_plot); all_y_concat = np.concatenate(all_y_plot); all_z_concat = np.concatenate(all_z_plot)
            if len(all_x_concat)>0 and len(all_y_concat)>0 and len(all_z_concat)>0: 
                max_range = np.array([all_x_concat.max()-all_x_concat.min(), all_y_concat.max()-all_y_concat.min(), all_z_concat.max()-all_z_concat.min()]).max() / 1.8
                if max_range < 0.1: max_range = 1.0 
                mid_x = (all_x_concat.max()+all_x_concat.min())*0.5; mid_y = (all_y_concat.max()+all_y_concat.min())*0.5; mid_z = (all_z_concat.max()+all_z_concat.min())*0.5
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

    print("Plotting Quaternion Comparison...")
    plot_quaternion_comparison(actual_time_rel, actual_orientations_filt, 
                               cmd_ref_time_rel, cmd_ref_orientations_filt, 
                               args.bag_file, time_base_label)
    
    print("Plotting RPY Comparison...")
    plot_rpy_comparison(
        actual_time_rel,
        bag_actual_roll, bag_actual_pitch, bag_actual_yaw,
        cmd_ref_time_rel,
        bag_log_roll_ref, bag_log_pitch_ref, bag_log_yaw_ref,
        args.bag_file, time_base_label
    )
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trajectory tracking performance from a rosbag for the entire simulation duration.")
    parser.add_argument('--bag_file', type=str, required=True, help='Path to the rosbag directory.')
    parser.add_argument('--pose_topic', type=str, default='/mavros/local_position/pose', help='Topic for vehicle pose.')
    parser.add_argument('--velocity_topic', type=str, default='/mavros/local_position/velocity_body', help='Topic for vehicle velocity (body frame).')
    parser.add_argument('--control_log_topic', type=str, default='/mlac_mission_node/control_log', help='Topic for controller log.')
    parser.add_argument('--storage_id', type=str, default='sqlite3', help='Rosbag storage ID.')
    cli_args = parser.parse_args()
    main(cli_args)
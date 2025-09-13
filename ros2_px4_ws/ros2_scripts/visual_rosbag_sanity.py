#!/usr/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
import math

# Conditional import for rosbag2_py
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
except ImportError:
    print("Failed to import rosbag2_py. Please ensure ROS 2 environment is sourced.")
    exit(1)

# Message type imports
from geometry_msgs.msg import PoseStamped as PoseStampedMsg
from mavros_msgs.msg import AttitudeTarget as AttitudeTargetMsg

def quaternion_to_rpy_degrees(q_w, q_x, q_y, q_z):
    """
    Convert a quaternion (w, x, y, z) into euler angles (roll, pitch, yaw) in degrees.
    """
    # Roll
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch
    sinp = 2 * (q_w * q_y - q_z * q_x)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
    # Yaw
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def plot_rpy_comparison(actual_time_rel, actual_roll, actual_pitch, actual_yaw,
                        setpoint_time_rel, setpoint_roll, setpoint_pitch, setpoint_yaw,
                        bag_file_name, time_base_label):
    """Plots the Roll, Pitch, and Yaw comparison."""
    fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'RPY Comparison vs. Time (Degrees)\nBag: {os.path.basename(bag_file_name)}', fontsize=16)

    # Plot Roll
    axs[0].plot(actual_time_rel, actual_roll, label='Actual Roll (/mavros/attitude)')
    axs[0].plot(setpoint_time_rel, setpoint_roll, label='Setpoint Roll (/mavros/setpoint_raw/attitude)', linestyle='--')
    axs[0].set_ylabel('Roll (degrees)')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Pitch
    axs[1].plot(actual_time_rel, actual_pitch, label='Actual Pitch (/mavros/attitude)')
    axs[1].plot(setpoint_time_rel, setpoint_pitch, label='Setpoint Pitch (/mavros/setpoint_raw/attitude)', linestyle='--')
    axs[1].set_ylabel('Pitch (degrees)')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Yaw
    axs[2].plot(actual_time_rel, actual_yaw, label='Actual Yaw (/mavros/attitude)')
    axs[2].plot(setpoint_time_rel, setpoint_yaw, label='Setpoint Yaw (/mavros/setpoint_raw/attitude)', linestyle='--')
    axs[2].set_ylabel('Yaw (degrees)')
    axs[2].legend()
    axs[2].grid(True)
    
    axs[2].set_xlabel(time_base_label)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def main(args):
    if not os.path.exists(args.bag_file):
        print(f"Error: Rosbag not found: {args.bag_file}")
        return

    # --- Setup rosbag reader ---
    storage_options = StorageOptions(uri=args.bag_file, storage_id=args.storage_id)
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening rosbag: {e}")
        return

    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    if args.actual_attitude_topic not in topic_types or args.setpoint_topic not in topic_types:
        print(f"Error: One or both specified topics not found in the rosbag.")
        print(f"Available topics: {list(topic_types.keys())}")
        return

    # --- Prepare lists to hold data ---
    bag_actual_times, bag_actual_roll, bag_actual_pitch, bag_actual_yaw = [], [], [], []
    bag_setpoint_times, bag_setpoint_roll, bag_setpoint_pitch, bag_setpoint_yaw = [], [], [], []
    first_timestamp_ns = -1

    # --- Read data from the bag ---
    print("Reading rosbag data...")
    while reader.has_next():
        (topic, data, timestamp_ns) = reader.read_next()
        if first_timestamp_ns == -1:
            first_timestamp_ns = timestamp_ns
        
        ros_time_sec = timestamp_ns / 1e9

        if topic == args.actual_attitude_topic:
            msg = deserialize_message(data, PoseStampedMsg)
            bag_actual_times.append(ros_time_sec)
            roll, pitch, yaw = quaternion_to_rpy_degrees(
                msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z
            )
            bag_actual_roll.append(roll)
            bag_actual_pitch.append(pitch)
            bag_actual_yaw.append(yaw)

        elif topic == args.setpoint_topic:
            msg = deserialize_message(data, PoseStampedMsg)
            bag_setpoint_times.append(ros_time_sec)
            roll, pitch, yaw = quaternion_to_rpy_degrees(
                msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z
            )
            bag_setpoint_roll.append(roll)
            bag_setpoint_pitch.append(pitch)
            bag_setpoint_yaw.append(yaw)

    if first_timestamp_ns == -1:
        print("No messages found on the specified topics. Exiting.")
        return

    # --- Convert to numpy arrays and align time ---
    first_timestamp_sec = first_timestamp_ns / 1e9
    actual_time_rel = np.array(bag_actual_times) - first_timestamp_sec
    setpoint_time_rel = np.array(bag_setpoint_times) - first_timestamp_sec
    
    # --- Plot the data ---
    print("Plotting RPY comparison...")
    time_base_label = f'Time since Bag Start (s)'
    plot_rpy_comparison(
        actual_time_rel, np.array(bag_actual_roll), np.array(bag_actual_pitch), np.array(bag_actual_yaw),
        setpoint_time_rel, np.array(bag_setpoint_roll), np.array(bag_setpoint_pitch), np.array(bag_setpoint_yaw),
        args.bag_file, time_base_label
    )
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot RPY comparison from a rosbag.")
    parser.add_argument('--bag_file', type=str, required=True, help='Path to the rosbag directory.')
    parser.add_argument('--actual_attitude_topic', type=str, default='/mavros/local_position/pose', help='Topic for actual vehicle attitude.')
    parser.add_argument('--setpoint_topic', type=str, default='/desired_attitude', help='Topic for attitude setpoints.')
    parser.add_argument('--storage_id', type=str, default='sqlite3', help='Rosbag storage ID.')
    cli_args = parser.parse_args()
    main(cli_args)
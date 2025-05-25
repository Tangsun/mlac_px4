#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import argparse
import os

def load_attitude_target_data(bag_file_path, topic_name):
    """
    Loads AttitudeTarget messages from a specific topic in a rosbag file.
    (Function remains the same as before)
    """
    if not os.path.exists(bag_file_path):
        print(f"Error: Bag file directory not found: {bag_file_path}")
        return None, None, None, None, None

    try:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_file_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        msg_type_str = None
        for tt in topic_types:
            if tt.name == topic_name:
                msg_type_str = tt.type
                break
        
        if not msg_type_str:
            print(f"Error: Topic '{topic_name}' not found in bag '{bag_file_path}'.")
            return None, None, None, None, None

        if "mavros_msgs/msg/AttitudeTarget" not in msg_type_str:
            print(f"Error: Topic '{topic_name}' in bag '{bag_file_path}' is of type '{msg_type_str}', "
                  f"but expected 'mavros_msgs/msg/AttitudeTarget'.")
            return None, None, None, None, None

        MessageClass = get_message(msg_type_str)

        times_list = []
        orientations_list = {'w': [], 'x': [], 'y': [], 'z': []}
        body_rates_list = {'x': [], 'y': [], 'z': []}
        thrusts_list = []
        first_type_mask = None
        
        storage_filter = rosbag2_py.StorageFilter(topics=[topic_name])
        reader.set_filter(storage_filter)

        print(f"Reading '{topic_name}' from '{bag_file_path}'...")
        count = 0
        while reader.has_next():
            (topic, data, timestamp_ns) = reader.read_next() 
            if topic == topic_name:
                msg = deserialize_message(data, MessageClass)
                time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                times_list.append(time_sec)
                
                orientations_list['w'].append(msg.orientation.w)
                orientations_list['x'].append(msg.orientation.x)
                orientations_list['y'].append(msg.orientation.y)
                orientations_list['z'].append(msg.orientation.z)
                
                body_rates_list['x'].append(msg.body_rate.x)
                body_rates_list['y'].append(msg.body_rate.y)
                body_rates_list['z'].append(msg.body_rate.z)
                
                thrusts_list.append(msg.thrust)

                if first_type_mask is None:
                    first_type_mask = msg.type_mask
                count +=1
        
        print(f"Read {count} messages.")
        if not times_list:
            print(f"No messages read from topic '{topic_name}' in '{bag_file_path}'.")
            return None, None, None, None, None

        times_np = np.array(times_list)
        if len(times_np) > 0:
             times_np = times_np - times_np[0] 
        
        orientations_np = {k: np.array(v) for k, v in orientations_list.items()}
        body_rates_np = {k: np.array(v) for k, v in body_rates_list.items()}
        thrusts_np = np.array(thrusts_list)

        return times_np, orientations_np, body_rates_np, thrusts_np, first_type_mask

    except Exception as e:
        print(f"Error processing bag file '{bag_file_path}': {e}")
        return None, None, None, None, None

def plot_attitude_comparison(data1, label1, data2, label2, data3, label3, title_prefix=""):
    """
    Plots comparison of AttitudeTarget components (orientation, body rates, thrust) for up to three datasets.
    data1, data2, data3 are tuples: (times, orientations, body_rates, thrusts)
    Pass (None, None, None, None) for dataX if not available.
    """
    times1, orientations1, body_rates1, thrusts1 = data1
    times2, orientations2, body_rates2, thrusts2 = data2
    times3, orientations3, body_rates3, thrusts3 = data3


    # --- Orientation Plot ---
    fig_orient, axs_orient = plt.subplots(4, 1, figsize=(14, 12), sharex=True) # Increased figsize slightly
    fig_orient.suptitle(f"{title_prefix}Orientation (Quaternion) Comparison", fontsize=16)
    
    q_components = ['w', 'x', 'y', 'z']
    # Define distinct colors for components if needed, or use default cycling
    # component_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Matplotlib default

    for i, comp in enumerate(q_components):
        if times1 is not None and orientations1 is not None and comp in orientations1:
            axs_orient[i].plot(times1, orientations1[comp], label=f'{label1} q{comp}', linestyle='-', alpha=0.9)
        if times2 is not None and orientations2 is not None and comp in orientations2:
            axs_orient[i].plot(times2, orientations2[comp], label=f'{label2} q{comp}', linestyle='--', alpha=0.9)
        if times3 is not None and orientations3 is not None and comp in orientations3:
            axs_orient[i].plot(times3, orientations3[comp], label=f'{label3} q{comp}', linestyle=':', alpha=0.9)
        
        axs_orient[i].set_ylabel(f'q{comp}')
        axs_orient[i].legend(loc='upper right', fontsize='small')
        axs_orient[i].grid(True)
    axs_orient[-1].set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Body Rates Plot ---
    fig_rates, axs_rates = plt.subplots(3, 1, figsize=(14, 10), sharex=True) # Increased figsize slightly
    fig_rates.suptitle(f"{title_prefix}Body Rates Comparison", fontsize=16)
    
    rate_components = ['x', 'y', 'z']
    rate_labels = ['Roll Rate (rad/s)', 'Pitch Rate (rad/s)', 'Yaw Rate (rad/s)']

    for i, comp in enumerate(rate_components):
        if times1 is not None and body_rates1 is not None and comp in body_rates1:
            axs_rates[i].plot(times1, body_rates1[comp], label=f'{label1} rate_{comp}', linestyle='-', alpha=0.9)
        if times2 is not None and body_rates2 is not None and comp in body_rates2:
            axs_rates[i].plot(times2, body_rates2[comp], label=f'{label2} rate_{comp}', linestyle='--', alpha=0.9)
        if times3 is not None and body_rates3 is not None and comp in body_rates3:
            axs_rates[i].plot(times3, body_rates3[comp], label=f'{label3} rate_{comp}', linestyle=':', alpha=0.9)

        axs_rates[i].set_ylabel(rate_labels[i])
        axs_rates[i].legend(loc='upper right', fontsize='small')
        axs_rates[i].grid(True)
    axs_rates[-1].set_xlabel('Time (s)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Thrust Plot ---
    fig_thrust, ax_thrust = plt.subplots(1, 1, figsize=(14, 6)) # Increased figsize slightly
    fig_thrust.suptitle(f"{title_prefix}Thrust Comparison", fontsize=16)

    if times1 is not None and thrusts1 is not None:
        ax_thrust.plot(times1, thrusts1, label=f'{label1} Thrust', linestyle='-', color='black', alpha=0.9)
    if times2 is not None and thrusts2 is not None:
        ax_thrust.plot(times2, thrusts2, label=f'{label2} Thrust', linestyle='--', color='dimgray', alpha=0.9)
    if times3 is not None and thrusts3 is not None:
        ax_thrust.plot(times3, thrusts3, label=f'{label3} Thrust', linestyle=':', color='darkgray', alpha=0.9)
        
    ax_thrust.set_xlabel('Time (s)')
    ax_thrust.set_ylabel('Thrust (0.0-1.0)')
    ax_thrust.legend(loc='upper right', fontsize='small')
    ax_thrust.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def main():
    parser = argparse.ArgumentParser(description="Compare AttitudeTarget commands from ROS bags.")
    parser.add_argument("bag1_path", type=str, help="Path to the first rosbag directory (e.g., original commands).")
    parser.add_argument("--topic1", type=str, default='/mavros/setpoint_raw/target_attitude', 
                        help="AttitudeTarget topic in Bag 1 (default: /mavros/setpoint_raw/target_attitude).")
    parser.add_argument("bag2_path", type=str, help="Path to the second rosbag directory (e.g., replay recording).")
    parser.add_argument("--topic2", type=str, default='/mavros/setpoint_raw/attitude', 
                        help="AttitudeTarget topic in Bag 2 for replayed commands sent to MAVROS (default: /mavros/setpoint_raw/attitude).")
    
    # Hardcoded topic for the MAVROS feedback from Bag 2
    bag2_target_attitude_topic = "/mavros/setpoint_raw/target_attitude" 
    
    args = parser.parse_args()

    print(f"--- Loading Data for Bag 1: {args.bag1_path}, Topic: {args.topic1} ---")
    times1, orientations1, body_rates1, thrusts1, type_mask1 = load_attitude_target_data(args.bag1_path, args.topic1)
    
    print(f"\n--- Loading Data for Bag 2 (Replayed Command): {args.bag2_path}, Topic: {args.topic2} ---")
    times2, orientations2, body_rates2, thrusts2, type_mask2 = load_attitude_target_data(args.bag2_path, args.topic2)

    print(f"\n--- Loading Data for Bag 2 (MAVROS Target Feedback): {args.bag2_path}, Topic: {bag2_target_attitude_topic} ---")
    times3, orientations3, body_rates3, thrusts3, type_mask3 = load_attitude_target_data(args.bag2_path, bag2_target_attitude_topic)


    if type_mask1 is not None:
        print(f"\nType mask from Bag 1, '{args.topic1}': {type_mask1} (binary: {type_mask1:08b})")
    if type_mask2 is not None:
        print(f"Type mask from Bag 2, '{args.topic2}': {type_mask2} (binary: {type_mask2:08b})")
    if type_mask3 is not None:
        print(f"Type mask from Bag 2, '{bag2_target_attitude_topic}': {type_mask3} (binary: {type_mask3:08b})")
    
    # Check for type mask consistency
    if type_mask1 is not None and type_mask2 is not None and type_mask1 != type_mask2:
        print(f"WARNING: Type mask mismatch between Bag 1 ('{args.topic1}') and Bag 2 ('{args.topic2}').")
    if type_mask2 is not None and type_mask3 is not None and type_mask2 != type_mask3:
        print(f"WARNING: Type mask mismatch between Bag 2 replayed command ('{args.topic2}') and MAVROS target feedback ('{bag2_target_attitude_topic}').")


    plots_generated = False
    data1_valid = times1 is not None # Check only time as other components might be empty but data is still "validly loaded"
    data2_valid = times2 is not None
    data3_valid = times3 is not None

    # Prepare data tuples, passing None for components if the entire dataset is invalid
    data_tuple1 = (times1, orientations1, body_rates1, thrusts1) if data1_valid else (None, None, None, None)
    data_tuple2 = (times2, orientations2, body_rates2, thrusts2) if data2_valid else (None, None, None, None)
    data_tuple3 = (times3, orientations3, body_rates3, thrusts3) if data3_valid else (None, None, None, None)
        
    # Plot only if at least one primary dataset is valid
    if data1_valid or data2_valid or data3_valid:
        plot_attitude_comparison(
            data_tuple1, f"Bag1: {os.path.basename(args.bag1_path)} ({args.topic1})",
            data_tuple2, f"Bag2: {os.path.basename(args.bag2_path)} ({args.topic2})",
            data_tuple3, f"Bag2: {os.path.basename(args.bag2_path)} ({bag2_target_attitude_topic})"
        )
        plots_generated = True
    else:
        print("\nCould not load sufficient data from any source. Skipping plots.")

    if plots_generated:
        print("\nDisplaying plots...")
        plt.show()
    else:
        print("\nNo plots were generated due to missing data.")

if __name__ == '__main__':
    main()

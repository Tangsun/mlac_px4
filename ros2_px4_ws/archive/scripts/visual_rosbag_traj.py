#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Keep this for 3D plotting functionality
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import argparse
import os

def load_pose_data_from_bag(bag_file_path, topic_name):
    """
    Loads PoseStamped messages from a specific topic in a rosbag file.
    (This function remains the same as the last corrected version)
    """
    if not os.path.exists(bag_file_path):
        print(f"Error: Bag file directory not found: {bag_file_path}")
        return None, None, None, None

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
            return None, None, None, None

        if "geometry_msgs/msg/PoseStamped" not in msg_type_str:
            print(f"Error: Topic '{topic_name}' in bag '{bag_file_path}' is of type '{msg_type_str}', "
                  f"but expected 'geometry_msgs/msg/PoseStamped'.")
            return None, None, None, None

        MessageClass = get_message(msg_type_str)

        times_list = []
        positions_list = {'x': [], 'y': [], 'z': []}

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
                positions_list['x'].append(msg.pose.position.x)
                positions_list['y'].append(msg.pose.position.y)
                positions_list['z'].append(msg.pose.position.z)
                count +=1
        
        print(f"Read {count} messages.")
        if not times_list:
            print(f"No messages read from topic '{topic_name}' in '{bag_file_path}'.")
            return None, None, None, None

        times = np.array(times_list)
        if len(times) == 0:
            return None, None, None, None
            
        times = times - times[0] 
        positions = {k: np.array(v) for k, v in positions_list.items()}

        velocities = {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}
        vel_times = np.array([])

        if len(times) > 1:
            dt = np.diff(times)
            vel_times = times[1:] 

            for axis in ['x', 'y', 'z']:
                if len(positions[axis]) > 1:
                    d_pos = np.diff(positions[axis])
                    vel_axis = np.zeros_like(d_pos, dtype=float)
                    non_zero_dt_mask = dt != 0
                    if np.any(non_zero_dt_mask): # Proceed if there's at least one non-zero dt
                        vel_axis[non_zero_dt_mask] = d_pos[non_zero_dt_mask] / dt[non_zero_dt_mask]
                    # Fill in velocities for zero dt (e.g., propagate previous velocity or set to 0)
                    # Here, they remain 0 if dt was 0 and d_pos was also 0, or become inf/nan if d_pos was non-zero
                    # A more robust way would be to handle dt=0 specifically, e.g., by forward filling
                    velocities[axis] = vel_axis
                else: 
                    velocities[axis] = np.zeros(len(vel_times)) # Should match vel_times length
        else:
            print(f"Warning: Not enough data points (only {len(times)}) to calculate velocity for {bag_file_path}")
            if len(times) == 1: 
                 vel_times = np.array([]) # vel_times should be empty if no diff possible
            for axis in ['x', 'y', 'z']:
                velocities[axis] = np.array([]) # Velocities should also be empty

        return times, positions, velocities, vel_times

    except Exception as e:
        print(f"Error processing bag file '{bag_file_path}': {e}")
        return None, None, None, None


def plot_3d_trajectories(pos1, label1, pos2, label2, title="3D Trajectory Comparison"):
    """Plots two 3D trajectories. (Function remains the same as the last corrected version)"""
    fig = plt.figure(figsize=(10, 8))
    try:
        ax = fig.add_subplot(111, projection='3d')
    except Exception as e:
        print(f"Failed to create 3D subplot. Ensure Matplotlib 3D support is correctly installed. Error: {e}")
        plt.close(fig)
        return

    if pos1 and pos1['x'].size > 0 : ax.plot(pos1['x'], pos1['y'], pos1['z'], label=label1, color='r', linestyle='-')
    if pos2 and pos2['x'].size > 0 : ax.plot(pos2['x'], pos2['y'], pos2['z'], label=label2, color='b', linestyle='--')
    
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Combine data for scaling only if both are present and non-empty
    all_x_list, all_y_list, all_z_list = [], [], []
    if pos1 and pos1['x'].size > 0:
        all_x_list.append(pos1['x'])
        all_y_list.append(pos1['y'])
        all_z_list.append(pos1['z'])
    if pos2 and pos2['x'].size > 0:
        all_x_list.append(pos2['x'])
        all_y_list.append(pos2['y'])
        all_z_list.append(pos2['z'])

    if not all_x_list: # If no valid data was added
        print("Warning: No valid position data for 3D plot scaling.")
        return

    all_x = np.concatenate(all_x_list)
    all_y = np.concatenate(all_y_list)
    all_z = np.concatenate(all_z_list)
    
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    if max_range == 0: max_range = 1.0 

    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_position_components_comparison(times1, pos1, label1, times2, pos2, label2):
    """Plots x, y, z position components from two trajectories against time for comparison."""
    if (times1 is None or pos1 is None) and (times2 is None or pos2 is None):
        print("Skipping position component plot: no data for either trajectory.")
        return

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Position Components Comparison vs. Time", fontsize=16)
    components = ['x', 'y', 'z']
    ylabels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']

    for i, comp in enumerate(components):
        if times1 is not None and pos1 is not None and comp in pos1 and pos1[comp].size > 0:
            axs[i].plot(times1, pos1[comp], label=f'{label1} - {comp.upper()}', linestyle='-', color='blue')
        if times2 is not None and pos2 is not None and comp in pos2 and pos2[comp].size > 0:
            axs[i].plot(times2, pos2[comp], label=f'{label2} - {comp.upper()}', linestyle='--', color='red')
        axs[i].set_ylabel(ylabels[i])
        axs[i].legend(loc='upper right')
        axs[i].grid(True)
    
    axs[-1].set_xlabel('Time (s) (normalized for each bag)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def plot_velocity_components_comparison(vel_times1, vel1, label1, vel_times2, vel2, label2):
    """Plots x_dot, y_dot, z_dot velocity components from two trajectories against time for comparison."""
    if (vel_times1 is None or vel1 is None) and (vel_times2 is None or vel2 is None):
        print("Skipping velocity component plot: no data for either trajectory.")
        return
        
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Velocity Components Comparison vs. Time", fontsize=16)
    components = ['x', 'y', 'z']
    ylabels = ['X Velocity (m/s)', 'Y Velocity (m/s)', 'Z Velocity (m/s)']

    for i, comp in enumerate(components):
        # Check if velocity data for this component exists and matches its time vector length
        plot1_valid = (vel_times1 is not None and vel1 is not None and comp in vel1 and 
                       vel1[comp].size > 0 and vel1[comp].size == vel_times1.size)
        plot2_valid = (vel_times2 is not None and vel2 is not None and comp in vel2 and 
                       vel2[comp].size > 0 and vel2[comp].size == vel_times2.size)

        if plot1_valid:
            axs[i].plot(vel_times1, vel1[comp], label=f'{label1} - {comp.upper()}_dot', linestyle='-', color='blue')
        if plot2_valid:
            axs[i].plot(vel_times2, vel2[comp], label=f'{label2} - {comp.upper()}_dot', linestyle='--', color='red')
        
        if plot1_valid or plot2_valid: # Only add legend if something was plotted
            axs[i].legend(loc='upper right')
        axs[i].set_ylabel(ylabels[i])
        axs[i].grid(True)
    
    axs[-1].set_xlabel('Time (s) (normalized for each bag, velocity time vector)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def main():
    parser = argparse.ArgumentParser(description="Compare two ROS bag trajectories from /mavros/local_position/pose.")
    parser.add_argument("bag1_path", type=str, help="Path to the first rosbag directory (e.g., original).")
    parser.add_argument("bag2_path", type=str, help="Path to the second rosbag directory (e.g., replay).")
    parser.add_argument("--topic", type=str, default="/mavros/local_position/pose",
                        help="Topic name for PoseStamped messages (default: /mavros/local_position/pose).")
    args = parser.parse_args()

    print("Loading data for Original Trajectory...")
    times1, pos1, vel1, vel_times1 = load_pose_data_from_bag(args.bag1_path, args.topic)
    
    print("\nLoading data for Replay Trajectory...")
    times2, pos2, vel2, vel_times2 = load_pose_data_from_bag(args.bag2_path, args.topic)

    plots_generated = False

    # Plot 1: 3D Trajectory Comparison
    if pos1 is not None and pos2 is not None: # Ensure both position datasets are loaded
        plot_3d_trajectories(pos1, "Original Trajectory", pos2, "Replay Trajectory")
        plots_generated = True
    else:
        print("Skipping 3D comparison plot due to missing position data for one or both trajectories.")

    # Plot 2: Position Components Comparison (x, y, z vs. time)
    # Only plot if at least one trajectory's position data is valid
    if (times1 is not None and pos1 is not None) or (times2 is not None and pos2 is not None):
        plot_position_components_comparison(times1, pos1, "Original", times2, pos2, "Replay")
        plots_generated = True
    else:
        print("Skipping position component comparison plot due to missing data.")
        
    # Plot 3: Velocity Components Comparison (x_dot, y_dot, z_dot vs. time)
    # Only plot if at least one trajectory's velocity data is valid
    if (vel_times1 is not None and vel1 is not None) or (vel_times2 is not None and vel2 is not None):
        plot_velocity_components_comparison(vel_times1, vel1, "Original", vel_times2, vel2, "Replay")
        plots_generated = True
    else:
        print("Skipping velocity component comparison plot due to missing data.")

    if plots_generated:
        print("\nDisplaying plots...")
        plt.show()
    else:
        print("\nNo data loaded successfully or insufficient data for plotting. No plots to display.")

if __name__ == '__main__':
    main()

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

    Args:
        bag_file_path (str): Path to the rosbag2 directory.
        topic_name (str): The topic to read PoseStamped messages from.

    Returns:
        tuple: (times, positions, velocities, vel_times)
               times (np.array): Timestamps in seconds.
               positions (dict): {'x': np.array, 'y': np.array, 'z': np.array}
               velocities (dict): {'x': np.array, 'y': np.array, 'z': np.array}
               vel_times (np.array): Timestamps for velocity data.
               Returns (None, None, None, None) if loading fails or topic not found.
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
        if len(times) == 0: # Should be caught by 'not times_list' but good to double check
            return None, None, None, None
            
        times = times - times[0] 
        positions = {k: np.array(v) for k, v in positions_list.items()}

        velocities = {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}
        vel_times = np.array([])

        if len(times) > 1:
            dt = np.diff(times)
            vel_times = times[1:] # Time vector for velocities

            for axis in ['x', 'y', 'z']:
                if len(positions[axis]) > 1:
                    d_pos = np.diff(positions[axis])
                    # Initialize velocity array for this axis
                    vel_axis = np.zeros_like(d_pos, dtype=float)
                    # Create a mask for non-zero dt to avoid division by zero
                    non_zero_dt_mask = dt != 0
                    # Calculate velocity only where dt is non-zero
                    vel_axis[non_zero_dt_mask] = d_pos[non_zero_dt_mask] / dt[non_zero_dt_mask]
                    # For dt == 0, vel_axis remains 0 (or you could choose to propagate previous, or NaN)
                    velocities[axis] = vel_axis
                else: # Should not happen if len(times) > 1
                    velocities[axis] = np.zeros(len(vel_times))
        else:
            print(f"Warning: Not enough data points (only {len(times)}) to calculate velocity for {bag_file_path}")
            # Ensure vel_times and velocities have consistent empty shapes if no velocity can be calculated
            if len(times) == 1: # if there's one point, diff makes it 0 length
                 vel_times = np.array([])
            for axis in ['x', 'y', 'z']:
                velocities[axis] = np.array([])


        return times, positions, velocities, vel_times

    except Exception as e:
        print(f"Error processing bag file '{bag_file_path}': {e}")
        return None, None, None, None


def plot_3d_trajectories(pos1, label1, pos2, label2, title="3D Trajectory Comparison"):
    """Plots two 3D trajectories."""
    # This function relies on Axes3D being available.
    # If the initial matplotlib import issue was only partially fixed, 3D plotting might still fail.
    fig = plt.figure(figsize=(10, 8))
    try:
        ax = fig.add_subplot(111, projection='3d')
    except Exception as e:
        print(f"Failed to create 3D subplot. Ensure Matplotlib 3D support is correctly installed. Error: {e}")
        plt.close(fig) # Close the figure if 3D projection fails
        return

    ax.plot(pos1['x'], pos1['y'], pos1['z'], label=label1, color='r')
    ax.plot(pos2['x'], pos2['y'], pos2['z'], label=label2, color='b')
    
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    all_x = np.concatenate((pos1['x'], pos2['x']))
    all_y = np.concatenate((pos1['y'], pos2['y']))
    all_z = np.concatenate((pos1['z'], pos2['z']))
    if all_x.size == 0 or all_y.size == 0 or all_z.size == 0: # Check if data is empty
        print("Warning: Empty position data for 3D plot scaling.")
        return

    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    if max_range == 0: max_range = 1.0 # Avoid zero range if all points are identical

    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_trajectory_details(times, positions, vel_times, velocities, trajectory_name):
    """Plots position vs. time and velocity vs. time for a single trajectory."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(f"Trajectory Details: {trajectory_name}", fontsize=16)

    axs[0].plot(times, positions['x'], label='X position', color='dodgerblue')
    axs[0].plot(times, positions['y'], label='Y position', color='darkorange')
    axs[0].plot(times, positions['z'], label='Z position', color='forestgreen')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position (m)')
    axs[0].set_title('Position vs. Time')
    axs[0].legend()
    axs[0].grid(True)

    if vel_times is not None and len(vel_times) > 0 and \
       velocities['x'].size == vel_times.size and \
       velocities['y'].size == vel_times.size and \
       velocities['z'].size == vel_times.size:
        axs[1].plot(vel_times, velocities['x'], label='X velocity (x_dot)', color='dodgerblue')
        axs[1].plot(vel_times, velocities['y'], label='Y velocity (y_dot)', color='darkorange')
        axs[1].plot(vel_times, velocities['z'], label='Z velocity (z_dot)', color='forestgreen')
        axs[1].set_xlabel('Time (s)') 
        axs[1].set_ylabel('Velocity (m/s)')
        axs[1].set_title('Velocity vs. Time')
        axs[1].legend()
        axs[1].grid(True)
    else:
        axs[1].text(0.5, 0.5, 'Velocity data not available, insufficient time points, or mismatched lengths.', 
                    horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].set_title('Velocity vs. Time')

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

    # Flag to track if any plot is generated
    plots_generated = False

    # Check if data for 3D plot is valid
    if pos1 is not None and pos2 is not None:
        plot_3d_trajectories(pos1, "Original Trajectory", pos2, "Replay Trajectory")
        plots_generated = True
    else:
        print("Could not load sufficient data for one or both trajectories. Skipping 3D comparison plot.")

    # Check if data for original trajectory details is valid
    if times1 is not None and pos1 is not None and vel1 is not None and vel_times1 is not None:
        plot_trajectory_details(times1, pos1, vel_times1, vel1, "Original Trajectory")
        plots_generated = True
    else:
        print("Could not load full data for Original Trajectory. Skipping detail plot.")
        
    # Check if data for replay trajectory details is valid
    if times2 is not None and pos2 is not None and vel2 is not None and vel_times2 is not None:
        plot_trajectory_details(times2, pos2, vel_times2, vel2, "Replay Trajectory")
        plots_generated = True
    else:
        print("Could not load full data for Replay Trajectory. Skipping detail plot.")

    if plots_generated:
        print("\nDisplaying plots...")
        plt.show()
    else:
        print("\nNo data loaded successfully or insufficient data for plotting. No plots to display.")

if __name__ == '__main__':
    main()

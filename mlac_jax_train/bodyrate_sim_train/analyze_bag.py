#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- ROS 2 Imports ---
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
except ImportError:
    print("Failed to import rosbag2_py. Please ensure your ROS 2 environment is sourced.")
    sys.exit(1)

def analyze_topic_frequency(rosbag_path, topic_name):
    """
    Analyzes the timestamps of a specific topic in a rosbag to determine its
    publishing frequency and consistency.

    Args:
        rosbag_path (str): Path to the rosbag directory.
        topic_name (str): The specific topic to analyze.
    """
    if not os.path.exists(rosbag_path):
        print(f"Error: Rosbag directory not found at '{rosbag_path}'")
        return

    # --- Setup Rosbag Reader ---
    storage_options = StorageOptions(uri=rosbag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    
    reader = SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening rosbag: {e}")
        return

    # Filter to read only the specified topic
    topic_filter = StorageFilter(topics=[topic_name])
    reader.set_filter(topic_filter)

    if not reader.has_next():
        print(f"Error: Topic '{topic_name}' not found or has no messages in the rosbag.")
        return

    # --- Extract Timestamps ---
    print(f"Reading timestamps for topic: {topic_name}...")
    timestamps_ns = []
    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()
        if topic == topic_name:
            timestamps_ns.append(t_ns)

    if len(timestamps_ns) < 2:
        print("Not enough messages to analyze frequency (found {} messages).".format(len(timestamps_ns)))
        return

    timestamps_s = np.array(timestamps_ns) / 1e9  # Convert nanoseconds to seconds
    
    # --- Calculate Statistics ---
    time_deltas = np.diff(timestamps_s)
    mean_delta = np.mean(time_deltas)
    std_delta = np.std(time_deltas)
    min_delta = np.min(time_deltas)
    max_delta = np.max(time_deltas)
    
    avg_freq = 1.0 / mean_delta if mean_delta > 0 else 0

    print("\n--- Frequency Analysis Results ---")
    print(f"Topic:                 {topic_name}")
    print(f"Number of Messages:    {len(timestamps_s)}")
    print(f"Total Duration (s):    {timestamps_s[-1] - timestamps_s[0]:.3f}")
    print(f"Average Frequency (Hz):{avg_freq:.2f}")
    print("------------------------------------")
    print(f"Time Deltas (s):")
    print(f"  - Mean:              {mean_delta:.4f}")
    print(f"  - Std Dev:           {std_delta:.4f}")
    print(f"  - Min:               {min_delta:.4f}")
    print(f"  - Max:               {max_delta:.4f}")
    print("------------------------------------\n")

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.hist(time_deltas * 1000, bins=50, edgecolor='black')
    plt.title(f'Histogram of Message Time Deltas for\n{topic_name}')
    plt.xlabel('Time Between Messages (milliseconds)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.axvline(mean_delta * 1000, color='r', linestyle='--', linewidth=2, label=f'Mean Delta: {mean_delta*1000:.2f} ms')
    plt.legend()
    
    print("Displaying plot of time delta distribution...")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze the message frequency of a topic in a ROS 2 rosbag.")
    parser.add_argument('rosbag', type=str, help="Path to the input rosbag directory.")
    parser.add_argument('topic', type=str, help="The full name of the topic to analyze (e.g., '/mlac_mission_node/control_log').")
    
    args = parser.parse_args()

    analyze_topic_frequency(args.rosbag, args.topic)
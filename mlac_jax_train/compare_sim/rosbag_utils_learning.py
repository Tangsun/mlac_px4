#!/usr/bin/env python3

"""
Utilities for reading rosbag2 files and producing aligned pose/control data
for simulation comparisons.
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# --- PLUMBING: Imports and Helpers ---
try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
except ImportError as exc:
    raise RuntimeError("rosbag_utils requires ROS 2 python packages.") from exc

def _lvh_forward(arr, invalid_mask):
    """Last-Value-Held helper."""
    arr = np.asarray(arr, dtype=float)
    valid = ~np.asarray(invalid_mask, dtype=bool)
    for i in range(arr.shape[0]):
        if not valid[i]:
            if i == 0:
                j = np.argmax(valid)
                arr[i] = arr[j] if valid[j] else 0.0
            else:
                arr[i] = arr[i - 1]
    return arr

def _get_rosbag_options(path, storage_id='sqlite3'):
    storage_options = StorageOptions(uri=path, storage_id=storage_id)
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )
    return storage_options, converter_options

# --- YOUR FOCUS AREA ---

def extract_attitude_data(
    rosbag_path,
    pose_topic="/mavros/local_position/pose",
    velocity_topic="/mavros/local_position/velocity_body",
    control_log_topic="/mlac_mission_node/control_log",
    att_setpoint_topic="/mavros/setpoint_raw/attitude",
):
    """
    TODO: Implement the reader logic here.
    
    1. Initialize the SequentialReader with StorageOptions (uri=rosbag_path, storage_id='sqlite3').
    2. Check if topics exist.
    3. Get message types using `get_message`.
    4. Loop through messages using `reader.read_next()`.
    5. Return the dictionary of numpy arrays.
    """
    # 1. Set up the SequentialReader
    reader = SequentialReader()
    storage_options, converter_options = _get_rosbag_options(rosbag_path)
    reader.open(storage_options, converter_options)
    
    # 2. CHECK TOPICS 
    # (The reader is now open and ready to inspect topics)
    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    required_topics = [pose_topic, velocity_topic, control_log_topic, att_setpoint_topic]
    for topic in required_topics:
        if topic not in topic_types:
            raise ValueError(f"Required topic '{topic}' not found in rosbag.")
        
    PoseMsg = get_message(topic_types[pose_topic])
    VelocityMsg = get_message(topic_types[velocity_topic])
    ControllerLogMsg = get_message(topic_types[control_log_topic])
    AttitudeTargetMsg = get_message(topic_types[att_setpoint_topic])

    def header_to_ns(header):
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
    
    # First pass: Find start and end times from scanning the control logs
    traj_exec_start_ns = None
    traj_exec_end_ns = None
    
    reader.set_filter(StorageFilter(topics=[control_log_topic]))
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, ControllerLogMsg)
        if traj_exec_start_ns == -1 and msg.trajectory_execution_start_ros_time.sec > 0:
            traj_exec_start_ns = (
                msg.trajectory_execution_start_ros_time.sec * 1e9
                + msg.trajectory_execution_start_ros_time.nanosec
            )
        if msg.trajectory_execution_end_ros_time.sec > 0:
            traj_exec_end_ns = (
                msg.trajectory_execution_end_ros_time.sec * 1e9
                + msg.trajectory_execution_end_ros_time.nanosec
            )
            break

    reader.set_filter(StorageFilter(topics=required_topics))
    reader.seek(0)

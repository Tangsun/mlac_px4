#!/usr/bin/env python3

import rclpy
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message

from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

import argparse
import os
import sys

# Import necessary message types (adjust if needed)
from mavros_msgs.msg import State

def get_rosbag_options(path, storage_id='sqlite3'):
    """Helper function to create storage and converter options."""
    storage_options = StorageOptions(uri=path, storage_id=storage_id)
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr')
    return storage_options, converter_options

def filter_bag_for_offboard(input_bag_dir: str, output_bag_dir: str):
    """
    Reads an input rosbag, identifies the first continuous time segment where
    the drone state (/mavros/state) is ARMED and in OFFBOARD mode, and writes
    all messages from all topics within that time segment to a new output bag.
    """
    print(f"Reading from: {input_bag_dir}")
    print(f"Writing filtered bag to: {output_bag_dir}")

    input_storage_options, input_converter_options = get_rosbag_options(input_bag_dir)
    output_storage_options, output_converter_options = get_rosbag_options(output_bag_dir)

    reader = SequentialReader()
    reader.open(input_storage_options, input_converter_options)

    writer = SequentialWriter()
    writer.open(output_storage_options, output_converter_options)

    # Copy topic metadata from input to output
    topic_types = reader.get_all_topics_and_types()
    type_map = {}
    for topic_metadata in topic_types:
        # Check if topic type is known locally before creating it
        try:
            get_message(topic_metadata.type) # Check if type is available
            type_map[topic_metadata.name] = topic_metadata.type
            writer.create_topic(topic_metadata)
            print(f"Registered topic: {topic_metadata.name} ({topic_metadata.type})")
        except ModuleNotFoundError:
            print(f"Warning: Message type '{topic_metadata.type}' for topic '{topic_metadata.name}' not found. Skipping topic.", file=sys.stderr)
        except Exception as e:
             print(f"Warning: Could not get message type for {topic_metadata.name} ({topic_metadata.type}): {e}. Skipping topic.", file=sys.stderr)


    # --- Find the Offboard Time Window ---
    offboard_start_time_ns = -1
    offboard_end_time_ns = -1
    in_offboard_segment = False
    print("Scanning bag to find Offboard time window...")

    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()
        if topic == '/mavros/state' and topic in type_map:
            try:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)

                is_offboard_and_armed = msg.armed and msg.mode == 'OFFBOARD'

                if is_offboard_and_armed and not in_offboard_segment:
                    # Start of the first segment
                    offboard_start_time_ns = t_ns
                    in_offboard_segment = True
                    print(f"Found Offboard start at t={t_ns} ns")
                elif not is_offboard_and_armed and in_offboard_segment:
                    # End of the segment
                    offboard_end_time_ns = t_ns
                    in_offboard_segment = False # Stop after the first segment ends
                    print(f"Found Offboard end at t={t_ns} ns")
                    break # Stop scanning once the first segment ends
            except Exception as e:
                print(f"Error deserializing /mavros/state message: {e}", file=sys.stderr)

    if offboard_start_time_ns == -1:
        print("Error: Did not find any ARMED+OFFBOARD state in the input bag.", file=sys.stderr)
        return

    if offboard_end_time_ns == -1:
        # If the bag ended while still in offboard mode
        print("Warning: Bag ended while still in Offboard mode. Using end of bag as end time.")
        # We need to know the actual end time of the bag, which requires another read or info.
        # For simplicity, we'll process until the end if no end time was found.

    # --- Reset reader and write the filtered data ---
    # Re-open the reader to start from the beginning again
    print("Rewinding bag and writing filtered data...")
    reader.seek(0) # Seek back to the start timestamp

    message_count = 0
    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()

        # Check if the message timestamp falls within the determined window
        is_within_window = False
        if offboard_start_time_ns != -1 and t_ns >= offboard_start_time_ns:
            if offboard_end_time_ns == -1 or t_ns <= offboard_end_time_ns:
                is_within_window = True

        if is_within_window and topic in type_map: # Only write topics we could register
            try:
                # No need to deserialize again, just write raw data
                writer.write(topic, data, t_ns)
                message_count += 1
            except Exception as e:
                 print(f"Error writing message for topic {topic}: {e}", file=sys.stderr)


    print(f"Finished writing filtered bag.")
    print(f"Offboard window: Start={offboard_start_time_ns} ns, End={offboard_end_time_ns if offboard_end_time_ns != -1 else 'End of Bag'} ns")
    print(f"Wrote {message_count} messages to {output_bag_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter ROS2 bag to keep only the first continuous ARMED+OFFBOARD segment.')
    parser.add_argument('input_bag', help='Path to the input bag directory')
    parser.add_argument('output_bag', help='Path for the output filtered bag directory')
    args = parser.parse_args()

    if not os.path.isdir(args.input_bag):
        print(f"Error: Input directory '{args.input_bag}' not found or not a directory.", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(args.output_bag):
        print(f"Error: Output directory '{args.output_bag}' already exists. Please remove it or choose a different name.", file=sys.stderr)
        sys.exit(1)

    # Ensure rclpy is initialized if message types need it implicitly (shouldn't for get_message)
    # rclpy.init()
    try:
        filter_bag_for_offboard(args.input_bag, args.output_bag)
    finally:
        # if rclpy.ok():
        #     rclpy.shutdown()
        pass


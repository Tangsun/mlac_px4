#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.serialization import deserialize_message
from rcl_interfaces.msg import ParameterDescriptor

from mavros_msgs.msg import AttitudeTarget # Make sure this is imported
from std_msgs.msg import Header 

import rosbag2_py 
from rosidl_runtime_py.utilities import get_message 
import os

class AttitudeReplayNode(Node):
    """
    A ROS 2 node to read AttitudeTarget messages directly from a rosbag,
    modify them to ignore body rates, and republish them to
    /mavros/setpoint_raw/attitude with fresh timestamps.
    """
    def __init__(self):
        super().__init__('attitude_replay_node')

        # --- Parameters ---
        self.declare_parameter(
            'bag_file_path', 
            '', 
            ParameterDescriptor(description='Absolute path to the rosbag2 file (directory). MANDATORY.')
        )
        self.declare_parameter(
            'recorded_topic_name', 
            '/mavros/setpoint_raw/target_attitude', 
            ParameterDescriptor(description='Topic name in the bag containing AttitudeTarget messages.')
        )
        self.declare_parameter(
            'publish_topic_name', 
            '/mavros/setpoint_raw/attitude', 
            ParameterDescriptor(description='Topic name to publish AttitudeTarget messages to.')
        )
        self.declare_parameter(
            'replay_rate_hz', 
            50.0, # Defaulting to 10Hz as per previous findings
            ParameterDescriptor(description='Rate at which to publish messages from the bag.')
        )
        self.declare_parameter(
            'loop_replay', 
            False,
            ParameterDescriptor(description='Whether to loop the replay when all messages are published.')
        )

        # Get parameter values
        self.bag_file_path = self.get_parameter('bag_file_path').get_parameter_value().string_value
        self.recorded_topic_name = self.get_parameter('recorded_topic_name').get_parameter_value().string_value
        self.publish_topic_name = self.get_parameter('publish_topic_name').get_parameter_value().string_value
        self.replay_rate = self.get_parameter('replay_rate_hz').get_parameter_value().double_value
        self.loop_replay = self.get_parameter('loop_replay').get_parameter_value().bool_value

        if not self.bag_file_path:
            self.get_logger().fatal("CRITICAL: Parameter 'bag_file_path' is not set. Shutting down.")
            rclpy.try_shutdown() 
            return 

        qos_profile_setpoint = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.attitude_publisher = self.create_publisher(
            AttitudeTarget,
            self.publish_topic_name,
            qos_profile_setpoint
        )

        self.messages_to_replay = []
        self.load_messages_from_bag() 

        if not self.messages_to_replay:
            self.get_logger().warn(f"No messages found for topic '{self.recorded_topic_name}' in '{self.bag_file_path}'. Node will not publish.")
            return

        self.current_message_index = 0
        if self.replay_rate <= 0:
            self.get_logger().warn("Replay rate is zero or negative, defaulting to 1 Hz.")
            self.replay_rate = 1.0
        self.timer_period = 1.0 / self.replay_rate
        self.replay_timer = self.create_timer(self.timer_period, self.publish_next_message_callback)

        self.get_logger().info("Attitude Replay Node (Direct Bag Reader, Ignoring Body Rates) Initialized.")
        self.get_logger().info(f"Reading from bag: '{self.bag_file_path}'")
        self.get_logger().info(f"Targeting recorded topic: '{self.recorded_topic_name}'")
        self.get_logger().info(f"Publishing to: '{self.publish_topic_name}' at {self.replay_rate} Hz with MODIFIED type_mask.")
        self.get_logger().info(f"Looping replay: {self.loop_replay}")

    def load_messages_from_bag(self):
        """
        Loads AttitudeTarget messages from the specified topic in the rosbag file.
        (This function remains largely the same as before)
        """
        try:
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=self.bag_file_path, storage_id='sqlite3')
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format='cdr', 
                output_serialization_format='cdr'
            )
            reader.open(storage_options, converter_options)

            topic_filter = rosbag2_py.StorageFilter(topics=[self.recorded_topic_name])
            reader.set_filter(topic_filter)

            topic_types = reader.get_all_topics_and_types()
            msg_type_str = None
            for topic_type in topic_types:
                if topic_type.name == self.recorded_topic_name:
                    msg_type_str = topic_type.type
                    break
            
            if not msg_type_str:
                self.get_logger().error(f"Topic '{self.recorded_topic_name}' not found in bag '{self.bag_file_path}'.")
                return
            
            if "mavros_msgs/msg/AttitudeTarget" not in msg_type_str:
                 self.get_logger().warn(
                    f"Topic '{self.recorded_topic_name}' is of type '{msg_type_str}', "
                    f"but expected 'mavros_msgs/msg/AttitudeTarget'."
                )
            
            MessageClass = get_message(msg_type_str)
            if MessageClass != AttitudeTarget: # Ensure we are working with AttitudeTarget
                 self.get_logger().error(f"Message class for topic '{self.recorded_topic_name}' is not AttitudeTarget.")
                 return


            self.get_logger().info(f"Reading messages from topic '{self.recorded_topic_name}' of type '{msg_type_str}'...")
            while reader.has_next():
                (topic, data, timestamp_ns) = reader.read_next()
                if topic == self.recorded_topic_name:
                    try:
                        msg = deserialize_message(data, MessageClass)
                        self.messages_to_replay.append(msg)
                    except Exception as e:
                        self.get_logger().error(f"Failed to deserialize message from topic '{topic}': {e}")
            
            self.get_logger().info(f"Loaded {len(self.messages_to_replay)} messages from '{self.recorded_topic_name}'.")

        except Exception as e:
            self.get_logger().error(f"Error while reading rosbag '{self.bag_file_path}': {e}")

    def publish_next_message_callback(self):
        """
        Called by the timer to publish the next message from the loaded list.
        MODIFIES the type_mask to ignore body rates.
        """
        if not self.messages_to_replay:
            if self.replay_timer and self.replay_timer.is_active():
                self.replay_timer.cancel()
            return

        if self.current_message_index < len(self.messages_to_replay):
            # It's good practice to work on a copy if the original message from the list
            # might be needed later in its pristine state, though for this replay, modifying is okay.
            # from copy import deepcopy
            # msg_to_publish = deepcopy(self.messages_to_replay[self.current_message_index])
            msg_to_publish = self.messages_to_replay[self.current_message_index]
            
            # *** KEY MODIFICATION: Set type_mask to ignore body rates ***
            # This tells PX4 to only use orientation (quaternion) and thrust.
            # PX4 will then determine the necessary body rates itself.
            msg_to_publish.type_mask = (
                AttitudeTarget.IGNORE_ROLL_RATE |
                AttitudeTarget.IGNORE_PITCH_RATE |
                AttitudeTarget.IGNORE_YAW_RATE
            ) # This evaluates to 1 | 2 | 4 = 7

            # Optionally, zero out the body rates in the message for clarity,
            # though PX4 should ignore them based on the type_mask.
            msg_to_publish.body_rate.x = 0.0
            msg_to_publish.body_rate.y = 0.0
            msg_to_publish.body_rate.z = 0.0
            
            # Update the header timestamp to current time for MAVROS
            msg_to_publish.header.stamp = self.get_clock().now().to_msg()
            
            self.attitude_publisher.publish(msg_to_publish)
            # self.get_logger().info(f"Published msg {self.current_message_index} with type_mask={msg_to_publish.type_mask}", throttle_duration_sec=1.0)
            self.current_message_index += 1
        else:
            if self.loop_replay:
                self.get_logger().info("All messages replayed. Looping...")
                self.current_message_index = 0 
            else:
                self.get_logger().info("All messages have been replayed. Stopping publisher.")
                if self.replay_timer and self.replay_timer.is_active(): 
                    self.replay_timer.cancel()

    def destroy_node(self):
        if hasattr(self, 'replay_timer') and self.replay_timer and self.replay_timer.is_active():
            self.replay_timer.cancel()
        self.get_logger().info("Attitude Replay Node (Direct Bag Reader, Ignoring Body Rates) shutting down.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    replay_node = None 
    try:
        replay_node = AttitudeReplayNode()
        if replay_node.bag_file_path and rclpy.ok(): 
            if not replay_node.messages_to_replay and not replay_node.loop_replay:
                 replay_node.get_logger().info("No messages loaded and not looping. Node will exit after setup if timer doesn't start.")
            rclpy.spin(replay_node)
        elif not replay_node.bag_file_path:
            print("Node initialization failed due to missing bag_file_path. Exiting.")

    except KeyboardInterrupt:
        print("Ctrl+C detected, shutting down replay node...")
    except Exception as e:
        if replay_node:
            replay_node.get_logger().fatal(f"Unhandled exception: {e}")
        else:
            print(f"Unhandled exception during node creation: {e}")
    finally:
        if replay_node and rclpy.ok(): 
            replay_node.destroy_node()
        if rclpy.ok(): 
            rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from mavros_msgs.msg import State # To check connection status

class MinimalSetpointPublisher(Node):

    def __init__(self):

        super().__init__('minimal_setpoint_publisher')

        # --- Parameters ---
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_z', 1.0) # Default target: 1m hover above origin
        self.declare_parameter('coordinate_frame', 'map') # Or 'odom', check your setup

        publish_rate = self.get_parameter('publish_rate_hz').value
        self.target_pose = PoseStamped()
        self.target_pose.pose.position.x = self.get_parameter('target_x').value
        self.target_pose.pose.position.y = self.get_parameter('target_y').value
        self.target_pose.pose.position.z = self.get_parameter('target_z').value
        # Set orientation to level flight (no roll/pitch), zero yaw
        self.target_pose.pose.orientation.w = 1.0
        self.target_pose.pose.orientation.x = 0.0
        self.target_pose.pose.orientation.y = 0.0
        self.target_pose.pose.orientation.z = 0.0
        self.coordinate_frame = self.get_parameter('coordinate_frame').value

        # --- QoS Profiles ---
        # Use reliable for state, best effort for high-rate setpoints/pose
        qos_profile_state = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_profile_setpoint = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        # --- MAVROS State Subscriber ---
        # We subscribe just to know when MAVROS is connected before starting
        self.state_sub = self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            qos_profile_state)

        # --- Setpoint Publisher ---
        self.setpoint_pub = self.create_publisher(
            PoseStamped,
            '/mavros/setpoint_position/local',
            qos_profile_setpoint) # Use compatible QoS

        # --- Node State ---
        self.mavros_connected = False
        self.publish_timer = None # Initialize timer variable

        self.get_logger().info("Minimal Setpoint Publisher Initialized.")
        self.get_logger().info(f"Target Pose: X={self.target_pose.pose.position.x:.2f}, "
                               f"Y={self.target_pose.pose.position.y:.2f}, "
                               f"Z={self.target_pose.pose.position.z:.2f} "
                               f"in '{self.coordinate_frame}' frame.")
        self.get_logger().info(f"Publishing rate: {publish_rate} Hz")

        # Store the rate for the timer
        self.timer_period = 1.0 / publish_rate

    def state_callback(self, msg):
        """Check MAVROS connection status and start publishing if connected."""
        if not self.mavros_connected and msg.connected:
            self.get_logger().info("MAVROS connected! Starting setpoint publishing.")
            self.mavros_connected = True
            # Start the publishing timer ONLY after MAVROS is connected
            if self.publish_timer is None:
                 self.publish_timer = self.create_timer(self.timer_period, self.publish_setpoint_loop)

        elif self.mavros_connected and not msg.connected:
             self.get_logger().warn("MAVROS disconnected! Stopping setpoint publishing.")
             self.mavros_connected = False
             # Stop the timer if MAVROS disconnects
             if self.publish_timer is not None:
                  self.publish_timer.cancel()
                  self.publish_timer = None # Reset timer variable


    def publish_setpoint_loop(self):
        """Publish the fixed target setpoint."""
        if not self.mavros_connected:
            # Should not happen if timer is managed correctly, but safety check
            self.get_logger().warn("Publish loop called but MAVROS not connected.", throttle_duration_sec=5.0)
            return

        # Update header timestamp and frame_id just before publishing
        self.target_pose.header = Header() # Create new header each time
        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.target_pose.header.frame_id = self.coordinate_frame

        self.setpoint_pub.publish(self.target_pose)
        # self.get_logger().info("Publishing setpoint", throttle_duration_sec=5.0) # Optional debug log

    def destroy_node(self):
        """Cleanup resources."""
        if self.publish_timer is not None:
            self.publish_timer.cancel()
        self.get_logger().info("Shutting down Minimal Setpoint Publisher.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        minimal_setpoint_publisher = MinimalSetpointPublisher()
        rclpy.spin(minimal_setpoint_publisher)
    except KeyboardInterrupt:
        print("Ctrl+C detected, shutting down...")
    finally:
        # Cleanup node resources
        if 'minimal_setpoint_publisher' in locals() and rclpy.ok():
             minimal_setpoint_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

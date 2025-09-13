#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time, Duration
from enum import Enum

from mavros_msgs.msg import AttitudeTarget, State as MavrosState
from geometry_msgs.msg import Quaternion, PoseStamped 
from std_msgs.msg import Header 
import numpy as np
import math
import traceback

from mlac_sim.bodyrate_conversion import BodyRateConverter
from mlac_sim.bodyrate_LQR import ConstantPositionTracker

"""
Note that a chunk of this code is similar to the `yaw_rotation_test.py` script.
This is to be changed to use body rate control to check Sunbochen's method.
           - Kai (09/03/2025)
"""

# --- Constants for State Machine ---
class MissionState(Enum):
    """Enumerates the different phases of the mission."""
    WAITING_FOR_OFFBOARD = 1
    POSITION_HOLD_ASCEND = 2
    HOVER_CALIBRATION = 3
    ROTATING_YAW = 4
    MISSION_COMPLETE = 5

# --- Utility Functions ---
def euler_from_quaternion(w, x, y, z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert euler angles to a quaternion.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return Quaternion(w=qw, x=qx, y=qy, z=qz)

class BodyRateSanityCheckNode(Node):
    def __init__(self):
        super().__init__('body_rate_sanity_check_node')

        self.declare_parameter('publish_rate_hz', 50.0)
        # self.declare_parameter('hover_thrust', 0.728) 
        self.declare_parameter('yaw_rate_dps', 15.0) 
        self.declare_parameter('rotation_duration_sec', 60.0)
        self.declare_parameter('initial_setpoint_x', 0.0) 
        self.declare_parameter('initial_setpoint_y', 0.0) 
        self.declare_parameter('initial_setpoint_z', 2.0) 
        self.declare_parameter('hover_calib_duration_sec', 8.0) 
        self.declare_parameter('position_tolerance_m', 0.2) 

        # PD control parameters
        self.declare_parameter('kp_z', 0.5) # PD control proportional gain
        self.declare_parameter('kd_z', 0.2) # PD control derivative gain
        self.declare_parameter('hover_thrust', 0.5) # Initial hover thrust

        # Proportional gain for body rate controller
        self.declare_parameter('bodyrate_kp', 0.5)

        self.publish_rate = self.get_parameter('publish_rate_hz').value
        self.hover_thrust = self.get_parameter('hover_thrust').value
        self.yaw_rate_rps = math.radians(self.get_parameter('yaw_rate_dps').value)
        self.rotation_duration_sec = self.get_parameter('rotation_duration_sec').value
        self.initial_setpoint_x = self.get_parameter('initial_setpoint_x').value
        self.initial_setpoint_y = self.get_parameter('initial_setpoint_y').value
        self.initial_setpoint_z = self.get_parameter('initial_setpoint_z').value
        self.hover_calib_duration_sec = self.get_parameter('hover_calib_duration_sec').value
        self.position_tolerance_m = self.get_parameter('position_tolerance_m').value

        self.kp_z = self.get_parameter('kp_z').value
        self.kd_z = self.get_parameter('kd_z').value

        self.bodyrate_kp = self.get_parameter('bodyrate_kp').value
        self.bodyrate_converter = BodyRateConverter(kp=self.bodyrate_kp)

        if self.publish_rate <= 0:
            self.get_logger().fatal("publish_rate_hz must be positive.")
            rclpy.try_shutdown(); return
        
        self.timer_period = 1.0 / self.publish_rate

        qos_profile_state = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_profile_setpoint = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.state_sub = self.create_subscription(MavrosState, '/mavros/state', self.mavros_state_callback, qos_profile_state)
        self.local_pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback, qos_profile_setpoint)

        # Publisher for Attitude Control (for the rotation phase)
        self.attitude_setpoint_pub = self.create_publisher(AttitudeTarget, '/mavros/setpoint_raw/attitude', qos_profile_setpoint)
        # NEW: Publisher for Position Control (for the ascent and hover phase)
        self.position_setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_profile_setpoint)
        # NEW: Publisher for desired attitude
        self.desired_attitude_pub = self.create_publisher(PoseStamped, '/desired_attitude', qos_profile_setpoint)

        self.current_mavros_state = MavrosState()
        self.current_local_pose = None 
        self.got_first_local_pose = False

        self.is_offboard_and_armed = False
        self.mission_start_time = None 
        self.hover_calib_start_time = None
        self.current_yaw_angle = 0.0 
        self.setpoint_streaming_active = False 

        # PD Controller variables
        self.last_error_z = 0.0

        # --- Mission State Machine Variables ---
        self.mission_state = MissionState.WAITING_FOR_OFFBOARD

        self.publish_timer = self.create_timer(self.timer_period, self.publish_setpoint_callback)
        
        self.get_logger().info("Body Rate Sanity Check Node Initialized.")

    def local_pose_callback(self, msg: PoseStamped):
        self.current_local_pose = msg
        if not self.got_first_local_pose:
            self.got_first_local_pose = True
            if self.current_mavros_state.connected and not self.setpoint_streaming_active:
                self.get_logger().info("MAVROS connected and pose received. Starting position setpoint streaming.")
                self.setpoint_streaming_active = True 

    def mavros_state_callback(self, msg: MavrosState):
        if not msg.connected and self.current_mavros_state.connected:
            self.get_logger().warn("MAVROS disconnected. Stopping setpoint streaming.")
            self.is_offboard_and_armed = False
            self.mission_start_time = None
            self.hover_calib_start_time = None
            self.setpoint_streaming_active = False 

        self.current_mavros_state = msg
        
        if self.current_mavros_state.connected and self.got_first_local_pose and not self.setpoint_streaming_active:
            self.get_logger().info("MAVROS connected and pose available. Starting initial setpoint streaming.")
            self.setpoint_streaming_active = True

        is_now_offboard_and_armed = msg.connected and msg.armed and msg.mode == "OFFBOARD"

        if is_now_offboard_and_armed and not self.is_offboard_and_armed:
            self.get_logger().info("Switched to OFFBOARD mode and ARMED. Starting ascent to initial position.")
            self.mission_state = MissionState.POSITION_HOLD_ASCEND
        
        self.is_offboard_and_armed = is_now_offboard_and_armed

    def publish_setpoint_callback(self):
        now = self.get_clock().now()

        # Step 1: Pre-offboard logic (before we are armed & in offboard mode)
        if not self.is_offboard_and_armed:
            if not self.current_mavros_state.connected or not self.setpoint_streaming_active:
                self.get_logger().warn("Waiting for connection and first pose...", throttle_duration_sec=5.0)
                return

            pos_msg = PoseStamped()
            pos_msg.header = Header(stamp=now.to_msg(), frame_id="map")
            pos_msg.pose.position.x = 0.0
            pos_msg.pose.position.y = 0.0
            pos_msg.pose.position.z = 0.0
            pos_msg.pose.orientation = euler_to_quaternion(0.0, 0.0, 0.0)
            self.position_setpoint_pub.publish(pos_msg)
            self.get_logger().info("STATE: WAITING_FOR_OFFBOARD - Streaming dummy position setpoint to enable OFFBOARD.", throttle_duration_sec=2.0)
            return

        # Step 2: Main mission logic after offboard mode is enabled
        if self.mission_state == MissionState.POSITION_HOLD_ASCEND:
            pos_msg = PoseStamped()
            pos_msg.header = Header(stamp=now.to_msg(), frame_id="map")
            pos_msg.pose.position.x = self.initial_setpoint_x
            pos_msg.pose.position.y = self.initial_setpoint_y
            pos_msg.pose.position.z = self.initial_setpoint_z
            pos_msg.pose.orientation = euler_to_quaternion(0.0, 0.0, 0.0)
            self.position_setpoint_pub.publish(pos_msg)
            self.get_logger().info(f"STATE: POSITION_HOLD_ASCEND - Sending position setpoint ({self.initial_setpoint_x}, {self.initial_setpoint_y}, {self.initial_setpoint_z})", throttle_duration_sec=2.0)
            
            if self.current_local_pose:
                current_pos = self.current_local_pose.pose.position
                dist_to_target = math.sqrt(
                    (current_pos.x - self.initial_setpoint_x)**2 +
                    (current_pos.y - self.initial_setpoint_y)**2 +
                    (current_pos.z - self.initial_setpoint_z)**2
                )
                if dist_to_target < self.position_tolerance_m:
                    self.mission_state = MissionState.HOVER_CALIBRATION
                    self.hover_calib_start_time = now
                    self.get_logger().info(f"STATE: HOVER_CALIBRATION - Reached position. Starting {self.hover_calib_duration_sec}s hover.")
                    
        elif self.mission_state == MissionState.HOVER_CALIBRATION:
            elapsed_hover_time = (now - self.hover_calib_start_time).nanoseconds / 1e9
            
            # Continue sending position setpoints to hold hover
            pos_msg = PoseStamped()
            pos_msg.header = Header(stamp=now.to_msg(), frame_id="map")
            pos_msg.pose.position.x = self.initial_setpoint_x
            pos_msg.pose.position.y = self.initial_setpoint_y
            pos_msg.pose.position.z = self.initial_setpoint_z
            pos_msg.pose.orientation = euler_to_quaternion(0.0, 0.0, 0.0)
            self.position_setpoint_pub.publish(pos_msg)
            
            self.get_logger().info(f"STATE: HOVER_CALIBRATION - Hovering... Elapsed: {elapsed_hover_time:.1f}s", throttle_duration_sec=2.0)
            
            if elapsed_hover_time > self.hover_calib_duration_sec:
                self.mission_state = MissionState.ROTATING_YAW
                self.mission_start_time = now
                self.get_logger().info(f"STATE: ROTATING_YAW - Hover complete. Switching to attitude control and starting yaw rotation.")
                
        elif self.mission_state == MissionState.ROTATING_YAW:
            elapsed_time_since_mission_start = (now - self.mission_start_time).nanoseconds / 1e9
            
            if elapsed_time_since_mission_start <= self.rotation_duration_sec:
                
                # PD control for thrust to maintain hover
                current_z = self.current_local_pose.pose.position.z
                error_z = self.initial_setpoint_z - current_z
                derivative_error_z = (error_z - self.last_error_z) / self.timer_period
                pd_thrust_correction = (self.kp_z * error_z) + (self.kd_z * derivative_error_z)
                final_thrust = self.hover_thrust + pd_thrust_correction
                final_thrust = max(0.0, min(1.0, final_thrust))
                self.last_error_z = error_z

                self.current_yaw_angle += self.yaw_rate_rps * self.timer_period
                target_yaw_this_step = self.current_yaw_angle
                target_yaw_this_step = (target_yaw_this_step + math.pi) % (2 * math.pi) - math.pi
                
                # ------------------------ Current & Target Attitudes ------------------------ #
                target_quat = euler_to_quaternion(0.0, 0.0, target_yaw_this_step)
                current_quat = self.current_local_pose.pose.orientation

                # ------------------------- Desired Attitude Logging ------------------------- #
                desired_att_log_msg = PoseStamped()
                desired_att_log_msg.header.stamp = now.to_msg()
                desired_att_log_msg.header.frame_id = "map"  # Use a consistent frame
                desired_att_log_msg.pose.orientation = target_quat
                # We only care about orientation, so position can be zero
                self.desired_attitude_pub.publish(desired_att_log_msg)

                # ----------- Compute body rate command using the BodyRateConverter ---------- #
                q_current = np.array([current_quat.w, current_quat.x, current_quat.y, current_quat.z])
                q_desired = np.array([target_quat.w, target_quat.x, target_quat.y, target_quat.z])
                body_rate_cmd = self.bodyrate_converter.attitude_to_bodyrate(q_current, q_desired)
                
                # -------------------------- Publish Attitude Target ------------------------- #
                att_msg = AttitudeTarget()
                att_msg.header = Header(stamp=now.to_msg(), frame_id="map") 
                att_msg.thrust = float(final_thrust)

                att_msg.body_rate.x = float(body_rate_cmd[0])
                att_msg.body_rate.y = float(body_rate_cmd[1])
                att_msg.body_rate.z = float(body_rate_cmd[2])
                att_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE
                self.attitude_setpoint_pub.publish(att_msg)
                
                if int(elapsed_time_since_mission_start * 10) % 20 == 0: 
                    self.get_logger().info(f"STATE: ROTATING WITH OMEGA={body_rate_cmd[0], body_rate_cmd[1], body_rate_cmd[2]} | t={elapsed_time_since_mission_start:.1f}s, Cmd Yaw={math.degrees(target_yaw_this_step):.1f}deg, Thrust={final_thrust:.2f}")
            else:
                self.mission_state = MissionState.MISSION_COMPLETE
                self.get_logger().info(f"STATE: MISSION_COMPLETE - Rotation complete. Holding attitude.")
                
        elif self.mission_state == MissionState.MISSION_COMPLETE:
            # Continue to publish the final attitude to hold
            att_msg = AttitudeTarget()
            att_msg.header = Header(stamp=now.to_msg(), frame_id="map") 
            att_msg.thrust = float(self.hover_thrust)
            att_msg.orientation = euler_to_quaternion(0.0, 0.0, self.current_yaw_angle)
            att_msg.body_rate.x = 0.0
            att_msg.body_rate.y = 0.0
            att_msg.body_rate.z = 0.0
            att_msg.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | AttitudeTarget.IGNORE_PITCH_RATE | AttitudeTarget.IGNORE_YAW_RATE
            self.attitude_setpoint_pub.publish(att_msg)
            self.get_logger().info("Mission complete. Holding final attitude.", throttle_duration_sec=5.0)

    def destroy_node(self):
        if self.publish_timer is not None:
            self.publish_timer.cancel()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BodyRateSanityCheckNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info("Ctrl+C detected, shutting down sanity check node.")
    except Exception as e:
        logger = rclpy.logging.get_logger("yaw_rotation_sanity_check_main")
        if node: logger = node.get_logger()
        logger.fatal(f"Unhandled exception: {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from ament_index_python.packages import get_package_share_directory
import numpy as np
import os
import traceback

from mavros_msgs.msg import AttitudeTarget, State as MavrosState
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String as StringMsg
from std_msgs.msg import Bool as BoolMsg

from mlac_msgs.msg import ControllerLog as ControllerLogMsg

from .outerloop_node import OuterLoop # Assuming the fixed version
from .structs import StateClass, GoalClass, ParametersClass, AttCmdClass, ControlLogClass
from .helpers import (quaternion_array_to_msg, vector_array_to_msg,
                       controllog_class_to_ros_msg, get_rpy)
from .utils import quaternion_to_rotation_matrix

class MlacMissionNode(Node):
    def __init__(self):
        super().__init__('mlac_mission_node')

        # Parameter Declarations (as before)
        self.declare_parameter(
            'controller_type', 'pid',
            ParameterDescriptor(description="Controller type: 'pid', 'coml', or 'coml_debug'")
        )
        self.declare_parameter(
            'control_loop_rate_hz', 50.0,
            ParameterDescriptor(description="Rate of the main control loop")
        )
        self.declare_parameter(
            'trajectory_file_name', 'takeoff_hover_land_8col_50hz.npy',
            ParameterDescriptor(description="Name of the .npy trajectory file in 'mlac_sim/traj_data/' folder")
        )
        self.declare_parameter(
            'vehicle_mass', 1.3,
            ParameterDescriptor(description="Vehicle mass (kg)")
        )
        self.declare_parameter(
            'Kp', [7.0, 7.0, 7.0],
            ParameterDescriptor(description="Proportional gains [Px, Py, Pz]")
        )
        self.declare_parameter(
            'Ki', [0.5, 0.5, 0.5],
            ParameterDescriptor(description="Integral gains [Ix, Iy, Iz]")
        )
        self.declare_parameter(
            'Kd', [4.0, 4.0, 4.0],
            ParameterDescriptor(description="Derivative gains [Dx, Dy, Dz]")
        )
        self.declare_parameter(
            'max_pos_err', [0.5, 0.5, 0.5],
            ParameterDescriptor(description="Max position error for PID saturation [err_x, err_y, err_z]")
        )
        self.declare_parameter(
            'max_vel_err', [1.0, 1.0, 1.0],
            ParameterDescriptor(description="Max velocity error for PID saturation [verr_x, verr_y, verr_z]")
        )
        self.declare_parameter(
            'max_thrust_N', 25.0,
            ParameterDescriptor(description="Max thrust capability of the vehicle in Newtons (e.g., mass * g * 2.0)")
        )

        # Get Parameters (as before)
        self.controller_type = self.get_parameter('controller_type').get_parameter_value().string_value
        self.trajectory_file_name = self.get_parameter('trajectory_file_name').get_parameter_value().string_value
        self.max_thrust_N = self.get_parameter('max_thrust_N').get_parameter_value().double_value
        self.control_loop_rate = self.get_parameter('control_loop_rate_hz').get_parameter_value().double_value

        if self.control_loop_rate <= 0:
            self.get_logger().warn("control_loop_rate_hz must be positive, defaulting to 50Hz.")
            self.control_loop_rate = 50.0
        if self.max_thrust_N <= 0:
            self.get_logger().fatal("max_thrust_N must be positive. Controller cannot function.")
            raise ValueError("max_thrust_N must be positive.")

        self.controller_params = ParametersClass()
        self.controller_params.mass = self.get_parameter('vehicle_mass').get_parameter_value().double_value
        self.controller_params.Kp = np.array(self.get_parameter('Kp').get_parameter_value().double_array_value)
        self.controller_params.Ki = np.array(self.get_parameter('Ki').get_parameter_value().double_array_value)
        self.controller_params.Kd = np.array(self.get_parameter('Kd').get_parameter_value().double_array_value)
        self.controller_params.maxPosErr = np.array(self.get_parameter('max_pos_err').get_parameter_value().double_array_value)
        self.controller_params.maxVelErr = np.array(self.get_parameter('max_vel_err').get_parameter_value().double_array_value)

        # Node State Variables
        self.current_vehicle_state_py = StateClass()
        self.current_goal_py = GoalClass() # This will be updated by the control loop
        self.is_vehicle_state_received = False
        self.is_controller_active = False # True if a mission command (HOLD/START_TRAJ) is active
        self.is_trajectory_loaded = False
        self.trajectory_execution_active = False # True if START_TRAJECTORY is active
        self.trajectory_start_time_ros: Time | None = None
        self.loaded_trajectory_data = None
        
        self.was_in_offboard_mode = False # Flag to track if we were in OFFBOARD
        self.was_armed = False # Flag to track if we were armed

        self._load_trajectory()

        self.outer_loop_ctrl = OuterLoop(
            params=self.controller_params,
            state0=self.current_vehicle_state_py, # Initial state (will be updated)
            goal0=self._get_hold_position_goal(),   # Initial goal (hold current, though state isn't known yet)
            controller=self.controller_type,
            package_name='mlac_sim'
        )

        # QoS Profiles and Subscribers/Publishers (as before)
        qos_sensor_data = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_reliable_volatile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_reliable_transientlocal = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.vehicle_pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.vehicle_pose_callback, qos_sensor_data)
        self.vehicle_velocity_sub = self.create_subscription(TwistStamped, '/mavros/local_position/velocity_body', self.vehicle_velocity_callback, qos_sensor_data)
        self.mavros_state_sub = self.create_subscription(MavrosState, '/mavros/state', self.mavros_state_callback, qos_reliable_volatile)
        self.mission_command_sub = self.create_subscription(StringMsg, '/mission_control/command', self.mission_command_callback, qos_reliable_volatile)

        self.attitude_setpoint_pub = self.create_publisher(AttitudeTarget, '/mavros/setpoint_raw/attitude', qos_sensor_data)
        self.controller_log_pub = self.create_publisher(ControllerLogMsg, '~/control_log', qos_reliable_volatile)
        self.trajectory_status_pub = self.create_publisher(BoolMsg, '~/trajectory_complete_status', qos_reliable_transientlocal)

        self.control_timer = self.create_timer(1.0 / self.control_loop_rate, self.control_loop_callback)
        self.get_logger().info(f"MLAC Mission Node ({self.controller_type}) initialized. Trajectory loaded: {self.is_trajectory_loaded}")

    def _load_trajectory(self):
        # (Same as before)
        try:
            package_share_path = get_package_share_directory('mlac_sim')
            traj_path = os.path.join(package_share_path, 'traj_data', self.trajectory_file_name)
            if os.path.exists(traj_path):
                self.loaded_trajectory_data = np.load(traj_path)
                if self.loaded_trajectory_data.ndim != 2 or self.loaded_trajectory_data.shape[0] < 2 or self.loaded_trajectory_data.shape[1] < 8:
                    self.get_logger().error(f"Trajectory file '{traj_path}' has incorrect shape: {self.loaded_trajectory_data.shape}. Needs at least 2 points and 8 columns (t,p,v,psi).")
                    self.loaded_trajectory_data = None
                    self.is_trajectory_loaded = False
                    return
                self.is_trajectory_loaded = True
                self.get_logger().info(f"Trajectory '{self.trajectory_file_name}' loaded successfully ({self.loaded_trajectory_data.shape[0]} points, {self.loaded_trajectory_data.shape[1]} cols).")
            else:
                self.get_logger().error(f"Trajectory file not found: {traj_path}")
                self.is_trajectory_loaded = False
        except Exception as e:
            self.get_logger().error(f"Failed to load trajectory '{self.trajectory_file_name}': {e}\n{traceback.format_exc()}")
            self.is_trajectory_loaded = False

    def mavros_state_callback(self, msg: MavrosState):
        self.get_logger().debug(
            f"MAVROS State CB: Mode='{msg.mode}', Armed={msg.armed} | "
            f"WasArmed={self.was_armed}, WasOffboard={self.was_in_offboard_mode}, "
            f"ControllerActive={self.is_controller_active}", 
            throttle_duration_sec=1.0
        )

        deactivate_controller_due_to_state_change = False
        deactivation_reason = ""

        if self.is_controller_active: # Only consider deactivating if a mission command was active
            if self.was_armed and not msg.armed: # It was armed, but now it's disarmed
                deactivate_controller_due_to_state_change = True
                deactivation_reason = "MAVROS Disarmed while controller was active"
            elif self.was_in_offboard_mode and msg.mode != "OFFBOARD": # It was in OFFBOARD, but now it's not
                deactivate_controller_due_to_state_change = True
                deactivation_reason = f"MAVROS Exited OFFBOARD mode to '{msg.mode}' while controller was active"
        
        if deactivate_controller_due_to_state_change:
            self.get_logger().warn(f"{deactivation_reason}. Deactivating user-commanded control.")
            self.is_controller_active = False         # Stop following specific mission commands
            self.trajectory_execution_active = False  # Stop trajectory following
            # The control_loop will now default to holding current position if vehicle state is available
            
            status_msg = BoolMsg()
            status_msg.data = True # Trajectory (if any) is considered stopped/complete
            self.trajectory_status_pub.publish(status_msg)

        # Update "was" states for the next callback iteration
        self.was_armed = msg.armed
        self.was_in_offboard_mode = (msg.mode == "OFFBOARD")


    def vehicle_pose_callback(self, msg: PoseStamped):
        # (Same as before, with debug log)
        self.current_vehicle_state_py.t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.current_vehicle_state_py.p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.current_vehicle_state_py.q = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        if not self.is_vehicle_state_received:
            self.is_vehicle_state_received = True
            self.get_logger().info("DEBUG: Vehicle state received for the first time (is_vehicle_state_received = True).")


    def vehicle_velocity_callback(self, msg: TwistStamped):
        # (Same as before)
        if not self.is_vehicle_state_received:
            return
        R_body_to_world = quaternion_to_rotation_matrix(self.current_vehicle_state_py.q).T
        self.current_vehicle_state_py.v = R_body_to_world @ np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.current_vehicle_state_py.w = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

    def mission_command_callback(self, msg: StringMsg):
        # (Same as before, with debug logs)
        command = msg.data.upper()
        self.get_logger().info(f"Received mission command: '{command}'")
        status_msg = BoolMsg()

        if command == "START_TRAJECTORY":
            if self.is_trajectory_loaded and self.is_vehicle_state_received:
                self.is_controller_active = True
                self.trajectory_execution_active = True
                self.trajectory_start_time_ros = self.get_clock().now()
                self.get_logger().info("DEBUG: START_TRAJECTORY: is_controller_active=True, trajectory_execution_active=True")
                initial_traj_goal = self._get_trajectory_goal_at_time(self.loaded_trajectory_data[0,0] if self.is_trajectory_loaded else 0.0)
                if initial_traj_goal:
                     self.outer_loop_ctrl.reset(self.current_vehicle_state_py, initial_traj_goal)
                status_msg.data = False
                self.trajectory_status_pub.publish(status_msg)
                self.get_logger().info("Controller activated by mission command. Executing trajectory.")
            else:
                self.get_logger().warn(f"Cannot START_TRAJECTORY: TrajectoryLoaded={self.is_trajectory_loaded}, VehicleStateRx={self.is_vehicle_state_received}.")
        elif command == "HOLD_POSITION":
            self.is_controller_active = True 
            self.get_logger().info("DEBUG: HOLD_POSITION: is_controller_active=True")
            if self.trajectory_execution_active: 
                status_msg.data = True
                self.trajectory_status_pub.publish(status_msg)
            self.trajectory_execution_active = False 
            self.get_logger().info("Controller activated by mission command. Holding current position.")
        elif command == "STOP_CONTROLLER":
            self.is_controller_active = False # This will make control_loop default to holding current pos
            self.get_logger().info("DEBUG: STOP_CONTROLLER: is_controller_active=False. Node will stream current pose if state available.")
            if self.trajectory_execution_active:
                status_msg.data = True
                self.trajectory_status_pub.publish(status_msg)
            self.trajectory_execution_active = False
            self.get_logger().info("Controller DEACTIVATED by mission command (will stream current pose).")
        else:
            self.get_logger().warn(f"Unknown mission command: '{command}'")

    def _get_trajectory_goal_at_time(self, target_time_in_trajectory_timeline: float) -> GoalClass | None:
        # (Same as before)
        if not self.is_trajectory_loaded or self.loaded_trajectory_data is None:
            self.get_logger().warn("_get_trajectory_goal_at_time called but no trajectory loaded.", throttle_duration_sec=5.0)
            return None
        traj_data = self.loaded_trajectory_data
        traj_file_times = traj_data[:, 0]
        clipped_target_time = np.clip(target_time_in_trajectory_timeline, traj_file_times[0], traj_file_times[-1])
        goal = GoalClass()
        goal.t = self.get_clock().now().nanoseconds / 1e9
        goal.p = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(1, 4)])
        goal.v = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(4, 7)])
        goal.psi = float(np.interp(clipped_target_time, traj_file_times, traj_data[:, 7]))
        num_cols = traj_data.shape[1]
        goal.a = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(8, 11)]) if num_cols >= 11 else np.zeros(3)
        goal.j = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(11, 14)]) if num_cols >= 14 else np.zeros(3)
        goal.dpsi = float(np.interp(clipped_target_time, traj_file_times, traj_data[:, 14])) if num_cols >= 15 else 0.0
        goal.mode_xy = GoalClass.Mode.POS_CTRL
        goal.mode_z = GoalClass.Mode.POS_CTRL
        return goal

    def _get_hold_position_goal(self) -> GoalClass:
        # (Same as before, with debug log)
        goal = GoalClass()
        if self.is_vehicle_state_received:
            goal.p = np.copy(self.current_vehicle_state_py.p)
            rpy_vector3 = get_rpy(self.current_vehicle_state_py.q)
            goal.psi = rpy_vector3.z
        else:
            goal.p = np.zeros(3) # Should ideally not happen if control_loop calls this
            goal.psi = 0.0
            self.get_logger().warn("DEBUG: Hold Goal created with zeros (no vehicle state yet when _get_hold_position_goal was called).")
        goal.v = np.zeros(3)
        goal.a = np.zeros(3)
        goal.j = np.zeros(3)
        goal.dpsi = 0.0
        goal.mode_xy = GoalClass.Mode.POS_CTRL
        goal.mode_z = GoalClass.Mode.POS_CTRL
        goal.t = self.get_clock().now().nanoseconds / 1e9
        return goal

    def control_loop_callback(self):
        current_ros_time = self.get_clock().now()
        
        self.get_logger().debug(
            f"Control Loop Tick: ActiveCmd={self.is_controller_active}, " # Renamed for clarity
            f"StateRx={self.is_vehicle_state_received}, "
            f"TrajExec={self.trajectory_execution_active}",
            throttle_duration_sec=1.0 
        )

        if not self.is_vehicle_state_received:
            self.get_logger().debug(
                f"Control Loop: Returning early. No vehicle state received yet.",
                throttle_duration_sec=2.0
            )
            return # Cannot do anything without vehicle state

        # --- Goal Determination ---
        # If a mission command is active (HOLD or START_TRAJECTORY)
        if self.is_controller_active:
            if self.trajectory_execution_active:
                # Trajectory following logic (same as before)
                if not self.is_trajectory_loaded or self.trajectory_start_time_ros is None or self.loaded_trajectory_data is None:
                    self.get_logger().warn("Trajectory execution active but no trajectory/start time. Switching to HOLD_POSITION commanded by mission.", throttle_duration_sec=2.0)
                    self.current_goal_py = self._get_hold_position_goal() # Hold current based on received state
                    self.trajectory_execution_active = False # Revert to hold, but controller is still "active" by command
                    status_msg = BoolMsg(); status_msg.data = True
                    self.trajectory_status_pub.publish(status_msg)
                else:
                    elapsed_execution_time_sec = (current_ros_time - self.trajectory_start_time_ros).nanoseconds / 1e9
                    target_time_in_traj_file = self.loaded_trajectory_data[0,0] + elapsed_execution_time_sec
                    current_traj_goal = self._get_trajectory_goal_at_time(target_time_in_traj_file)
                    if current_traj_goal is None:
                        self.get_logger().error("Failed to get trajectory goal point during execution. Switching to HOLD_POSITION commanded by mission.")
                        self.trajectory_execution_active = False
                        self.current_goal_py = self._get_hold_position_goal()
                        status_msg = BoolMsg(); status_msg.data = True
                        self.trajectory_status_pub.publish(status_msg)
                    else:
                        self.current_goal_py = current_traj_goal

                    if target_time_in_traj_file >= self.loaded_trajectory_data[-1, 0] - 1e-3:
                        if self.trajectory_execution_active:
                             self.get_logger().info(f"Trajectory time ended. Holding last trajectory point (commanded by mission).")
                             self.trajectory_execution_active = False 
                             # self.is_controller_active remains true, so it will hold the last point
                             status_msg = BoolMsg(); status_msg.data = True 
                             self.trajectory_status_pub.publish(status_msg)
            else: # self.is_controller_active is True, but not trajectory_execution_active -> HOLD_POSITION command
                self.current_goal_py = self._get_hold_position_goal()
                self.get_logger().debug("Control Loop: HOLD_POSITION command active. Goal is current pose.", throttle_duration_sec=1.0)
        else: 
            # No active mission command (is_controller_active is False)
            # Default to streaming setpoints to hold current position for OFFBOARD readiness
            self.current_goal_py = self._get_hold_position_goal()
            self.get_logger().debug("Control Loop: No active mission command. Streaming current pose for OFFBOARD readiness.", throttle_duration_sec=1.0)
        
        # --- Compute and Publish ---
        try:
            att_cmd_py: AttCmdClass = self.outer_loop_ctrl.compute_attitude_command(
                t=(current_ros_time.nanoseconds / 1e9),
                state=self.current_vehicle_state_py,
                goal=self.current_goal_py
            )
        except Exception as e:
            self.get_logger().error(f"Error in outer_loop_ctrl.compute_attitude_command: {e}\n{traceback.format_exc()}")
            return 

        att_msg = AttitudeTarget()
        att_msg.header.stamp = current_ros_time.to_msg()
        att_msg.orientation = quaternion_array_to_msg(att_cmd_py.q)
        att_msg.body_rate = vector_array_to_msg(att_cmd_py.w)
        
        R_body_to_world_desired = quaternion_to_rotation_matrix(att_cmd_py.q).T
        desired_body_z_axis_in_world = R_body_to_world_desired[:, 2]
        thrust_force_along_desired_z = np.dot(att_cmd_py.F_W, desired_body_z_axis_in_world)
        normalized_thrust = np.clip(thrust_force_along_desired_z / self.max_thrust_N, 0.0, 1.0)
        att_msg.thrust = float(normalized_thrust)
        att_msg.type_mask = 0 

        self.attitude_setpoint_pub.publish(att_msg)
        self.get_logger().debug("Control Loop: AttitudeTarget published.", throttle_duration_sec=1.0)

        # Logging (as before)
        log_data_py = self.outer_loop_ctrl.get_log()
        log_data_py.p_ref = np.copy(self.current_goal_py.p)
        log_data_py.v_ref = np.copy(self.current_goal_py.v)
        log_data_py.a_ff = np.copy(self.current_goal_py.a)
        log_data_py.j_ff = np.copy(self.current_goal_py.j)
        if hasattr(log_data_py, 'psi_ref'):
            log_data_py.psi_ref = self.current_goal_py.psi
        if hasattr(log_data_py, 'dpsi_ref'):
            log_data_py.dpsi_ref = self.current_goal_py.dpsi
        log_msg = controllog_class_to_ros_msg(log_data_py, current_ros_time.to_msg())
        self.controller_log_pub.publish(log_msg)

    def destroy_node(self):
        # (Same as before)
        self.get_logger().info("Shutting down MLAC Mission Node.")
        if self.control_timer:
            self.control_timer.cancel()
        super().destroy_node()

def main(args=None):
    # (Same as before)
    rclpy.init(args=args)
    node = None
    try:
        node = MlacMissionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info("Ctrl+C detected, shutting down MLAC Mission Node...")
    except Exception as e:
        logger = rclpy.logging.get_logger("mlac_mission_node_main")
        if node: logger = node.get_logger()
        logger.fatal(f"Unhandled exception in MLAC Mission Node: {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

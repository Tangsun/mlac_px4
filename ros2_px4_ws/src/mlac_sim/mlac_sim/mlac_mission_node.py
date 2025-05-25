import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time, Duration
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

from .outerloop_node import OuterLoop
from .structs import StateClass, GoalClass, ParametersClass, AttCmdClass # ControlLogClass removed
from .helpers import (quaternion_array_to_msg, vector_array_to_msg,
                       controllog_class_to_ros_msg, get_rpy)
from .utils import quaternion_to_rotation_matrix
from .mlac_fsm import MissionFiniteStateMachine, MissionPhase # Import FSM

class MlacMissionNode(Node):
    def __init__(self):
        super().__init__('mlac_mission_node')

        # --- Parameter Declarations ---
        # Existing Parameters
        self.declare_parameter('controller_type', 'pid', ParameterDescriptor(description="Controller type: 'pid', 'coml', or 'coml_debug'"))
        self.declare_parameter('control_loop_rate_hz', 50.0, ParameterDescriptor(description="Rate of the main control loop"))
        self.declare_parameter('trajectory_file_name', 'circle_trajectory_8col_50hz.npy', ParameterDescriptor(description="Name of the .npy trajectory file in 'mlac_sim/traj_data/' folder"))
        # self.declare_parameter('vehicle_mass', 4.562, ParameterDescriptor(description="Vehicle mass (kg)"))
        self.declare_parameter('vehicle_mass', 2.0, ParameterDescriptor(description="Vehicle mass (kg)"))
        self.declare_parameter('Kp', [7.0, 7.0, 7.0], ParameterDescriptor(description="Proportional gains [Px, Py, Pz]"))
        self.declare_parameter('Ki', [0.5, 0.5, 0.5], ParameterDescriptor(description="Integral gains [Ix, Iy, Iz]"))
        self.declare_parameter('Kd', [4.0, 4.0, 4.0], ParameterDescriptor(description="Derivative gains [Dx, Dy, Dz]"))
        self.declare_parameter('max_pos_err', [0.5, 0.5, 0.5], ParameterDescriptor(description="Max position error for PID saturation [err_x, err_y, err_z]"))
        self.declare_parameter('max_vel_err', [1.0, 1.0, 1.0], ParameterDescriptor(description="Max velocity error for PID saturation [verr_x, verr_y, verr_z]"))
        # self.declare_parameter('max_thrust_N', 2.2*25.0, ParameterDescriptor(description="Max thrust capability (N)"))
        self.declare_parameter('max_thrust_N', 25.0, ParameterDescriptor(description="Max thrust capability (N)"))

        # New Mission Logic Parameters for FSM
        self.declare_parameter('initial_hover_position', [0.0, 0.0, 2.0], ParameterDescriptor(description="Initial hover position [x, y, z] (m)"))
        self.declare_parameter('final_hover_position', [0.0, 0.0, 2.0], ParameterDescriptor(description="Final hover position [x, y, z] (m)"))
        self.declare_parameter('landing_position', [0.0, 0.0, 0.05], ParameterDescriptor(description="Landing target position [x, y, z] (m), z is target altitude before disarm"))
        self.declare_parameter('position_reached_threshold', 0.2, ParameterDescriptor(description="Threshold to consider a position reached (m)"))
        self.declare_parameter('hover_duration_sec', 5.0, ParameterDescriptor(description="Duration to hover at initial/final points (s)"))
        self.declare_parameter('landing_descent_rate_mps', 0.3, ParameterDescriptor(description="Descent rate for landing (m/s positive value)"))
        self.declare_parameter('wait_for_offboard_arm_timeout_sec', 30.0, ParameterDescriptor(description="Timeout (seconds) to wait for OFFBOARD and ARM after START command"))


        # --- Get Parameters ---
        self.controller_type = self.get_parameter('controller_type').get_parameter_value().string_value
        self.trajectory_file_name = self.get_parameter('trajectory_file_name').get_parameter_value().string_value
        self.max_thrust_N = self.get_parameter('max_thrust_N').get_parameter_value().double_value
        self.control_loop_rate = self.get_parameter('control_loop_rate_hz').get_parameter_value().double_value

        if self.control_loop_rate <= 0: self.control_loop_rate = 50.0
        if self.max_thrust_N <= 0: raise ValueError("max_thrust_N must be positive.")

        self.controller_params = ParametersClass()
        self.controller_params.mass = self.get_parameter('vehicle_mass').get_parameter_value().double_value
        self.controller_params.Kp = np.array(self.get_parameter('Kp').get_parameter_value().double_array_value)
        self.controller_params.Ki = np.array(self.get_parameter('Ki').get_parameter_value().double_array_value)
        self.controller_params.Kd = np.array(self.get_parameter('Kd').get_parameter_value().double_array_value)
        self.controller_params.maxPosErr = np.array(self.get_parameter('max_pos_err').get_parameter_value().double_array_value)
        self.controller_params.maxVelErr = np.array(self.get_parameter('max_vel_err').get_parameter_value().double_array_value)
        
        # --- Node State Variables ---
        self.current_vehicle_state_py = StateClass()
        # self.current_goal_py = GoalClass() # Now managed by FSM
        self.is_vehicle_state_received = False
        self.loaded_trajectory_data: np.ndarray | None = None # Keep this for loading
        self.current_mavros_state = MavrosState()

        self._load_trajectory() # Load trajectory and inform FSM

        # --- Initialize FSM ---
        initial_hover_pos = self.get_parameter('initial_hover_position').get_parameter_value().double_array_value
        final_hover_pos = self.get_parameter('final_hover_position').get_parameter_value().double_array_value
        landing_pos = self.get_parameter('landing_position').get_parameter_value().double_array_value
        pos_reached_thresh = self.get_parameter('position_reached_threshold').get_parameter_value().double_value
        hover_duration_sec = self.get_parameter('hover_duration_sec').get_parameter_value().double_value
        landing_descent_rate_mps = self.get_parameter('landing_descent_rate_mps').get_parameter_value().double_value
        wait_timeout = self.get_parameter('wait_for_offboard_arm_timeout_sec').get_parameter_value().double_value

        self.mission_fsm = MissionFiniteStateMachine(
            logger=self.get_logger(), clock=self.get_clock(),
            initial_hover_pos=initial_hover_pos, final_hover_pos=final_hover_pos, landing_pos=landing_pos,
            pos_reached_thresh=pos_reached_thresh, hover_duration_sec=hover_duration_sec,
            landing_descent_rate_mps=landing_descent_rate_mps,
            wait_for_offboard_arm_timeout_sec=wait_timeout
        )
        self.mission_fsm.set_trajectory_data(self.loaded_trajectory_data)


        # --- Initialize OuterLoop Controller ---
        # Create a default initial goal for the controller, FSM will override quickly
        default_initial_goal = GoalClass()
        if self.is_vehicle_state_received: # Should be false here, but defensive
            default_initial_goal.p = np.copy(self.current_vehicle_state_py.p)
        else: # If no state yet, use (0,0,0) or initial hover if available
             default_initial_goal.p = np.array(initial_hover_pos) if initial_hover_pos else np.zeros(3)


        self.outer_loop_ctrl = OuterLoop(
            params=self.controller_params,
            state0=self.current_vehicle_state_py, # Will be updated quickly
            goal0=default_initial_goal,
            controller=self.controller_type,
            package_name='mlac_sim'
        )

        # QoS Profiles
        qos_sensor_data = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_reliable_volatile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_reliable_transientlocal = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)

        # Subscribers/Publishers
        self.vehicle_pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.vehicle_pose_callback, qos_sensor_data)
        self.vehicle_velocity_sub = self.create_subscription(TwistStamped, '/mavros/local_position/velocity_body', self.vehicle_velocity_callback, qos_sensor_data)
        self.mavros_state_sub = self.create_subscription(MavrosState, '/mavros/state', self.mavros_state_callback, qos_reliable_volatile)
        self.mission_command_sub = self.create_subscription(StringMsg, '/mission_control/command', self.mission_command_callback, qos_reliable_volatile)

        self.attitude_setpoint_pub = self.create_publisher(AttitudeTarget, '/mavros/setpoint_raw/attitude', qos_sensor_data)
        self.controller_log_pub = self.create_publisher(ControllerLogMsg, '~/control_log', qos_reliable_volatile)
        self.trajectory_status_pub = self.create_publisher(BoolMsg, '~/trajectory_complete_status', qos_reliable_transientlocal)

        self.control_timer = self.create_timer(1.0 / self.control_loop_rate, self.control_loop_callback)
        self.get_logger().info(f"MLAC Mission Node ({self.controller_type}) initialized with FSM. Trajectory loaded: {self.mission_fsm.is_trajectory_loaded_fsm}")


    def _load_trajectory(self):
        try:
            package_share_path = get_package_share_directory('mlac_sim')
            traj_path = os.path.join(package_share_path, 'traj_data', self.trajectory_file_name)
            if os.path.exists(traj_path):
                self.loaded_trajectory_data = np.load(traj_path)
                if self.loaded_trajectory_data.ndim != 2 or self.loaded_trajectory_data.shape[0] < 2 or self.loaded_trajectory_data.shape[1] < 8:
                    self.get_logger().error(f"Trajectory file '{traj_path}' has incorrect shape: {self.loaded_trajectory_data.shape}. Needs at least 2 points and 8 columns (t,p,v,psi).")
                    self.loaded_trajectory_data = None
                else:
                    self.get_logger().info(f"Trajectory '{self.trajectory_file_name}' loaded successfully ({self.loaded_trajectory_data.shape[0]} points).")
            else:
                self.get_logger().error(f"Trajectory file not found: {traj_path}")
                self.loaded_trajectory_data = None
        except Exception as e:
            self.get_logger().error(f"Failed to load trajectory '{self.trajectory_file_name}': {e}\n{traceback.format_exc()}")
            self.loaded_trajectory_data = None
        
        # Inform FSM about the (potentially new) trajectory data
        if hasattr(self, 'mission_fsm'): # Ensure FSM is initialized
             self.mission_fsm.set_trajectory_data(self.loaded_trajectory_data)


    def mavros_state_callback(self, msg: MavrosState):
        self.current_mavros_state = msg
        # FSM's update loop will handle mode/arm changes.
        # We could add an explicit call here if immediate reaction outside FSM update cycle is needed for critical MAVROS changes.
        # e.g., self.mission_fsm.notify_mavros_state_change(msg)

    def vehicle_pose_callback(self, msg: PoseStamped):
        self.current_vehicle_state_py.t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.current_vehicle_state_py.p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.current_vehicle_state_py.q = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        if not self.is_vehicle_state_received:
            self.is_vehicle_state_received = True
            self.get_logger().info("Vehicle state received for the first time.")
            # Reset controller with initial state if FSM isn't immediately providing a goal
            # This is tricky because FSM provides the goal. Initial goal for controller is less critical now.

    def vehicle_velocity_callback(self, msg: TwistStamped):
        if not self.is_vehicle_state_received: return
        # Assuming velocity is inFLU frame, needs conversion to world (NED or ENU)
        # For ENU world frame (like MAVROS default local_position):
        R_body_to_world = quaternion_to_rotation_matrix(self.current_vehicle_state_py.q).T # This is R_frd_to_ned or R_flu_to_enu
        self.current_vehicle_state_py.v = R_body_to_world @ np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.current_vehicle_state_py.w = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]) # Body rates, usually fine as is

    def mission_command_callback(self, msg: StringMsg):
        command = msg.data.upper()
        self.get_logger().info(f"Received mission command: '{command}' for FSM.")
        if not self.is_vehicle_state_received:
            self.get_logger().warn("Cannot process FSM command yet, vehicle state not received.")
            return
        self.mission_fsm.process_command(command, self.current_vehicle_state_py)

    def control_loop_callback(self):
        current_ros_time = self.get_clock().now()
        
        if not self.is_vehicle_state_received:
            self.get_logger().debug("Control Loop: No vehicle state yet.", throttle_duration_sec=2.0)
            return

        # --- Update FSM and Get Goal ---
        current_goal_from_fsm, is_traj_phase_completed = self.mission_fsm.update(
            self.current_vehicle_state_py, self.current_mavros_state
        )
        
        # Publish trajectory status
        status_msg = BoolMsg()
        status_msg.data = is_traj_phase_completed 
        self.trajectory_status_pub.publish(status_msg)

        if current_goal_from_fsm is None or not self.mission_fsm.is_active():
            # self.get_logger().debug(f"Control Loop: FSM not active or no goal. Phase: {self.mission_fsm.current_phase.name}", throttle_duration_sec=1.0)
            # Optionally, could send a zero thrust / disarm command or just stop publishing setpoints
            # For now, if FSM is idle/landed, it provides a "stream current pose" goal.
            if current_goal_from_fsm is None: # Should not happen if FSM is well-behaved
                self.get_logger().warn(f"FSM returned None goal in phase {self.mission_fsm.current_phase.name}. Streaming current pose as fallback.")
                current_goal_from_fsm = self.mission_fsm._create_goal_from_position_array(pos_array=self.current_vehicle_state_py.p)


        # --- Compute and Publish Attitude Command ---
        try:
            # Ensure controller's internal state (if any) is reset if the goal source/type changes significantly.
            # The FSM transitions should ideally trigger controller resets if needed (e.g., PID integral reset).
            # The current OuterLoop.reset() is called by FSM logic implicitly if it changes the goal significantly.
            # Consider if outer_loop_ctrl.reset() needs to be called more explicitly based on FSM phase changes.
            # For now, assume FSM gives continuous enough goals or handles resets.

            # On initial call after vehicle state received, ensure controller has good state0
            if self.outer_loop_ctrl.t_last_ == 0.0: # Proxy for first run or after reset
                 self.outer_loop_ctrl.reset(self.current_vehicle_state_py, current_goal_from_fsm)


            att_cmd_py: AttCmdClass = self.outer_loop_ctrl.compute_attitude_command(
                t=(current_ros_time.nanoseconds / 1e9),
                state=self.current_vehicle_state_py,
                goal=current_goal_from_fsm
            )
        except Exception as e:
            self.get_logger().error(f"Error in outer_loop_ctrl.compute_attitude_command: {e}\n{traceback.format_exc()}")
            return

        att_msg = AttitudeTarget()
        att_msg.header.stamp = current_ros_time.to_msg()
        att_msg.orientation = quaternion_array_to_msg(att_cmd_py.q)
        att_msg.body_rate = vector_array_to_msg(att_cmd_py.w)
        
        # Thrust calculation (same as before)
        R_body_to_world_desired = quaternion_to_rotation_matrix(att_cmd_py.q).T
        desired_body_z_axis_in_world = R_body_to_world_desired[:, 2]
        thrust_force_along_desired_z = np.dot(att_cmd_py.F_W, desired_body_z_axis_in_world)
        normalized_thrust = np.clip(thrust_force_along_desired_z / self.max_thrust_N, 0.0, 1.0)
        att_msg.thrust = float(normalized_thrust)
        
        att_msg.type_mask = ( # Ignore body rates, PX4 will generate them
            AttitudeTarget.IGNORE_ROLL_RATE |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )
        self.attitude_setpoint_pub.publish(att_msg)

        # --- Logging ControllerLog ---
        log_data_py = self.outer_loop_ctrl.get_log() # Get latest log from controller
        # Populate reference values from the FSM's goal for logging
        log_data_py.p_ref = np.copy(current_goal_from_fsm.p)
        log_data_py.v_ref = np.copy(current_goal_from_fsm.v)
        log_data_py.a_ff = np.copy(current_goal_from_fsm.a) if current_goal_from_fsm.a is not None else np.zeros(3)
        log_data_py.j_ff = np.copy(current_goal_from_fsm.j) if current_goal_from_fsm.j is not None else np.zeros(3)
        log_data_py.psi_ref = current_goal_from_fsm.psi
        log_data_py.dpsi_ref = current_goal_from_fsm.dpsi if current_goal_from_fsm.dpsi is not None else 0.0
        
        log_msg = controllog_class_to_ros_msg(log_data_py, current_ros_time.to_msg())
        self.controller_log_pub.publish(log_msg)


    def destroy_node(self):
        self.get_logger().info("Shutting down MLAC Mission Node with FSM.")
        if self.control_timer: self.control_timer.cancel()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MlacMissionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info("Ctrl+C detected, shutting down MLAC Mission Node (FSM)...")
    except Exception as e:
        logger = rclpy.logging.get_logger("mlac_mission_node_main_fsm")
        if node: logger = node.get_logger()
        logger.fatal(f"Unhandled exception in MLAC Mission Node (FSM): {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok(): node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
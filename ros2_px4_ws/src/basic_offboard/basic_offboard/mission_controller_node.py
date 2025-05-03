#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
import asyncio # Keep asyncio if you might add async service calls later

# --- Import our state definitions and manager ---
# Ensure you are using the UPDATED state_manager.py with the SIMPLER DroneState Enum
from .state_manager import DroneState, StateManager

# --- Import ROS message/service types ---
from geometry_msgs.msg import PoseStamped, Point # Keep Point for helper function
from std_msgs.msg import Header
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import math
import traceback
import copy

# --- NEW: Import NumPy and OS ---
import numpy as np
import os

# ++ NEW: Import Enum and auto for Sub-State ++
from enum import Enum, auto

# ++ NEW: Define internal sub-states for the OFFBOARD_ACTIVE phase ++
class OffboardSubState(Enum):
    NONE = auto()                   # Not in an active offboard sub-state
    REACHING_INITIAL_HOVER = auto() # Moving to takeoff altitude/position
    AT_INITIAL_HOVER = auto()       # Stable at initial hover (e.g., if no trajectory loaded)
    REACHING_TRAJ_START = auto()    # Moving from initial hover to trajectory start point
    TRACKING_TRAJECTORY = auto()    # Following the loaded trajectory points
    AT_FINAL_HOVER = auto()         # Reached end of trajectory, hovering at final point
# ++ END NEW SECTION ++


class MissionControllerNode(Node):

    def __init__(self):
        super().__init__('mission_controller_node')

        # --- Parameters ---
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('state_machine_rate_hz', 50.0)
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_z', 1.5)
        self.declare_parameter('coordinate_frame', 'map')
        # -- REMOVED: altitude_reach_threshold --
        self.declare_parameter('descent_rate', 0.3)
        self.declare_parameter('landing_x', 0.0)
        self.declare_parameter('landing_y', 0.0)
        # ++ ADDED: Parameter for hover check ++
        self.declare_parameter('hover_pos_threshold', 0.15) # meters tolerance for position hold
        # ++ Path used from user's provided code ++
        self.declare_parameter('trajectory_file', 'src/basic_offboard/basic_offboard/traj_data/circle_trajectory_50hz.npy')

        # --- Get Parameters ---
        publish_rate = self.get_parameter('publish_rate_hz').value
        state_machine_rate = self.get_parameter('state_machine_rate_hz').value
        if publish_rate <= 0 or state_machine_rate <= 0:
             raise ValueError("publish_rate_hz and state_machine_rate_hz must be positive")
        state_machine_period = 1.0 / state_machine_rate

        self.target_altitude = self.get_parameter('target_z').value
        self.coordinate_frame = self.get_parameter('coordinate_frame').value
        # -- REMOVED: altitude_reach_threshold --
        self.descent_rate = self.get_parameter('descent_rate').value
        self.descent_step = self.descent_rate * state_machine_period
        self.landing_target_x = self.get_parameter('landing_x').value
        self.landing_target_y = self.get_parameter('landing_y').value
        # ++ Get hover threshold ++
        self.hover_pos_threshold = self.get_parameter('hover_pos_threshold').value
        self.hover_pos_threshold_sq = self.hover_pos_threshold ** 2 # Pre-calculate squared value
        self.trajectory_file_path = self.get_parameter('trajectory_file').value

        # --- Load Pre-computed Trajectory ---
        self.precomputed_trajectory = None # Full loaded data
        self.trajectory_duration = 0.0     # Duration derived from file
        self.traj_times = None
        self.traj_pos_x, self.traj_pos_y, self.traj_pos_z = None, None, None
        self.traj_vel_x, self.traj_vel_y, self.traj_vel_z = None, None, None
        # ++ ADDED: Store Start/Final Poses ++
        self.trajectory_start_pose: PoseStamped | None = None
        self.trajectory_final_pose: PoseStamped | None = None
        try:
            # Resolve path (updated logic)
            ws_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # Navigate up from basic_offboard/basic_offboard
            if not os.path.isabs(self.trajectory_file_path):
                 potential_path = os.path.join(ws_root, self.trajectory_file_path)
                 if os.path.exists(potential_path):
                      self.trajectory_file_path = potential_path
                 else:
                      self.trajectory_file_path = os.path.abspath(self.trajectory_file_path)

            if os.path.exists(self.trajectory_file_path):
                self.get_logger().info(f"Loading trajectory from: {self.trajectory_file_path}")
                loaded_data = np.load(self.trajectory_file_path)

                if loaded_data.ndim != 2 or loaded_data.shape[1] < 7: raise ValueError(f"Trajectory file needs >= 7 columns. Found shape: {loaded_data.shape}")
                if loaded_data.shape[0] < 2: raise ValueError(f"Trajectory file needs >= 2 points. Found shape: {loaded_data.shape}")

                self.precomputed_trajectory = loaded_data
                self.traj_times = loaded_data[:, 0]
                self.traj_pos_x, self.traj_pos_y, self.traj_pos_z = loaded_data[:, 1], loaded_data[:, 2], loaded_data[:, 3]
                self.traj_vel_x, self.traj_vel_y, self.traj_vel_z = loaded_data[:, 4], loaded_data[:, 5], loaded_data[:, 6]

                self.trajectory_duration = self.traj_times[-1] - self.traj_times[0]
                if self.trajectory_duration < 0: raise ValueError("Trajectory time data not monotonic.")

                time_steps = np.diff(self.traj_times)
                mean_dt, std_dt = np.mean(time_steps), np.std(time_steps)
                estimated_freq = 1.0 / mean_dt if mean_dt > 1e-9 else 0.0

                self.get_logger().info(f"  Trajectory Loaded: Duration={self.trajectory_duration:.3f}s, Steps={len(self.traj_times)}")
                self.get_logger().info(f"  Detected Sampling: Mean dt={mean_dt:.4f}s (~{estimated_freq:.2f} Hz), Std dt={std_dt:.6f}s")
                if std_dt > mean_dt * 0.1 and mean_dt > 1e-9: self.get_logger().warn(f"  Inconsistent time steps detected.")
                if abs(estimated_freq - state_machine_rate) > 0.1 * state_machine_rate: self.get_logger().warn(f"  Trajectory sampling rate (~{estimated_freq:.2f} Hz) differs significantly from node logic rate ({state_machine_rate:.2f} Hz).")

                # ++ Store start/final poses ++
                start_pos = Point(x=self.traj_pos_x[0], y=self.traj_pos_y[0], z=self.traj_pos_z[0])
                self.trajectory_start_pose = self._create_pose_stamped_from_point(start_pos)
                final_pos = Point(x=self.traj_pos_x[-1], y=self.traj_pos_y[-1], z=self.traj_pos_z[-1])
                self.trajectory_final_pose = self._create_pose_stamped_from_point(final_pos)
                self.get_logger().info(f"  Trajectory Start: ({start_pos.x:.2f}, {start_pos.y:.2f}, {start_pos.z:.2f})")
                self.get_logger().info(f"  Trajectory End:   ({final_pos.x:.2f}, {final_pos.y:.2f}, {final_pos.z:.2f})")

            else:
                 self.get_logger().error(f"Trajectory file not found at {self.trajectory_file_path}! Cannot follow trajectory.")

        except Exception as e:
            self.get_logger().error(f"Failed to load or parse trajectory file '{self.trajectory_file_path}': {e}")
            self.precomputed_trajectory = self.trajectory_start_pose = self.trajectory_final_pose = None
            self.traj_times = self.traj_pos_x = self.traj_pos_y = self.traj_pos_z = None
            self.traj_vel_x = self.traj_vel_y = self.traj_vel_z = None
            self.trajectory_duration = 0.0

        # Target pose for initial hover (using parameters)
        self.initial_hover_target = PoseStamped()
        self.initial_hover_target.header.frame_id = self.coordinate_frame
        self.initial_hover_target.pose.position.x = self.get_parameter('target_x').value
        self.initial_hover_target.pose.position.y = self.get_parameter('target_y').value
        self.initial_hover_target.pose.position.z = self.target_altitude # Use target_altitude parameter
        self.initial_hover_target.pose.orientation.w = 1.0

        # --- QoS Profiles ---
        qos_profile_state = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_profile_setpoint = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_profile_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        # --- Instantiate State Manager ---
        # Pass the clock object!
        self.state_manager = StateManager(self.get_logger(), self.get_clock(), ground_threshold=0.15)

        # --- MAVROS Subscribers/Publisher/Service Clients ---
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, qos_profile_state)
        self.local_pos_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pos_callback, qos_profile_best_effort)
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_profile_setpoint)
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # --- Node State / Data ---
        self.current_mavros_state = State()
        self.current_local_pos = PoseStamped()
        self.current_drone_state = DroneState.INIT # Main state from StateManager
        # ++ NEW: Internal sub-state variable ++
        self.offboard_sub_state = OffboardSubState.NONE
        self.setpoint_to_publish = PoseStamped()
        self.setpoint_streaming_active = False
        # -- REMOVED: self.altitude_reached --
        self.landing_setpoint_pose = None
        self.trajectory_start_time: Time | None = None # Used only for TRACKING state timing

        # Inform state manager about trajectory status
        self.state_manager.set_trajectory_status(loaded=(self.precomputed_trajectory is not None), finished=False)

        # --- Timers ---
        self.setpoint_timer = self.create_timer(1.0 / publish_rate, self.publish_setpoint_loop)
        self.mission_logic_timer = self.create_timer(state_machine_period, self.run_mission_logic)

        # --- Logging ---
        self.get_logger().info(f"Mission Controller Node Initialized.")
        self.get_logger().info(f"  Node Rates: Logic={state_machine_rate:.1f}Hz, Publish={publish_rate:.1f}Hz")
        if self.precomputed_trajectory is None:
             self.get_logger().warn("  No trajectory loaded/found. Will only hover if Offboard is activated.")
        self.get_logger().info("  Please use QGroundControl to ARM and set OFFBOARD mode.")


    # --- Callback Functions ---
    def state_callback(self, msg):
        self.current_mavros_state = msg

    def local_pos_callback(self, msg):
        # Store the latest position
        self.current_local_pos = msg

    # --- Setpoint Publishing Loop & Helpers ---
    def publish_setpoint_loop(self):
        if not self.setpoint_streaming_active: return
        # Update timestamp just before publishing
        self.setpoint_to_publish.header.stamp = self.get_clock().now().to_msg()
        self.setpoint_to_publish.header.frame_id = self.coordinate_frame
        self.setpoint_pub.publish(self.setpoint_to_publish)

    def start_setpoint_streaming(self, target_setpoint: PoseStamped):
        # Always update the target first
        self.update_streaming_setpoint(target_setpoint)
        # Start streaming only if not already active
        if not self.setpoint_streaming_active:
            self.get_logger().debug(f"Starting setpoint streaming to target: P({target_setpoint.pose.position.x:.2f}, {target_setpoint.pose.position.y:.2f}, {target_setpoint.pose.position.z:.2f})")
            self.setpoint_streaming_active = True

    def update_streaming_setpoint(self, new_setpoint_pose: PoseStamped):
         self.setpoint_to_publish = copy.deepcopy(new_setpoint_pose)

    def stop_setpoint_streaming(self):
        if self.setpoint_streaming_active:
            self.get_logger().info("Stopping setpoint streaming.")
            self.setpoint_streaming_active = False

    # --- MAVROS Service Call Helpers --- (Keep synchronous versions for now)
    def call_arming_service_sync(self, value: bool):
        if not self.arming_client.service_is_ready(): self.get_logger().error("Arming service client not available!"); return False
        req = CommandBool.Request(); req.value = value
        self.get_logger().info(f"Requesting {'Arming' if value else 'Disarming'} (sync)...")
        try:
            response = self.arming_client.call(req) # Blocking call
            if response is None: self.get_logger().error("Arming service call failed (sync - no response/timeout)."); return False
            if response.success: self.get_logger().info(f"Arming request successful (sync): Result={response.result}"); return True
            else: self.get_logger().error(f"Arming request failed (sync): Success={response.success}, Result={response.result}"); return False
        except Exception as e: self.get_logger().error(f"Arming service call failed (sync): {e}\n{traceback.format_exc()}"); return False

    def call_set_mode_service_sync(self, mode: str):
        if not self.set_mode_client.service_is_ready(): self.get_logger().error("Set mode service client not available!"); return False
        req = SetMode.Request(); req.custom_mode = mode; req.base_mode = 0
        self.get_logger().info(f"Requesting mode: {mode} (sync)...")
        try:
            response = self.set_mode_client.call(req) # Blocking call
            if response is None: self.get_logger().error("Set_mode service call failed (sync - no response/timeout)."); return False
            if response.mode_sent: self.get_logger().info(f"Mode change request '{mode}' sent successfully by MAVROS (sync)."); return True
            else: self.get_logger().error(f"MAVROS failed to send mode change request '{mode}' (sync)."); return False
        except Exception as e: self.get_logger().error(f"Set_mode service call failed (sync): {e}\n{traceback.format_exc()}"); return False


    # --- Trajectory Interpolation Function ---
    def _lookup_trajectory_point(self, elapsed_time_sec: float) -> tuple[float, float, float, float, float, float] | None:
        """Interpolates the trajectory point (pos+vel) for the given time."""
        if self.traj_times is None: return None
        elapsed_time_sec = np.clip(elapsed_time_sec, self.traj_times[0], self.traj_times[-1])
        try:
            interp_x = np.interp(elapsed_time_sec, self.traj_times, self.traj_pos_x)
            interp_y = np.interp(elapsed_time_sec, self.traj_times, self.traj_pos_y)
            interp_z = np.interp(elapsed_time_sec, self.traj_times, self.traj_pos_z)
            interp_vx = np.interp(elapsed_time_sec, self.traj_times, self.traj_vel_x)
            interp_vy = np.interp(elapsed_time_sec, self.traj_times, self.traj_vel_y)
            interp_vz = np.interp(elapsed_time_sec, self.traj_times, self.traj_vel_z)
            return (float(interp_x), float(interp_y), float(interp_z), float(interp_vx), float(interp_vy), float(interp_vz))
        except Exception as e: self.get_logger().error(f"Interpolation failed: {e}"); return None

    # ++ NEW HELPER: Check if close to a target pose ++
    def _is_at_target_pose(self, target_pose: PoseStamped, pos_threshold_sq: float) -> bool:
        """Checks if the drone is close to the target XY and Z position."""
        if not self.current_local_pos.header.stamp.sec > 0: return False
        dx = self.current_local_pos.pose.position.x - target_pose.pose.position.x
        dy = self.current_local_pos.pose.position.y - target_pose.pose.position.y
        dz = self.current_local_pos.pose.position.z - target_pose.pose.position.z
        dist_sq = dx*dx + dy*dy + dz*dz
        return dist_sq < pos_threshold_sq


    # --- Main Mission Logic Loop (MODIFIED to use Sub-States) ---
    def run_mission_logic(self):
        # 1. Get overall state from StateManager
        new_state = self.state_manager.update_state(
            self.current_mavros_state, self.current_local_pos
        )

        # --- State Transition Logic ---
        if new_state != self.current_drone_state:
            self.get_logger().info(f"Mission Controller: State change {self.current_drone_state.name} -> {new_state.name}")

            # Reset sub-state when entering OFFBOARD_ACTIVE
            if new_state == DroneState.OFFBOARD_ACTIVE:
                self.offboard_sub_state = OffboardSubState.REACHING_INITIAL_HOVER
                self.trajectory_start_time = None
                self.state_manager.set_trajectory_status(loaded=(self.precomputed_trajectory is not None), finished=False)
                self.get_logger().info(f"  Entering Offboard - Initial SubState: {self.offboard_sub_state.name}")
            elif self.current_drone_state == DroneState.OFFBOARD_ACTIVE: # Leaving OFFBOARD_ACTIVE
                 self.get_logger().info(f"  Leaving Offboard - Resetting SubState from {self.offboard_sub_state.name}")
                 self.offboard_sub_state = OffboardSubState.NONE

            # Initialize landing setpoint when entering LANDING state
            if new_state == DroneState.LANDING and self.current_drone_state != DroneState.LANDING:
                # Land at final trajectory XY if available, else current XY
                landing_x = self.trajectory_final_pose.pose.position.x if self.trajectory_final_pose else self.current_local_pos.pose.position.x
                landing_y = self.trajectory_final_pose.pose.position.y if self.trajectory_final_pose else self.current_local_pos.pose.position.y
                self.landing_setpoint_pose = PoseStamped()
                self.landing_setpoint_pose.header.frame_id = self.coordinate_frame
                self.landing_setpoint_pose.pose.position.x = landing_x
                self.landing_setpoint_pose.pose.position.y = landing_y
                self.landing_setpoint_pose.pose.position.z = self.current_local_pos.pose.position.z if self.current_local_pos.header.stamp.sec > 0 else self.target_altitude
                self.landing_setpoint_pose.pose.orientation.w = 1.0
                self.get_logger().info(f"  Initiating Landing towards ({landing_x:.2f}, {landing_y:.2f})")

            self.current_drone_state = new_state # Update main state

        # --- Action Logic based on Main State ---
        try:
            # Handle ARMED_WAITING_FOR_MODE
            if self.current_drone_state == DroneState.ARMED_WAITING_FOR_MODE:
                self.start_setpoint_streaming(self.initial_hover_target)

            # ++ Handle OFFBOARD_ACTIVE using internal sub-states ++
            elif self.current_drone_state == DroneState.OFFBOARD_ACTIVE:
                next_sub_state = self.offboard_sub_state

                # --- SubState: REACHING_INITIAL_HOVER ---
                if self.offboard_sub_state == OffboardSubState.REACHING_INITIAL_HOVER:
                    self.start_setpoint_streaming(self.initial_hover_target)
                    self.get_logger().debug(f"SubState: REACHING_INITIAL_HOVER. Target Z: {self.initial_hover_target.pose.position.z:.2f}m", throttle_duration_sec=2.0)
                    if self._is_at_target_pose(self.initial_hover_target, self.hover_pos_threshold_sq):
                        self.get_logger().info("  Reached initial hover point.")
                        if self.precomputed_trajectory is not None and self.trajectory_start_pose is not None:
                            next_sub_state = OffboardSubState.REACHING_TRAJ_START
                        else:
                            self.get_logger().warn("  No valid trajectory loaded/start point found. Staying at initial hover.")
                            next_sub_state = OffboardSubState.AT_INITIAL_HOVER

                # --- SubState: AT_INITIAL_HOVER ---
                elif self.offboard_sub_state == OffboardSubState.AT_INITIAL_HOVER:
                     self.start_setpoint_streaming(self.initial_hover_target)
                     self.get_logger().info("SubState: AT_INITIAL_HOVER (Holding position).", throttle_duration_sec=5.0)

                # --- SubState: REACHING_TRAJ_START ---
                elif self.offboard_sub_state == OffboardSubState.REACHING_TRAJ_START:
                    if self.trajectory_start_pose:
                        self.start_setpoint_streaming(self.trajectory_start_pose)
                        self.get_logger().debug(f"SubState: REACHING_TRAJ_START. Target: ({self.trajectory_start_pose.pose.position.x:.2f}, {self.trajectory_start_pose.pose.position.y:.2f}, {self.trajectory_start_pose.pose.position.z:.2f})", throttle_duration_sec=2.0)
                        if self._is_at_target_pose(self.trajectory_start_pose, self.hover_pos_threshold_sq):
                            self.get_logger().info("  Reached trajectory start point.")
                            next_sub_state = OffboardSubState.TRACKING_TRAJECTORY
                    else:
                        self.get_logger().error("In REACHING_TRAJ_START but trajectory_start_pose is None! Holding initial hover.")
                        self.start_setpoint_streaming(self.initial_hover_target)
                        next_sub_state = OffboardSubState.AT_INITIAL_HOVER

                # --- SubState: TRACKING_TRAJECTORY ---
                elif self.offboard_sub_state == OffboardSubState.TRACKING_TRAJECTORY:
                    if self.trajectory_start_time is None:
                        self.get_logger().info("  SubState: TRACKING_TRAJECTORY starting.")
                        self.trajectory_start_time = self.get_clock().now()
                        # Add start_rosbag_recording() here later

                    elapsed_time = (self.get_clock().now() - self.trajectory_start_time).nanoseconds / 1e9

                    if elapsed_time >= self.trajectory_duration:
                        if not self.state_manager.trajectory_finished:
                            self.get_logger().info(f"  SubState: TRACKING_TRAJECTORY finished (elapsed {elapsed_time:.2f}s >= duration {self.trajectory_duration:.2f}s).")
                            self.state_manager.set_trajectory_status(loaded=True, finished=True)
                            # Add stop_rosbag_recording() here later
                            next_sub_state = OffboardSubState.AT_FINAL_HOVER
                    else:
                        target_data = self._lookup_trajectory_point(elapsed_time)
                        if target_data:
                            px, py, pz = target_data[0:3]
                            target_pose_msg = self._create_pose_stamped_from_point(Point(x=px, y=py, z=pz))
                            self.update_streaming_setpoint(target_pose_msg)
                            self.start_setpoint_streaming(target_pose_msg)
                            self.get_logger().debug(f"SubState: TRACKING_TRAJECTORY t={elapsed_time:.2f}s", throttle_duration_sec=1.0)
                        else:
                            self.get_logger().warn("Trajectory lookup failed during tracking! Holding last valid setpoint.", throttle_duration_sec=2.0)
                            self.start_setpoint_streaming(self.setpoint_to_publish)

                # --- SubState: AT_FINAL_HOVER ---
                elif self.offboard_sub_state == OffboardSubState.AT_FINAL_HOVER:
                     if self.trajectory_final_pose:
                        self.start_setpoint_streaming(self.trajectory_final_pose)
                        self.get_logger().info("SubState: AT_FINAL_HOVER. Waiting for StateManager to trigger LANDING.", throttle_duration_sec=5.0)
                     else:
                        self.get_logger().warn("In AT_FINAL_HOVER but trajectory_final_pose is None! Holding initial hover.", throttle_duration_sec=5.0)
                        self.start_setpoint_streaming(self.initial_hover_target)

                # --- Update internal sub-state if it changed ---
                if next_sub_state != self.offboard_sub_state:
                     self.get_logger().info(f"  Offboard SubState change: {self.offboard_sub_state.name} -> {next_sub_state.name}")
                     self.offboard_sub_state = next_sub_state
            # ++ END OFFBOARD_ACTIVE ++


            # Handle LANDING state
            elif self.current_drone_state == DroneState.LANDING:
                if self.landing_setpoint_pose is not None:
                    self.landing_setpoint_pose.pose.position.z -= self.descent_step
                    current_target_z = max(self.landing_setpoint_pose.pose.position.z, -0.1)
                    self.landing_setpoint_pose.pose.position.z = current_target_z
                    self.update_streaming_setpoint(self.landing_setpoint_pose)
                    self.start_setpoint_streaming(self.landing_setpoint_pose)
                    self.get_logger().info(f"Mission Controller: LANDING - Commanding Z: {current_target_z:.2f}m", throttle_duration_sec=1.0)
                else:
                    self.get_logger().error("Mission Controller: In LANDING state but landing_setpoint_pose is None!")
                    self.stop_setpoint_streaming()

            # Handle other states
            elif self.current_drone_state in [DroneState.MISSION_COMPLETE, DroneState.INIT, DroneState.IDLE]:
                 self.stop_setpoint_streaming()
                 if self.current_drone_state == DroneState.MISSION_COMPLETE:
                     self.get_logger().info("Mission Controller: Mission complete.", throttle_duration_sec=10.0)

        except Exception as e:
             self.get_logger().error(f"Error in mission logic loop: {e}\n{traceback.format_exc()}")
             self.stop_setpoint_streaming()


    # --- Helper to create PoseStamped ---
    def _create_pose_stamped_from_point(self, point: Point) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.coordinate_frame
        pose.pose.position.x = point.x
        pose.pose.position.y = point.y
        pose.pose.position.z = point.z
        pose.pose.orientation.w = 1.0 # Keep yaw constant for now
        return pose

    # --- destroy_node ---
    def destroy_node(self):
        self.get_logger().info("Shutting down Mission Controller Node...")
        self.stop_setpoint_streaming()
        # Add rosbag stop call here later
        if hasattr(self, 'setpoint_timer') and self.setpoint_timer: self.setpoint_timer.cancel()
        if hasattr(self, 'mission_logic_timer') and self.mission_logic_timer: self.mission_logic_timer.cancel()
        super().destroy_node()


# --- main function ---
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        # Import StateManager here to ensure it uses the code from the install space if running installed
        from basic_offboard.state_manager import DroneState, StateManager # Use package name

        node = MissionControllerNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            # Graceful shutdown
            executor.shutdown()
            if node and rclpy.ok(): node.destroy_node()
    except KeyboardInterrupt:
        print("Ctrl+C detected, shutting down...")
    except Exception as e:
         # Log any other exceptions during init or spin
         logger = rclpy.logging.get_logger("mission_controller_main")
         logger.fatal(f"Unhandled exception: {e}\n{traceback.format_exc()}")
    finally:
        # Ensure ROS cleanup
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
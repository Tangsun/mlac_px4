#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
import asyncio # Keep asyncio if you might add async service calls later

# --- Import our state definitions and manager ---
from .state_manager import DroneState, StateManager # Uses the updated state_manager.py

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
        self.declare_parameter('altitude_reach_threshold', 0.2)
        self.declare_parameter('descent_rate', 0.3)
        self.declare_parameter('landing_x', 0.0)
        self.declare_parameter('landing_y', 0.0)
        self.declare_parameter('trajectory_file', 'src/basic_offboard/basic_offboard/traj_data/circle_trajectory_50hz.npy')

        # --- Get Parameters ---
        # Node operating rates are set by parameters
        publish_rate = self.get_parameter('publish_rate_hz').value
        state_machine_rate = self.get_parameter('state_machine_rate_hz').value
        # Ensure rates are positive
        if publish_rate <= 0 or state_machine_rate <= 0:
             raise ValueError("publish_rate_hz and state_machine_rate_hz must be positive")
        state_machine_period = 1.0 / state_machine_rate

        # Other parameters
        self.target_altitude = self.get_parameter('target_z').value
        self.coordinate_frame = self.get_parameter('coordinate_frame').value
        self.altitude_reach_threshold = self.get_parameter('altitude_reach_threshold').value
        self.descent_rate = self.get_parameter('descent_rate').value
        self.descent_step = self.descent_rate * state_machine_period
        self.landing_target_x = self.get_parameter('landing_x').value
        self.landing_target_y = self.get_parameter('landing_y').value
        self.trajectory_file_path = self.get_parameter('trajectory_file').value

        # --- Load Pre-computed Trajectory ---
        self.precomputed_trajectory = None # Full loaded data
        self.trajectory_duration = 0.0     # Duration derived from file
        self.traj_times = None
        self.traj_pos_x = None
        self.traj_pos_y = None
        self.traj_pos_z = None
        self.traj_vel_x = None
        self.traj_vel_y = None
        self.traj_vel_z = None
        try:
            # Resolve path
            if not os.path.isabs(self.trajectory_file_path):
                 self.trajectory_file_path = os.path.abspath(self.trajectory_file_path)

            if os.path.exists(self.trajectory_file_path):
                self.get_logger().info(f"Loading trajectory from: {self.trajectory_file_path}")
                loaded_data = np.load(self.trajectory_file_path)

                # --- Validation ---
                if loaded_data.ndim != 2 or loaded_data.shape[1] < 7:
                    raise ValueError(f"Trajectory file needs >= 7 columns: [time, px, py, pz, vx, vy, vz]. Found shape: {loaded_data.shape}")
                if loaded_data.shape[0] < 2:
                    raise ValueError(f"Trajectory file needs at least 2 points to calculate dt. Found shape: {loaded_data.shape}")

                # --- Store Data Columns ---
                self.precomputed_trajectory = loaded_data
                self.traj_times = loaded_data[:, 0]
                self.traj_pos_x = loaded_data[:, 1]
                self.traj_pos_y = loaded_data[:, 2]
                self.traj_pos_z = loaded_data[:, 3]
                self.traj_vel_x = loaded_data[:, 4]
                self.traj_vel_y = loaded_data[:, 5]
                self.traj_vel_z = loaded_data[:, 6]

                # --- Calculate and Log Trajectory Properties ---
                # Duration
                self.trajectory_duration = self.traj_times[-1] - self.traj_times[0] # Use difference for accuracy
                if self.trajectory_duration < 0: # Basic sanity check
                     raise ValueError("Trajectory time data is not monotonically increasing.")

                # Sampling Info
                time_steps = np.diff(self.traj_times)
                mean_dt = np.mean(time_steps)
                std_dt = np.std(time_steps)
                estimated_freq = 1.0 / mean_dt if mean_dt > 1e-9 else 0.0 # Avoid division by zero

                self.get_logger().info(f"  Trajectory Loaded: Duration={self.trajectory_duration:.3f}s, Steps={len(self.traj_times)}")
                self.get_logger().info(f"  Detected Sampling: Mean dt={mean_dt:.4f}s (~{estimated_freq:.2f} Hz), Std dt={std_dt:.6f}s")

                # Warn if sampling seems inconsistent
                if std_dt > mean_dt * 0.1 and mean_dt > 1e-9: # Warning if std dev > 10% of mean dt
                    self.get_logger().warn(f"  Inconsistent time steps detected in trajectory file (std dev > 10% of mean). Interpolation might be less accurate.")
                # Warn if detected frequency differs significantly from node's logic rate
                if abs(estimated_freq - state_machine_rate) > 0.1 * state_machine_rate:
                     self.get_logger().warn(f"  Trajectory file sampling rate (~{estimated_freq:.2f} Hz) differs significantly from node's logic rate ({state_machine_rate:.2f} Hz).")

                # --- End Calculation/Logging ---

            else:
                 self.get_logger().error(f"Trajectory file not found at {self.trajectory_file_path}! Cannot follow trajectory.")

        except Exception as e:
            self.get_logger().error(f"Failed to load or parse trajectory file '{self.trajectory_file_path}': {e}")
            # Ensure all related attributes are None if loading fails
            self.precomputed_trajectory = None
            self.trajectory_duration = 0.0
            self.traj_times = self.traj_pos_x = self.traj_pos_y = self.traj_pos_z = None
            self.traj_vel_x = self.traj_vel_y = self.traj_vel_z = None

        # Target pose for initial hover
        self.hover_pose = PoseStamped()
        self.hover_pose.header.frame_id = self.coordinate_frame
        self.hover_pose.pose.position.x = self.get_parameter('target_x').value
        self.hover_pose.pose.position.y = self.get_parameter('target_y').value
        self.hover_pose.pose.position.z = self.target_altitude
        self.hover_pose.pose.orientation.w = 1.0 # Level flight

        # --- QoS Profiles --- (Unchanged)
        qos_profile_state = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_profile_setpoint = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_profile_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        # --- Instantiate State Manager --- (Unchanged)
        self.state_manager = StateManager(self.get_logger(),
                                           target_altitude=self.target_altitude,
                                           altitude_threshold=self.altitude_reach_threshold)

        # --- MAVROS Subscribers --- (Unchanged)
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, qos_profile_state)
        self.local_pos_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pos_callback, qos_profile_best_effort)

        # --- MAVROS Publisher --- (Unchanged)
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_profile_setpoint)

        # --- MAVROS Service Clients --- (Unchanged)
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # --- Node State / Data ---
        self.current_mavros_state = State()
        self.current_local_pos = PoseStamped()
        self.current_drone_state = DroneState.INIT
        self.setpoint_to_publish = PoseStamped()
        self.setpoint_streaming_active = False
        self.altitude_reached = False
        self.landing_setpoint_pose = None
        # ++ MODIFIED: trajectory_start_time only ++
        self.trajectory_start_time: Time | None = None

        # -- REMOVED: Dummy trajectory variables --
        # self.dummy_trajectory = self._create_dummy_trajectory()
        # self.current_trajectory_index = 0

        # ++ MODIFIED: Inform state manager based on actual loading ++
        self.state_manager.set_trajectory_status(loaded=(self.precomputed_trajectory is not None), finished=False)

        # --- Timers --- (Rates updated)
        self.setpoint_timer = self.create_timer(1.0 / publish_rate, self.publish_setpoint_loop)
        self.mission_logic_timer = self.create_timer(state_machine_period, self.run_mission_logic)

        # ++ MODIFIED: Logging ++
        self.get_logger().info("Mission Controller Node (Precomputed Trajectory @ 50Hz) Initialized.")
        if self.precomputed_trajectory is None:
             self.get_logger().warn("No trajectory loaded, will only hover if Offboard is activated.")
        else:
             self.get_logger().info(f"Trajectory loaded. Duration: {self.trajectory_duration:.2f}s.")
        self.get_logger().info("Please use QGroundControl to ARM and set OFFBOARD mode.")
        # ++ END MODIFICATION ++


    # -- REMOVED: _create_dummy_trajectory --

    # --- Callback Functions --- (Unchanged)
    def state_callback(self, msg):
        self.current_mavros_state = msg

    def local_pos_callback(self, msg):
        self.current_local_pos = msg
        # Altitude check moved to run_mission_logic

    # --- Setpoint Publishing Loop & Helpers --- (Unchanged)
    def publish_setpoint_loop(self):
        if not self.setpoint_streaming_active: return
        self.setpoint_to_publish.header.stamp = self.get_clock().now().to_msg()
        self.setpoint_to_publish.header.frame_id = self.coordinate_frame
        self.setpoint_pub.publish(self.setpoint_to_publish)

    def start_setpoint_streaming(self, initial_setpoint: PoseStamped):
        if not self.setpoint_streaming_active:
            self.get_logger().debug(f"Starting setpoint streaming to target: X={initial_setpoint.pose.position.x:.2f}, Y={initial_setpoint.pose.position.y:.2f}, Z={initial_setpoint.pose.position.z:.2f}")
            self.setpoint_to_publish = copy.deepcopy(initial_setpoint)
            self.setpoint_streaming_active = True
        self.update_streaming_setpoint(initial_setpoint) # Ensure target is updated

    def update_streaming_setpoint(self, new_setpoint_pose: PoseStamped):
         self.setpoint_to_publish = copy.deepcopy(new_setpoint_pose)

    def stop_setpoint_streaming(self):
        if self.setpoint_streaming_active:
            self.get_logger().info("Stopping setpoint streaming.")
            self.setpoint_streaming_active = False

    # --- MAVROS Service Call Helpers --- (Unchanged - still synchronous and unused in main flow)
    def call_arming_service_sync(self, value: bool):
        if not self.arming_client.service_is_ready(): self.get_logger().error("Arming service client not available!"); return False
        req = CommandBool.Request(); req.value = value
        self.get_logger().info(f"Requesting {'Arming' if value else 'Disarming'} (sync)...")
        try:
            # Note: Using synchronous call here, potentially blocking. Consider async if used.
            response = self.arming_client.call(req)
            if response is None: self.get_logger().error("Arming service call failed (sync - no response/timeout)."); return False
            if response.success: self.get_logger().info(f"Arming request successful (sync): Result={response.result}"); return True
            else: self.get_logger().error(f"Arming request failed (sync): Success={response.success}, Result={response.result}"); return False
        except Exception as e: self.get_logger().error(f"Arming service call failed (sync): {e}\n{traceback.format_exc()}"); return False

    def call_set_mode_service_sync(self, mode: str):
        if not self.set_mode_client.service_is_ready(): self.get_logger().error("Set mode service client not available!"); return False
        req = SetMode.Request(); req.custom_mode = mode; req.base_mode = 0
        self.get_logger().info(f"Requesting mode: {mode} (sync)...")
        try:
            # Note: Using synchronous call here, potentially blocking. Consider async if used.
            response = self.set_mode_client.call(req)
            if response is None: self.get_logger().error("Set_mode service call failed (sync - no response/timeout)."); return False
            if response.mode_sent: self.get_logger().info(f"Mode change request '{mode}' sent successfully by MAVROS (sync)."); return True
            else: self.get_logger().error(f"MAVROS failed to send mode change request '{mode}' (sync)."); return False
        except Exception as e: self.get_logger().error(f"Set_mode service call failed (sync): {e}\n{traceback.format_exc()}"); return False


    # ++ NEW: Trajectory Interpolation Function ++
    def _lookup_trajectory_point(self, elapsed_time_sec: float) -> tuple[float, float, float, float, float, float] | None:
        """
        Interpolates the trajectory point (pos+vel) for the given time
        from the precomputed data arrays.
        """
        if self.traj_times is None: # Check if data was loaded successfully
            self.get_logger().warn("_lookup_trajectory_point called but no trajectory data available!", throttle_duration_sec=5.0)
            return None # Indicate failure

        # Clamp elapsed time to the trajectory's time range
        # np.clip is important to handle potential slight timing overruns
        elapsed_time_sec = np.clip(elapsed_time_sec, self.traj_times[0], self.traj_times[-1])

        # Use NumPy's linear interpolation for each component
        try:
            interp_x = np.interp(elapsed_time_sec, self.traj_times, self.traj_pos_x)
            interp_y = np.interp(elapsed_time_sec, self.traj_times, self.traj_pos_y)
            interp_z = np.interp(elapsed_time_sec, self.traj_times, self.traj_pos_z)
            interp_vx = np.interp(elapsed_time_sec, self.traj_times, self.traj_vel_x)
            interp_vy = np.interp(elapsed_time_sec, self.traj_times, self.traj_vel_y)
            interp_vz = np.interp(elapsed_time_sec, self.traj_times, self.traj_vel_z)
            # Return tuple: (px, py, pz, vx, vy, vz)
            return (float(interp_x), float(interp_y), float(interp_z),
                    float(interp_vx), float(interp_vy), float(interp_vz))
        except Exception as e:
             self.get_logger().error(f"Interpolation failed: {e}")
             return None # Indicate failure
    # ++ END NEW FUNCTION ++


    # --- Main Mission Logic Loop (Modified for Precomputed Trajectory) ---
    def run_mission_logic(self):
        """
        Periodically checks the state manager and performs actions
        based on the *observed* current state. Uses precomputed trajectory.
        """
        # 1. Ask the state manager to determine the current state
        new_state = self.state_manager.update_state(
            self.current_mavros_state,
            self.current_local_pos
        )

        # --- Handle state transitions --- (largely unchanged, ensures flags reset)
        if new_state != self.current_drone_state:
            self.get_logger().info(f"Mission Controller: State change {self.current_drone_state.name} -> {new_state.name}")
            if new_state == DroneState.OFFBOARD_ACTIVE:
                self.altitude_reached = False
                self.trajectory_start_time = None
                self.state_manager.set_trajectory_status(loaded=(self.precomputed_trajectory is not None), finished=False)
            if self.current_drone_state == DroneState.OFFBOARD_ACTIVE and new_state != DroneState.OFFBOARD_ACTIVE:
                 self.trajectory_start_time = None # Clear start time if we leave the state for any reason
            if new_state == DroneState.DESCENDING:
                self.trajectory_start_time = None # Ensure timer is cleared before descent
                # Initialize landing setpoint (Unchanged)
                self.landing_setpoint_pose = PoseStamped()
                self.landing_setpoint_pose.header.frame_id = self.coordinate_frame
                self.landing_setpoint_pose.pose.position.x = self.landing_target_x
                self.landing_setpoint_pose.pose.position.y = self.landing_target_y
                self.landing_setpoint_pose.pose.position.z = self.current_local_pos.pose.position.z
                self.landing_setpoint_pose.pose.orientation.w = 1.0
                self.get_logger().info(f"Initiating descent towards ({self.landing_target_x:.2f}, {self.landing_target_y:.2f}) from Z={self.landing_setpoint_pose.pose.position.z:.2f}m")

            self.current_drone_state = new_state


        # 2. Perform actions based on the *current observed* state
        try:
            # Start streaming hover pose when ready for Offboard (Unchanged)
            if self.current_drone_state == DroneState.ARMED_WAITING_FOR_MODE:
                self.start_setpoint_streaming(self.hover_pose)

            # ++ MODIFIED: OFFBOARD_ACTIVE Logic ++
            elif self.current_drone_state == DroneState.OFFBOARD_ACTIVE:
                # Check if target altitude has been reached (Unchanged)
                if not self.altitude_reached:
                    altitude_error = abs(self.current_local_pos.pose.position.z - self.target_altitude)
                    self.altitude_reached = altitude_error < self.altitude_reach_threshold
                    if self.altitude_reached:
                        self.get_logger().info("Mission Controller: Altitude reached.")
                        if self.precomputed_trajectory is None: # Check if trajectory is available
                             self.get_logger().warn("Altitude reached, but no trajectory loaded. Holding hover.")
                    else:
                        # Still climbing/holding hover altitude
                        self.start_setpoint_streaming(self.hover_pose)
                        self.get_logger().debug(f"Mission Controller: OFFBOARD ACTIVE - Holding hover / climbing. Current Z: {self.current_local_pos.pose.position.z:.2f}m", throttle_duration_sec=2.0)

                # --- Altitude reached: Follow precomputed trajectory (if loaded) ---
                if self.altitude_reached and self.precomputed_trajectory is not None:
                    if self.trajectory_start_time is None:
                        # Start the timer only once when trajectory begins
                        self.get_logger().info("Mission Controller: Starting precomputed trajectory.")
                        self.trajectory_start_time = self.get_clock().now()

                    # Calculate elapsed time since trajectory started
                    elapsed_time = (self.get_clock().now() - self.trajectory_start_time).nanoseconds / 1e9

                    # Check if trajectory duration exceeded
                    if elapsed_time >= self.trajectory_duration:
                        if not self.state_manager.trajectory_finished: # Prevent multiple logs/calls
                            self.get_logger().info(f"Mission Controller: Precomputed trajectory finished (elapsed: {elapsed_time:.2f}s >= duration: {self.trajectory_duration:.2f}s).")
                            self.state_manager.set_trajectory_status(loaded=True, finished=True)
                            # Publish the very last point from the data as the final setpoint before descent starts
                            target_data = self._lookup_trajectory_point(self.trajectory_duration)
                            if target_data:
                                final_pose = self._create_pose_stamped_from_point(Point(x=target_data[0], y=target_data[1], z=target_data[2]))
                                self.update_streaming_setpoint(final_pose) # Update target
                                self.start_setpoint_streaming(final_pose) # Ensure it's streaming
                        # No need to reset trajectory_start_time here, state manager handles transition
                    else:
                        # Lookup/interpolate the trajectory point for the current time
                        target_data = self._lookup_trajectory_point(elapsed_time)

                        if target_data:
                             # Extract position for PoseStamped
                             px, py, pz = target_data[0], target_data[1], target_data[2]
                             # vx, vy, vz are available in target_data[3:6] if needed later

                             target_pose_msg = self._create_pose_stamped_from_point(Point(x=px, y=py, z=pz))
                             self.update_streaming_setpoint(target_pose_msg)
                             self.start_setpoint_streaming(target_pose_msg) # Ensure streaming is active

                             self.get_logger().debug(f"Mission Controller: Following trajectory - t={elapsed_time:.2f}s -> Target(X:{px:.2f}, Y:{py:.2f}, Z:{pz:.2f})", throttle_duration_sec=1.0)
                        else:
                            # If lookup failed, hold hover pose
                            self.get_logger().warn("Trajectory lookup failed, holding hover pose.", throttle_duration_sec=5.0)
                            self.update_streaming_setpoint(self.hover_pose)
                            self.start_setpoint_streaming(self.hover_pose)

                # --- Altitude reached but NO trajectory loaded: Just hover --- (Unchanged)
                elif self.altitude_reached and self.precomputed_trajectory is None:
                     self.start_setpoint_streaming(self.hover_pose)
                     self.get_logger().info("Mission Controller: OFFBOARD ACTIVE - Altitude reached, hovering (no trajectory loaded).", throttle_duration_sec=5.0)
            # ++ END MODIFIED SECTION ++


            # Handle controlled descent (Unchanged)
            elif self.current_drone_state == DroneState.DESCENDING:
                if self.landing_setpoint_pose is not None:
                    self.landing_setpoint_pose.pose.position.z -= self.descent_step
                    self.landing_setpoint_pose.pose.position.z = max(self.landing_setpoint_pose.pose.position.z, -0.1)
                    self.landing_setpoint_pose.pose.position.x = self.landing_target_x
                    self.landing_setpoint_pose.pose.position.y = self.landing_target_y
                    self.update_streaming_setpoint(self.landing_setpoint_pose)
                    self.start_setpoint_streaming(self.landing_setpoint_pose) # Ensure streaming
                    self.get_logger().info(f"Mission Controller: DESCENDING - Target Z: {self.landing_setpoint_pose.pose.position.z:.2f}m", throttle_duration_sec=1.0)
                else:
                    self.get_logger().error("Mission Controller: In DESCENDING state but landing_setpoint_pose is None!")
                    self.stop_setpoint_streaming()

            # Keep streaming when grounded (Unchanged)
            elif self.current_drone_state == DroneState.GROUNDED:
                if self.landing_setpoint_pose is not None:
                    self.landing_setpoint_pose.pose.position.z = max(self.landing_setpoint_pose.pose.position.z, -0.1)
                    self.landing_setpoint_pose.pose.position.x = self.landing_target_x
                    self.landing_setpoint_pose.pose.position.y = self.landing_target_y
                    self.start_setpoint_streaming(self.landing_setpoint_pose) # Keep streaming
                    self.get_logger().info("Mission Controller: Landed. Publishing final setpoint. Waiting for manual DISARM via QGC.", throttle_duration_sec=5.0)
                else:
                    self.stop_setpoint_streaming()


            # Stop streaming when mission complete or in initial states (Unchanged)
            elif self.current_drone_state in [DroneState.MISSION_COMPLETE, DroneState.INIT, DroneState.IDLE, DroneState.WAITING_FOR_ARM]:
                self.stop_setpoint_streaming()
                if self.current_drone_state == DroneState.MISSION_COMPLETE:
                    self.get_logger().info("Mission Controller: Mission complete. Setpoint streaming stopped.", throttle_duration_sec=10.0)

        except Exception as e:
            self.get_logger().error(f"Error in mission logic loop: {e}\n{traceback.format_exc()}")
            self.stop_setpoint_streaming() # Stop streaming on error


    # -- REMOVED: _is_close_to_target method --

    # --- Helper to create PoseStamped (Unchanged) ---
    def _create_pose_stamped_from_point(self, point: Point) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.coordinate_frame
        # Stamp is updated in publish_setpoint_loop
        pose.pose.position.x = point.x
        pose.pose.position.y = point.y
        pose.pose.position.z = point.z
        pose.pose.orientation.w = 1.0 # Keep yaw constant for now
        return pose

    # --- destroy_node (Unchanged) ---
    def destroy_node(self):
        self.get_logger().info("Shutting down Mission Controller Node...")
        self.stop_setpoint_streaming()
        # Cancel timers explicitly on destroy
        if hasattr(self, 'setpoint_timer') and self.setpoint_timer:
            self.setpoint_timer.cancel()
        if hasattr(self, 'mission_logic_timer') and self.mission_logic_timer:
            self.mission_logic_timer.cancel()
        super().destroy_node()


# --- main function (Unchanged) ---
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MissionControllerNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
            # Ensure node is destroyed if it exists and rclpy is still ok
            if node and rclpy.ok():
                 node.destroy_node()
    except KeyboardInterrupt:
        print("Ctrl+C detected, shutting down...")
    # Optional: Catch broader exceptions during spin
    # except Exception as e:
    #      if node: node.get_logger().fatal(f"Unhandled exception during spin: {e}\n{traceback.format_exc()}")
    finally:
        # Ensure rclpy shutdown happens even if node destruction failed
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
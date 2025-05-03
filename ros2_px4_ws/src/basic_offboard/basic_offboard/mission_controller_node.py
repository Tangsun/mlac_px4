#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time # Import Time
import asyncio

# --- Import our state definitions and manager ---
from .state_manager import DroneState, StateManager # Uses the updated state_manager.py

# --- Import ROS message/service types ---
from geometry_msgs.msg import PoseStamped, Point # Point can be removed if not needed
from std_msgs.msg import Header
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import math # For distance calculations and sin/cos
import traceback # Import traceback module
import copy # To copy pose message for landing

class MissionControllerNode(Node):

    def __init__(self):
        super().__init__('mission_controller_node')

        # --- Parameters ---
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('state_machine_rate_hz', 50.0) # Increase rate for smoother trajectory sampling
        self.declare_parameter('target_x', 0.0) # Initial hover X
        self.declare_parameter('target_y', 0.0) # Initial hover Y
        self.declare_parameter('target_z', 1.5) # Target hover altitude
        self.declare_parameter('coordinate_frame', 'map')
        # REMOVE: self.declare_parameter('waypoint_reach_threshold', 0.2)
        self.declare_parameter('altitude_reach_threshold', 0.2) # meters
        self.declare_parameter('descent_rate', 0.3) # meters per second for landing
        self.declare_parameter('landing_x', 0.0) # Target X for landing (can be updated later)
        self.declare_parameter('landing_y', 0.0) # Target Y for landing (can be updated later)

        # --- New Parameters for Timed Trajectory ---
        self.declare_parameter('trajectory_duration_sec', 20.0) # Example: 20 second trajectory
        self.declare_parameter('trajectory_radius', 2.0) # Example: 2 meter radius circle

        publish_rate = self.get_parameter('publish_rate_hz').value
        state_machine_period = 1.0 / self.get_parameter('state_machine_rate_hz').value
        self.target_altitude = self.get_parameter('target_z').value
        self.coordinate_frame = self.get_parameter('coordinate_frame').value
        # REMOVE: self.waypoint_reach_threshold_sq = self.get_parameter('waypoint_reach_threshold').value ** 2
        self.altitude_reach_threshold = self.get_parameter('altitude_reach_threshold').value
        self.descent_rate = self.get_parameter('descent_rate').value
        self.descent_step = self.descent_rate * state_machine_period
        self.landing_target_x = self.get_parameter('landing_x').value
        self.landing_target_y = self.get_parameter('landing_y').value

        # --- Get Timed Trajectory Parameters ---
        self.trajectory_duration = self.get_parameter('trajectory_duration_sec').value
        self.trajectory_radius = self.get_parameter('trajectory_radius').value


        # Target pose for initial hover (remains the same)
        self.hover_pose = PoseStamped()
        self.hover_pose.header.frame_id = self.coordinate_frame
        self.hover_pose.pose.position.x = self.get_parameter('target_x').value
        self.hover_pose.pose.position.y = self.get_parameter('target_y').value
        self.hover_pose.pose.position.z = self.target_altitude
        self.hover_pose.pose.orientation.w = 1.0 # Level flight

        # --- QoS Profiles --- (remain the same)
        qos_profile_state = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_profile_setpoint = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_profile_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        # --- Instantiate State Manager --- (remains the same)
        self.state_manager = StateManager(self.get_logger(),
                                           target_altitude=self.target_altitude,
                                           altitude_threshold=self.altitude_reach_threshold)

        # --- MAVROS Subscribers --- (remain the same)
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, qos_profile_state)
        self.local_pos_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pos_callback, qos_profile_best_effort)

        # --- MAVROS Publisher --- (remains the same)
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_profile_setpoint)

        # --- MAVROS Service Clients --- (remain the same)
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

        # --- Timed Trajectory State ---
        self.trajectory_start_time: Time | None = None # Use rclpy Time object

        # REMOVE: self.dummy_trajectory = self._create_dummy_trajectory()
        # REMOVE: self.current_trajectory_index = 0

        # Inform state manager (can set loaded=True if we assume trajectory is always ready)
        self.state_manager.set_trajectory_status(loaded=True, finished=False)

        # --- Timers --- (remain the same)
        self.setpoint_timer = self.create_timer(1.0 / publish_rate, self.publish_setpoint_loop)
        self.mission_logic_timer = self.create_timer(state_machine_period, self.run_mission_logic)

        self.get_logger().info("Mission Controller Node (Timed Trajectory Observer) Initialized.")
        self.get_logger().info("Please use QGroundControl to ARM and set OFFBOARD mode.")
        self.get_logger().info(f"Trajectory: Circle (R={self.trajectory_radius:.1f}m) for {self.trajectory_duration:.1f}s at Z={self.target_altitude:.1f}m.")


    # REMOVE: def _create_dummy_trajectory(self) -> list[Point]:

    # --- New Timed Trajectory Function ---
    def _get_timed_trajectory_point(self, elapsed_time_sec: float) -> tuple[float, float, float]:
        """Calculates the desired (x, y, z) for a given elapsed time."""
        # Example: Circular trajectory
        # Ensure we don't divide by zero if duration is zero
        if self.trajectory_duration <= 0:
            return (self.hover_pose.pose.position.x,
                    self.hover_pose.pose.position.y,
                    self.target_altitude)

        # Normalize time to fraction of trajectory duration
        time_fraction = elapsed_time_sec / self.trajectory_duration
        angle = 2 * math.pi * time_fraction # Complete one circle over the duration

        # Calculate position on the circle (offset from initial hover X/Y)
        center_x = self.hover_pose.pose.position.x
        center_y = self.hover_pose.pose.position.y
        radius = self.trajectory_radius

        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        z = self.target_altitude # Maintain constant altitude

        return x, y, z


    # --- Callback Functions --- (remain the same)
    def state_callback(self, msg):
        self.current_mavros_state = msg

    def local_pos_callback(self, msg):
        self.current_local_pos = msg
        # We check altitude reached condition inside run_mission_logic now

    # --- Setpoint Publishing Loop --- (remain the same)
    def publish_setpoint_loop(self):
        if not self.setpoint_streaming_active: return
        # Ensure header is updated just before publishing
        self.setpoint_to_publish.header.stamp = self.get_clock().now().to_msg()
        self.setpoint_to_publish.header.frame_id = self.coordinate_frame
        self.setpoint_pub.publish(self.setpoint_to_publish)

    # --- start/update/stop_setpoint_streaming --- (remain the same)
    def start_setpoint_streaming(self, initial_setpoint: PoseStamped):
        if not self.setpoint_streaming_active:
            self.get_logger().info(f"Starting setpoint streaming to target: X={initial_setpoint.pose.position.x:.2f}, Y={initial_setpoint.pose.position.y:.2f}, Z={initial_setpoint.pose.position.z:.2f}")
            self.setpoint_to_publish = copy.deepcopy(initial_setpoint) # Use deepcopy for safety
            self.setpoint_streaming_active = True
        # Always update the target even if already streaming
        self.update_streaming_setpoint(initial_setpoint)


    def update_streaming_setpoint(self, new_setpoint_pose: PoseStamped):
         self.setpoint_to_publish = copy.deepcopy(new_setpoint_pose)


    def stop_setpoint_streaming(self):
        if self.setpoint_streaming_active:
            self.get_logger().info("Stopping setpoint streaming.")
            self.setpoint_streaming_active = False

    # --- MAVROS Service Call Helpers --- (remain the same, still unused in main flow)
    def call_arming_service_sync(self, value: bool):
        # ... (sync implementation) ...
        pass # Keeping implementation for brevity, but it's the same

    def call_set_mode_service_sync(self, mode: str):
        # ... (sync implementation) ...
        pass # Keeping implementation for brevity, but it's the same

    # --- Main Mission Logic Loop (Modified for Timed Trajectory) ---
    def run_mission_logic(self):
        """
        Periodically checks the state manager and performs actions
        based on the *observed* current state. Includes timed trajectory logic.
        """
        # 1. Ask the state manager to determine the current state
        new_state = self.state_manager.update_state(
            self.current_mavros_state,
            self.current_local_pos
        )

        # Log state transitions and reset flags
        if new_state != self.current_drone_state:
            self.get_logger().info(f"Mission Controller: Detected state change {self.current_drone_state.name} -> {new_state.name}")

            # Reset flags/state when entering relevant states
            if new_state == DroneState.OFFBOARD_ACTIVE:
                self.altitude_reached = False # Reset altitude flag on entering
                self.trajectory_start_time = None # Reset trajectory timer
                self.state_manager.set_trajectory_status(loaded=True, finished=False) # Mark trajectory as not finished

            # Reset flags/state when leaving OFFBOARD_ACTIVE
            if self.current_drone_state == DroneState.OFFBOARD_ACTIVE and new_state != DroneState.OFFBOARD_ACTIVE:
                 self.trajectory_start_time = None # Clear start time if we leave the state

            # Initialize landing setpoint when starting descent
            if new_state == DroneState.DESCENDING:
                self.trajectory_start_time = None # Ensure trajectory timer is cleared
                self.landing_setpoint_pose = PoseStamped()
                self.landing_setpoint_pose.header.frame_id = self.coordinate_frame
                # Start descent from current XY, target landing XY
                self.landing_setpoint_pose.pose.position.x = self.landing_target_x
                self.landing_setpoint_pose.pose.position.y = self.landing_target_y
                self.landing_setpoint_pose.pose.position.z = self.current_local_pos.pose.position.z # Start from current Z
                self.landing_setpoint_pose.pose.orientation.w = 1.0
                self.get_logger().info(f"Initiating descent towards ({self.landing_target_x:.2f}, {self.landing_target_y:.2f}) from Z={self.landing_setpoint_pose.pose.position.z:.2f}m")

            self.current_drone_state = new_state


        # 2. Perform actions based on the *current observed* state
        try:
            # Start streaming setpoints when armed and waiting for offboard
            if self.current_drone_state == DroneState.ARMED_WAITING_FOR_MODE:
                self.start_setpoint_streaming(self.hover_pose) # Stream hover pose

            # Handle hover and trajectory following when offboard is active
            elif self.current_drone_state == DroneState.OFFBOARD_ACTIVE:
                # Check if target altitude has been reached
                if not self.altitude_reached:
                    altitude_error = abs(self.current_local_pos.pose.position.z - self.target_altitude)
                    self.altitude_reached = altitude_error < self.altitude_reach_threshold
                    if self.altitude_reached:
                        self.get_logger().info("Mission Controller: Altitude reached.")
                    else:
                        # Still climbing/holding hover altitude
                        self.start_setpoint_streaming(self.hover_pose) # Continue publishing hover setpoint
                        self.get_logger().info(f"Mission Controller: OFFBOARD ACTIVE - Holding hover / climbing. Current Z: {self.current_local_pos.pose.position.z:.2f}m", throttle_duration_sec=2.0)

                # Altitude is reached, start/continue timed trajectory
                if self.altitude_reached:
                    if self.trajectory_start_time is None:
                        self.get_logger().info("Mission Controller: Starting timed trajectory.")
                        self.trajectory_start_time = self.get_clock().now()

                    # Calculate elapsed time
                    elapsed_time = (self.get_clock().now() - self.trajectory_start_time).nanoseconds / 1e9

                    # Check if trajectory duration exceeded
                    if elapsed_time > self.trajectory_duration:
                        self.get_logger().info(f"Mission Controller: Timed trajectory finished (elapsed: {elapsed_time:.2f}s).")
                        self.state_manager.set_trajectory_status(loaded=True, finished=True)
                        self.trajectory_start_time = None # Reset start time
                        # State manager will transition to DESCENDING on the next loop
                        # Optionally, publish the final trajectory point or hover pose here briefly
                        last_x, last_y, last_z = self._get_timed_trajectory_point(self.trajectory_duration)
                        final_pose = self._create_pose_stamped_from_point(Point(x=last_x, y=last_y, z=last_z))
                        self.update_streaming_setpoint(final_pose)

                    else:
                        # Get the trajectory point for the current elapsed time
                        target_x, target_y, target_z = self._get_timed_trajectory_point(elapsed_time)

                        # Create and publish the setpoint
                        target_pose_msg = self._create_pose_stamped_from_point(Point(x=target_x, y=target_y, z=target_z))
                        self.update_streaming_setpoint(target_pose_msg)
                        self.start_setpoint_streaming(target_pose_msg) # Ensure streaming is active

                        self.get_logger().info(f"Mission Controller: Following trajectory - t={elapsed_time:.2f}s -> Target(X:{target_x:.2f}, Y:{target_y:.2f}, Z:{target_z:.2f})", throttle_duration_sec=1.0)


            # Handle controlled descent (remains the same)
            elif self.current_drone_state == DroneState.DESCENDING:
                if self.landing_setpoint_pose is not None:
                    self.landing_setpoint_pose.pose.position.z -= self.descent_step
                    # Ensure Z doesn't go too low and stays at landing XY
                    self.landing_setpoint_pose.pose.position.z = max(self.landing_setpoint_pose.pose.position.z, -0.1)
                    self.landing_setpoint_pose.pose.position.x = self.landing_target_x
                    self.landing_setpoint_pose.pose.position.y = self.landing_target_y

                    self.update_streaming_setpoint(self.landing_setpoint_pose)
                    self.start_setpoint_streaming(self.landing_setpoint_pose) # Ensure streaming
                    self.get_logger().info(f"Mission Controller: DESCENDING - Target Z: {self.landing_setpoint_pose.pose.position.z:.2f}m", throttle_duration_sec=1.0)
                else:
                    self.get_logger().error("Mission Controller: In DESCENDING state but landing_setpoint_pose is None!")
                    self.stop_setpoint_streaming()

            # Keep streaming when grounded (remains the same)
            elif self.current_drone_state == DroneState.GROUNDED:
                if self.landing_setpoint_pose is not None:
                    self.landing_setpoint_pose.pose.position.z = max(self.landing_setpoint_pose.pose.position.z, -0.1)
                    self.landing_setpoint_pose.pose.position.x = self.landing_target_x
                    self.landing_setpoint_pose.pose.position.y = self.landing_target_y
                    self.start_setpoint_streaming(self.landing_setpoint_pose)
                    self.get_logger().info("Mission Controller: Landed. Publishing final setpoint. Waiting for manual DISARM via QGC.", throttle_duration_sec=5.0)
                else:
                    self.stop_setpoint_streaming()


            # Stop streaming only when mission complete or in initial states (remains the same)
            elif self.current_drone_state in [DroneState.MISSION_COMPLETE, DroneState.INIT, DroneState.IDLE, DroneState.WAITING_FOR_ARM]:
                self.stop_setpoint_streaming()
                if self.current_drone_state == DroneState.MISSION_COMPLETE:
                    self.get_logger().info("Mission Controller: Mission complete. Setpoint streaming stopped.", throttle_duration_sec=10.0)


        except Exception as e:
            self.get_logger().error(f"Error in mission logic loop: {e}\n{traceback.format_exc()}")
            self.stop_setpoint_streaming() # Stop streaming on error

    # REMOVE: def _is_close_to_target(self, target_position: Point) -> bool:

    # --- create_pose_stamped_from_point --- (remain the same)
    def _create_pose_stamped_from_point(self, point: Point) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.coordinate_frame
        # Stamp is updated in publish_setpoint_loop
        pose.pose.position.x = point.x
        pose.pose.position.y = point.y
        pose.pose.position.z = point.z
        pose.pose.orientation.w = 1.0 # Keep level orientation for simplicity
        return pose

    # --- destroy_node --- (remains the same)
    def destroy_node(self):
        self.get_logger().info("Shutting down Mission Controller Node...")
        self.stop_setpoint_streaming()
        if hasattr(self, 'setpoint_timer') and self.setpoint_timer: self.setpoint_timer.cancel()
        if hasattr(self, 'mission_logic_timer') and self.mission_logic_timer: self.mission_logic_timer.cancel()
        super().destroy_node()

# --- main function --- (remains the same)
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MissionControllerNode()
        # Use MultiThreadedExecutor if you have multiple callbacks/timers/service calls
        # If not strictly needed, SingleThreadedExecutor might be simpler.
        # MultiThreadedExecutor is generally safer for nodes doing many things.
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            # Important: Shutdown the executor and destroy the node explicitly
            executor.shutdown()
            if node and rclpy.ok(): # Check if node exists and rclpy is still running
                 node.destroy_node()
    except KeyboardInterrupt:
        print("Ctrl+C detected, shutting down...")
    # except Exception as e: # Catch other potential exceptions during spin
    #      if node:
    #          node.get_logger().error(f"Unhandled exception during spin: {e}\n{traceback.format_exc()}")
    finally:
        # Ensure rclpy is shutdown cleanly
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
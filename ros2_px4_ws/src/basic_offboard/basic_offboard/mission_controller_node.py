#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
import asyncio

# --- Import our state definitions and manager ---
from .state_manager import DroneState, StateManager # Uses the updated state_manager.py

# --- Import ROS message/service types ---
from geometry_msgs.msg import PoseStamped, Point # Point for dummy trajectory
from std_msgs.msg import Header
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import math # For distance calculations
import traceback # Import traceback module
import copy # To copy pose message for landing

class MissionControllerNode(Node):

    def __init__(self):
        super().__init__('mission_controller_node')

        # --- Parameters ---
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('state_machine_rate_hz', 5.0) # Rate for logic checks
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_z', 1.5) # Target hover altitude
        self.declare_parameter('coordinate_frame', 'map')
        self.declare_parameter('waypoint_reach_threshold', 0.2) # meters
        self.declare_parameter('altitude_reach_threshold', 0.2) # meters
        self.declare_parameter('descent_rate', 0.3) # meters per second for landing
        self.declare_parameter('landing_x', 0.0) # Target X for landing
        self.declare_parameter('landing_y', 0.0) # Target Y for landing

        publish_rate = self.get_parameter('publish_rate_hz').value
        state_machine_period = 1.0 / self.get_parameter('state_machine_rate_hz').value # Store period
        self.target_altitude = self.get_parameter('target_z').value
        self.coordinate_frame = self.get_parameter('coordinate_frame').value
        self.waypoint_reach_threshold_sq = self.get_parameter('waypoint_reach_threshold').value ** 2
        self.altitude_reach_threshold = self.get_parameter('altitude_reach_threshold').value
        self.descent_rate = self.get_parameter('descent_rate').value
        self.descent_step = self.descent_rate * state_machine_period # Z decrease per logic cycle
        self.landing_target_x = self.get_parameter('landing_x').value
        self.landing_target_y = self.get_parameter('landing_y').value


        # Target pose for initial hover
        self.hover_pose = PoseStamped()
        self.hover_pose.header.frame_id = self.coordinate_frame
        self.hover_pose.pose.position.x = self.get_parameter('target_x').value
        self.hover_pose.pose.position.y = self.get_parameter('target_y').value
        self.hover_pose.pose.position.z = self.target_altitude
        self.hover_pose.pose.orientation.w = 1.0 # Level flight

        # --- QoS Profiles ---
        qos_profile_state = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_profile_setpoint = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_profile_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        # --- Instantiate State Manager ---
        self.state_manager = StateManager(self.get_logger(),
                                           target_altitude=self.target_altitude,
                                           altitude_threshold=self.altitude_reach_threshold)

        # --- MAVROS Subscribers ---
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, qos_profile_state)
        self.local_pos_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pos_callback, qos_profile_best_effort)

        # --- MAVROS Publisher ---
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_profile_setpoint)

        # --- MAVROS Service Clients (Defined but not called in main loop) ---
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # --- Node State / Data ---
        self.current_mavros_state = State()       # Latest state from MAVROS
        self.current_local_pos = PoseStamped()    # Latest position from MAVROS
        self.current_drone_state = DroneState.INIT # Get initial state from manager
        self.setpoint_to_publish = PoseStamped()  # The setpoint currently being sent
        self.setpoint_streaming_active = False    # Controls the setpoint timer
        self.altitude_reached = False             # Flag for reaching hover altitude
        self.landing_setpoint_pose = None         # Store pose used during descent

        # --- Dummy Trajectory Data ---
        self.dummy_trajectory = self._create_dummy_trajectory()
        self.current_trajectory_index = 0
        # Inform state manager about trajectory status
        self.state_manager.set_trajectory_status(loaded=(self.dummy_trajectory is not None), finished=False)

        # --- Timers ---
        self.setpoint_timer = self.create_timer(1.0 / publish_rate, self.publish_setpoint_loop)
        # Main logic timer - calls the state manager and performs actions
        self.mission_logic_timer = self.create_timer(state_machine_period, self.run_mission_logic) # Use sync version for now

        self.get_logger().info("Mission Controller Node (Controlled Descent Observer) Initialized.")
        self.get_logger().info("Please use QGroundControl to ARM and set OFFBOARD mode.")


    def _create_dummy_trajectory(self) -> list[Point]:
        """Creates a simple square trajectory."""
        points = []
        alt = self.target_altitude
        points.append(Point(x=0.0, y=0.0, z=alt)) # Start (same as hover)
        points.append(Point(x=2.0, y=0.0, z=alt))
        points.append(Point(x=2.0, y=2.0, z=alt))
        points.append(Point(x=0.0, y=2.0, z=alt))
        points.append(Point(x=0.0, y=0.0, z=alt)) # Return to start
        self.get_logger().info(f"Created dummy trajectory with {len(points)} points.")
        return points

    # --- Callback Functions ---
    def state_callback(self, msg):
        self.current_mavros_state = msg

    def local_pos_callback(self, msg):
        self.current_local_pos = msg
        if self.current_drone_state in [DroneState.OFFBOARD_ACTIVE]:
             altitude_error = abs(self.current_local_pos.pose.position.z - self.target_altitude)
             self.altitude_reached = altitude_error < self.altitude_reach_threshold

    # --- Setpoint Publishing Loop ---
    def publish_setpoint_loop(self):
        if not self.setpoint_streaming_active: return
        self.setpoint_to_publish.header.stamp = self.get_clock().now().to_msg()
        self.setpoint_to_publish.header.frame_id = self.coordinate_frame
        self.setpoint_pub.publish(self.setpoint_to_publish)

    def start_setpoint_streaming(self, initial_setpoint: PoseStamped):
        if not self.setpoint_streaming_active:
            self.get_logger().info(f"Starting setpoint streaming to target: X={initial_setpoint.pose.position.x:.2f}, Y={initial_setpoint.pose.position.y:.2f}, Z={initial_setpoint.pose.position.z:.2f}")
            self.setpoint_to_publish = initial_setpoint
            self.setpoint_streaming_active = True

    def update_streaming_setpoint(self, new_setpoint_pose: PoseStamped):
         # Deep copy might be safer if the original object is modified elsewhere
         self.setpoint_to_publish = copy.deepcopy(new_setpoint_pose)
         # Ensure header is set correctly (might be overwritten by deepcopy if not careful)
         self.setpoint_to_publish.header.frame_id = self.coordinate_frame


    def stop_setpoint_streaming(self):
        if self.setpoint_streaming_active:
            self.get_logger().info("Stopping setpoint streaming.")
            self.setpoint_streaming_active = False

    # --- MAVROS Service Call Helpers (Defined but not called in main loop) ---
    # Keep the synchronous versions for potential manual debugging if needed later
    def call_arming_service_sync(self, value: bool):
        if not self.arming_client.service_is_ready(): self.get_logger().error("Arming service client not available!"); return False
        req = CommandBool.Request(); req.value = value
        self.get_logger().info(f"Requesting {'Arming' if value else 'Disarming'} (sync)...")
        try:
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
            response = self.set_mode_client.call(req)
            if response is None: self.get_logger().error("Set_mode service call failed (sync - no response/timeout)."); return False
            if response.mode_sent: self.get_logger().info(f"Mode change request '{mode}' sent successfully by MAVROS (sync)."); return True
            else: self.get_logger().error(f"MAVROS failed to send mode change request '{mode}' (sync)."); return False
        except Exception as e: self.get_logger().error(f"Set_mode service call failed (sync): {e}\n{traceback.format_exc()}"); return False


    # --- Main Mission Logic Loop (Observer Actions with Controlled Descent) ---
    def run_mission_logic(self):
        """
        Periodically checks the state manager and performs actions
        based on the *observed* current state. Commands descent via setpoints.
        """
        # 1. Ask the state manager to determine the current state
        new_state = self.state_manager.update_state(
            self.current_mavros_state,
            self.current_local_pos
        )
        # Only update internal state if state manager calculation changed it
        if new_state != self.current_drone_state:
            self.get_logger().info(f"Mission Controller: Detected state change to {new_state.name}")
            # Reset flags/indices when entering relevant states
            if new_state == DroneState.OFFBOARD_ACTIVE:
                self.current_trajectory_index = 0
                self.altitude_reached = False # Reset altitude flag
                self.state_manager.set_trajectory_status(loaded=True, finished=False) # Reset finished flag
            # Initialize landing setpoint when starting descent
            if new_state == DroneState.DESCENDING:
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
            # Start streaming setpoints when armed and waiting for offboard
            if self.current_drone_state == DroneState.ARMED_WAITING_FOR_MODE:
                self.start_setpoint_streaming(self.hover_pose)

            # Handle hover and trajectory following when offboard is active
            elif self.current_drone_state == DroneState.OFFBOARD_ACTIVE:
                # ... (trajectory following logic remains the same) ...
                if not self.altitude_reached:
                    self.start_setpoint_streaming(self.hover_pose)
                    self.get_logger().info(f"Mission Controller: OFFBOARD ACTIVE - Holding hover / climbing. Alt reached: {self.altitude_reached}", throttle_duration_sec=2.0)
                else:
                    if self.dummy_trajectory and not self.state_manager.trajectory_finished:
                        if self.current_trajectory_index < len(self.dummy_trajectory):
                            target_point = self.dummy_trajectory[self.current_trajectory_index]
                            target_pose_msg = self._create_pose_stamped_from_point(target_point)
                            self.update_streaming_setpoint(target_pose_msg)
                            self.start_setpoint_streaming(target_pose_msg)
                            self.get_logger().debug(f"Mission Controller: Following waypoint {self.current_trajectory_index}")
                            if self._is_close_to_target(target_point):
                                self.get_logger().info(f"Mission Controller: Reached waypoint {self.current_trajectory_index}.")
                                self.current_trajectory_index += 1
                                if self.current_trajectory_index >= len(self.dummy_trajectory):
                                    self.get_logger().info("Mission Controller: Dummy trajectory finished.")
                                    self.state_manager.set_trajectory_status(loaded=True, finished=True)
                        else:
                            self.get_logger().warn("Mission Controller: Trajectory index out of bounds but not marked finished?")
                            self.state_manager.set_trajectory_status(loaded=True, finished=True)
                    else:
                        self.get_logger().info("Mission Controller: OFFBOARD ACTIVE - Altitude reached, hovering.", throttle_duration_sec=2.0)
                        self.start_setpoint_streaming(self.hover_pose)


            # Handle controlled descent
            elif self.current_drone_state == DroneState.DESCENDING:
                if self.landing_setpoint_pose is not None:
                    self.landing_setpoint_pose.pose.position.z -= self.descent_step
                    self.landing_setpoint_pose.pose.position.z = max(self.landing_setpoint_pose.pose.position.z, -0.1)
                    self.update_streaming_setpoint(self.landing_setpoint_pose)
                    self.start_setpoint_streaming(self.landing_setpoint_pose) # Ensure streaming
                    self.get_logger().info(f"Mission Controller: DESCENDING - Target Z: {self.landing_setpoint_pose.pose.position.z:.2f}m", throttle_duration_sec=1.0)
                else:
                    self.get_logger().error("Mission Controller: In DESCENDING state but landing_setpoint_pose is None!")
                    self.stop_setpoint_streaming()

            # --- MODIFIED: Keep streaming when grounded ---
            elif self.current_drone_state == DroneState.GROUNDED:
                # Keep publishing the last landing setpoint to prevent failsafe
                # while waiting for manual disarm.
                if self.landing_setpoint_pose is not None:
                    # Ensure Z is very low
                    self.landing_setpoint_pose.pose.position.z = max(self.landing_setpoint_pose.pose.position.z, -0.1)
                    self.start_setpoint_streaming(self.landing_setpoint_pose) # Keep streaming
                    self.get_logger().info("Mission Controller: Landed. Publishing final setpoint. Waiting for manual DISARM via QGC.", throttle_duration_sec=5.0)
                else:
                    # Fallback if landing_setpoint is somehow None
                    self.stop_setpoint_streaming()


            # --- MODIFIED: Only stop streaming when mission complete or in initial states ---
            elif self.current_drone_state in [DroneState.MISSION_COMPLETE, DroneState.INIT, DroneState.IDLE, DroneState.WAITING_FOR_ARM]:
                self.stop_setpoint_streaming()
                if self.current_drone_state == DroneState.MISSION_COMPLETE:
                    self.get_logger().info("Mission Controller: Mission complete. Setpoint streaming stopped.", throttle_duration_sec=10.0)


        except Exception as e:
            self.get_logger().error(f"Error in mission logic loop: {e}\n{traceback.format_exc()}")
            self.stop_setpoint_streaming() # Stop streaming on error


    def _is_close_to_target(self, target_position: Point) -> bool:
        if self.current_local_pos.header.stamp.sec == 0: return False
        dx = self.current_local_pos.pose.position.x - target_position.x
        dy = self.current_local_pos.pose.position.y - target_position.y
        dz = self.current_local_pos.pose.position.z - target_position.z
        dist_sq = dx*dx + dy*dy + dz*dz
        return dist_sq < self.waypoint_reach_threshold_sq

    def _create_pose_stamped_from_point(self, point: Point) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.coordinate_frame
        pose.pose.position.x = point.x
        pose.pose.position.y = point.y
        pose.pose.position.z = point.z
        pose.pose.orientation.w = 1.0
        return pose

    def destroy_node(self):
        self.get_logger().info("Shutting down Mission Controller Node...")
        self.stop_setpoint_streaming()
        if hasattr(self, 'setpoint_timer') and self.setpoint_timer: self.setpoint_timer.cancel()
        if hasattr(self, 'mission_logic_timer') and self.mission_logic_timer: self.mission_logic_timer.cancel()
        super().destroy_node()


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
            if node:
                 node.destroy_node()
    except KeyboardInterrupt:
        print("Ctrl+C detected, shutting down...")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
# File: state_manager.py
# Purpose: Defines states and handles state transition logic, including controlled descent.

from enum import Enum
from rclpy.node import Node # Import Node only for get_logger, not full node functionality
from rclpy.clock import Clock, ClockType

# Import message types used for checking conditions
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
import math # For distance calculations

class DroneState(Enum):
    INIT = 0                # Initial state, waiting for connection
    IDLE = 1                # Connected, waiting for external arm command
    WAITING_FOR_ARM = 2     # Signifies intent, waiting for armed confirmation
    ARMED_WAITING_FOR_MODE = 3 # Armed, waiting for external Offboard mode command
    OFFBOARD_ACTIVE = 4     # In offboard, ready to execute mission (hover/trajectory)
    DESCENDING = 5          # Trajectory finished, actively commanding descent in Offboard mode
    GROUNDED = 6            # Detected low altitude after descent, waiting for manual disarm
    MISSION_COMPLETE = 7    # Disarmed after landing (final state for observation)

class StateManager():
    def __init__(self, node_logger, target_altitude=1.5, altitude_threshold=0.2, ground_threshold=0.15):
        """
        Initializes the StateManager for observing drone state and controlled descent.
        """
        self.current_state = DroneState.INIT
        self.logger = node_logger
        self.target_altitude = target_altitude
        self.altitude_threshold = altitude_threshold
        self.ground_threshold = ground_threshold
        self.trajectory_loaded = False # Flag controlled by the main node
        self.trajectory_finished = False # Flag controlled by the main node
        self.clock = Clock(clock_type=ClockType.ROS_TIME) # Use ROS time

        self.logger.info("StateManager (Observer Mode with Controlled Descent) initialized.")

    def set_trajectory_status(self, loaded: bool, finished: bool):
        """Allows the main node to inform the state manager about the trajectory."""
        if self.trajectory_loaded != loaded or self.trajectory_finished != finished:
             self.logger.debug(f"StateManager: Trajectory status updated - Loaded={loaded}, Finished={finished}")
        self.trajectory_loaded = loaded
        self.trajectory_finished = finished

    def update_state(self, current_mavros_state: State, current_local_pos: PoseStamped) -> DroneState:
        """
        Determines the next state based purely on observed MAVROS state and position,
        and trajectory status.
        """
        next_state = self.current_state # Assume no change unless conditions met

        # --- State Transition Logic (Observer with Controlled Descent) ---
        if self.current_state == DroneState.INIT:
            if current_mavros_state.connected:
                self.logger.info("StateManager: Condition met - MAVROS connected.")
                next_state = DroneState.IDLE
        elif self.current_state == DroneState.IDLE:
            self.logger.debug("StateManager: In IDLE, proceeding to wait for arm.")
            next_state = DroneState.WAITING_FOR_ARM
        elif self.current_state == DroneState.WAITING_FOR_ARM:
            if current_mavros_state.armed:
                self.logger.info("StateManager: Condition met - Drone ARMED externally.")
                next_state = DroneState.ARMED_WAITING_FOR_MODE
            elif not current_mavros_state.connected: # Failsafe
                 self.logger.warn("StateManager: MAVROS disconnected while waiting for arm!")
                 next_state = DroneState.INIT
        elif self.current_state == DroneState.ARMED_WAITING_FOR_MODE:
            if current_mavros_state.mode == "OFFBOARD":
                self.logger.info("StateManager: Condition met - OFFBOARD mode activated externally.")
                next_state = DroneState.OFFBOARD_ACTIVE
            elif not current_mavros_state.armed: # If disarmed before Offboard
                 self.logger.warn("StateManager: Drone DISARMED externally before Offboard mode set.")
                 next_state = DroneState.IDLE # Go back to IDLE

        elif self.current_state == DroneState.OFFBOARD_ACTIVE:
            # Check if trajectory finished OR if we lost Offboard/Armed status
            if not current_mavros_state.armed:
                 self.logger.warn("StateManager: Drone DISARMED externally while Offboard active! Switching to GROUNDED.")
                 next_state = DroneState.GROUNDED # Go directly to wait for final disarm confirmation
            elif current_mavros_state.mode != "OFFBOARD":
                 self.logger.warn(f"StateManager: Mode changed externally from OFFBOARD to {current_mavros_state.mode}. Switching to GROUNDED.")
                 next_state = DroneState.GROUNDED # Assume landing happened or intervention occurred
            elif self.trajectory_finished:
                 self.logger.info("StateManager: Condition met - Trajectory finished. Initiating controlled descent.")
                 next_state = DroneState.DESCENDING # <-- Transition to DESCENDING
            # else: remain in OFFBOARD_ACTIVE (main node handles hover/trajectory logic)

        elif self.current_state == DroneState.DESCENDING:
            # Check if we've reached the ground while descending
            is_on_ground = current_local_pos.pose.position.z < self.ground_threshold # Use current Z
            # Also check if we somehow lost Offboard/Armed state during descent
            if not current_mavros_state.armed:
                self.logger.warn("StateManager: Drone DISARMED externally during descent! Switching to GROUNDED.")
                next_state = DroneState.GROUNDED
            elif current_mavros_state.mode != "OFFBOARD":
                 self.logger.warn(f"StateManager: Mode changed externally from OFFBOARD to {current_mavros_state.mode} during descent. Switching to GROUNDED.")
                 next_state = DroneState.GROUNDED
            elif is_on_ground:
                self.logger.info("StateManager: Condition met - Drone appears landed (low altitude during descent).")
                next_state = DroneState.GROUNDED # Proceed to wait for manual disarm
            # else: remain in DESCENDING (main node continues sending lower Z setpoints)

        elif self.current_state == DroneState.GROUNDED:
            # Wait for external disarm command
            if not current_mavros_state.armed:
                self.logger.info("StateManager: Condition met - Drone DISARMED externally after landing.")
                next_state = DroneState.MISSION_COMPLETE
            # else: remain in GROUNDED

        elif self.current_state == DroneState.MISSION_COMPLETE:
            # Final state
            pass

        # --- Update and return the state ---
        if next_state != self.current_state:
            self.logger.info(f"StateManager: Calculated state transition: {self.current_state.name} -> {next_state.name}")
            self.current_state = next_state

        return self.current_state
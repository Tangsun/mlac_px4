# state_manager.py (Corrected Version)

from enum import Enum, auto
# ++ Import Clock type for type hinting ++
from rclpy.clock import Clock
# ++ Import message types used for checks ++
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
# ++ Import Logger for type hinting ++
from rclpy.impl.rcutils_logger import RcutilsLogger

class DroneState(Enum):
    INIT = auto()
    IDLE = auto()
    ARMED_WAITING_FOR_MODE = auto()
    OFFBOARD_ACTIVE = auto()
    LANDING = auto()
    MISSION_COMPLETE = auto()

class StateManager:
    # ++ CORRECTED __init__ signature and usage ++
    def __init__(self, logger: RcutilsLogger, clock: Clock, ground_threshold=0.15): # Accept clock
        """
        Manages high-level drone state based on MAVROS feedback.

        Args:
            logger: ROS 2 Logger object from the parent node.
            clock: ROS 2 Clock object from the parent node. # Added clock arg
            ground_threshold (float): Altitude threshold (meters) to consider drone near ground.
        """
        self.logger = logger
        self._clock = clock # Store the passed clock object
        self.ground_threshold = ground_threshold

        # Status flags updated by the controller node
        self.trajectory_loaded = False
        self.trajectory_finished = False

        # Internal state tracking
        self.current_state = DroneState.INIT
        # CORRECTED: Use the stored clock object now
        self.last_state_log_time = self._clock.now()
        self.last_state_logged = None
    # ++ End __init__ correction ++

    def set_trajectory_status(self, loaded: bool, finished: bool):
        """Allows the controller node to signal trajectory status."""
        status_changed = (self.trajectory_loaded != loaded or self.trajectory_finished != finished)
        self.trajectory_loaded = loaded
        self.trajectory_finished = finished
        # Optionally log change if needed
        # if status_changed:
        #     self.logger.info(f"StateManager: Trajectory status updated -> Loaded={loaded}, Finished={finished}")

    def _log_state_change(self, old_state, new_state):
        """Logs state transitions."""
        if new_state != old_state:
             # Use debug level for frequent state logging if desired
             self.logger.info(f"StateManager: State Transition: {old_state.name} -> {new_state.name}")

    def update_state(self, mavros_state: State, local_pos: PoseStamped) -> DroneState:
        """
        Determines the current high-level drone state based on inputs.
        (Logic from previous correct version - No changes needed here if __init__ is fixed)
        """
        old_state = self.current_state
        next_state = old_state

        if not mavros_state.connected:
            next_state = DroneState.INIT
            self._log_state_change(old_state, next_state)
            self.current_state = next_state
            return next_state

        current_altitude = 0.0
        position_valid = local_pos.header.stamp.sec > 0
        if position_valid: current_altitude = local_pos.pose.position.z
        is_near_ground = position_valid and (current_altitude < self.ground_threshold)

        if old_state == DroneState.INIT:
            if mavros_state.connected: next_state = DroneState.IDLE
        elif old_state == DroneState.IDLE:
            if mavros_state.armed: next_state = DroneState.ARMED_WAITING_FOR_MODE
        elif old_state == DroneState.ARMED_WAITING_FOR_MODE:
            if not mavros_state.armed:
                self.logger.warn("StateManager: Disarmed externally while waiting for Offboard.")
                next_state = DroneState.IDLE
            elif mavros_state.mode == "OFFBOARD":
                self.trajectory_finished = False # Reset flag
                next_state = DroneState.OFFBOARD_ACTIVE
        elif old_state == DroneState.OFFBOARD_ACTIVE:
            if not mavros_state.armed:
                self.logger.warn("StateManager: Disarmed externally during Offboard Active!")
                next_state = DroneState.LANDING
            elif mavros_state.mode != "OFFBOARD":
                self.logger.warn(f"StateManager: Mode changed externally from OFFBOARD to {mavros_state.mode}!")
                next_state = DroneState.LANDING # Default to landing on mode change
            elif self.trajectory_finished:
                 self.logger.info("StateManager: Trajectory finished signaled by controller.")
                 next_state = DroneState.LANDING
        elif old_state == DroneState.LANDING:
            if not mavros_state.armed:
                self.logger.info("StateManager: Disarmed after/during landing.")
                next_state = DroneState.MISSION_COMPLETE
            elif mavros_state.mode != "OFFBOARD" and mavros_state.mode != "AUTO.LAND":
                 self.logger.warn(f"StateManager: Mode changed to {mavros_state.mode} during LANDING state!")
                 # Stay landing unless disarmed
        elif old_state == DroneState.MISSION_COMPLETE:
            if mavros_state.armed:
                self.logger.info("StateManager: Re-armed after mission complete. Resetting state.")
                next_state = DroneState.ARMED_WAITING_FOR_MODE

        self._log_state_change(old_state, next_state)
        self.current_state = next_state
        return next_state

    def get_current_state(self) -> DroneState:
        """Returns the last calculated state."""
        return self.current_state
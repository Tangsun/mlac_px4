import numpy as np
from enum import Enum, auto
from rclpy.time import Time, Duration
from rclpy.node import Node # For type hinting logger, clock

from .structs import GoalClass, StateClass # Assuming these are in the same package
from mavros_msgs.msg import State as MavrosState # For MAVROS state type hint

class MissionPhase(Enum):
    IDLE = auto()                           # Node is running but no command active, streams current pose
    AWAITING_OFFBOARD_AND_ARM = auto()      # Received START, waiting for MAVROS state
    TAKING_OFF_TO_INITIAL_HOVER = auto()
    AT_INITIAL_HOVER = auto()               # Hovering for a defined duration
    MOVING_TO_TRAJECTORY_START = auto()
    EXECUTING_TRAJECTORY = auto()
    MOVING_TO_FINAL_HOVER = auto()
    AT_FINAL_HOVER = auto()                 # Hovering for a defined duration
    LANDING = auto()
    LANDED = auto()                         # Ground detected after landing sequence
    USER_COMMANDED_HOLD = auto()            # User issued "HOLD_POSITION"

class MissionFiniteStateMachine:
    def __init__(self, logger: Node.get_logger, clock: Node.get_clock,
                 initial_hover_pos: list[float], final_hover_pos: list[float], landing_pos: list[float],
                 pos_reached_thresh: float, hover_duration_sec: float, landing_descent_rate_mps: float,
                 wait_for_offboard_arm_timeout_sec: float):
        self.logger = logger
        self.clock = clock

        self.initial_hover_pos_param = np.array(initial_hover_pos)
        self.final_hover_pos_param = np.array(final_hover_pos)
        self.landing_pos_param = np.array(landing_pos) # Z component is target ground altitude
        self.pos_reached_thresh_sq = pos_reached_thresh**2 # Use squared for efficiency
        self.hover_duration = Duration(seconds=hover_duration_sec)
        self.landing_descent_rate = abs(landing_descent_rate_mps)
        self.wait_for_offboard_arm_timeout = Duration(seconds=wait_for_offboard_arm_timeout_sec)

        self.current_phase = MissionPhase.IDLE
        self.phase_start_time: Time | None = None
        self.command_pending_start_time: Time | None = None

        self.current_goal_py = GoalClass() # The goal to be sent to the controller
        self.hold_position_goal_py: GoalClass | None = None # For USER_COMMANDED_HOLD

        # Trajectory related data (to be set by the main node)
        self.trajectory_data: np.ndarray | None = None
        self.trajectory_start_file_time_offset: float = 0.0
        self.trajectory_start_point_goal_py: GoalClass | None = None
        self.trajectory_execution_start_time_ros: Time | None = None
        self.is_trajectory_loaded_fsm = False

        self.trajectory_completed_in_fsm = False # True when EXECUTE_TRAJECTORY finishes

        # Pre-defined goals for fixed points
        self.initial_hover_goal_py = self._create_goal_from_position_array(self.initial_hover_pos_param)
        self.final_hover_goal_py = self._create_goal_from_position_array(self.final_hover_pos_param)
        # Landing goal Z is dynamic

        self.logger.info("MissionFiniteStateMachine initialized.")

    def _create_goal_from_position_array(self, pos_array: np.ndarray, psi: float = 0.0,
                                         vel_array: np.ndarray = np.zeros(3),
                                         acc_array: np.ndarray = np.zeros(3),
                                         jerk_array: np.ndarray | None = None, # None for jerk
                                         dpsi: float | None = None) -> GoalClass: # None for dpsi
        goal = GoalClass()
        goal.t = self.clock.now().nanoseconds / 1e9 # Current time for goal
        goal.p = np.array(pos_array)
        goal.v = np.array(vel_array)
        goal.a = np.array(acc_array)
        goal.j = jerk_array
        goal.psi = psi
        goal.dpsi = dpsi
        goal.mode_xy = GoalClass.Mode.POS_CTRL
        goal.mode_z = GoalClass.Mode.POS_CTRL
        return goal

    def _is_at_target_pose(self, current_p_np: np.ndarray, target_p_np: np.ndarray) -> bool:
        dist_sq = np.sum((current_p_np - target_p_np)**2)
        return dist_sq < self.pos_reached_thresh_sq

    def set_trajectory_data(self, trajectory_data: np.ndarray | None):
        self.trajectory_data = trajectory_data
        if self.trajectory_data is not None and self.trajectory_data.shape[0] > 0:
            self.is_trajectory_loaded_fsm = True
            self.trajectory_start_file_time_offset = self.trajectory_data[0, 0]
            start_pos = self.trajectory_data[0, 1:4]
            start_vel = self.trajectory_data[0, 4:7]
            start_psi = self.trajectory_data[0, 7]
            start_acc = np.zeros(3)
            start_jerk = None
            start_dpsi = None
            if self.trajectory_data.shape[1] >= 11: # Accel
                start_acc = self.trajectory_data[0, 8:11]
            if self.trajectory_data.shape[1] >= 14: # Jerk
                start_jerk = self.trajectory_data[0, 11:14]
            if self.trajectory_data.shape[1] >= 15: # dPsi
                start_dpsi = self.trajectory_data[0, 14]

            self.trajectory_start_point_goal_py = self._create_goal_from_position_array(
                pos_array=start_pos, psi=start_psi, vel_array=start_vel, acc_array=start_acc, jerk_array=start_jerk, dpsi=start_dpsi
            )
            self.logger.info(f"FSM: Trajectory data set. Start point: P={start_pos}, V={start_vel}, Psi={start_psi:.2f}")
        else:
            self.is_trajectory_loaded_fsm = False
            self.trajectory_start_point_goal_py = None
            self.logger.warn("FSM: Cleared trajectory data or received empty data.")


    def _get_trajectory_goal_at_time_fsm(self, target_time_in_trajectory_timeline: float) -> GoalClass | None:
        if not self.is_trajectory_loaded_fsm or self.trajectory_data is None:
            return None
        traj_data = self.trajectory_data
        traj_file_times = traj_data[:, 0]
        clipped_target_time = np.clip(target_time_in_trajectory_timeline, traj_file_times[0], traj_file_times[-1])

        goal = GoalClass()
        goal.t = self.clock.now().nanoseconds / 1e9
        goal.p = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(1, 4)])
        goal.v = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(4, 7)])
        goal.psi = float(np.interp(clipped_target_time, traj_file_times, traj_data[:, 7]))

        num_cols = traj_data.shape[1]
        goal.a = np.zeros(3) # Default
        if num_cols >= 11: goal.a = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(8, 11)])
        goal.j = None
        if num_cols >= 14: goal.j = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(11, 14)])
        goal.dpsi = None
        if num_cols >= 15: goal.dpsi = float(np.interp(clipped_target_time, traj_file_times, traj_data[:, 14]))
        
        goal.mode_xy = GoalClass.Mode.POS_CTRL
        goal.mode_z = GoalClass.Mode.POS_CTRL
        return goal

    def process_command(self, command: str, current_vehicle_state: StateClass):
        self.logger.info(f"FSM received command: '{command}' in state {self.current_phase.name}")
        if command == "START_TRAJECTORY":
            if self.current_phase == MissionPhase.IDLE or self.current_phase == MissionPhase.LANDED:
                self._transition_to_phase(MissionPhase.AWAITING_OFFBOARD_AND_ARM)
                self.command_pending_start_time = self.clock.now()
                self.trajectory_completed_in_fsm = False # Reset for new mission
            else:
                self.logger.warn(f"FSM: Cannot START_TRAJECTORY from {self.current_phase.name}")
        elif command == "HOLD_POSITION":
            self.hold_position_goal_py = self._create_goal_from_position_array(
                pos_array=current_vehicle_state.p, psi=0 # TODO: Get current psi
            ) # Update hold goal to current pos
            self._transition_to_phase(MissionPhase.USER_COMMANDED_HOLD)
        elif command == "STOP_CONTROLLER":
            self._transition_to_phase(MissionPhase.IDLE)
            self.trajectory_completed_in_fsm = True # Mission aborted
        else:
            self.logger.warn(f"FSM: Unknown command '{command}'")

    def _transition_to_phase(self, new_phase: MissionPhase):
        if self.current_phase != new_phase:
            self.logger.info(f"FSM: Transitioning from {self.current_phase.name} to {new_phase.name}")
            self.current_phase = new_phase
            self.phase_start_time = self.clock.now()

    def is_active(self) -> bool:
        """Returns true if the FSM is in a state that requires controller output."""
        return self.current_phase not in [MissionPhase.IDLE, MissionPhase.LANDED]

    def update(self, current_vehicle_state: StateClass, mavros_state: MavrosState) -> tuple[GoalClass | None, bool]:
        """
        Updates the FSM state and returns the current goal and trajectory completion status.
        Returns: (current_goal_to_send, is_trajectory_phase_completed_or_aborted)
        """
        now = self.clock.now()
        current_p_np = current_vehicle_state.p
        active_goal = None # The goal to be sent out

        # --- Safety: External disarm or mode change ---
        if self.current_phase not in [MissionPhase.IDLE, MissionPhase.LANDED, MissionPhase.AWAITING_OFFBOARD_AND_ARM]:
            if not mavros_state.armed or mavros_state.mode != "OFFBOARD":
                self.logger.warn(f"FSM: MAVROS not ARMED or not OFFBOARD (Armed: {mavros_state.armed}, Mode: {mavros_state.mode}) during active phase {self.current_phase.name}. Transitioning to IDLE.")
                self._transition_to_phase(MissionPhase.IDLE)
                self.trajectory_completed_in_fsm = True # Aborted

        # --- State Logic ---
        if self.current_phase == MissionPhase.IDLE:
            active_goal = self._create_goal_from_position_array(pos_array=current_p_np) # Stream current pose

        elif self.current_phase == MissionPhase.AWAITING_OFFBOARD_AND_ARM:
            active_goal = self._create_goal_from_position_array(pos_array=current_p_np) # Stream current pose
            if mavros_state.armed and mavros_state.mode == "OFFBOARD":
                self.logger.info("FSM: Armed and in OFFBOARD mode. Starting mission.")
                self._transition_to_phase(MissionPhase.TAKING_OFF_TO_INITIAL_HOVER)
                self.command_pending_start_time = None
            elif self.command_pending_start_time and (now - self.command_pending_start_time) > self.wait_for_offboard_arm_timeout:
                self.logger.warn("FSM: Timeout waiting for OFFBOARD and ARM. Returning to IDLE.")
                self._transition_to_phase(MissionPhase.IDLE)
                self.command_pending_start_time = None
                self.trajectory_completed_in_fsm = True # Failed to start

        elif self.current_phase == MissionPhase.TAKING_OFF_TO_INITIAL_HOVER:
            active_goal = self.initial_hover_goal_py
            if self._is_at_target_pose(current_p_np, self.initial_hover_goal_py.p):
                self._transition_to_phase(MissionPhase.AT_INITIAL_HOVER)

        elif self.current_phase == MissionPhase.AT_INITIAL_HOVER:
            active_goal = self.initial_hover_goal_py
            if self.phase_start_time and (now - self.phase_start_time) >= self.hover_duration:
                if self.is_trajectory_loaded_fsm and self.trajectory_start_point_goal_py:
                    self._transition_to_phase(MissionPhase.MOVING_TO_TRAJECTORY_START)
                else: # No trajectory, or not properly loaded
                    self.logger.warn("FSM: No trajectory loaded. Proceeding to final hover (which might be same as initial).")
                    self._transition_to_phase(MissionPhase.MOVING_TO_FINAL_HOVER)


        elif self.current_phase == MissionPhase.MOVING_TO_TRAJECTORY_START:
            if not self.is_trajectory_loaded_fsm or not self.trajectory_start_point_goal_py:
                self.logger.error("FSM: In MOVING_TO_TRAJECTORY_START but trajectory not ready. Switching to IDLE.")
                self._transition_to_phase(MissionPhase.IDLE)
                self.trajectory_completed_in_fsm = True # Error
                return active_goal, self.trajectory_completed_in_fsm

            active_goal = self.trajectory_start_point_goal_py
            if self._is_at_target_pose(current_p_np, self.trajectory_start_point_goal_py.p):
                self._transition_to_phase(MissionPhase.EXECUTING_TRAJECTORY)
                self.trajectory_execution_start_time_ros = self.clock.now()

        elif self.current_phase == MissionPhase.EXECUTING_TRAJECTORY:
            if not self.is_trajectory_loaded_fsm or self.trajectory_data is None or self.trajectory_execution_start_time_ros is None:
                self.logger.error("FSM: In EXECUTING_TRAJECTORY but trajectory/start time not ready. Switching to IDLE.")
                self._transition_to_phase(MissionPhase.IDLE)
                self.trajectory_completed_in_fsm = True # Error
                return active_goal, self.trajectory_completed_in_fsm

            elapsed_execution_time_sec = (now - self.trajectory_execution_start_time_ros).nanoseconds / 1e9
            target_time_in_traj_file = self.trajectory_start_file_time_offset + elapsed_execution_time_sec
            
            active_goal = self._get_trajectory_goal_at_time_fsm(target_time_in_traj_file)
            if active_goal is None: # Should not happen if checks above pass
                 self.logger.error("FSM: Failed to get trajectory point during execution. Going to IDLE.")
                 self._transition_to_phase(MissionPhase.IDLE)
                 self.trajectory_completed_in_fsm = True # Error
                 return active_goal, self.trajectory_completed_in_fsm

            if target_time_in_traj_file >= self.trajectory_data[-1, 0] - 1e-3: # Trajectory time ended
                self.logger.info("FSM: Trajectory execution finished.")
                self.trajectory_completed_in_fsm = True
                self._transition_to_phase(MissionPhase.MOVING_TO_FINAL_HOVER)

        elif self.current_phase == MissionPhase.MOVING_TO_FINAL_HOVER:
            active_goal = self.final_hover_goal_py
            if self._is_at_target_pose(current_p_np, self.final_hover_goal_py.p):
                self._transition_to_phase(MissionPhase.AT_FINAL_HOVER)

        elif self.current_phase == MissionPhase.AT_FINAL_HOVER:
            active_goal = self.final_hover_goal_py
            if self.phase_start_time and (now - self.phase_start_time) >= self.hover_duration:
                self._transition_to_phase(MissionPhase.LANDING)
                # Initialize landing Z to current Z to ensure smooth start of descent
                self.current_goal_py = self._create_goal_from_position_array( # Temporary goal for landing
                     pos_array=np.array([self.landing_pos_param[0], self.landing_pos_param[1], current_p_np[2]]),
                     psi=0 # Consider using final hover psi or current psi
                )

        elif self.current_phase == MissionPhase.LANDING:
            # Gradually decrease Z component of self.current_goal_py
            # Target X, Y are from landing_pos_param
            target_z_next = self.current_goal_py.p[2] - self.landing_descent_rate * (1.0 / 20.0) # Assuming 20Hz update for descent step, adjust if FSM runs slower
            target_z_next = max(target_z_next, self.landing_pos_param[2]) # Don't go below target landing Z

            self.current_goal_py.p[0] = self.landing_pos_param[0]
            self.current_goal_py.p[1] = self.landing_pos_param[1]
            self.current_goal_py.p[2] = target_z_next
            active_goal = self.current_goal_py
            
            # Check if landed: close to landing XY and Z near landing_pos_param[2]
            # More robust: MAVROS might disarm automatically, or check relative altitude if available
            if abs(current_p_np[2] - self.landing_pos_param[2]) < self.pos_reached_thresh_sq**0.5 and \
               self._is_at_target_pose(current_p_np[0:2], self.landing_pos_param[0:2]):
                 self.logger.info(f"FSM: Assumed landed based on position. Current Z: {current_p_np[2]:.2f}, Target Z: {self.landing_pos_param[2]:.2f}")
                 self._transition_to_phase(MissionPhase.LANDED)
            # Or if MAVROS disarms during landing phase
            elif not mavros_state.armed:
                 self.logger.info("FSM: Disarmed during landing phase. Assuming landed.")
                 self._transition_to_phase(MissionPhase.LANDED)


        elif self.current_phase == MissionPhase.LANDED:
            active_goal = self._create_goal_from_position_array(pos_array=current_p_np) # Stream current pose
            self.trajectory_completed_in_fsm = True # Mission fully complete

        elif self.current_phase == MissionPhase.USER_COMMANDED_HOLD:
            if self.hold_position_goal_py:
                active_goal = self.hold_position_goal_py
            else: # Fallback if hold_position_goal_py not set
                active_goal = self._create_goal_from_position_array(pos_array=current_p_np)

        return active_goal, self.trajectory_completed_in_fsm
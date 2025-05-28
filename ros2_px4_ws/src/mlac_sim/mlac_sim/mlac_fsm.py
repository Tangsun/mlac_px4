# In tangsun/mlac_px4/mlac_px4-65b5af8de243304ffb69315e826df1ac5a5041a3/ros2_px4_ws/src/mlac_sim/mlac_sim/mlac_fsm.py
import numpy as np
from enum import Enum, auto
from rclpy.time import Time, Duration
from rclpy.node import Node 

from .structs import GoalClass, StateClass 
from mavros_msgs.msg import State as MavrosState 
import math 

class MissionPhase(Enum):
    IDLE = auto()                           
    AWAITING_OFFBOARD_AND_ARM = auto()      
    TAKING_OFF_TO_INITIAL_HOVER = auto()
    AT_INITIAL_HOVER = auto()               
    DEBUG_ROTATING_YAW_HOVER = auto()
    MOVING_TO_TRAJECTORY_START = auto()
    EXECUTING_TRAJECTORY = auto()
    MOVING_TO_FINAL_HOVER = auto()
    AT_FINAL_HOVER = auto()                 
    LANDING = auto()
    LANDED = auto()                         
    USER_COMMANDED_HOLD = auto()            

class MissionFiniteStateMachine:
    def __init__(self, logger: Node.get_logger, clock: Node.get_clock,
                 initial_hover_pos: list[float], final_hover_pos: list[float], landing_pos: list[float],
                 pos_reached_thresh: float, hover_duration_sec: float, landing_descent_rate_mps: float,
                 wait_for_offboard_arm_timeout_sec: float):
        self.logger = logger
        self.clock = clock

        self.initial_hover_pos_param = np.array(initial_hover_pos)
        self.final_hover_pos_param = np.array(final_hover_pos)
        self.landing_pos_param = np.array(landing_pos) 
        self.pos_reached_thresh_sq = pos_reached_thresh**2 
        self.hover_duration = Duration(seconds=hover_duration_sec)
        self.landing_descent_rate = abs(landing_descent_rate_mps)
        self.wait_for_offboard_arm_timeout = Duration(seconds=wait_for_offboard_arm_timeout_sec)

        self.current_phase = MissionPhase.IDLE
        self.phase_start_time: Time | None = None
        self.command_pending_start_time: Time | None = None

        self.current_goal_py = GoalClass() 
        self.hold_position_goal_py: GoalClass | None = None 

        self.trajectory_data: np.ndarray | None = None
        self.trajectory_start_file_time_offset: float = 0.0
        self.trajectory_start_point_goal_py: GoalClass | None = None
        self.trajectory_execution_start_time_ros: Time | None = None
        self.is_trajectory_loaded_fsm = False
        self.trajectory_completed_in_fsm = False

        self.last_landing_update_time: Time | None = None # Initialize here

        # Ensure these are created with force_zero_feedback=False by default for normal operation
        self.initial_hover_goal_py = self._create_goal_from_position_array(
            self.initial_hover_pos_param, force_zero_feedback=False
        )
        self.final_hover_goal_py = self._create_goal_from_position_array(
            self.final_hover_pos_param, force_zero_feedback=False
        )
        
        self.debug_mode_active = False 
        self.debug_initial_yaw_rad = 0.0
        self.debug_yaw_rate_rps = 0.0
        self.debug_duration_sec = 20.0 
        self.debug_hover_position = self.initial_hover_pos_param 

        self.logger.info("MissionFiniteStateMachine initialized.")

    def _create_goal_from_position_array(self, pos_array: np.ndarray, psi: float = 0.0,
                                         vel_array: np.ndarray = np.zeros(3),
                                         acc_array: np.ndarray = np.zeros(3),
                                         jerk_array: np.ndarray | None = None, 
                                         dpsi: float | None = None,
                                         force_zero_feedback: bool = False) -> GoalClass: # Default is False
        goal = GoalClass()
        goal.t = self.clock.now().nanoseconds / 1e9 
        goal.p = np.array(pos_array)
        goal.v = np.array(vel_array)
        goal.a = np.array(acc_array)
        goal.j = jerk_array
        goal.psi = psi
        goal.dpsi = dpsi
        goal.mode_xy = GoalClass.Mode.POS_CTRL
        goal.mode_z = GoalClass.Mode.POS_CTRL
        goal.force_zero_feedback_contribution = force_zero_feedback # Set the flag
        return goal

    def configure_debug_rotating_yaw(self, active: bool, 
                                     initial_yaw_rad: float, yaw_rate_rps: float, 
                                     duration_sec: float, hover_pos: np.ndarray):
        self.debug_mode_active = active
        self.debug_initial_yaw_rad = initial_yaw_rad
        self.debug_yaw_rate_rps = yaw_rate_rps
        self.debug_duration_sec = duration_sec
        self.debug_hover_position = np.array(hover_pos)
        # Update initial_hover_goal_py if debug mode is active to use debug params for its psi
        if active:
            self.initial_hover_goal_py = self._create_goal_from_position_array(
                self.debug_hover_position, # Use debug hover position for initial hover too
                psi=self.debug_initial_yaw_rad,
                force_zero_feedback=False # PID active for reaching this initial hover
            )
            self.logger.info(f"FSM: Debug Rotating Yaw Mode CONFIGURED. Initial Hover/Debug Pos: {self.debug_hover_position}, Initial Yaw: {math.degrees(initial_yaw_rad):.1f}deg, Rate: {math.degrees(yaw_rate_rps):.1f}dps, Duration: {duration_sec}s")
        else:
            # Revert initial_hover_goal_py to use default parameters if debug mode is turned off
            self.initial_hover_goal_py = self._create_goal_from_position_array(
                self.initial_hover_pos_param, force_zero_feedback=False
            )


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
            if self.trajectory_data.shape[1] >= 11: 
                start_acc = self.trajectory_data[0, 8:11]
            if self.trajectory_data.shape[1] >= 14: 
                start_jerk = self.trajectory_data[0, 11:14]
            if self.trajectory_data.shape[1] >= 15: 
                start_dpsi = self.trajectory_data[0, 14]

            self.trajectory_start_point_goal_py = self._create_goal_from_position_array(
                pos_array=start_pos, psi=start_psi, vel_array=start_vel, 
                acc_array=start_acc, jerk_array=start_jerk, dpsi=start_dpsi,
                force_zero_feedback=False # Normal PID for trajectory tracking
            )
            self.logger.info(f"FSM: Trajectory data set. Start point: P={start_pos}, V={start_vel}, Psi={start_psi:.2f}")
        else:
            self.is_trajectory_loaded_fsm = False
            self.trajectory_start_point_goal_py = None
            self.logger.warn("FSM: Cleared trajectory data or received empty data.")

    def _get_trajectory_goal_at_time_fsm(self, target_time_in_trajectory_timeline: float) -> GoalClass | None:
        if not self.is_trajectory_loaded_fsm or self.trajectory_data is None:
            return None
        # ... (interpolation logic as before) ...
        # Ensure the created goal has force_zero_feedback=False for normal trajectory following
        goal = GoalClass()
        # ... (populate p, v, psi, a, j, dpsi from interpolated trajectory_data) ...
        traj_data = self.trajectory_data
        traj_file_times = traj_data[:, 0]
        clipped_target_time = np.clip(target_time_in_trajectory_timeline, traj_file_times[0], traj_file_times[-1])
        goal.t = self.clock.now().nanoseconds / 1e9
        goal.p = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(1, 4)])
        goal.v = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(4, 7)])
        goal.psi = float(np.interp(clipped_target_time, traj_file_times, traj_data[:, 7]))
        num_cols = traj_data.shape[1]
        goal.a = np.zeros(3) 
        if num_cols >= 11: goal.a = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(8, 11)])
        goal.j = None
        if num_cols >= 14: goal.j = np.array([np.interp(clipped_target_time, traj_file_times, traj_data[:, i]) for i in range(11, 14)])
        goal.dpsi = None
        if num_cols >= 15: goal.dpsi = float(np.interp(clipped_target_time, traj_file_times, traj_data[:, 14]))
        goal.mode_xy = GoalClass.Mode.POS_CTRL
        goal.mode_z = GoalClass.Mode.POS_CTRL
        goal.force_zero_feedback_contribution = False # Normal PID for trajectory
        return goal


    def _transition_to_phase(self, new_phase: MissionPhase):
        if self.current_phase != new_phase:
            self.logger.info(f"FSM: Transitioning from {self.current_phase.name} to {new_phase.name}")
            if new_phase == MissionPhase.DEBUG_ROTATING_YAW_HOVER:
                self.logger.warn("FSM: ***** ENTERED DEBUG_ROTATING_YAW_HOVER PHASE *****")
            self.current_phase = new_phase
            self.phase_start_time = self.clock.now()

    def is_active(self) -> bool:
        return self.current_phase not in [MissionPhase.IDLE, MissionPhase.LANDED]

    def process_command(self, command: str, current_vehicle_state: StateClass):
        self.logger.info(f"FSM received command: '{command}' in state {self.current_phase.name}")
        if command == "START_MISSION":
            if self.current_phase == MissionPhase.IDLE or self.current_phase == MissionPhase.LANDED:
                self._transition_to_phase(MissionPhase.AWAITING_OFFBOARD_AND_ARM)
                self.command_pending_start_time = self.clock.now()
                self.trajectory_completed_in_fsm = False 
            else:
                self.logger.warn(f"FSM: Cannot START_MISSION from {self.current_phase.name}")
        elif command == "HOLD_POSITION":
            current_yaw_for_hold = 0.0
            try: # Attempt to get current yaw for a smoother hold
                q_for_yaw = current_vehicle_state.q
                # Simple yaw extraction (assuming standard quaternion to Euler)
                # t3 = +2.0 * (q_for_yaw[0] * q_for_yaw[3] + q_for_yaw[1] * q_for_yaw[2])
                # t4 = +1.0 - 2.0 * (q_for_yaw[2] * q_for_yaw[2] + q_for_yaw[3] * q_for_yaw[3])
                # current_yaw_for_hold = math.atan2(t3, t4)
                # Using a more robust helper if available, or keeping it simple
                # For now, let's assume a simple default or that it's not critical for this flow
                pass # Keep default 0 or implement more robust get_rpy if needed
            except Exception as e:
                self.logger.warn(f"FSM: Could not get current RPY for HOLD_POSITION, defaulting psi to 0. Error: {e}")


            self.hold_position_goal_py = self._create_goal_from_position_array(
                pos_array=current_vehicle_state.p, psi=current_yaw_for_hold,
                force_zero_feedback=False # Active PID for HOLD
            ) 
            self._transition_to_phase(MissionPhase.USER_COMMANDED_HOLD)
        elif command == "STOP_CONTROLLER": 
            self._transition_to_phase(MissionPhase.IDLE)
            self.trajectory_completed_in_fsm = True 


    def update(self, current_vehicle_state: StateClass, mavros_state: MavrosState) -> tuple[GoalClass | None, bool]:
        now = self.clock.now()
        current_p_np = current_vehicle_state.p
        active_goal = None 

        # Safety check: if not armed or not in OFFBOARD, go to IDLE (unless already IDLE/LANDED)
        if self.current_phase not in [MissionPhase.IDLE, MissionPhase.LANDED, MissionPhase.AWAITING_OFFBOARD_AND_ARM]:
            if not mavros_state.armed or mavros_state.mode != "OFFBOARD":
                self.logger.warn(f"FSM: MAVROS not ARMED or not OFFBOARD (Mode: {mavros_state.mode}, Armed: {mavros_state.armed}). Transitioning to IDLE.")
                self._transition_to_phase(MissionPhase.IDLE)
                self.trajectory_completed_in_fsm = True # Consider mission failed/aborted

        # Phase logic
        if self.current_phase == MissionPhase.IDLE:
            active_goal = self._create_goal_from_position_array(pos_array=current_p_np, force_zero_feedback=True)

        elif self.current_phase == MissionPhase.AWAITING_OFFBOARD_AND_ARM:
            active_goal = self._create_goal_from_position_array(pos_array=current_p_np, force_zero_feedback=True) 
            if mavros_state.armed and mavros_state.mode == "OFFBOARD":
                self.logger.info("FSM: Armed and in OFFBOARD mode. Starting mission.")
                # Determine initial hover position based on debug mode
                if self.debug_mode_active:
                    self.initial_hover_goal_py = self._create_goal_from_position_array(
                        self.debug_hover_position, 
                        psi=self.debug_initial_yaw_rad, 
                        force_zero_feedback=False
                    )
                    self.logger.info(f"FSM: Targeting debug initial hover at {self.debug_hover_position} with yaw {self.debug_initial_yaw_rad:.2f} rad.")
                else:
                     self.initial_hover_goal_py = self._create_goal_from_position_array(
                        self.initial_hover_pos_param, 
                        psi=0.0, # Default initial yaw for normal mission
                        force_zero_feedback=False
                    )
                     self.logger.info(f"FSM: Targeting standard initial hover at {self.initial_hover_pos_param} with yaw 0.0 rad.")

                self._transition_to_phase(MissionPhase.TAKING_OFF_TO_INITIAL_HOVER)
                self.command_pending_start_time = None
            elif self.command_pending_start_time and (now - self.command_pending_start_time) > self.wait_for_offboard_arm_timeout:
                self.logger.warn("FSM: Timeout waiting for OFFBOARD and ARM. Returning to IDLE.")
                self._transition_to_phase(MissionPhase.IDLE)
                self.command_pending_start_time = None
                self.trajectory_completed_in_fsm = True 

        elif self.current_phase == MissionPhase.TAKING_OFF_TO_INITIAL_HOVER:
            active_goal = self.initial_hover_goal_py # This was already set with force_zero_feedback=False
            if self._is_at_target_pose(current_p_np, active_goal.p):
                self._transition_to_phase(MissionPhase.AT_INITIAL_HOVER)

        elif self.current_phase == MissionPhase.AT_INITIAL_HOVER:
            active_goal = self.initial_hover_goal_py # Still force_zero_feedback=False
            if self.phase_start_time and (now - self.phase_start_time) >= self.hover_duration:
                if self.debug_mode_active:
                    self._transition_to_phase(MissionPhase.DEBUG_ROTATING_YAW_HOVER)
                elif self.is_trajectory_loaded_fsm and self.trajectory_start_point_goal_py:
                    self._transition_to_phase(MissionPhase.MOVING_TO_TRAJECTORY_START)
                else: 
                    self.logger.warn("FSM: No trajectory or debug mode. Proceeding to final hover.")
                    # Ensure final_hover_goal_py is for normal PID
                    self.final_hover_goal_py = self._create_goal_from_position_array(
                        self.final_hover_pos_param, force_zero_feedback=False
                    )
                    self._transition_to_phase(MissionPhase.MOVING_TO_FINAL_HOVER)
        
        elif self.current_phase == MissionPhase.DEBUG_ROTATING_YAW_HOVER:
            elapsed_debug_time_sec = (now - self.phase_start_time).nanoseconds / 1e9
            current_debug_psi = (self.debug_initial_yaw_rad + self.debug_yaw_rate_rps * elapsed_debug_time_sec)
            current_debug_psi_normalized = (current_debug_psi + math.pi) % (2 * math.pi) - math.pi

            active_goal = self._create_goal_from_position_array(
                pos_array=self.debug_hover_position, 
                psi=current_debug_psi_normalized,
                dpsi=self.debug_yaw_rate_rps,
                force_zero_feedback=True # Key for this debug mode
            )
            if elapsed_debug_time_sec >= self.debug_duration_sec:
                self.logger.info("FSM: Debug rotating yaw finished. Proceeding to landing.")
                self.trajectory_completed_in_fsm = True 
                # Prepare for normal landing
                current_yaw_for_landing = active_goal.psi # Land with the final debug yaw
                self.current_goal_py = self._create_goal_from_position_array( 
                     pos_array=np.array([self.landing_pos_param[0], self.landing_pos_param[1], current_p_np[2]]), # Start descent from current XY of landing_pos
                     psi=current_yaw_for_landing,
                     force_zero_feedback=False # PID for landing
                )
                self.last_landing_update_time = now # ++ SET IT HERE ++
                self._transition_to_phase(MissionPhase.LANDING) 

        elif self.current_phase == MissionPhase.MOVING_TO_TRAJECTORY_START:
            if not self.is_trajectory_loaded_fsm or not self.trajectory_start_point_goal_py:
                self.logger.error("FSM: In MOVING_TO_TRAJECTORY_START but trajectory not ready. Switching to IDLE.")
                self._transition_to_phase(MissionPhase.IDLE); self.trajectory_completed_in_fsm = True 
                return self._create_goal_from_position_array(pos_array=current_p_np, force_zero_feedback=True), True

            # trajectory_start_point_goal_py should have force_zero_feedback=False set during its creation
            active_goal = self.trajectory_start_point_goal_py 
            if self._is_at_target_pose(current_p_np, self.trajectory_start_point_goal_py.p):
                self._transition_to_phase(MissionPhase.EXECUTING_TRAJECTORY)
                self.trajectory_execution_start_time_ros = self.clock.now()

        elif self.current_phase == MissionPhase.EXECUTING_TRAJECTORY:
            if not self.is_trajectory_loaded_fsm or self.trajectory_data is None or self.trajectory_execution_start_time_ros is None:
                self.logger.error("FSM: In EXECUTING_TRAJECTORY but trajectory/start time not ready. Switching to IDLE.")
                self._transition_to_phase(MissionPhase.IDLE); self.trajectory_completed_in_fsm = True
                return self._create_goal_from_position_array(pos_array=current_p_np, force_zero_feedback=True), True
            
            elapsed_execution_time_sec = (now - self.trajectory_execution_start_time_ros).nanoseconds / 1e9
            target_time_in_traj_file = self.trajectory_start_file_time_offset + elapsed_execution_time_sec
            
            active_goal = self._get_trajectory_goal_at_time_fsm(target_time_in_traj_file)
            # _get_trajectory_goal_at_time_fsm now sets force_zero_feedback=False internally
            if active_goal is None: 
                 self.logger.error("FSM: Failed to get trajectory point during execution. Going to IDLE.")
                 self._transition_to_phase(MissionPhase.IDLE); self.trajectory_completed_in_fsm = True
                 return self._create_goal_from_position_array(pos_array=current_p_np, force_zero_feedback=True), True

            if target_time_in_traj_file >= self.trajectory_data[-1, 0] - 1e-3: 
                self.logger.info("FSM: Trajectory execution finished.")
                self.trajectory_completed_in_fsm = True
                # Ensure final_hover_goal_py is for normal PID
                self.final_hover_goal_py = self._create_goal_from_position_array(
                    self.final_hover_pos_param, force_zero_feedback=False
                )
                self._transition_to_phase(MissionPhase.MOVING_TO_FINAL_HOVER)

        elif self.current_phase == MissionPhase.MOVING_TO_FINAL_HOVER:
            # final_hover_goal_py should have force_zero_feedback=False
            active_goal = self.final_hover_goal_py 
            if self._is_at_target_pose(current_p_np, self.final_hover_goal_py.p):
                self._transition_to_phase(MissionPhase.AT_FINAL_HOVER)

        elif self.current_phase == MissionPhase.AT_FINAL_HOVER:
            # final_hover_goal_py should have force_zero_feedback=False
            active_goal = self.final_hover_goal_py 
            if self.phase_start_time and (now - self.phase_start_time) >= self.hover_duration:
                self._transition_to_phase(MissionPhase.LANDING)
                self.current_goal_py = self._create_goal_from_position_array( 
                     pos_array=np.array([self.landing_pos_param[0], self.landing_pos_param[1], current_p_np[2]]),
                     psi=self.final_hover_goal_py.psi, 
                     force_zero_feedback=False # PID for landing
                )
                self.last_landing_update_time = now # Initialize for descent calculation


        elif self.current_phase == MissionPhase.LANDING:
            target_z_next = self.current_goal_py.p[2] 
            if self.phase_start_time: 
                 dt_land = (now - self.last_landing_update_time).nanoseconds / 1e9
                 target_z_next = self.current_goal_py.p[2] - self.landing_descent_rate * dt_land
            self.last_landing_update_time = now

            target_z_next = max(target_z_next, self.landing_pos_param[2]) 
            self.current_goal_py.p[0] = self.landing_pos_param[0]
            self.current_goal_py.p[1] = self.landing_pos_param[1]
            self.current_goal_py.p[2] = target_z_next
            self.current_goal_py.force_zero_feedback_contribution = False # Ensure PID is active
            active_goal = self.current_goal_py
            
            # Check if landed based on Z position and drone being disarmed or very low Z error
            # Use a slightly more generous threshold for Z to ensure it actually reaches near target landing altitude
            is_at_landing_xy = self._is_at_target_pose(current_p_np[:2], self.landing_pos_param[:2])
            is_at_landing_z = abs(current_p_np[2] - self.landing_pos_param[2]) < 0.1 # e.g. 10cm for landing Z

            if (is_at_landing_xy and is_at_landing_z) or not mavros_state.armed:
                 log_reason = "position criteria" if (is_at_landing_xy and is_at_landing_z) else "disarmed"
                 self.logger.info(f"FSM: Assumed landed based on {log_reason}.")
                 self._transition_to_phase(MissionPhase.LANDED)


        elif self.current_phase == MissionPhase.LANDED:
            active_goal = self._create_goal_from_position_array(pos_array=current_p_np, force_zero_feedback=True) 
            self.trajectory_completed_in_fsm = True 

        elif self.current_phase == MissionPhase.USER_COMMANDED_HOLD:
            if self.hold_position_goal_py: # Should have been created with force_zero_feedback=False
                active_goal = self.hold_position_goal_py 
            else: 
                active_goal = self._create_goal_from_position_array(pos_array=current_p_np, force_zero_feedback=False) # Default to PID if hold_goal somehow not set

        if active_goal is None:
             self.logger.warn(f"FSM: active_goal was None in phase {self.current_phase.name}. Defaulting to current pose with force_zero_feedback=True.")
             active_goal = self._create_goal_from_position_array(pos_array=current_p_np, force_zero_feedback=True)

        return active_goal, self.trajectory_completed_in_fsm
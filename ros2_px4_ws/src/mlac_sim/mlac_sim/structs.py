# In tangsun/mlac_px4/mlac_px4-65b5af8de243304ffb69315e826df1ac5a5041a3/ros2_px4_ws/src/mlac_sim/mlac_sim/structs.py

from enum import Enum
import numpy as np

class AttCmdClass:
    # ... (no changes here) ...
    def __init__(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
        self.w = np.zeros(3)
        self.F_W = np.zeros(3)

class ParametersClass:
    # ... (no changes here) ...
    def __init__(self):
        self.mass = 0.0
        self.Kp = np.zeros(3)
        self.Ki = np.zeros(3)
        self.Kd = np.zeros(3)
        self.maxPosErr = np.zeros(3)
        self.maxVelErr = np.zeros(3)

class StateClass:
    # ... (no changes here) ...
    def __init__(self):
        self.t = -1.0
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
        self.w = np.zeros(3)

class GoalClass:
    class Mode(Enum):
        POS_CTRL = 0
        VEL_CTRL = 1
        ACC_CTRL = 2

    def __init__(self):
        self.mode_xy = self.Mode.POS_CTRL
        self.mode_z = self.Mode.POS_CTRL
        self.t = -1.0
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.a = np.zeros(3) 
        self.j = None        
        self.psi = 0.0
        self.dpsi = None     
        # ++ NEW FLAG ++
        self.force_zero_feedback_contribution: bool = False 
        # ++ END NEW FLAG ++

class ControlLogClass:
    # ... (no changes here) ...
    def __init__(self):
        self.p = np.zeros(3)
        self.p_ref = np.zeros(3)
        self.p_err = np.zeros(3)
        self.p_err_int = np.zeros(3)
        self.v = np.zeros(3)
        self.v_ref = np.zeros(3)
        self.v_err = np.zeros(3)
        self.a_ff = np.zeros(3)
        self.a_fb = np.zeros(3)
        self.j_ff = np.zeros(3) 
        self.j_fb = np.zeros(3) 
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) 
        self.q_ref = np.array([1.0, 0.0, 0.0, 0.0]) 
        self.w = np.zeros(3)
        self.w_ref = np.zeros(3)
        self.F_W = np.zeros(3)
        self.mode_xy = GoalClass.Mode.POS_CTRL 
        self.mode_z = GoalClass.Mode.POS_CTRL  

        self.psi_ref = 0.0
        self.dpsi_ref = 0.0

        self.roll_ref = 0.0
        self.pitch_ref = 0.0

        self.P_norm = 0.0
        self.A_norm = 0.0
        self.y_norm = 0.0
        self.f_hat = np.zeros(3)

        self.trajectory_execution_start_ros_time = None # Will store rclpy.time.Time object
        self.trajectory_execution_end_ros_time = None   # Will store rclpy.time.Time object

class ModeClass(Enum): 
    # ... (no changes here) ...
    Preflight = 0
    SpinningUp = 1
    Flying = 2
    EmergencyStop = 3
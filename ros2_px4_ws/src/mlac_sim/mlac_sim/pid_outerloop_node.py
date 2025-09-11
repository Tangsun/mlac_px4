## This is a temporary file to debug the outer loop controller
## The goal is to compare COML implementation with no adaptation with the current PID Design

import numpy as np
import os
import pickle
from ament_index_python.packages import get_package_share_directory # ROS 2 equivalent for rospkg

# Assuming these are local modules within your ROS 2 package (mlac_sim)
from .dynamics import prior
from .structs import AttCmdClass, ControlLogClass, GoalClass
from .helpers import quaternion_multiply
from .utils import params_to_posdef, quaternion_to_rotation_matrix, flat_rotation_matrix_to_quaternion

def convert_p_qbar(p):
    return np.sqrt(1/(1 - 1/p) - 1.1)

class IntegratorClass:
    def __init__(self):
        self.value_ = 0.0 

    def increment(self, inc, dt):
        self.value_ += inc * dt

    def reset(self):
        self.value_ = 0.0

    def value(self):
        return self.value_

class PIDOuterLoop:
    def __init__(self, params, state0, goal0, controller='pid', package_name='mlac_sim'): 
        self.controller = controller
        self.package_name = package_name 

        if self.controller != 'pid':
            raise NotImplementedError("Only 'pid' controller is implemented in this OuterLoop class.")

        self.params_ = params
        self.GRAVITY = np.array([0.0, 0.0, -9.80665]) 

        self.Ix_ = IntegratorClass()
        self.Iy_ = IntegratorClass()
        self.Iz_ = IntegratorClass()

        self.log_ = ControlLogClass()
        self.a_fb_last_ = np.zeros(3)
        self.j_fb_last_ = np.zeros(3)
        self.t_last_ = 0.0 

        self.mode_xy_last_ = GoalClass.Mode.POS_CTRL
        self.mode_z_last_ = GoalClass.Mode.POS_CTRL
        
        self.reset(state0, goal0)

    def reset(self, state0, goal0):
        self.Ix_.reset()
        self.Iy_.reset()
        self.Iz_.reset()

        self.log_ = ControlLogClass()
        self.a_fb_last_ = np.zeros(3)
        self.j_fb_last_ = np.zeros(3)
        self.t_last_ = 0.0 
    
    def update_log(self, state): 
        self.log_ = ControlLogClass()
        self.log_.p = state.p
        self.log_.v = state.v
        self.log_.q = state.q
        self.log_.w = state.w

    def compute_attitude_command(self, t, state, goal):
        dt = 1e-2 if self.t_last_ == 0.0 else t - self.t_last_ 

        if dt > 1e-9: # More robust check for positive dt
            self.t_last_ = t
        else:
            print(f"Warning: non-positive or very small dt: {dt} [s]. Using previous command or safe hover.")
            cmd = AttCmdClass()
            if hasattr(self, 'last_computed_q_ref_') and hasattr(self, 'last_computed_w_ref_') and hasattr(self, 'last_computed_F_W_'):
                cmd.q = self.last_computed_q_ref_
                cmd.w = self.last_computed_w_ref_
                cmd.F_W = self.last_computed_F_W_
            else: 
                cmd.q = np.array([np.cos(goal.psi / 2.0), 0.0, 0.0, np.sin(goal.psi / 2.0)]) 
                cmd.q = cmd.q / np.linalg.norm(cmd.q)
                cmd.w = np.zeros(3)
                cmd.F_W = -self.params_.mass * self.GRAVITY 
            return cmd
        
        f_hat = np.zeros(3) 

        F_W = self.get_force(dt, state, goal, f_hat)
        q_ref = self.get_attitude(state, goal, F_W)
        w_ref = self.get_rates(dt, state, goal, F_W, self.log_.a_fb, q_ref)

        cmd = AttCmdClass()
        cmd.q = q_ref
        cmd.w = w_ref
        cmd.F_W = F_W

        self.last_computed_q_ref_ = q_ref
        self.last_computed_w_ref_ = w_ref
        self.last_computed_F_W_ = F_W
        return cmd

    def get_force(self, dt, state, goal, f_hat):
        # Assuming state.t is available for logging, if not, pass 't' as an argument
        current_time_for_log = state.t 
        logger_fn = self.get_logger().debug

        a_fb_calculated = np.zeros(3) 
        e = goal.p - state.p # Calculate errors for logging, even if not used for F_W in debug
        edot = goal.v - state.v

        if goal.force_zero_feedback_contribution:
            logger_fn(f"DEBUG MODE (get_force @ t={current_time_for_log:.3f}): Forcing zero feedback. Goal p={goal.p}, psi={goal.psi:.2f}, dpsi={goal.dpsi}")
            # For pure attitude command, desired world acceleration is only to counteract gravity
            # F_W = m * (a_des - g_world)
            # If a_des (from goal.a + a_fb) should be 0 for pure hover, then F_W = m * (-g_world)
            thrust_factor = 1.00
            F_W = -self.params_.mass * self.GRAVITY * thrust_factor
            
            # Log that feedback contributions are zeroed for this cycle
            self.log_.p_err = e # Log actual error
            self.log_.v_err = edot # Log actual error
            self.log_.p_err_int = np.array([self.Ix_.value(), self.Iy_.value(), self.Iz_.value()]) # Log current integral, though not used for F_W
            self.log_.a_fb = np.zeros(3) # Feedback acceleration is forced to zero
            if 'coml' in self.controller:
                self.log_.f_hat = np.zeros(3) # Effective f_hat is zero for this F_W calc
        
        else:
            # ---------------------------------------------------------------------------- #
            #                                PID Controller                                #
            # ---------------------------------------------------------------------------- #
            e_clamped = np.minimum(np.maximum(e, -self.params_.maxPosErr), self.params_.maxPosErr)      # maxPosErr is [0.5, 0.5, 0.5] by default
            edot_clamped = np.minimum(np.maximum(edot, -self.params_.maxVelErr), self.params_.maxVelErr)    # maxVelErr is [1.0, 1.0, 1.0] by default

            # ----------- Check the mode changes to reset integrators if needed ---------- #
            if goal.mode_xy != self.mode_xy_last_:
                self.Ix_.reset(); self.Iy_.reset()
                self.mode_xy_last_ = goal.mode_xy
            if goal.mode_z != self.mode_z_last_:
                self.Iz_.reset()
                self.mode_z_last_ = goal.mode_z
            
            # ---------------------- PID Integral logic for XY mode ---------------------- #
            if goal.mode_xy == GoalClass.Mode.POS_CTRL:
                self.Ix_.increment(e_clamped[0], dt); self.Iy_.increment(e_clamped[1], dt)
            # ... (rest of PID integral logic as before) ...
            elif goal.mode_xy == GoalClass.Mode.VEL_CTRL: e_clamped[0] = e_clamped[1] = 0.0 
            elif goal.mode_xy == GoalClass.Mode.ACC_CTRL: e_clamped[0] = e_clamped[1] = 0.0; edot_clamped[0] = edot_clamped[1] = 0.0
            # ---------------------- PID Integral logic for Z mode ----------------------- #
            if goal.mode_z == GoalClass.Mode.POS_CTRL: self.Iz_.increment(e_clamped[2], dt)
            elif goal.mode_z == GoalClass.Mode.VEL_CTRL: e_clamped[2] = 0.0
            elif goal.mode_z == GoalClass.Mode.ACC_CTRL: e_clamped[2] = 0.0; edot_clamped[2] = 0.0

            eint = np.array([self.Ix_.value(), self.Iy_.value(), self.Iz_.value()])

            # ----------------------------- MAIN PID formula ----------------------------- #
            a_fb_calculated = self.params_.Kp * e_clamped \
                            + self.params_.Ki * eint \
                            + self.params_.Kd * edot_clamped
            F_W = a_fb_calculated + self.params_.mass * (goal.a - self.GRAVITY)     # NOTE(KAI): `goal.a` seems to be zero (09/10/2025)
            self.log_.p_err_int = eint # Log PID integral term
            self.log_.a_fb = a_fb_calculated

        # ----------- Common logging after F_W is determined for all cases ----------- #
        self.log_.p = state.p
        self.log_.p_ref = goal.p
        self.log_.p_err = e # This is actual error, not necessarily what drove F_W if in debug mode
        self.log_.v = state.v
        self.log_.v_ref = goal.v
        self.log_.v_err = edot # Actual error
        self.log_.a_ff = goal.a 
        # self.log_.a_fb is set within each controller block or by debug block
        self.log_.F_W = F_W
        
        logger_fn(f"get_force FINAL @ t={current_time_for_log:.3f}: F_W_cmd={F_W}, P_err_x={e[0]:.3f}, V_err_x={edot[0]:.3f}, A_fb_x={self.log_.a_fb[0]:.3f}")
        return F_W

    def get_attitude(self, state, goal, F_W):
        F_W_norm = np.linalg.norm(F_W)
        if F_W_norm < 1e-8: 
            # If the force is too small, use a default level attitude
            q_ref = np.array([np.cos(goal.psi / 2.0), 0.0, 0.0, np.sin(goal.psi / 2.0)])
        else:
            b_3d = F_W / F_W_norm  # Normalize the force vector to get the body z-axis
            b_1d_tilde = np.array([np.cos(goal.psi), np.sin(goal.psi), 0.0])
            b_2d = np.cross(b_3d, b_1d_tilde)
            b_2d /= np.linalg.norm(b_2d)
            b_1d = np.cross(b_2d, b_3d)
            b_1d /= np.linalg.norm(b_1d)

            R_d = np.column_stack((b_1d, b_2d, b_3d))

            flat_Rd = R_d.flatten()

            q_ref = flat_rotation_matrix_to_quaternion(flat_Rd)

        self.log_.q = state.q
        self.log_.q_ref = q_ref

        return q_ref

    def get_rates(self, dt, state, goal, F_W, a_fb, q_ref):
        # Use goal.j if available, otherwise use zeros
        goal_j_for_calc = goal.j if goal.j is not None else np.zeros(3)
        # Use goal.dpsi if available, otherwise use zero
        goal_dpsi_for_calc = goal.dpsi if goal.dpsi is not None else 0.0

        j_fb = np.zeros(3)
        if dt > 1e-6: 
            current_j_fb = (a_fb - self.a_fb_last_) / dt 
            tau = 0.1
            alpha = dt / (tau + dt)
            j_fb = alpha * current_j_fb + (1.0 - alpha) * self.j_fb_last_
        else:
            j_fb = self.j_fb_last_

        self.a_fb_last_ = a_fb
        self.j_fb_last_ = j_fb

        Fdot_W_term = self.params_.mass * (goal_j_for_calc + j_fb)

        xi = F_W / self.params_.mass  
        norm_xi = np.linalg.norm(xi)
        
        rates = np.zeros(3) 

        if norm_xi < 1e-6: 
            rates[2] = goal_dpsi_for_calc
        else:
            abc = xi / norm_xi 
            xi_dot = Fdot_W_term / self.params_.mass 
            
            I = np.eye(3)
            abcdot = ((norm_xi**2 * I - np.outer(xi, xi)) / (norm_xi**3 + 1e-9)) @ xi_dot  
            
            a, b, c = abc
            adot, bdot, cdot = abcdot
            psi = goal.psi # Use current goal's psi for rate calculation consistency
            # psidot is goal_dpsi_for_calc

            den_1_plus_c = 1.0 + c
            if np.abs(den_1_plus_c) < 1e-9: 
                # print(f"Warning: c is close to -1 (value: {c}) in get_rates. Singularity. Using yaw rate only.")
                rates[2] = goal_dpsi_for_calc
            else:
                rates[0] = np.sin(psi) * adot - np.cos(psi) * bdot - (a * np.sin(psi) - b * np.cos(psi)) * (cdot / den_1_plus_c)
                rates[1] = np.cos(psi) * adot + np.sin(psi) * bdot - (a * np.cos(psi) + b * np.sin(psi)) * (cdot / den_1_plus_c)
                rates[2] = (b * adot - a * bdot) / den_1_plus_c + goal_dpsi_for_calc
        
        self.log_.p = state.p 
        self.log_.p_ref = goal.p
        self.log_.p_err = goal.p - state.p 
        self.log_.v = state.v
        self.log_.v_ref = goal.v
        self.log_.v_err = goal.v - state.v 
        self.log_.a_ff = goal.a # Log the actual goal.a used
        self.log_.a_fb = a_fb   
        self.log_.F_W = F_W
        self.log_.j_ff = goal_j_for_calc # Log the jerk used in calculation
        self.log_.j_fb = j_fb
        self.log_.w = state.w
        self.log_.w_ref = rates
        return rates

    def get_log(self):
        return self.log_

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
from .utils import params_to_posdef, quaternion_to_rotation_matrix

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

class OuterLoop:
    def __init__(self, params, state0, goal0, controller='coml', package_name='mlac_sim'): 
        self.controller = controller
        self.package_name = package_name 

        if self.controller == 'coml':
            package_share_path = get_package_share_directory(self.package_name)
            trial_name = 'reg_P_1.0_reg_k_R_0.001_k_R_scale_1_k_R_z_1.26'
            filename = 'seed=0_M=50_E=1000_pinit=2.00_pfreq=2000_regP=1.0000.pkl'
            model_dir = os.path.join(package_share_path, 'models', trial_name)
            model_pkl_loc = os.path.join(model_dir, filename)
            try:
                with open(model_pkl_loc, 'rb') as f:
                    train_results = pickle.load(f)
                print(f'COML model loaded from: {model_pkl_loc}')
            except FileNotFoundError:
                print(f"ERROR: COML model file not found at {model_pkl_loc}")
                raise 
            self.pnorm = convert_p_qbar(train_results['pnorm'])
            self.W = train_results['model']['W']
            self.b = train_results['model']['b']
            self.Λ = params_to_posdef(train_results['controller']['Λ'])
            self.K = params_to_posdef(train_results['controller']['K'])
            self.P = params_to_posdef(train_results['controller']['P'])
        elif self.controller == 'coml_debug':
            print('COML debug model: no model is loaded!')
            self.pnorm = convert_p_qbar(2.0)
            self.W = None
            self.b = None
            self.Λ = np.eye(3)
            self.K = np.eye(3)
            self.P = np.eye(3)
        
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
        if self.controller == 'coml':
            q0 = state0.p
            dq0 = state0.v
            R_flatten0 = quaternion_to_rotation_matrix(state0.q).flatten()
            Omega0 = state0.w
            r0 = goal0.p
            dr0 = goal0.v
            
            self.dA_prev, y0 = self.adaptation_law(q0, dq0, R_flatten0, Omega0, r0, dr0)
            self.pA_prev = np.zeros((q0.size, y0.size))
            
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
        if self.controller == 'coml':
            qn = 1.1 + self.pnorm**2
            R_flatten = quaternion_to_rotation_matrix(state.q).flatten()
            dA, y = self.adaptation_law(state.p, state.v, R_flatten, state.w, goal.p, goal.v)
            
            if not hasattr(self, 'pA_prev') or not hasattr(self, 'dA_prev'):
                 print("CRITICAL WARNING: pA_prev or dA_prev not initialized in COML. Reset might not have run correctly.")
                 self.dA_prev, y0_init = self.adaptation_law(state.p, state.v, R_flatten, state.w, goal.p, goal.v)
                 self.pA_prev = np.zeros((state.p.size, y0_init.size))
                 dA = self.dA_prev 

            pA = self.pA_prev + (dt/2.0)*(self.dA_prev + dA) 
            
            current_P_matrix = self.P 
            A_adapt = (np.maximum(np.abs(pA), 1e-6 * np.ones_like(pA))**(qn-1) * np.sign(pA) * (np.ones_like(pA) - np.isclose(pA, 0.0, atol=1e-6)) 
                      ) @ current_P_matrix
            f_hat = A_adapt @ y

            self.log_.P_norm = np.linalg.norm(current_P_matrix)
            self.log_.A_norm = np.linalg.norm(A_adapt)
            self.log_.y_norm = np.linalg.norm(y)
            self.log_.f_hat = f_hat

            self.pA_prev = pA
            self.dA_prev = dA
        
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

    def adaptation_law(self, q, dq, R_flatten, Omega, r, dr):
        y = np.concatenate((q, dq, R_flatten, Omega))
        if self.W is not None and self.b is not None: 
            for W_layer, b_layer in zip(self.W, self.b): 
                y = np.tanh(W_layer@y + b_layer)

        current_Lambda = self.Λ 
        current_P_adapt_law = self.P 

        if current_Lambda is None or current_P_adapt_law is None: 
            current_Lambda = np.eye(q.size)
            current_P_adapt_law = np.eye(y.size)

        e, de = q - r, dq - dr
        s = de + current_Lambda@e
        dA = np.outer(s, y) @ current_P_adapt_law
        return dA, y

    def get_force(self, dt, state, goal, f_hat):
        a_fb_calculated = np.zeros(3) 

        # Use goal.a directly, which is already zeros if not specified by trajectory
        goal_a_for_calc = goal.a 

        if self.controller == 'coml':
            current_Lambda = self.Λ 
            current_K_feedback = self.K    

            e, edot = state.p - goal.p, state.v - goal.v
            s = edot + current_Lambda@e
            v_ref_terms, dv_ref_terms = goal.v - current_Lambda@e, goal_a_for_calc - current_Lambda@edot

            H, C, g_dyn, B = prior(state.p, state.v) 
            τ = H@dv_ref_terms + C@v_ref_terms + g_dyn - f_hat - current_K_feedback@s
            F_W = np.linalg.solve(B, τ)
            try:
                H_inv = np.linalg.inv(H)
                a_fb_calculated = H_inv @ (-current_K_feedback @ s - f_hat)
            except np.linalg.LinAlgError:
                a_fb_calculated = np.zeros(3)

        elif self.controller  == 'coml_debug':
            Kp_diag = np.diag(self.params_.Kp)
            Kv_diag = np.diag(self.params_.Kd) 
            current_K_feedback = Kv_diag
            current_Lambda = Kp_diag @ np.linalg.inv(current_K_feedback) if np.linalg.det(current_K_feedback) !=0 else np.eye(state.p.size)

            e, edot = state.p - goal.p, state.v - goal.v
            s = edot + current_Lambda@e
            v_ref_terms, dv_ref_terms = goal.v - current_Lambda@e, goal_a_for_calc - current_Lambda@edot

            H, C, g_dyn, B = prior(state.p, state.v)
            τ = H@dv_ref_terms + C@v_ref_terms + g_dyn - current_K_feedback@s 
            F_W = np.linalg.solve(B, τ)
            try:
                H_inv = np.linalg.inv(H)
                a_fb_calculated = H_inv @ (-current_K_feedback @ s)
            except np.linalg.LinAlgError:
                a_fb_calculated = np.zeros(3)
        else: # PID controller
            e = goal.p - state.p
            edot = goal.v - state.v

            e = np.minimum(np.maximum(e, -self.params_.maxPosErr), self.params_.maxPosErr)
            edot = np.minimum(np.maximum(edot, -self.params_.maxVelErr), self.params_.maxVelErr)

            if goal.mode_xy != self.mode_xy_last_:
                self.Ix_.reset()
                self.Iy_.reset()
                self.mode_xy_last_ = goal.mode_xy

            if goal.mode_z != self.mode_z_last_:
                self.Iz_.reset()
                self.mode_z_last_ = goal.mode_z

            if goal.mode_xy == GoalClass.Mode.POS_CTRL:
                self.Ix_.increment(e[0], dt)
                self.Iy_.increment(e[1], dt)
            elif goal.mode_xy == GoalClass.Mode.VEL_CTRL:
                e[0] = e[1] = 0.0 
            elif goal.mode_xy == GoalClass.Mode.ACC_CTRL:
                e[0] = e[1] = 0.0
                edot[0] = edot[1] = 0.0

            if goal.mode_z == GoalClass.Mode.POS_CTRL:
                self.Iz_.increment(e[2], dt)
            elif goal.mode_z == GoalClass.Mode.VEL_CTRL:
                e[2] = 0.0
            elif goal.mode_z == GoalClass.Mode.ACC_CTRL:
                e[2] = 0.0
                edot[2] = 0.0

            eint = np.array([self.Ix_.value(), self.Iy_.value(), self.Iz_.value()])
            Kp_arr = np.array(self.params_.Kp)
            Ki_arr = np.array(self.params_.Ki)
            Kd_arr = np.array(self.params_.Kd)
            a_fb_calculated = Kp_arr * e + Ki_arr * eint + Kd_arr * edot
            
            F_W = self.params_.mass * (goal_a_for_calc + a_fb_calculated - self.GRAVITY)

        self.log_.p = state.p
        self.log_.p_ref = goal.p
        self.log_.p_err = e 
        if self.controller != 'coml' and self.controller != 'coml_debug': 
             self.log_.p_err_int = eint
        self.log_.v = state.v
        self.log_.v_ref = goal.v
        self.log_.v_err = edot 
        self.log_.a_ff = goal_a_for_calc # Log the acceleration used (could be zeros)
        self.log_.a_fb = a_fb_calculated 
        self.log_.F_W = F_W
        return F_W

    def get_attitude(self, state, goal, F_W):
        norm_F_W = np.linalg.norm(F_W)
        
        q_ref_calculated = np.array([np.cos(goal.psi / 2.0), 0.0, 0.0, np.sin(goal.psi / 2.0)])
        norm_default_q_ref = np.linalg.norm(q_ref_calculated)
        if norm_default_q_ref < 1e-9:
            q_ref_calculated = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            q_ref_calculated = q_ref_calculated / norm_default_q_ref

        if norm_F_W >= 1e-6: # Only calculate from F_W if force is significant
            xi = F_W / self.params_.mass
            abc = xi / norm_F_W # This is xi / ||xi|| because norm_F_W = ||xi|| * mass, so this is wrong.
                                # Correct: abc = xi / np.linalg.norm(xi)
            
            # Re-calculate abc correctly
            norm_xi_val = np.linalg.norm(xi)
            if norm_xi_val < 1e-9: # If xi is zero, abc cannot be calculated from it
                 # q_ref_calculated remains the default (level attitude)
                 pass
            else:
                abc = xi / norm_xi_val
                a, b, c = abc
                psi = goal.psi

                val_for_sqrt = 2 * (1 + c)
                if val_for_sqrt < 1e-9: 
                    # q_ref_calculated remains the default
                    # print(f"Warning: c is close to -1 (value: {c}), problematic for invsqrt21pc. Using default q_ref.")
                    pass
                else:
                    invsqrt21pc = 1 / np.sqrt(val_for_sqrt)
                    quaternion0 = np.array([invsqrt21pc*(1+c), invsqrt21pc*(-b), invsqrt21pc*a, 0.0])
                    quaternion1 = np.array([np.cos(psi/2.0), 0.0, 0.0, np.sin(psi/2.0)]) 
                    q_ref_calculated = quaternion_multiply(quaternion0, quaternion1)
        # else: q_ref_calculated remains the default level attitude

        q_ref_final = q_ref_calculated # It should be normalized by now or is a unit quaternion
        norm_q_ref_final = np.linalg.norm(q_ref_final)
        if abs(norm_q_ref_final - 1.0) > 1e-6 : # Re-normalize if not already unit
            if norm_q_ref_final < 1e-9:
                q_ref_final = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                q_ref_final = q_ref_final / norm_q_ref_final
            
        self.log_.q = state.q
        self.log_.q_ref = q_ref_final
        return q_ref_final

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

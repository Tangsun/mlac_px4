## This is a temporary file to debug the outer loop controller
## The goal is to compare COML implementation with no adaptation with the current PID Design

# import jax.numpy as jnp # jax.numpy is not used in the provided snippet
import numpy as np
import os
import pickle
# Removed: import rospkg # ROS 1 specific
from ament_index_python.packages import get_package_share_directory # ROS 2 equivalent

# Assuming these are local modules within your ROS 2 package or installed correctly
from .dynamics import prior # Using relative import if dynamics is in the same package
from .structs import AttCmdClass, ControlLogClass, GoalClass # Using relative import
from .helpers import quaternion_multiply # Using relative import
from .utils import params_to_posdef, quaternion_to_rotation_matrix # Using relative import

# Helper function (remains unchanged)
def convert_p_qbar(p):
    return np.sqrt(1/(1 - 1/p) - 1.1)

class IntegratorClass:
    def __init__(self):
        self.value_ = 0

    def increment(self, inc, dt):
        self.value_ += inc * dt

    def reset(self):
        self.value_ = 0

    def value(self):
        return self.value_

class OuterLoop:
    def __init__(self, params, state0, goal0, controller='coml', package_name='outer_loop_python'):
        """
        Initializes the OuterLoop controller.

        Args:
            params: Controller parameters.
            state0: Initial state of the vehicle.
            goal0: Initial goal for the vehicle.
            controller (str): Type of controller to use ('coml', 'coml_debug', 'pid').
            package_name (str): The name of the ROS 2 package where model files are located.
                                This is used for COML controllers.
        """
        self.controller = controller
        self.package_name = package_name # Store package name for model loading

        if self.controller == 'coml':
            # ROS 2: Get the share directory of the package
            try:
                package_share_path = get_package_share_directory(self.package_name)
            except Exception as e:
                print(f"Error: Could not find package '{self.package_name}'. Please ensure it's built and sourced.")
                print(f"Details: {e}")
                # Fallback or raise error, depending on desired behavior
                # For now, let's try to proceed assuming models might be in a local path if package not found
                package_share_path = "." # Fallback to current directory, not ideal
                print(f"Warning: Using fallback path '{package_share_path}' for models.")


            # Define trial and filename for the COML model
            trial_name = 'reg_P_1.0_reg_k_R_0.001_k_R_scale_1_k_R_z_1.26'
            filename = 'seed=0_M=50_E=1000_pinit=2.00_pfreq=2000_regP=1.0000.pkl'
            
            # Construct the model directory and file path
            # Assumes 'models' directory is directly under 'share/<package_name>/'
            model_dir = os.path.join(package_share_path, 'models', trial_name)
            model_pkl_loc = os.path.join(model_dir, filename)

            try:
                with open(model_pkl_loc, 'rb') as f:
                    train_results = pickle.load(f)
                print(f'COML model loaded from: {model_pkl_loc}')
                
                self.pnorm = convert_p_qbar(train_results['pnorm'])
                self.W = train_results['model']['W']
                self.b = train_results['model']['b']
                self.Λ = params_to_posdef(train_results['controller']['Λ'])
                self.K = params_to_posdef(train_results['controller']['K'])
                self.P = params_to_posdef(train_results['controller']['P'])
            except FileNotFoundError:
                print(f"Error: COML model file not found at {model_pkl_loc}")
                # Handle error appropriately, e.g., raise an exception or switch to a default controller
                raise
            except Exception as e:
                print(f"Error loading COML model: {e}")
                raise

        elif self.controller == 'coml_debug':
            # ROS 2: Get the share directory of the package (though not strictly needed for empty_model.pkl if not used)
            # package_share_path = get_package_share_directory(self.package_name)
            # trial_name = 'coml_debug'
            # filename = 'empty_model.pkl' # This file might not be strictly necessary if it's truly empty
            # model_dir = os.path.join(package_share_path, 'models', trial_name)
            # model_pkl_loc = os.path.join(model_dir, filename)
            # print(f"COML debug model: Looking for empty model at {model_pkl_loc} (not loaded).")

            print('COML debug model: no model is loaded!')
            self.pnorm = convert_p_qbar(2.0)
            # Initialize other COML-specific attributes to defaults or None if they won't be used
            self.W = None 
            self.b = None
            self.Λ = np.eye(3) # Example default
            self.K = np.eye(3) # Example default
            self.P = np.eye(3) # Example default
        
        self.params_ = params
        self.GRAVITY = np.array([0, 0, -9.80665])

        self.Ix_ = IntegratorClass()
        self.Iy_ = IntegratorClass()
        self.Iz_ = IntegratorClass()

        self.log_ = ControlLogClass()
        self.a_fb_last_ = np.zeros(3)
        self.j_fb_last_ = np.zeros(3)
        self.t_last_ = 0.0 # Initialize t_last_

        self.mode_xy_last_ = GoalClass.Mode.POS_CTRL
        self.mode_z_last_ = GoalClass.Mode.POS_CTRL
        
        self.reset(state0, goal0)

    def reset(self, state0, goal0):
        if self.controller == 'coml':
            # Assume starting position, velocity, attitude, and angular velocity of 0
            q0 = state0.p
            dq0 = state0.v
            R_flatten0 = quaternion_to_rotation_matrix(state0.q).flatten()
            Omega0 = state0.w
            r0 = goal0.p
            dr0 = goal.v # Corrected from goal0.v to goal.v if goal is the current goal object
            
            self.dA_prev, y0 = self.adaptation_law(q0, dq0, R_flatten0, Omega0, r0, dr0)
            self.pA_prev = np.zeros((q0.size, y0.size))
        elif self.controller == 'coml_debug':
            # Initialize for coml_debug if necessary, e.g., dA_prev and pA_prev
            # For simplicity, assuming they are not strictly needed or handled if None
            self.dA_prev = None 
            self.pA_prev = None
            
        self.Ix_.reset()
        self.Iy_.reset()
        self.Iz_.reset()

        self.log_ = ControlLogClass() # Re-initialize log
        self.a_fb_last_ = np.zeros(3)
        self.j_fb_last_ = np.zeros(3)
        self.t_last_ = 0.0 # Reset t_last_

    def update_log(self, state):
        # This function re-creates the log object each time.
        # Consider if it should update fields of the existing self.log_
        self.log_ = ControlLogClass()
        self.log_.p = state.p
        self.log_.v = state.v
        self.log_.q = state.q
        self.log_.w = state.w

    def compute_attitude_command(self, t, state, goal):
        dt = 1e-2 if self.t_last_ == 0 else t - self.t_last_

        # It's generally better to ensure dt is positive before proceeding
        if dt <= 0:
            print(f"Warning: non-positive dt: {dt} [s]. Using previous log and returning previous command or a safe command.")
            # Decide how to handle this: re-use last command, or a neutral command.
            # For now, just log and proceed, but this could lead to issues.
            # If t_last_ is not updated, dt will remain non-positive in subsequent calls if t doesn't change.
            # A robust solution might involve returning the last computed command or a default safe command.
            # For this adaptation, we'll let it proceed but update t_last_ only if dt > 0
            if self.log_.F_W is None: # If no command ever computed
                 # Fallback to a very basic hover-like command if F_W not initialized
                cmd = AttCmdClass()
                cmd.q = np.array([1.0, 0.0, 0.0, 0.0]) # Level attitude
                cmd.w = np.zeros(3)
                cmd.F_W = -self.params_.mass * self.GRAVITY # Thrust to counteract gravity
                return cmd
            else: # Return last command
                cmd = AttCmdClass()
                cmd.q = self.log_.q_ref if self.log_.q_ref is not None else np.array([1.0, 0.0, 0.0, 0.0])
                cmd.w = self.log_.w_ref if self.log_.w_ref is not None else np.zeros(3)
                cmd.F_W = self.log_.F_W
                return cmd


        self.t_last_ = t # Update t_last_ here since dt > 0 is now guaranteed for computation
        
        if self.controller == 'coml':
            if self.P is None: # Check if COML model was loaded properly
                print("Error: COML controller selected but P matrix is None. Model might not have loaded.")
                f_hat = np.zeros(3) # Fallback
            else:
                qn = 1.1 + self.pnorm**2

                # Integrate adaptation law via trapezoidal rule
                R_flatten = quaternion_to_rotation_matrix(state.q).flatten()
                dA, y = self.adaptation_law(state.p, state.v, R_flatten, state.w, goal.p, goal.v)
                
                # Ensure pA_prev and dA_prev were initialized
                if self.pA_prev is None or self.dA_prev is None:
                    print("Error: pA_prev or dA_prev not initialized for COML controller.")
                    # Initialize them here if it's the first proper run after a reset
                    self.dA_prev, y0 = self.adaptation_law(state.p, state.v, R_flatten, state.w, goal.p, goal.v) # Re-call with current state
                    self.pA_prev = np.zeros((state.p.size, y0.size))
                    dA = self.dA_prev # Use current dA for the first step

                pA = self.pA_prev + (dt)*(self.dA_prev + dA)/2.0 # Ensure float division
                P_matrix = self.P # Use a more descriptive name
                # Original: A = (np.maximum(np.abs(pA), 1e-6 * np.ones_like(pA))**(qn-1) * np.sign(pA) * (np.ones_like(pA) - np.isclose(pA, 0, atol=1e-6)) ) @ P_matrix
                
                # Simplified and potentially more robust calculation for A:
                abs_pA = np.abs(pA)
                pA_power_term = np.power(np.maximum(abs_pA, 1e-6), qn - 1) * np.sign(pA)
                # Ensure not to multiply by zero if pA is close to zero, effectively zeroing out the term
                pA_power_term[abs_pA < 1e-6] = 0.0 
                A = pA_power_term @ P_matrix

                f_hat = A @ y

                # Log adaptation values
                self.log_.P_norm = np.linalg.norm(P_matrix)
                self.log_.A_norm = np.linalg.norm(A)
                self.log_.y_norm = np.linalg.norm(y)
                self.log_.f_hat = f_hat

                # Update prev values
                self.pA_prev = pA
                self.dA_prev = dA
        else: # PID or coml_debug
            f_hat = np.zeros(3) # No adaptation

        # Compute force, attitude, and rates
        F_W = self.get_force(dt, state, goal, f_hat)
        q_ref = self.get_attitude(state, goal, F_W)
        
        # For PID, a_fb is calculated inside get_force. For COML, it's not explicitly passed.
        # The get_rates function expects a_fb. We need to ensure it's available.
        # If using COML, a_fb isn't directly computed in the same way as PID.
        # Let's use self.log_.a_fb which should be populated by get_force if it's PID.
        # If COML, a_fb might conceptually be zero or implicitly handled.
        # For now, let's assume self.log_.a_fb is the correct one to use.
        # If it's not set by get_force for COML, then j_fb will be based on its last value or zero.
        current_a_fb = self.log_.a_fb if hasattr(self.log_, 'a_fb') and self.log_.a_fb is not None else np.zeros(3)
        w_ref = self.get_rates(dt, state, goal, F_W, current_a_fb, q_ref)


        cmd = AttCmdClass()
        cmd.q = q_ref
        cmd.w = w_ref
        cmd.F_W = F_W # Store the computed world frame force

        return cmd

    def adaptation_law(self, q, dq, R_flatten, Omega, r, dr):
        # Regressor features
        y = np.concatenate((q, dq, R_flatten, Omega))
        
        # Check if W and b are initialized (e.g. for coml_debug)
        if self.W is None or self.b is None:
            # This case should ideally only happen for coml_debug if no model is used.
            # The regressor y might not be meaningful here, or a default behavior is needed.
            # For now, let's assume y is passed through if W,b are not set (effectively no feature transformation)
            pass # y remains as concatenated state
        else:
            for W_layer, b_layer in zip(self.W, self.b): # Use more descriptive names
                y = np.tanh(W_layer @ y + b_layer)

        # Auxiliary signals
        # Ensure Λ and P are valid matrices
        current_Lambda = self.Λ if self.Λ is not None else np.eye(q.size) # Default if None
        current_P_adapt = self.P if self.P is not None else np.eye(y.size)  # Default if None, size might be an issue

        e, de = q - r, dq - dr
        s = de + current_Lambda @ e
        
        # Check dimensions for np.outer(s, y) @ current_P_adapt
        # s is (3,), y can be (N,). np.outer(s,y) is (3,N)
        # current_P_adapt should be (N, M) or (N,N) if M=N
        # dA should be (3,M)
        # If current_P_adapt is (y.size, y.size), then dA is (s.size, y.size)
        if s.ndim == 1 and y.ndim == 1:
            if current_P_adapt.shape[0] != y.size:
                 print(f"Warning: P matrix for adaptation law has mismatched dimensions. P.shape[0]={current_P_adapt.shape[0]}, y.size={y.size}")
                 # Fallback or error. For now, create a compatible P if possible, or return zero dA.
                 # This indicates a setup issue with the COML model parameters.
                 dA = np.zeros((s.size, current_P_adapt.shape[1] if current_P_adapt.ndim > 1 else y.size ))
            else:
                 dA = np.outer(s, y) @ current_P_adapt
        else:
            print("Warning: s or y in adaptation_law are not 1D arrays as expected.")
            dA = np.zeros((s.size if hasattr(s, 'size') else 3, y.size if hasattr(y, 'size') else current_P_adapt.shape[1]))


        return dA, y

    def get_force(self, dt, state, goal, f_hat):
        e = np.zeros(3) # Initialize to ensure it's always defined
        edot = np.zeros(3) # Initialize
        a_fb = np.zeros(3) # Initialize a_fb, used for logging and get_rates

        if self.controller == 'coml' or self.controller == 'coml_debug':
            current_Lambda = self.Λ
            current_K = self.K

            if current_Lambda is None or current_K is None:
                print(f"Warning: COML matrices (Lambda or K) are None for controller '{self.controller}'. Using PID-like defaults.")
                # Fallback to PID-like calculation if COML matrices are not set
                # This is a simplified PID for fallback, not the full PID from the 'else' branch
                Kp_diag = self.params_.Kp if hasattr(self.params_, 'Kp') else np.array([1.0, 1.0, 1.0])
                Kd_diag = self.params_.Kd if hasattr(self.params_, 'Kd') else np.array([1.0, 1.0, 1.0])
                K_p = np.diag(Kp_diag)
                K_v = np.diag(Kd_diag)
                current_K = K_v
                current_Lambda = K_p @ np.linalg.inv(K_v) if np.linalg.det(K_v) != 0 else np.eye(3)


            e, edot = state.p - goal.p, state.v - goal.v
            s = edot + current_Lambda @ e
            v_ref_terms, dv_ref_terms = goal.v - current_Lambda @ e, goal.a - current_Lambda @ edot

            H, C, g, B = prior(state.p, state.v) # Assuming prior function is available and correct
            tau = H @ dv_ref_terms + C @ v_ref_terms + g - f_hat - current_K @ s
            
            try:
                F_W = np.linalg.solve(B, tau)
            except np.linalg.LinAlgError:
                print("Error: Singular matrix B in get_force. Using pseudo-inverse.")
                F_W = np.linalg.pinv(B) @ tau
            
            # For COML, a_fb is not explicitly PID-style.
            # We can conceptualize the feedback portion of 'tau' as related to a_fb.
            # For logging consistency, let's estimate an equivalent a_fb.
            # The term -K@s is the primary feedback part related to acceleration.
            # And f_hat is also a feedback term.
            # So, H @ a_fb_equivalent = -K@s - f_hat
            # a_fb_equivalent = inv(H) @ (-K@s - f_hat)
            try:
                H_inv = np.linalg.inv(H)
                a_fb = H_inv @ (-current_K @ s - f_hat) # This is an *effective* a_fb for logging
            except np.linalg.LinAlgError:
                a_fb = np.zeros(3) # Fallback if H is singular

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
            # Ensure Kp, Ki, Kd are numpy arrays for element-wise multiplication
            Kp_arr = np.array(self.params_.Kp)
            Ki_arr = np.array(self.params_.Ki)
            Kd_arr = np.array(self.params_.Kd)
            a_fb = Kp_arr * e + Ki_arr * eint + Kd_arr * edot
            
            F_W = self.params_.mass * (goal.a + a_fb - self.GRAVITY)

        # Log control signals
        self.log_.p = state.p
        self.log_.p_ref = goal.p
        self.log_.p_err = e # Note: for COML, e is state.p - goal.p; for PID, e is goal.p - state.p. Standardize if needed.
                           # Current COML e = state.p - goal.p. Let's stick to that for consistency with s.
                           # If PID e was goal.p - state.p, then p_err for PID should be -e from COML's perspective.
                           # For now, logging the 'e' as defined in each block.
        if hasattr(self, 'eint') and self.controller != 'coml' and self.controller != 'coml_debug': # Only log eint for PID
             self.log_.p_err_int = eint
        self.log_.v = state.v
        self.log_.v_ref = goal.v
        self.log_.v_err = edot # Similar to e, edot definition consistency.
        self.log_.a_ff = goal.a
        self.log_.a_fb = a_fb # This is now set for all controller types
        self.log_.F_W = F_W

        return F_W

    def get_attitude(self, state, goal, F_W):
        # Ensure F_W has a minimum magnitude to avoid division by zero or instability
        norm_F_W = np.linalg.norm(F_W)
        if norm_F_W < 1e-6: # If force is negligible, command level attitude
            # print("Warning: Desired force F_W is near zero. Commanding level attitude.")
            q_ref = np.array([np.cos(goal.psi / 2.0), 0.0, 0.0, np.sin(goal.psi / 2.0)]) # Level with desired yaw
            q_ref = q_ref / np.linalg.norm(q_ref)
            self.log_.q = state.q
            self.log_.q_ref = q_ref
            return q_ref

        # xi = F_W / self.params_.mass  # Eq. 26 (Original)
        # Normalized desired force direction vector (thrust vector in world frame)
        # abc = F_W / norm_F_W # This is the desired z-axis of the body in world frame
        
        # Let b_z_d be the desired body z-axis in the world frame
        b_z_d = F_W / norm_F_W

        # Desired yaw angle
        psi = goal.psi

        # Construct desired rotation matrix R_WB_d (World to Body Desired)
        # The z-axis of this R_WB_d is b_z_d
        # The x-axis can be derived using desired yaw:
        # b_x_d_projection_on_xy_plane = np.array([np.cos(psi), np.sin(psi), 0.0])
        # However, b_x_d must be orthogonal to b_z_d.
        # A common way:
        # 1. Desired body z-axis: b_z_d = F_W / ||F_W||
        # 2. Desired body x-axis projection on world XY plane: b_x_d_xy_proj = [cos(psi), sin(psi), 0]
        # 3. Desired body y-axis: b_y_d = normalize(cross(b_z_d, b_x_d_xy_proj))
        #    Handle case where b_z_d is aligned with b_x_d_xy_proj (e.g. F_W is purely horizontal with yaw aligned)
        #    This happens if F_W is [F_wx, F_wy, 0] and psi = atan2(F_wy, F_wx)
        # 4. Desired body x-axis: b_x_d = normalize(cross(b_y_d, b_z_d))

        # Simpler approach from many geometric controllers:
        # Target z-axis of body in world frame
        z_b_des = F_W / norm_F_W

        # Target x-axis of body in world frame (construct from yaw)
        # Project desired heading onto XY plane, then make it orthogonal to z_b_des
        x_c_des = np.array([np.cos(psi), np.sin(psi), 0.0]) # Commanded x-axis in world XY plane
        
        # Ensure z_b_des is not perfectly vertical for cross product stability with x_c_des if x_c_des is used directly
        # If z_b_des is [0,0,1] or [0,0,-1], x_c_des is fine.
        
        # Desired y-axis of body in world frame
        # y_b_des = normalize( cross(z_b_des, x_c_des) )
        # Need to handle if z_b_des and x_c_des are parallel (e.g. F_W is purely horizontal and aligned with yaw)
        # This happens if z_b_des[2] is near zero.
        
        dot_prod_zc = np.dot(z_b_des, x_c_des)
        if np.abs(np.linalg.norm(np.cross(z_b_des, x_c_des))) < 1e-6 : # If z_b_des and x_c_des are nearly parallel
            # This case is tricky. If F_W is horizontal and aligned with yaw, y_b_des is ambiguous from this construction.
            # For example, if F_W = [1,0,0] and psi = 0. Then z_b_des = [1,0,0], x_c_des = [1,0,0]. Cross product is zero.
            # This implies a 90-degree pitch up/down.
            # Let's use a fallback or alternative construction for this edge case.
            # If F_W is purely horizontal, z_b_des is in XY plane.
            # We want the body x-axis to be "as close as possible" to x_c_des while being perp to z_b_des.
            # And body y-axis perp to both.
            # This usually means body x-axis will be vertical if z_b_des is horizontal.
            # The original paper's formulation (Eq. 19-24) might be more robust here.
            # For now, let's use the original formulation's quaternion construction which avoids explicit R matrix.
            
            a, b, c = z_b_des # Renaming for consistency with original paper's notation where abc = xi / ||xi||
            
            # Handle c == -1 case (force pointing straight down)
            if np.isclose(c, -1.0): # F_W is pointing straight down
                # Quaternion for 180-degree rotation around an arbitrary axis in XY plane, e.g., world X-axis [1,0,0]
                # This results in body z-axis pointing down, body x-axis pointing along -world_x
                # q0 = [0, 1, 0, 0] # 180 deg rotation around x-axis
                # To align with desired yaw psi: we want body_x to align with [cos(psi), sin(psi), 0] projected
                # and body_z to be [0,0,-1].
                # R_z(psi) followed by R_y(pi)
                # q_yaw = [cos(psi/2), 0,0,sin(psi/2)]
                # q_pitch_180 = [0,0,1,0] (180 deg around y)
                # q_ref = quaternion_multiply(q_pitch_180, q_yaw) # Check order
                # This means: R = R_y(pi) * R_z(psi). Body_z = R * [0,0,1]_B = [-sin(psi), cos(psi), 0]_W (incorrect)
                # We want body_z_W = [0,0,-1]. Body_x_W should align with [cos(psi), sin(psi), 0]
                # So, R_WB_d = [ [cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0,0,-1] ] (if x-fwd, y-left, z-up body frame convention was used for R)
                # Or for x-fwd, y-right, z-down: R_WB_d = [ [cos(psi), sin(psi), 0], [-sin(psi), cos(psi), 0], [0,0,1] ] where F_W points along +z_down
                # If F_W points straight down, z_b_des = [0,0,-1] (NED frame for z_b_des) or [0,0,1] (ENU frame for z_b_des)
                # Assuming F_W is in world frame, and GRAVITY is [0,0,-9.8], so F_W for hover is [0,0,mg]
                # Then z_b_des for hover is [0,0,1] (world Z up).
                # If c = z_b_des[2] approx -1, it means F_W is pointing straight down in a Z-up world. (e.g. F_W = [0,0,-mg])
                # This is unusual unless inverted flight.
                # Let's assume c is based on a world frame where +Z is up.
                # If c ~ -1 (force straight down), invsqrt21pc has 1+c ~ 0 in denominator.
                # Original paper formulation: if c = -1, q = [0, sin(psi/2), cos(psi/2), 0] (Eq. 23)
                # This is for a specific choice. A common choice for c=-1 (thrust vector [0,0,-1]):
                # body x points along world x, body y along world -y, body z along world -z. (180 deg pitch)
                # q = [0, 1, 0, 0] (180 deg around x-axis) then apply yaw.
                q_pitch180 = np.array([0.0, 1.0, 0.0, 0.0]) # 180 deg rotation around world x-axis
                q_yaw = np.array([np.cos(psi/2.0), 0.0, 0.0, np.sin(psi/2.0)])
                q_ref = quaternion_multiply(q_pitch180, q_yaw) # Order might matter: yaw first then pitch, or vice-versa
                                                              # R = R_x(pi) R_z(psi). Check this.
            else: # Standard case c > -1
                invsqrt21pc = 1.0 / np.sqrt(2.0 * (1.0 + c))
                quaternion0 = np.array([invsqrt21pc*(1.0+c), invsqrt21pc*(-b), invsqrt21pc*a, 0.0]) # (w,x,y,z)
                quaternion1 = np.array([np.cos(psi/2.0), 0.0, 0.0, np.sin(psi/2.0)])
                q_ref = quaternion_multiply(quaternion0, quaternion1)

        q_ref = q_ref / np.linalg.norm(q_ref)

        self.log_.q = state.q
        self.log_.q_ref = q_ref
        return q_ref

    def get_rates(self, dt, state, goal, F_W, a_fb, q_ref):
        j_fb = np.zeros(3)
        if dt > 1e-6: # Avoid division by zero if dt is too small
            # Numeric derivative for j_fb
            current_j_fb = (a_fb - self.a_fb_last_) / dt

            # Low-pass filter differentiation
            tau = 0.1 # LPF time constant for jerk
            alpha = dt / (tau + dt)
            j_fb = alpha * current_j_fb + (1.0 - alpha) * self.j_fb_last_
        else:
            j_fb = self.j_fb_last_ # Reuse last value if dt is too small or zero

        self.a_fb_last_ = a_fb # Save for the next iteration
        self.j_fb_last_ = j_fb # Save for the next iteration

        # Construct angular rates consistent with trajectory dynamics
        # Fdot_W = self.params_.mass * (goal.j + j_fb) # Original had goal.j, but goal.j is jerk_ff.
        # F_W = m * (a_ff + a_fb - g). So Fdot_W = m * (j_ff + j_fb - 0) if g is constant.
        Fdot_W = self.params_.mass * (goal.j + j_fb) # goal.j is feedforward jerk
        
        norm_F_W = np.linalg.norm(F_W)
        if norm_F_W < 1e-6: # If F_W is zero, desired rates are also zero (or based on psi_dot only)
            # print("Warning: F_W is near zero in get_rates. Setting body rates based on dpsi only.")
            rates = np.array([0.0, 0.0, goal.dpsi])
            # Log here if necessary, or ensure log population at the end covers this.
            self.log_.w = state.w
            self.log_.w_ref = rates
            # Populate other relevant log fields for this case
            self.log_.j_ff = goal.j
            self.log_.j_fb = j_fb
            return rates

        # xi = F_W / self.params_.mass (already computed effectively as b_z_d)
        # abc = F_W / norm_F_W
        # xi_dot = Fdot_W / self.params_.mass
        
        b_z_d = F_W / norm_F_W # Current body z-axis direction in world
        b_z_d_dot = Fdot_W / self.params_.mass # Derivative of unnormalized thrust vector xi
        
        # abcdot = ( (norm_xi**2 * I - outer(xi,xi)) / norm_xi**3 ) @ xi_dot (Eq. 20)
        # This is derivative of the normalized vector b_z_d
        # d(v/||v||)/dt = (||v||^2 * dv/dt - (v . dv/dt) * v) / ||v||^3
        # Here, v = F_W, dv/dt = Fdot_W. Let m = self.params_.mass
        # v = m*xi, dv/dt = m*xi_dot
        # So, b_z_d_dot_calculated = (norm_F_W**2 * Fdot_W - np.dot(F_W, Fdot_W) * F_W) / (norm_F_W**3)
        # This is abcdot in the paper's notation.
        
        # If F_W is used directly (not xi):
        # Let v = F_W. Then b_z_d = v / ||v||.
        # d(b_z_d)/dt = (Fdot_W * ||F_W|| - F_W * (np.dot(F_W, Fdot_W) / ||F_W||) ) / ||F_W||^2
        #             = (Fdot_W * ||F_W||^2 - F_W * np.dot(F_W, Fdot_W)) / ||F_W||^3
        
        norm_F_W_sq = norm_F_W * norm_F_W
        # Derivative of the normalized thrust vector direction
        b_z_d_deriv = (Fdot_W * norm_F_W_sq - F_W * np.dot(F_W, Fdot_W)) / (norm_F_W_sq * norm_F_W + 1e-9) # Add epsilon for safety

        # Assert b_z_d . b_z_d_deriv should be approximately 0.0
        # if not np.allclose(np.dot(b_z_d, b_z_d_deriv), 0.0, atol=1e-3): # Relaxed tolerance
            # print(f"Warning: b_z_d and b_z_d_deriv are not orthogonal: {np.dot(b_z_d, b_z_d_deriv)}")

        a, b, c = b_z_d
        adot, bdot, cdot = b_z_d_deriv
        psi, dpsi = goal.psi, goal.dpsi # dpsi is yaw_rate_desired

        rates = np.zeros(3)
        # Denominator term 1+c can be zero if c = -1 (thrust vector is [0,0,-1])
        den_1_plus_c = 1.0 + c
        if np.isclose(den_1_plus_c, 0.0): # c is close to -1
            # This case corresponds to F_W pointing straight down.
            # Original paper Eq. 25 provides specific formulas for c = -1.
            # Omega_x = adot * sin(psi) - bdot * cos(psi)
            # Omega_y = adot * cos(psi) + bdot * sin(psi)
            # Omega_z = dpsi
            # This assumes a particular body frame orientation when c=-1.
            # Let's use a more general approach by projecting angular velocity.
            # w_body = R_BW @ w_world. R_BW = q_ref_conj * q_dot_ref (kinematic relation)
            # This is more complex. The paper's derivation is specific.
            # If c = -1, q_ref is approx [0, sin(psi/2), cos(psi/2), 0] (from Eq.23, if a=0, b=1)
            # or [0, cos(psi/2), -sin(psi/2), 0] (if a=1, b=0)
            # For now, if c is very close to -1, the rates might become unstable.
            # A robust solution would involve deriving w_ref from R_dot = S(w_ref)R
            # R_ref = quaternion_to_rotation_matrix(q_ref)
            # R_dot_ref can be found by differentiating q_ref or by differentiating the R components.
            # For simplicity, if c is problematic, we might limit rates or use dpsi only.
            print("Warning: c is close to -1 in get_rates, rates calculation might be unstable. Using yaw rate only.")
            rates[2] = dpsi
        else:
            rates[0] = (np.sin(psi) * adot - np.cos(psi) * bdot - (a * np.sin(psi) - b * np.cos(psi)) * (cdot / den_1_plus_c))
            rates[1] = (np.cos(psi) * adot + np.sin(psi) * bdot - (a * np.cos(psi) + b * np.sin(psi)) * (cdot / den_1_plus_c))
            rates[2] = ((b * adot - a * bdot) / den_1_plus_c + dpsi)

        # Log control signals
        self.log_.j_ff = goal.j # Feedforward jerk from goal
        self.log_.j_fb = j_fb   # Feedback jerk computed
        self.log_.w = state.w   # Current vehicle angular rates
        self.log_.w_ref = rates # Desired angular rates

        return rates

    def get_log(self):
        return self.log_


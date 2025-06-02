import argparse
import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# Assuming utils.py and dynamics.py are in the same directory or accessible
from dynamics import prior 
from utils import (
    random_ragged_spline, spline,
    params_to_posdef, hat, vee,
    odeint_fixed_step
)

# --- Configuration for Matplotlib ---
plt.rcParams.update({'font.size': 12})

# --- Helper: Model Performance, Visualization, and Loss Saving ---
def analyze_best_model_and_training_history(raw_data, output_dir, 
                                            plot_trajectory=True, 
                                            save_loss_plot=False, 
                                            save_loss_data=False):
    # Assuming 'best_step_meta' and other keys exist if this function is called.
    best_epoch_idx = raw_data['best_step_meta']
    print(f"--- Analyzing Model (Best Trained Epoch: {best_epoch_idx}) ---")

    hparams_meta = raw_data['hparams']['meta']
    dt = hparams_meta['dt']
    T = hparams_meta['T']
    
    best_epoch_data = raw_data['train_lossaux_history'][best_epoch_idx]
    print("\nPerformance Metrics for Best Trained Epoch:")
    print(f"\tMeta-Training Loss (at best epoch update): {best_epoch_data['loss']:.4f}")
    if 'reg_P_penalty' in best_epoch_data: # Optional keys can still be checked
        print(f"\tReg P Penalty (term): {best_epoch_data['reg_P_penalty']:.4f}")
    if 'reg_k_R_penalty' in best_epoch_data:
        print(f"\tReg k_R Penalty (term): {best_epoch_data['reg_k_R_penalty']:.4f}")
    if 'k_R' in best_epoch_data:
        print(f"\tk_R values: {best_epoch_data['k_R']}")
    if 'eigs_P' in best_epoch_data:
        print(f"\tEigenvalues of P (controller gains): {best_epoch_data['eigs_P']}")

    q_bar = raw_data['pnorm']
    p_val = 1.0 / (1.0 - 1.0 / (1.1 + q_bar ** 2)) if (1.1 + q_bar ** 2) != 0 and \
                                                     (1.0 - 1.0 / (1.1 + q_bar ** 2)) != 0 else float('inf')
    print(f"\tP-norm (q_bar): {q_bar:.4f} (p-value: {p_val:.4f})")


    if plot_trajectory and 'x' in best_epoch_data :
        print("\nPlotting and saving a training trajectory for the best trained epoch...")
        # Direct access assuming keys exist
        x_traj = best_epoch_data['x']
        coefs_train = raw_data['coefs']
        t_knots_train = raw_data['t_knots']
        min_ref_train = jnp.array(raw_data['min_ref'])
        max_ref_train = jnp.array(raw_data['max_ref'])
        mystery_idx_plot = 0

        def reference_train_func(t_val):
            r_components = []
            for c_idx in range(len(coefs_train)):
                r_components.append(
                    spline(t_val, t_knots_train[mystery_idx_plot], coefs_train[c_idx][mystery_idx_plot]))
            r = jnp.array(r_components) + jnp.array([0, 0, 1])
            r = jnp.clip(r, min_ref_train, max_ref_train)
            return r

        num_steps_plot = int(np.maximum(np.abs((T - 0) / dt), 1))
        ts_plot = jnp.linspace(0, T, num_steps_plot + 1)
        r_traj = jax.vmap(reference_train_func)(ts_plot)
        dr_traj = jax.vmap(jax.jacfwd(reference_train_func))(ts_plot)

        fig_pos, ax_pos = plt.subplots(1, 3, figsize=(15, 5))
        labels_pos = ['x', 'y', 'z']
        for i_ax in range(3):
            ax_pos[i_ax].plot(ts_plot, x_traj[:, i_ax], 'r-', label=f'q_{labels_pos[i_ax]} (actual)')
            ax_pos[i_ax].plot(ts_plot, r_traj[:, i_ax], 'b--', label=f'r_{labels_pos[i_ax]} (ref)')
            ax_pos[i_ax].set_ylabel(labels_pos[i_ax]); ax_pos[i_ax].set_xlabel('t (s)'); ax_pos[i_ax].legend()
        fig_pos.suptitle(f'Position Tracking (Training Traj, Epoch {best_epoch_idx})')
        fig_pos.tight_layout()
        plot_path_pos = os.path.join(output_dir, f"training_pos_tracking_epoch{best_epoch_idx}.png")
        fig_pos.savefig(plot_path_pos); plt.close(fig_pos)
        print(f"Saved training position plot to {plot_path_pos}")

        fig_vel, ax_vel = plt.subplots(1, 3, figsize=(15, 5))
        labels_vel = ['dx', 'dy', 'dz']
        for i_ax in range(3):
            ax_vel[i_ax].plot(ts_plot, x_traj[:, i_ax + 3], 'r-', label=f'dq_{labels_pos[i_ax]} (actual)')
            ax_vel[i_ax].plot(ts_plot, dr_traj[:, i_ax], 'b--', label=f'dr_{labels_pos[i_ax]} (ref)')
            ax_vel[i_ax].set_ylabel(labels_vel[i_ax]); ax_vel[i_ax].set_xlabel('t (s)'); ax_vel[i_ax].legend()
        fig_vel.suptitle(f'Velocity Tracking (Training Traj, Epoch {best_epoch_idx})')
        fig_vel.tight_layout()
        plot_path_vel = os.path.join(output_dir, f"training_vel_tracking_epoch{best_epoch_idx}.png")
        fig_vel.savefig(plot_path_vel); plt.close(fig_vel)
        print(f"Saved training velocity plot to {plot_path_vel}")
    elif plot_trajectory:
         print("\nSkipping training trajectory plotting: 'x' not in best_epoch_data.")


    if save_loss_plot or save_loss_data:
        train_loss_aux_history = raw_data['train_lossaux_history']
        validation_loss_history = raw_data['valid_loss_history']
        
        epochs = range(len(train_loss_aux_history))
        total_meta_training_losses = [item['loss'] for item in train_loss_aux_history]
        
        if save_loss_data:
            loss_data_path = os.path.join(output_dir, "loss_history_data.txt")
            with open(loss_data_path, "w") as f:
                f.write("Epoch,MetaTrainingLoss,ValidationLoss\n")
                max_epochs = max(len(total_meta_training_losses), len(validation_loss_history))
                for i in range(max_epochs):
                    train_loss_val = total_meta_training_losses[i] if i < len(total_meta_training_losses) else 'N/A'
                    valid_loss_val = validation_loss_history[i] if i < len(validation_loss_history) else 'N/A'
                    f.write(f"{i},{train_loss_val},{valid_loss_val}\n")
            print(f"Saved loss data to {loss_data_path}")

        if save_loss_plot and train_loss_aux_history:
            print("\nPlotting and saving loss histories...")
            fig_loss, ax_loss = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            ax_loss[0].plot(epochs, total_meta_training_losses, label='Meta-Training Loss')
            ax_loss[0].set_xlabel('Epoch'); ax_loss[0].set_ylabel('Loss')
            ax_loss[0].set_title('Meta-Training Loss History'); ax_loss[0].legend(); ax_loss[0].set_yscale('log')

            if validation_loss_history:
                ax_loss[1].plot(range(len(validation_loss_history)), validation_loss_history, label='Validation Loss')
            ax_loss[1].set_xlabel('Epoch'); ax_loss[1].set_title('Validation Loss History')
            ax_loss[1].legend(); ax_loss[1].set_yscale('log')

            fig_loss.tight_layout()
            plot_path_loss = os.path.join(output_dir, "loss_histories.png")
            fig_loss.savefig(plot_path_loss); plt.close(fig_loss)
            print(f"Saved loss history plot to {plot_path_loss}")
        elif save_loss_plot:
            print("\nSkipping loss history plotting: No training loss data found.")

# --- ODE for simulation ---
@partial(jax.jit, static_argnums=(6, 7)) # reference_func and prior_dyn are static
def controlled_ode_sim_step3(z, t, controller_params_W_b, controller_gains, controller_pnorm_val,
                             plant_dynamics_ensemble_params, reference_func, prior_dyn):
    x_state, R_flatten_state, Omega_state, pA_state, c_tracking_error_sq_state = z
    num_dof_x = x_state.size // 2
    q, dq = x_state[:num_dof_x], x_state[num_dof_x:]

    r_val = reference_func(t)
    dr_val = jax.jacfwd(reference_func)(t)
    ddr_val = jax.jacfwd(jax.jacfwd(reference_func))(t)

    y_features_nn_input = jnp.concatenate([x_state, R_flatten_state, Omega_state], axis=0)
    y_features_nn_output = y_features_nn_input 
    # Assuming controller_params_W_b['W'] and ['b'] exist and are non-empty lists if NN is used
    if controller_params_W_b['W']: 
        temp_features = y_features_nn_input
        for W_layer, b_layer in zip(controller_params_W_b['W'], controller_params_W_b['b']):
            temp_features = jnp.tanh(W_layer @ temp_features + b_layer)
        y_features_nn_output = temp_features

    Λ_gain = params_to_posdef(controller_gains['Λ'])
    K_gain = params_to_posdef(controller_gains['K'])
    P_gain_mat = params_to_posdef(controller_gains['P'])
    k_R_gain = controller_gains['k_R']
    k_Omega_gain = controller_gains['k_Omega']
    qn_val = 1.1 + controller_pnorm_val ** 2
    
    A_adaptive_term = (jnp.maximum(jnp.abs(pA_state), 1e-6) ** (qn_val - 1) * jnp.sign(pA_state) *
                      (1 - jnp.isclose(pA_state, 0, atol=1e-6))) 
    A_adaptive = A_adaptive_term @ P_gain_mat 
    f_ext_hat_val = A_adaptive @ y_features_nn_output

    e_pos, de_pos = q - r_val, dq - dr_val
    v_aux, dv_aux = dr_val - Λ_gain @ e_pos, ddr_val - Λ_gain @ de_pos
    s_err = de_pos + Λ_gain @ e_pos

    H_dyn, C_dyn, g_dyn, B_dyn = prior_dyn(q, dq)
    tau_val = H_dyn @ dv_aux + C_dyn @ v_aux + g_dyn - f_ext_hat_val - K_gain @ s_err
    u_d_val = jnp.linalg.solve(B_dyn, tau_val)
    dpA_state = jnp.outer(s_err, y_features_nn_output) @ P_gain_mat

    R_mat = R_flatten_state.reshape((3, 3))
    J_inertia = jnp.diag(jnp.array([0.02167, 0.02167, 0.04000]))
    f_d_val = jnp.linalg.norm(u_d_val)
    b_3d_val = u_d_val / (jnp.linalg.norm(u_d_val) + 1e-6)
    b2_d_val_temp = jnp.cross(b_3d_val, jnp.array([1.0, 0.0, 0.0]))
    b2_d_val = b2_d_val_temp / (jnp.linalg.norm(b2_d_val_temp) + 1e-6)
    b1_d_val = jnp.cross(b2_d_val, b_3d_val)
    R_d_mat = jnp.column_stack((b1_d_val, b2_d_val, b_3d_val))
    Omega_d_val = jnp.zeros(3); dOmega_d_val = jnp.zeros(3)
    e_R_val = 0.5 * vee(R_d_mat.T @ R_mat - R_mat.T @ R_d_mat)
    e_Omega_val = Omega_state - R_mat.T @ R_d_mat @ Omega_d_val
    M_val = (- k_R_gain * e_R_val - k_Omega_gain * e_Omega_val + 
             jnp.cross(Omega_state, J_inertia @ Omega_state) - 
             J_inertia @ (hat(Omega_state) @ R_mat.T @ R_d_mat @ Omega_d_val - R_mat.T @ R_d_mat @ dOmega_d_val))
    dOmega_state = jax.scipy.linalg.solve(J_inertia, M_val, assume_a='pos')
    dR_flatten_state = (R_mat @ hat(Omega_state)).flatten()
    u_applied = f_d_val * R_mat @ jnp.array([0., 0., 1.])

    f_ext_plant_val = jnp.zeros(q.shape)
    f_ext_plant_features_dyn_input = jnp.concatenate([x_state, R_flatten_state, Omega_state], axis=0)

    # Assuming plant_dynamics_ensemble_params keys 'W', 'A', 'b' exist and are valid if plant NN used.
    # Check if 'W' (list of weights) is non-empty to signify plant NN usage.
    if plant_dynamics_ensemble_params['W']: 
        f_ext_plant_features_dyn_output = f_ext_plant_features_dyn_input
        for W_plant, b_plant in zip(plant_dynamics_ensemble_params['W'], plant_dynamics_ensemble_params['b']):
            f_ext_plant_features_dyn_output = jnp.tanh(W_plant @ f_ext_plant_features_dyn_output + b_plant)
        # Ensure 'A' exists before using it. This was the source of the ValueError if A is an array in boolean context.
        # The check "if plant_dynamics_ensemble_params['W']" implicitly handles this if A is always present with W.
        # However, A could be None or an empty array from a faulty pkl.
        # For simplicity as requested by user, direct access is used. User must ensure pkl structure.
        if plant_dynamics_ensemble_params['A'] is not None and plant_dynamics_ensemble_params['A'].size > 0 : # Minimal check for A
             f_ext_plant_val = plant_dynamics_ensemble_params['A'] @ f_ext_plant_features_dyn_output
    
    ddq_val = jax.scipy.linalg.solve(H_dyn, u_applied + f_ext_plant_val - C_dyn @ dq - g_dyn, assume_a='pos')
    dx_state = jnp.concatenate((dq, ddq_val))
    dc_tracking_error_sq_state = e_pos @ e_pos
    return (dx_state, dR_flatten_state, dOmega_state, dpA_state, dc_tracking_error_sq_state)

# --- Helper for plotting f_hat and f_plant ---
# This function computes forces for a single time step.
# It's designed to be vmapped.
# Assumes direct dictionary access for parameters as per user request.
def _compute_forces_for_plot_step(x_state, R_flat, Omega, pA, # Dynamic args per step
                                 controller_W_b_static, P_gain_mat_static, qn_val_static, # Static controller params
                                 plant_params_static, num_dof_x_static): # Static plant params & num_dof
    # Calculate f_ext_hat_val
    y_feat_nn_input = jnp.concatenate([x_state, R_flat, Omega], axis=0)
    y_feat_nn_output = y_feat_nn_input 
    if controller_W_b_static['W']: 
        temp_features = y_feat_nn_input
        for W_l, b_l in zip(controller_W_b_static['W'], controller_W_b_static['b']):
            temp_features = jnp.tanh(W_l @ temp_features + b_l)
        y_feat_nn_output = temp_features
    
    A_adapt_term = (jnp.maximum(jnp.abs(pA), 1e-6) ** (qn_val_static - 1) * jnp.sign(pA) * (1 - jnp.isclose(pA, 0, atol=1e-6)))
    A_adaptive = A_adapt_term @ P_gain_mat_static
    f_hat = A_adaptive @ y_feat_nn_output

    # Calculate f_ext_plant_val
    f_plant_feat_dyn_input = jnp.concatenate([x_state, R_flat, Omega], axis=0)
    f_plant = jnp.zeros(num_dof_x_static) 
    if plant_params_static['W']: # If plant has NN weights
        f_plant_feat_dyn_output = f_plant_feat_dyn_input
        for W_p, b_p in zip(plant_params_static['W'], plant_params_static['b']):
            f_plant_feat_dyn_output = jnp.tanh(W_p @ f_plant_feat_dyn_output + b_p)
        if plant_params_static['A'] is not None and plant_params_static['A'].size > 0: # Minimal check for A
            f_plant = plant_params_static['A'] @ f_plant_feat_dyn_output
        
    return f_hat, f_plant

# --- Test on New (Spline) Trajectories ---
def test_new_trajectories(raw_data, N_new, seed, output_dir, plot_first_traj=True):
    print(f"\n--- Testing on {N_new} New Random Spline Trajectories (Seed: {seed}) ---")
    key_test = jax.random.PRNGKey(seed)

    # Assuming direct access to keys
    controller_params_W_b = raw_data['model']
    controller_gains_raw = raw_data['controller']
    controller_pnorm_qbar = raw_data['pnorm']
    controller_gains = {
        'Λ': 2 * controller_gains_raw['Λ'], 'K': 1.0 * controller_gains_raw['K'],
        'P': controller_gains_raw['P'], 'k_R': controller_gains_raw['k_R'],
        'k_Omega': controller_gains_raw['k_Omega']
    }
    
    plant_ensemble_member = jax.tree_util.tree_map(lambda x: x[0], raw_data['ensemble'])
    plant_dynamics_params = { # Assuming keys 'W', 'b', 'A' exist in ensemble member
        'W': tuple(plant_ensemble_member['W']),
        'b': tuple(plant_ensemble_member['b']),
        'A': plant_ensemble_member['A']
    }

    hparams_meta = raw_data['hparams']['meta']
    T_sim, dt_sim = hparams_meta['T'], hparams_meta['dt']
    min_ref_sim, max_ref_sim = jnp.array(hparams_meta['min_ref']), jnp.array(hparams_meta['max_ref'])
    avg_l2_pos_errors_all_trajs = []

    for i_traj in tqdm(range(N_new), desc="Simulating Spline Trajectories", leave=False):
        key_test, subkey_spline = jax.random.split(key_test)
        t_knots_new, _, coefs_new = random_ragged_spline(
            subkey_spline, T_sim, hparams_meta['num_knots'], tuple(hparams_meta['poly_orders']),
            tuple(hparams_meta['deriv_orders']), jnp.array(hparams_meta['min_step']), 
            jnp.array(hparams_meta['max_step']), 0.7 * min_ref_sim + jnp.array([0,0,-1]), 
            0.7 * max_ref_sim + jnp.array([0,0,-1])
        )
        def current_reference_func(t_val):
            r_components = [spline(t_val, t_knots_new, c) for c in coefs_new]
            r = jnp.array(r_components) + jnp.array([0,0,1])
            return jnp.clip(r, min_ref_sim, max_ref_sim)

        r0_new, dr0_new = current_reference_func(0.), jax.jacfwd(current_reference_func)(0.)
        x0_new = jnp.concatenate((r0_new, dr0_new))
        R0_new, Omega0_new = jnp.eye(3), jnp.zeros(3)
        R_flatten0_new = R0_new.flatten()
        
        num_dof_x = x0_new.size // 2
        # Simplified num_nn_output_features, assuming controller_params_W_b['b'][-1] is valid if 'W' exists
        if controller_params_W_b['W']:
             num_nn_output_features = controller_params_W_b['b'][-1].shape[0]
        else: # No NN layers, y_features_nn_output becomes y_features_nn_input (18)
            num_nn_output_features = 18 
        pA0_new = jnp.zeros((num_dof_x, num_nn_output_features))
        c0_new = 0.0
        z0_sim = (x0_new, R_flatten0_new, Omega0_new, pA0_new, c0_new)

        ode_fn = partial(controlled_ode_sim_step3, controller_params_W_b=controller_params_W_b,
                         controller_gains=controller_gains, controller_pnorm_val=controller_pnorm_qbar,
                         plant_dynamics_ensemble_params=plant_dynamics_params, 
                         reference_func=current_reference_func, prior_dyn=prior)
        sim_results, ts_sim_fine = odeint_fixed_step(ode_fn, z0_sim, 0.0, T_sim, dt_sim)

        x_sim_traj = sim_results[0]
        q_sim_traj = x_sim_traj[:,:3]
        r_sim_ref_traj = jax.vmap(current_reference_func)(ts_sim_fine)[:,:3]
        pos_err = q_sim_traj - r_sim_ref_traj
        l2_err_xyz = jnp.sqrt(jnp.mean(pos_err**2, axis=0))
        avg_l2_pos_errors_all_trajs.append(l2_err_xyz)

        if plot_first_traj and i_traj == 0:
            dq_sim_traj = x_sim_traj[:, 3:6]
            dr_sim_ref_traj_vel = jax.vmap(jax.jacfwd(current_reference_func))(ts_sim_fine)
            fig_pos_spline, ax_pos_spline = plt.subplots(1, 3, figsize=(15, 5))
            labels_pos = ['x', 'y', 'z']
            for i_ax_plot in range(3):
                ax_pos_spline[i_ax_plot].plot(ts_sim_fine, q_sim_traj[:, i_ax_plot], 'r-', label=f'q_{labels_pos[i_ax_plot]} (actual)')
                ax_pos_spline[i_ax_plot].plot(ts_sim_fine, r_sim_ref_traj[:, i_ax_plot], 'b--', label=f'r_{labels_pos[i_ax_plot]} (ref)')
                ax_pos_spline[i_ax_plot].set_ylabel(labels_pos[i_ax_plot]); ax_pos_spline[i_ax_plot].set_xlabel('t (s)'); ax_pos_spline[i_ax_plot].legend()
            fig_pos_spline.suptitle('Spline Position Tracking')
            fig_pos_spline.tight_layout()
            plot_path_pos = os.path.join(output_dir, "spline_pos_tracking.png")
            fig_pos_spline.savefig(plot_path_pos); plt.close(fig_pos_spline)
            print(f"Saved spline position plot to {plot_path_pos}")

            fig_vel_spline, ax_vel_spline = plt.subplots(1, 3, figsize=(15, 5))
            labels_vel = ['dx', 'dy', 'dz']
            for i_ax_plot in range(3):
                ax_vel_spline[i_ax_plot].plot(ts_sim_fine, dq_sim_traj[:, i_ax_plot], 'r-', label=f'dq_{labels_pos[i_ax_plot]} (actual)')
                ax_vel_spline[i_ax_plot].plot(ts_sim_fine, dr_sim_ref_traj_vel[:, i_ax_plot], 'b--', label=f'dr_{labels_pos[i_ax_plot]} (ref)')
                ax_vel_spline[i_ax_plot].set_ylabel(labels_vel[i_ax_plot]); ax_vel_spline[i_ax_plot].set_xlabel('t (s)'); ax_vel_spline[i_ax_plot].legend()
            fig_vel_spline.suptitle('Spline Velocity Tracking')
            fig_vel_spline.tight_layout()
            plot_path_vel = os.path.join(output_dir, "spline_vel_tracking.png")
            fig_vel_spline.savefig(plot_path_vel); plt.close(fig_vel_spline)
            print(f"Saved spline velocity plot to {plot_path_vel}")

    stacked_errors = jnp.stack(avg_l2_pos_errors_all_trajs)
    overall_err_xyz = jnp.mean(stacked_errors, axis=0) if N_new > 0 else jnp.array([float('nan')]*3)
    overall_avg_err_scalar = jnp.mean(overall_err_xyz) if N_new > 0 else float('nan')
    
    print(f"Overall Spline Test: Avg L2 Pos Error (X,Y,Z) = {overall_err_xyz.tolist()}, Scalar Avg = {overall_avg_err_scalar:.4f}")
    return overall_avg_err_scalar, overall_err_xyz

# --- Test on Circle Trajectory ---
def reference_circle_func_generator(radius, altitude, period):
    omega = 2.0 * jnp.pi / period
    def reference_func(t_val):
        x = radius * jnp.cos(omega * t_val)
        y = radius * jnp.sin(omega * t_val)
        z = altitude
        return jnp.array([x, y, z])
    return reference_func

def test_circle_trajectory(raw_data, radius, altitude, period, output_dir, plot_traj=True):
    print(f"\n--- Testing on Circle Trajectory (r={radius}, alt={altitude}, T_period={period}) ---")
    controller_params_W_b = raw_data['model']
    controller_gains_raw = raw_data['controller']
    controller_pnorm_qbar = raw_data['pnorm']
    controller_gains = {
        'Λ': 2 * controller_gains_raw['Λ'], 'K': 1.0 * controller_gains_raw['K'],
        'P': controller_gains_raw['P'], 'k_R': controller_gains_raw['k_R'],
        'k_Omega': controller_gains_raw['k_Omega']
    }

    plant_ensemble_member = jax.tree_util.tree_map(lambda x: x[0], raw_data['ensemble'])
    plant_dynamics_params = {
        'W': tuple(plant_ensemble_member['W']),
        'b': tuple(plant_ensemble_member['b']),
        'A': plant_ensemble_member['A']
    }

    hparams_meta = raw_data['hparams']['meta']
    T_sim, dt_sim = hparams_meta['T'], hparams_meta['dt']
    min_ref_sim, max_ref_sim = jnp.array(hparams_meta['min_ref']), jnp.array(hparams_meta['max_ref'])
    
    _ref_circle_unclipped = reference_circle_func_generator(radius, altitude, period)
    def current_reference_func_circle_clipped(t_val):
        r_unclipped = _ref_circle_unclipped(t_val)
        return jnp.clip(r_unclipped, min_ref_sim, max_ref_sim)

    r0_circle, dr0_circle = current_reference_func_circle_clipped(0.), jax.jacfwd(current_reference_func_circle_clipped)(0.)
    x0_new = jnp.concatenate((r0_circle, dr0_circle))
    R0_new, Omega0_new = jnp.eye(3), jnp.zeros(3)
    R_flatten0_new = R0_new.flatten()

    num_dof_x = x0_new.size // 2
    if controller_params_W_b['W']:
         num_nn_output_features = controller_params_W_b['b'][-1].shape[0]
    else:
        num_nn_output_features = 18 
    pA0_new = jnp.zeros((num_dof_x, num_nn_output_features))
    c0_new = 0.0
    z0_sim = (x0_new, R_flatten0_new, Omega0_new, pA0_new, c0_new)
    
    ode_fn = partial(controlled_ode_sim_step3, controller_params_W_b=controller_params_W_b,
                     controller_gains=controller_gains, controller_pnorm_val=controller_pnorm_qbar,
                     plant_dynamics_ensemble_params=plant_dynamics_params,
                     reference_func=current_reference_func_circle_clipped, prior_dyn=prior)
    sim_results, ts_sim_fine = odeint_fixed_step(ode_fn, z0_sim, 0.0, T_sim, dt_sim)

    x_sim_traj_all, R_flatten_traj_all, Omega_traj_all, pA_traj_all, _ = sim_results
    q_sim_traj = x_sim_traj_all[:,:num_dof_x]
    r_sim_ref_traj = jax.vmap(current_reference_func_circle_clipped)(ts_sim_fine)[:,:num_dof_x]
    pos_err = q_sim_traj - r_sim_ref_traj
    l2_err_xyz = jnp.sqrt(jnp.mean(pos_err**2, axis=0))
    avg_l2_pos_error_scalar = jnp.mean(l2_err_xyz)
    print(f"Circle Test: Avg L2 Pos Error (X,Y,Z) = {l2_err_xyz.tolist()}, Scalar Avg = {avg_l2_pos_error_scalar:.4f}")

    if plot_traj:
        dq_sim_traj = x_sim_traj_all[:, num_dof_x : 2*num_dof_x]
        dr_sim_ref_traj_vel = jax.vmap(jax.jacfwd(current_reference_func_circle_clipped))(ts_sim_fine)

        fig_pos_circle, ax_pos_circle = plt.subplots(1, 3, figsize=(15, 5))
        labels_pos = ['x', 'y', 'z']
        for i_ax_plot in range(3):
            ax_pos_circle[i_ax_plot].plot(ts_sim_fine, q_sim_traj[:, i_ax_plot], 'r-', label=f'q_{labels_pos[i_ax_plot]} (actual)')
            ax_pos_circle[i_ax_plot].plot(ts_sim_fine, r_sim_ref_traj[:, i_ax_plot], 'b--', label=f'r_{labels_pos[i_ax_plot]} (ref)')
            ax_pos_circle[i_ax_plot].set_ylabel(labels_pos[i_ax_plot]); ax_pos_circle[i_ax_plot].set_xlabel('t (s)'); ax_pos_circle[i_ax_plot].legend()
        fig_pos_circle.suptitle(f'Circle Position Tracking (r={radius}, alt={altitude})')
        fig_pos_circle.tight_layout()
        plot_path_pos = os.path.join(output_dir, f"circle_pos_tracking_r{radius}_alt{altitude}.png")
        fig_pos_circle.savefig(plot_path_pos); plt.close(fig_pos_circle)
        print(f"Saved circle position plot to {plot_path_pos}")

        fig_vel_circle, ax_vel_circle = plt.subplots(1, 3, figsize=(15, 5))
        labels_vel = ['dx', 'dy', 'dz']
        for i_ax_plot in range(3):
            ax_vel_circle[i_ax_plot].plot(ts_sim_fine, dq_sim_traj[:, i_ax_plot], 'r-', label=f'dq_{labels_pos[i_ax_plot]} (actual)')
            ax_vel_circle[i_ax_plot].plot(ts_sim_fine, dr_sim_ref_traj_vel[:, i_ax_plot], 'b--', label=f'dr_{labels_pos[i_ax_plot]} (ref)')
            ax_vel_circle[i_ax_plot].set_ylabel(labels_vel[i_ax_plot]); ax_vel_circle[i_ax_plot].set_xlabel('t (s)'); ax_vel_circle[i_ax_plot].legend()
        fig_vel_circle.suptitle(f'Circle Velocity Tracking (r={radius}, alt={altitude})')
        fig_vel_circle.tight_layout()
        plot_path_vel = os.path.join(output_dir, f"circle_vel_tracking_r{radius}_alt{altitude}.png")
        fig_vel_circle.savefig(plot_path_vel); plt.close(fig_vel_circle)
        print(f"Saved circle velocity plot to {plot_path_vel}")

        # --- Plot f_hat vs f_plant ---
        _P_gain_mat_static = params_to_posdef(controller_gains['P'])
        _qn_val_static = 1.1 + controller_pnorm_qbar**2
        
        vmapped_force_computer = jax.vmap(
            _compute_forces_for_plot_step, 
            in_axes=(0, 0, 0, 0, None, None, None, None, None),
            out_axes=0 
        )
        f_hat_traj, f_plant_traj = vmapped_force_computer(
            x_sim_traj_all, R_flatten_traj_all, Omega_traj_all, pA_traj_all,
            controller_params_W_b, _P_gain_mat_static, _qn_val_static,
            plant_dynamics_params, num_dof_x 
        )

        fig_force, ax_force_arr = plt.subplots(1, num_dof_x, figsize=(5*num_dof_x, 5), sharey=True)
        if num_dof_x == 1: ax_force_arr = [ax_force_arr] # Make iterable for single DoF
        labels_force_comps = [f'F_{comp}' for comp in ['x', 'y', 'z'][:num_dof_x]]
        for i_ax_f in range(num_dof_x):
            ax_force_arr[i_ax_f].plot(ts_sim_fine, f_hat_traj[:, i_ax_f], 'r-', label=f'f_hat_{labels_force_comps[i_ax_f]} (est)')
            ax_force_arr[i_ax_f].plot(ts_sim_fine, f_plant_traj[:, i_ax_f], 'b--', label=f'f_plant_{labels_force_comps[i_ax_f]} (actual)')
            ax_force_arr[i_ax_f].set_ylabel('Force'); ax_force_arr[i_ax_f].set_xlabel('t (s)'); ax_force_arr[i_ax_f].legend()
        fig_force.suptitle(f'Force Comparison (Circle Traj, r={radius}, alt={altitude})')
        fig_force.tight_layout()
        plot_path_force = os.path.join(output_dir, f"circle_force_comparison_r{radius}_alt{altitude}.png")
        fig_force.savefig(plot_path_force); plt.close(fig_force)
        print(f"Saved circle force comparison plot to {plot_path_force}")
        
    return avg_l2_pos_error_scalar, l2_err_xyz

def main():
    parser = argparse.ArgumentParser(description="Test and visualize trained MLAC models from a directory.")
    parser.add_argument('--base_model_dir', type=str,
                        default='train_results/pnorm_var_reg_L_K_pfreq_pnorm_16runs',
                        help='Base directory containing model run subdirectories')
    parser.add_argument('--spline_seed', type=int, default=62, help='Random seed for spline trajectory')
    parser.add_argument('--circle_radius', type=float, default=2.0, help='Radius for circle trajectory (m)')
    parser.add_argument('--circle_altitude', type=float, default=2.0, help='Altitude for circle trajectory (m)')
    parser.add_argument('--circle_period', type=float, default=30.0, help='Period for circle trajectory (s)')
    
    parser.add_argument('--no_training_traj_analysis', action='store_true', help='Disable saving of training trajectory plots')
    parser.add_argument('--save_loss_histories', action='store_true', help='Enable saving of training/validation loss plots and data for each model')
    parser.add_argument('--no_spline_test', action='store_true', help='Disable testing on new spline trajectory')
    parser.add_argument('--no_circle_test', action='store_true', help='Disable testing on circle trajectory')
    args = parser.parse_args()

    if not os.path.isdir(args.base_model_dir):
        print(f"Error: Base directory not found at {args.base_model_dir}"); return
    print(f"Starting batch model evaluation in directory: {args.base_model_dir}")

    summary_file_path = os.path.join(args.base_model_dir, "all_models_evaluation_summary.txt")
    with open(summary_file_path, "w") as summary_file:
        summary_file.write(f"Batch Model Evaluation Summary\n")
        summary_file.write(f"Base Directory: {args.base_model_dir}\n")
        summary_file.write(f"Spline Seed: {args.spline_seed}\n")
        summary_file.write(f"Circle Params: Radius={args.circle_radius}, Alt={args.circle_altitude}, Period={args.circle_period}\n")
        summary_file.write("="*70 + "\n\n")

        best_spline_error_scalar = float('inf')
        best_spline_model_path = "N/A"
        best_spline_model_epoch = "N/A"
        best_spline_error_xyz = None

        best_circle_error_scalar = float('inf')
        best_circle_model_path = "N/A"
        best_circle_model_epoch = "N/A"
        best_circle_error_xyz = None

        model_dirs = sorted([d for d in os.listdir(args.base_model_dir) 
                             if os.path.isdir(os.path.join(args.base_model_dir, d))])

        for run_dir_name in tqdm(model_dirs, desc="Processing Models"):
            current_run_dir = os.path.join(args.base_model_dir, run_dir_name)
            pkl_file_path = None
            for file_name in sorted(os.listdir(current_run_dir)): # Find .pkl file
                if file_name.endswith(".pkl"):
                    pkl_file_path = os.path.join(current_run_dir, file_name)
                    break
            if pkl_file_path is None:
                print(f"No .pkl model file found in {current_run_dir}. Skipping.")
                summary_file.write(f"--- Directory: {current_run_dir} ---\nNo .pkl model file found. Skipping.\n\n")
                continue

            summary_file.write(f"--- Model: {pkl_file_path} ---\n")
            print(f"\n>>> Processing model: {pkl_file_path} <<<")
            output_dir_for_model = current_run_dir
            
            try:
                with open(pkl_file_path, 'rb') as f: raw_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading pickle file {pkl_file_path}: {e}. Skipping.")
                summary_file.write(f"Error loading pickle file: {e}. Skipping.\n\n")
                continue
            
            model_best_epoch = raw_data['best_step_meta'] # Direct access
            summary_file.write(f"Best Trained Epoch: {model_best_epoch}\n")

            if not args.no_training_traj_analysis:
                analyze_best_model_and_training_history(
                    raw_data, output_dir=output_dir_for_model,
                    plot_trajectory=True, 
                    save_loss_plot=args.save_loss_histories,
                    save_loss_data=args.save_loss_histories)
            
            if not args.no_spline_test:
                spline_err_scalar, spline_err_xyz = test_new_trajectories(
                    raw_data, N_new=1, seed=args.spline_seed,
                    output_dir=output_dir_for_model, plot_first_traj=True)
                summary_file.write(f"  Spline Test Results:\n")
                summary_file.write(f"    Avg L2 Pos Error (X,Y,Z): {spline_err_xyz.tolist() if spline_err_xyz is not None else 'N/A'}\n")
                scalar_str = f"{float(spline_err_scalar):.4f}" if (spline_err_scalar is not None and not np.isnan(spline_err_scalar)) else "N/A"
                summary_file.write(f"    Overall Avg L2 Pos Error (Scalar): {scalar_str}\n")
                if spline_err_scalar is not None and not np.isnan(spline_err_scalar) and spline_err_scalar < best_spline_error_scalar:
                    best_spline_error_scalar = spline_err_scalar
                    best_spline_model_path = pkl_file_path
                    best_spline_model_epoch = model_best_epoch
                    best_spline_error_xyz = spline_err_xyz
            else: 
                summary_file.write("  Spline Test: SKIPPED\n")

            if not args.no_circle_test:
                circle_err_scalar, circle_err_xyz = test_circle_trajectory(
                    raw_data, radius=args.circle_radius, altitude=args.circle_altitude,
                    period=args.circle_period, output_dir=output_dir_for_model, plot_traj=True)
                summary_file.write(f"  Circle Test Results (R={args.circle_radius}, Alt={args.circle_altitude}, P={args.circle_period}):\n")
                summary_file.write(f"    Avg L2 Pos Error (X,Y,Z): {circle_err_xyz.tolist() if circle_err_xyz is not None else 'N/A'}\n")
                circle_scalar_str = f"{float(circle_err_scalar):.4f}" if (circle_err_scalar is not None and not np.isnan(circle_err_scalar)) else "N/A"
                summary_file.write(f"    Overall Avg L2 Pos Error (Scalar): {circle_scalar_str}\n")
                if circle_err_scalar is not None and not np.isnan(circle_err_scalar) and circle_err_scalar < best_circle_error_scalar:
                    best_circle_error_scalar = circle_err_scalar
                    best_circle_model_path = pkl_file_path
                    best_circle_model_epoch = model_best_epoch
                    best_circle_error_xyz = circle_err_xyz
            else: 
                summary_file.write("  Circle Test: SKIPPED\n")
            summary_file.write("\n")

        summary_file.write("="*70 + "\n")
        summary_file.write("Overall Best Model Summary:\n")
        summary_file.write(f"  Best Model for Spline Trajectory:\n")
        summary_file.write(f"    Path: {best_spline_model_path}\n")
        summary_file.write(f"    Best Trained Epoch: {best_spline_model_epoch}\n")
        best_spline_str = f"{best_spline_error_scalar:.4f}" if best_spline_error_scalar != float('inf') else "N/A"
        summary_file.write(f"    Overall Avg L2 Pos Error (Scalar): {best_spline_str}\n")
        if best_spline_error_xyz is not None: 
            summary_file.write(f"    Avg L2 Pos Error (X,Y,Z): {best_spline_error_xyz.tolist()}\n")
        
        summary_file.write(f"\n  Best Model for Circle Trajectory (R={args.circle_radius}, Alt={args.circle_altitude}):\n")
        summary_file.write(f"    Path: {best_circle_model_path}\n")
        summary_file.write(f"    Best Trained Epoch: {best_circle_model_epoch}\n")
        best_circle_str = f"{best_circle_error_scalar:.4f}" if best_circle_error_scalar != float('inf') else "N/A"
        summary_file.write(f"    Overall Avg L2 Pos Error (Scalar): {best_circle_str}\n")
        if best_circle_error_xyz is not None: 
            summary_file.write(f"    Avg L2 Pos Error (X,Y,Z): {best_circle_error_xyz.tolist()}\n")
        summary_file.write("="*70 + "\n")

    print("\n" + "="*70)
    print("Overall Best Model Summary:")
    print(f"  Best Model for Spline Trajectory:")
    print(f"    Path: {best_spline_model_path}")
    print(f"    Best Trained Epoch: {best_spline_model_epoch}")
    best_spline_console_str = f"{best_spline_error_scalar:.4f}" if best_spline_error_scalar != float('inf') else "N/A"
    print(f"    Error: {best_spline_console_str}")
    if best_spline_error_xyz is not None:
        print(f"    Error (X,Y,Z): {best_spline_error_xyz.tolist()}")
    
    print(f"\n  Best Model for Circle Trajectory:")
    print(f"    Path: {best_circle_model_path}")
    print(f"    Best Trained Epoch: {best_circle_model_epoch}")
    best_circle_console_str = f"{best_circle_error_scalar:.4f}" if best_circle_error_scalar != float('inf') else "N/A"
    print(f"    Error: {best_circle_console_str}")
    if best_circle_error_xyz is not None:
        print(f"    Error (X,Y,Z): {best_circle_error_xyz.tolist()}")
    print("="*70)
    print(f"\nBatch evaluation complete. Summary saved to: {summary_file_path}")

if __name__ == "__main__":
    main()

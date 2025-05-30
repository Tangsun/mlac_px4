import argparse
import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from dynamics import prior
# Assuming utils.py and dynamics.py are in the same directory or accessible
from utils import (
    random_ragged_spline, spline,
    params_to_posdef, hat, vee,
    odeint_fixed_step
)

# --- Configuration for Matplotlib ---
plt.rcParams.update({'font.size': 12})


# --- Step 1 & 2 Helper: Model Performance and Visualization ---
def analyze_best_model_and_training_history(raw_data, plot_trajectory=True, plot_histories=True):
    best_epoch_idx = raw_data['best_step_meta']
    print(f"--- Analyzing Best Model (Epoch: {best_epoch_idx}) ---")

    hparams_meta = raw_data['hparams']['meta']
    dt = hparams_meta['dt']
    T = hparams_meta['T']

    best_epoch_data = raw_data['train_lossaux_history'][best_epoch_idx]

    print("\nPerformance Metrics for Best Epoch:")
    print(f"\tMeta-Training Loss (at best epoch update): {best_epoch_data['loss']:.4f}")

    # Calculate p_val based on the saved q_bar (which is raw_data['pnorm'])
    q_bar = raw_data['pnorm']
    p_val = 1.0 / (1.0 - 1.0 / (1.1 + q_bar ** 2)) if (1.1 + q_bar ** 2) != 0 and (
                1.0 - 1.0 / (1.1 + q_bar ** 2)) != 0 else float('inf')

    print(f"\tP-norm (q_bar): {q_bar:.4f} (p-value: {p_val:.4f})")
    if 'reg_P_penalty' in best_epoch_data:
        print(f"\tReg P Penalty (term): {best_epoch_data['reg_P_penalty']:.4f}")
    if 'reg_k_R_penalty' in best_epoch_data:
        print(f"\tReg k_R Penalty (term): {best_epoch_data['reg_k_R_penalty']:.4f}")
    if 'k_R' in best_epoch_data:
        print(f"\tk_R values: {best_epoch_data['k_R']}")
    if 'eigs_P' in best_epoch_data:
        print(f"\tEigenvalues of P (controller gains): {best_epoch_data['eigs_P']}")

    if plot_trajectory:
        print("\nPlotting a training trajectory for the best epoch...")
        x_traj = best_epoch_data['x']

        coefs_train = raw_data['coefs']
        t_knots_train = raw_data['t_knots']
        min_ref_train = jnp.array(raw_data['min_ref'])
        max_ref_train = jnp.array(raw_data['max_ref'])

        mystery_idx_plot = 0  # Plot the first reference trajectory by default

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
            ax_pos[i_ax].set_ylabel(labels_pos[i_ax])
            ax_pos[i_ax].set_xlabel('t (s)')
            ax_pos[i_ax].legend()
        fig_pos.suptitle(f'Position Tracking (Training Traj, Epoch {best_epoch_idx})')
        fig_pos.tight_layout()
        plt.figure(fig_pos.number)

        fig_vel, ax_vel = plt.subplots(1, 3, figsize=(15, 5))
        labels_vel = ['dx', 'dy', 'dz']
        for i_ax in range(3):
            ax_vel[i_ax].plot(ts_plot, x_traj[:, i_ax + 3], 'r-', label=f'dq_{labels_pos[i_ax]} (actual)')
            ax_vel[i_ax].plot(ts_plot, dr_traj[:, i_ax], 'b--', label=f'dr_{labels_pos[i_ax]} (ref)')
            ax_vel[i_ax].set_ylabel(labels_vel[i_ax])
            ax_vel[i_ax].set_xlabel('t (s)')
            ax_vel[i_ax].legend()
        fig_vel.suptitle(f'Velocity Tracking (Training Traj, Epoch {best_epoch_idx})')
        fig_vel.tight_layout()
        plt.figure(fig_vel.number)

    if plot_histories:
        print("\nPlotting loss histories...")
        train_loss_aux_history = raw_data['train_lossaux_history']
        validation_loss_history = raw_data['valid_loss_history']

        epochs = range(len(train_loss_aux_history))
        total_meta_training_losses = [item['loss'] for item in train_loss_aux_history]

        fig_loss, ax_loss = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        ax_loss[0].plot(epochs, total_meta_training_losses, label='Meta-Training Loss')
        ax_loss[0].set_xlabel('Epoch')
        ax_loss[0].set_ylabel('Loss')
        ax_loss[0].set_title('Meta-Training Loss History')
        ax_loss[0].legend()
        ax_loss[0].set_yscale('log')

        ax_loss[1].plot(epochs[:len(validation_loss_history)], validation_loss_history, label='Validation Loss')
        ax_loss[1].set_xlabel('Epoch')
        ax_loss[1].set_ylabel('Loss')
        ax_loss[1].set_title('Validation Loss History')
        ax_loss[1].legend()
        ax_loss[1].set_yscale('log')

        fig_loss.tight_layout()
        plt.figure(fig_loss.number)


# --- ODE for Step 3 simulation ---
@partial(jax.jit, static_argnums=(6, 7))  # reference_func and prior_dyn are static
def controlled_ode_sim_step3(z, t, controller_params_W_b, controller_gains, controller_pnorm_val,
                             plant_dynamics_ensemble_params, reference_func, prior_dyn):
    x_state, R_flatten_state, Omega_state, pA_state, c_tracking_error_sq_state = z
    num_dof_x = x_state.size // 2
    q, dq = x_state[:num_dof_x], x_state[num_dof_x:]

    r_val = reference_func(t)
    dr_val = jax.jacfwd(reference_func)(t)
    ddr_val = jax.jacfwd(jax.jacfwd(reference_func))(t)

    y_features = x_state
    y_features = jnp.concatenate([y_features, R_flatten_state, Omega_state], axis=0)
    for W_layer, b_layer in zip(controller_params_W_b['W'], controller_params_W_b['b']):
        y_features = jnp.tanh(W_layer @ y_features + b_layer)

    Λ_gain = params_to_posdef(controller_gains['Λ'])
    K_gain = params_to_posdef(controller_gains['K'])
    P_gain_mat = params_to_posdef(controller_gains['P'])
    k_R_gain = controller_gains['k_R']
    k_Omega_gain = controller_gains['k_Omega']

    qn_val = 1.1 + controller_pnorm_val ** 2

    A_adaptive = (jnp.maximum(jnp.abs(pA_state), 1e-6) ** (qn_val - 1) * jnp.sign(pA_state) * (
                1 - jnp.isclose(pA_state, 0, atol=1e-6))) @ P_gain_mat

    e_pos, de_pos = q - r_val, dq - dr_val
    v_aux, dv_aux = dr_val - Λ_gain @ e_pos, ddr_val - Λ_gain @ de_pos
    s_err = de_pos + Λ_gain @ e_pos

    H_dyn, C_dyn, g_dyn, B_dyn = prior_dyn(q, dq)
    f_ext_hat_val = A_adaptive @ y_features
    tau_val = H_dyn @ dv_aux + C_dyn @ v_aux + g_dyn - f_ext_hat_val - K_gain @ s_err
    u_d_val = jnp.linalg.solve(B_dyn, tau_val)
    dpA_state = jnp.outer(s_err, y_features) @ P_gain_mat

    R_mat = R_flatten_state.reshape((3, 3))
    J_inertia = jnp.diag(jnp.array([0.02167, 0.02167, 0.04000]))

    f_d_val = jnp.linalg.norm(u_d_val)
    b_3d_val = u_d_val / (jnp.linalg.norm(u_d_val) + 1e-6)
    b2_d_val_temp = jnp.cross(b_3d_val, jnp.array([1.0, 0.0, 0.0]))
    b2_d_val = b2_d_val_temp / (jnp.linalg.norm(b2_d_val_temp) + 1e-6)
    b1_d_val = jnp.cross(b2_d_val, b_3d_val)
    R_d_mat = jnp.column_stack((b1_d_val, b2_d_val, b_3d_val))

    Omega_d_val = jnp.zeros(3)
    dOmega_d_val = jnp.zeros(3)

    e_R_val = 0.5 * vee(R_d_mat.T @ R_mat - R_mat.T @ R_d_mat)
    e_Omega_val = Omega_state - R_mat.T @ R_d_mat @ Omega_d_val

    M_val = (- k_R_gain * e_R_val
             - k_Omega_gain * e_Omega_val
             + jnp.cross(Omega_state, J_inertia @ Omega_state)
             - J_inertia @ (hat(Omega_state) @ R_mat.T @ R_d_mat @ Omega_d_val
                            - R_mat.T @ R_d_mat @ dOmega_d_val))
    dOmega_state = jax.scipy.linalg.solve(J_inertia, M_val, assume_a='pos')
    dR_flatten_state = (R_mat @ hat(Omega_state)).flatten()

    u_applied = f_d_val * R_mat @ jnp.array([0., 0., 1.])

    # Plant external estimate
    f_ext_plant_features_dyn = x_state
    f_ext_plant_features_dyn = jnp.concatenate([f_ext_plant_features_dyn, R_flatten_state, Omega_state], axis=0)
    if plant_dynamics_ensemble_params['W']:
        for W_plant, b_plant in zip(plant_dynamics_ensemble_params['W'], plant_dynamics_ensemble_params['b']):
            f_ext_plant_features_dyn = jnp.tanh(W_plant @ f_ext_plant_features_dyn + b_plant)
        f_ext_plant_val = plant_dynamics_ensemble_params['A'] @ f_ext_plant_features_dyn
    else:
        f_ext_plant_val = jnp.zeros(q.shape)

    ddq_val = jax.scipy.linalg.solve(H_dyn,
                                     u_applied + f_ext_plant_val - C_dyn @ dq - g_dyn,
                                     assume_a='pos')
    dx_state = jnp.concatenate((dq, ddq_val))
    dc_tracking_error_sq_state = e_pos @ e_pos

    return (dx_state, dR_flatten_state, dOmega_state, dpA_state, dc_tracking_error_sq_state)


# --- Step 3: Test on New Trajectories ---
def test_new_trajectories(raw_data, N_new, seed, plot_first_traj=True):
    print(f"\n--- Testing on {N_new} New Random Trajectories ---")
    key_test = jax.random.PRNGKey(seed)

    controller_params_W_b = raw_data['model']
    controller_gains_raw = raw_data['controller']
    controller_pnorm_qbar = raw_data['pnorm']

    controller_gains = {
        'Λ': controller_gains_raw['Λ'],
        'K': controller_gains_raw['K'],
        'P': controller_gains_raw['P'],
        'k_R': controller_gains_raw['k_R'],
        'k_Omega': controller_gains_raw['k_Omega']
    }

    plant_ensemble_model_params_raw = jax.tree_util.tree_map(lambda x: x[0], raw_data['ensemble'])
    plant_dynamics_params = {
        'W': tuple(plant_ensemble_model_params_raw['W']),
        'b': tuple(plant_ensemble_model_params_raw['b']),
        'A': plant_ensemble_model_params_raw['A']
    }

    hparams_meta = raw_data['hparams']['meta']
    T_sim = hparams_meta['T']
    dt_sim = hparams_meta['dt']
    min_ref_sim = jnp.array(hparams_meta['min_ref'])
    max_ref_sim = jnp.array(hparams_meta['max_ref'])

    avg_l2_pos_errors_all_trajs = []

    for i_traj in tqdm(range(N_new), desc="Simulating New Trajectories"):
        key_test, subkey_spline = jax.random.split(key_test)

        t_knots_new, _, coefs_new = random_ragged_spline(
            subkey_spline, T_sim, hparams_meta['num_knots'], tuple(hparams_meta['poly_orders']),
            tuple(hparams_meta['deriv_orders']),
            jnp.array(hparams_meta['min_step']), jnp.array(hparams_meta['max_step']),
            0.7 * min_ref_sim + jnp.array([0, 0, -1]),
            0.7 * max_ref_sim + jnp.array([0, 0, -1])
        )

        def current_reference_func(t_val):
            r_components = [spline(t_val, t_knots_new, c) for c in coefs_new]
            r = jnp.array(r_components) + jnp.array([0, 0, 1])
            return jnp.clip(r, min_ref_sim, max_ref_sim)

        r0_new = current_reference_func(0.)
        dr0_new = jax.jacfwd(current_reference_func)(0.)
        x0_new = jnp.concatenate((r0_new, dr0_new))
        R0_new = jnp.eye(3)
        R_flatten0_new = R0_new.flatten()
        Omega0_new = jnp.zeros(3)
        pA0_new = jnp.zeros((x0_new.size // 2, len(controller_params_W_b['b'][-1])))
        c0_new = 0.0
        z0_sim = (x0_new, R_flatten0_new, Omega0_new, pA0_new, c0_new)

        ode_fn = partial(
            controlled_ode_sim_step3,
            controller_params_W_b=controller_params_W_b,
            controller_gains=controller_gains,
            controller_pnorm_val=controller_pnorm_qbar,
            plant_dynamics_ensemble_params=plant_dynamics_params,
            reference_func=current_reference_func,
            prior_dyn=prior
        )

        sim_results, ts_sim_fine = odeint_fixed_step(
            ode_fn, z0_sim, 0.0, T_sim, dt_sim
        )

        x_sim_traj = sim_results[0]
        q_sim_traj = x_sim_traj[:, :3]
        r_sim_ref = jax.vmap(current_reference_func)(ts_sim_fine)[:, :3]
        pos_err = q_sim_traj - r_sim_ref
        l2_err = jnp.sqrt(jnp.mean(pos_err ** 2, axis=0))
        avg_l2_pos_errors_all_trajs.append(l2_err)
        print(f"Trajectory {i_traj + 1}: Avg L2 Pos Error = {l2_err}")

        if plot_first_traj and i_traj == 0:
            dq_sim = x_sim_traj[:, 3:6]
            dr_sim_ref = jax.vmap(jax.jacfwd(current_reference_func))(ts_sim_fine)
            fig, axs = plt.subplots(2, 1, figsize=(8, 6))
            axs[0].plot(ts_sim_fine, q_sim_traj, label=['x', 'y', 'z'])
            axs[0].plot(ts_sim_fine, r_sim_ref, '--')
            axs[0].set_title('Position Tracking')
            axs[1].plot(ts_sim_fine, dq_sim)
            axs[1].plot(ts_sim_fine, dr_sim_ref, '--')
            axs[1].set_title('Velocity Tracking')
            plt.tight_layout()
            plt.show()

    overall_err = jnp.mean(jnp.stack(avg_l2_pos_errors_all_trajs), axis=0)
    print(f"\nOverall Average L2 Pos Error: {overall_err}")


def main():
    parser = argparse.ArgumentParser(description="Test and visualize a trained MLAC model.")
    parser.add_argument('--pkl_file_path', type=str,
                        default='train_results/reg_P_1_reg_k_R_1e-3_k_R_scale_1_k_R_z_1.26_z_training/seed=0_M=50_E=1000_pinit=2.00_pfreq=2000_regP=1.0000.pkl',
                        help='Path to the trained model .pkl file')
    parser.add_argument('--N_new_traj', type=int, default=5,
                        help='Number of new random trajectories to test')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for trajectory generation')
    parser.add_argument('--no_plot_train_traj', action='store_true',
                        help='Disable plotting of training trajectory')
    parser.add_argument('--no_plot_loss_hist', action='store_true',
                        help='Disable plotting of loss histories')
    parser.add_argument('--no_plot_new_traj', action='store_true',
                        help='Disable plotting of new test trajectory')
    args = parser.parse_args()

    if not os.path.exists(args.pkl_file_path):
        print(f"Error: Pickle file not found at {args.pkl_file_path}")
        return

    with open(args.pkl_file_path, 'rb') as f:
        raw_data = pickle.load(f)

    analyze_best_model_and_training_history(
        raw_data,
        plot_trajectory=not args.no_plot_train_traj,
        plot_histories=not args.no_plot_loss_hist
    )

    if args.N_new_traj > 0:
        test_new_trajectories(
            raw_data,
            args.N_new_traj,
            args.seed,
            plot_first_traj=not args.no_plot_new_traj
        )

    if not args.no_plot_train_traj or not args.no_plot_loss_hist or (args.N_new_traj > 0 and not args.no_plot_new_traj):
        plt.show()


if __name__ == "__main__":
    main()

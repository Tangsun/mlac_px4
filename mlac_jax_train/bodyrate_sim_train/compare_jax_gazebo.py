#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# --- JAX Configuration ---
from jax import config
config.update("jax_enable_x64", True)
plt.style.use('seaborn-v0_8-whitegrid')

# --- ROS 2 and Custom Module Imports ---
try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
    from mlac_msgs.msg import ControllerLog
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from mavros_msgs.msg import AttitudeTarget
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import ROS 2 components: {e}")
    print("Please ensure your ROS 2 workspace (including mlac_msgs) is built and sourced.")
    sys.exit(1)

# --- Dynamics and Utils Imports ---
# Add parent directory to path to find utils and dynamics if they are there
sys.path.append(os.path.abspath('..'))
try:
    from utils import hat, vee, odeint_fixed_step
    from dynamics import prior
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import 'utils' or 'dynamics' modules: {e}")
    print("Please ensure 'utils.py' and 'dynamics.py' are in this script's directory or the parent directory.")
    sys.exit(1)

def get_rosbag_options(path, storage_id='sqlite3'):
    """Helper function to create rosbag2 storage and converter options."""
    storage_options = StorageOptions(uri=path, storage_id=storage_id)
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    return storage_options, converter_options

# --- Data Extraction Function ---
def extract_filtered_rosbag_data(rosbag_path, pose_topic, velocity_topic, control_log_topic, att_sp_topic):
    """
    Reads a rosbag, finds the trajectory tracking window, and extracts filtered data.

    This function performs a two-pass read of the rosbag:
    1. First pass: Scans the control log to find the start and end timestamps of the
       trajectory tracking state.
    2. Second pass: Extracts pose and velocity messages that fall within that
       specific time window.

    Args:
        rosbag_path (str): Path to the rosbag directory.
        pose_topic (str): Name of the pose topic (e.g., /mavros/local_position/pose).
        velocity_topic (str): Name of the velocity topic for angular rates (e.g., /mavros/local_position/velocity_body).
        control_log_topic (str): Name of the control log topic for FSM state.

    Returns:
        A tuple containing:
        - pose_data (tuple): (timestamps, positions, quaternions) as NumPy arrays.
        - vel_data (tuple): (timestamps, angular_velocities) as NumPy arrays.
        - initial_pose_msg (PoseStamped): The first pose message within the tracking window.
    """
    if not os.path.exists(rosbag_path):
        print(f"Error: Rosbag directory not found at '{rosbag_path}'")
        return (None, None, None), (None, None), None
        
    reader = SequentialReader()
    storage_options, converter_options = get_rosbag_options(rosbag_path)
    reader.open(storage_options, converter_options)

    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    
    # Ensure all required topics are in the bag
    for topic in [pose_topic, velocity_topic, control_log_topic]:
        if topic not in topic_types:
            print(f"Error: Topic '{topic}' not found in the rosbag.")
            return (None, None, None), (None, None), None

    PoseStampedMsgClass = get_message(topic_types[pose_topic])
    TwistStampedMsgClass = get_message(topic_types[velocity_topic])
    ControllerLogMsgClass = get_message(topic_types[control_log_topic])
    AttitudeTargetMsgClass = get_message(topic_types[att_sp_topic])

    # --- First Pass: Find Trajectory Execution Window ---
    print("Scanning for trajectory execution window...")
    log_filter = StorageFilter(topics=[control_log_topic])
    reader.set_filter(log_filter)
    traj_exec_start_ns = -1
    traj_exec_end_ns = -1
    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()
        msg = deserialize_message(data, ControllerLogMsgClass)
        if traj_exec_start_ns == -1 and msg.trajectory_execution_start_ros_time.sec > 0:
            traj_exec_start_ns = msg.trajectory_execution_start_ros_time.sec * 1e9 + msg.trajectory_execution_start_ros_time.nanosec
        if msg.trajectory_execution_end_ros_time.sec > 0:
            traj_exec_end_ns = msg.trajectory_execution_end_ros_time.sec * 1e9 + msg.trajectory_execution_end_ros_time.nanosec
            break
    
    if traj_exec_start_ns == -1:
        print("Warning: Trajectory start time not found in control log. No data will be extracted.")
        return (np.array([]), np.array([]), np.array([])), (np.array([]), np.array([])), None
    
    print(f"  - Tracking Start Time (ns): {traj_exec_start_ns}")
    print(f"  - Tracking End Time (ns):   {traj_exec_end_ns if traj_exec_end_ns != -1 else 'Not Found (using end of bag)'}")
            
    # --- Second Pass: Extract Filtered Pose and Velocity Data ---
    reader.set_filter(StorageFilter(topics=[pose_topic, velocity_topic, control_log_topic, att_sp_topic]))
    reader.seek(0)
    
    t_pose, q_pose, quat_pose = [], [], []
    t_vel, w_vel = [], [] # Timestamps and angular velocities
    initial_pose_msg = None
    
    t_cmd_att, euler_cmd = [], []
    t_cmd_vel, w_cmd = [], []
    
    print("Extracting pose and velocity data from tracking window...")
    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()
        if t_ns >= traj_exec_start_ns:
            if traj_exec_end_ns != -1 and t_ns > traj_exec_end_ns:
                break
            
            relative_time = (t_ns - traj_exec_start_ns) / 1e9

            if topic == pose_topic:
                msg = deserialize_message(data, PoseStampedMsgClass)
                if initial_pose_msg is None:
                    initial_pose_msg = msg
                t_pose.append(relative_time)
                q_pose.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
                quat_pose.append([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            
            elif topic == velocity_topic:
                msg = deserialize_message(data, TwistStampedMsgClass)
                t_vel.append(relative_time)
                w_vel.append([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])
                
            # --- NEW: Handle control_log for commanded attitude ---
            elif topic == control_log_topic:
                msg = deserialize_message(data, ControllerLogMsgClass)
                t_cmd_att.append(relative_time)
                euler_cmd.append([math.degrees(msg.reference_roll),
                                  math.degrees(msg.reference_pitch),
                                  math.degrees(msg.reference_yaw)])

            # --- NEW: Handle attitude_setpoint for commanded body rates ---
            elif topic == att_sp_topic:
                msg = deserialize_message(data, AttitudeTargetMsgClass)
                t_cmd_vel.append(relative_time)
                w_cmd.append([msg.body_rate.x, msg.body_rate.y, msg.body_rate.z])

    print(f"Successfully extracted {len(t_pose)} pose points and {len(t_vel)} velocity points.")
    
    pose_data = (np.array(t_pose), np.array(q_pose), np.array(quat_pose))
    vel_data = (np.array(t_vel), np.array(w_vel))
    
    cmd_data = ((np.array(t_cmd_att), np.array(euler_cmd)), 
                (np.array(t_cmd_vel), np.array(w_cmd)))

    return pose_data, vel_data, cmd_data, initial_pose_msg


# --- JAX Simulation Functions (SMC) ---
def npy_reference_func(t, ts_ref, r_ref, dr_ref, ddr_ref, yaw_ref, yaw_rate_ref):
    r = jnp.array([jnp.interp(t, ts_ref, r_ref[:, i]) for i in range(3)])
    dr = jnp.array([jnp.interp(t, ts_ref, dr_ref[:, i]) for i in range(3)])
    ddr = jnp.array([jnp.interp(t, ts_ref, ddr_ref[:, i]) for i in range(3)])
    yaw = jnp.interp(t, ts_ref, yaw_ref)
    yaw_rate = jnp.interp(t, ts_ref, yaw_rate_ref)
    return r, dr, ddr, yaw, yaw_rate

def simulation_ode(z, t, k_R, K_mat, Lambda_mat, reference_func, dt):
    x, R_flatten, Omega_state = z
    q, dq = x[:3], x[3:]
    R = R_flatten.reshape((3, 3))
    r, dr, ddr, yaw_d, yaw_rate_d = reference_func(t)
    e, de = q - r, dq - dr
    s = de + Lambda_mat @ e
    v, dv = dr - Lambda_mat @ e, ddr - Lambda_mat @ de
    H, C, g, B = prior(q, dq)
    tau = H @ dv + C @ v + g - K_mat @ s
    u_d = jnp.linalg.solve(B, tau)
    f_d = jnp.linalg.norm(u_d)
    b_3d = u_d / (f_d + 1e-6)
    
    b_1d_desired = jnp.array([jnp.cos(yaw_d), jnp.sin(yaw_d), 0.])
    # b_1d_desired = dr / (jnp.linalg.norm(dr) + 1e-6)
    b_2d_temp = jnp.cross(b_3d, b_1d_desired)
    b_2d = b_2d_temp / (jnp.linalg.norm(b_2d_temp) + 1e-6)
    b_1d = jnp.cross(b_2d, b_3d)
    R_d = jnp.column_stack((b_1d, b_2d, b_3d))
    e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
    
    # Feedforward (for yaw rate ONLY NOW)
    world_yaw_rate = jnp.array([0., 0., yaw_rate_d])
    Omega_ff = R.T @ world_yaw_rate
    
    Omega_cmd = -k_R * e_R + Omega_ff
    
    dR = R @ hat(Omega_cmd)
    u_applied = f_d * R @ jnp.array([0., 0., 1.])
    ddq = jnp.linalg.solve(H, u_applied - C @ dq - g)
    dx = jnp.concatenate((dq, ddq))
    dOmega = (Omega_cmd - Omega_state) / dt
    return dx, dR.flatten(), dOmega

@partial(jax.jit, static_argnums=(5, 7))
def jax_flatten_wrapper(z_flat, t, k_R, K_mat, Lambda_mat, reference_func, dt, z_unravel_func, ts_ref, r_ref, dr_ref, ddr_ref, yaw_ref, yaw_rate_ref):
    z_tree = z_unravel_func(z_flat)
    ref_func_partial = partial(reference_func, ts_ref=ts_ref, r_ref=r_ref, dr_ref=dr_ref, ddr_ref=ddr_ref, yaw_ref=yaw_ref, yaw_rate_ref=yaw_rate_ref)
    dz_tree = simulation_ode(z_tree, t, k_R, K_mat, Lambda_mat, ref_func_partial, dt)
    return jnp.concatenate(jax.tree_util.tree_leaves(dz_tree))


def run_jax_simulation(gt_data, initial_pose_msg, gains):
    """
    Runs the JAX simulation using the provided ground truth and initial conditions.
    """
    ts_ref, r_ref, dr_ref, ddr_ref, yaw_ref, yaw_rate_ref = gt_data
    k_R, K_mat, Lambda_mat = gains
    
    T_FINAL = ts_ref[-1]
    # DT = ts_ref[1] - ts_ref[0]
    DT = 0.02

    if initial_pose_msg:
        p = initial_pose_msg.pose.position
        o = initial_pose_msg.pose.orientation
        r0_jax = jnp.array([p.x, p.y, p.z])
        q_init = np.array([o.x, o.y, o.z, o.w])
        R0_jax = jnp.array(Rotation.from_quat(q_init).as_matrix())
    else:
        print("Warning: No initial pose from rosbag. Using trajectory file for initial conditions.")
        r0_jax, R0_jax = jnp.array(r_ref[0]), jnp.eye(3)
        
    dr0_jax = jnp.array(dr_ref[0])
    x0_jax = jnp.concatenate([r0_jax, dr0_jax])
    z0_tree = (x0_jax, R0_jax.flatten(), jnp.zeros(3))
    z0_flat, z_unravel_func = jax.flatten_util.ravel_pytree(z0_tree)

    print("\n--- Starting JAX Simulation ---")
    ode_for_solver = partial(jax_flatten_wrapper, k_R=k_R, K_mat=K_mat, Lambda_mat=Lambda_mat, reference_func=npy_reference_func, dt=DT, z_unravel_func=z_unravel_func, ts_ref=jnp.array(ts_ref), r_ref=jnp.array(r_ref), dr_ref=jnp.array(dr_ref), ddr_ref=jnp.array(ddr_ref), yaw_ref=jnp.array(yaw_ref), yaw_rate_ref=jnp.array(yaw_rate_ref))
    
    z_history_flat, ts_jax = odeint_fixed_step(ode_for_solver, z0_flat, 0.0, T_FINAL, DT)
    
    z_history = jax.vmap(z_unravel_func)(z_history_flat)
    x_hist, R_flat_hist, Omega_hist = z_history # <-- MODIFIED: Capture Omega_hist
    q_jax = x_hist[:, :3]
    R_jax = R_flat_hist.reshape(-1, 3, 3)
    euler_jax = Rotation.from_matrix(np.asarray(R_jax)).as_euler('xyz', degrees=True)
    print("--- JAX Simulation Complete ---")

    return ts_jax, q_jax, euler_jax, Omega_hist # <-- MODIFIED: Return Omega_hist


# --- JAX Simulation Functions (PID Controller) ---

def simulation_ode_pid(z, t, kr, kp, ki, kd, integral_limit, reference_func, dt):
    """
    Simulates the drone dynamics using a PID position controller.
    The state 'z' is now (x, R_flatten, Omega_state, integral_error).
    """
    x, R_flatten, Omega_state, integral_error = z
    q, dq = x[:3], x[3:]
    R = R_flatten.reshape((3, 3))

    r, dr, ddr, yaw_d, yaw_rate_d = reference_func(t)

    # PID Controller Logic
    pos_err = q - r
    vel_err = dq - dr

    # Update and clamp integral error
    new_integral_error = integral_error + pos_err * dt
    new_integral_error = jnp.clip(new_integral_error, -integral_limit, integral_limit)

    # Calculate desired acceleration based on PID law
    m, g_acc = 2.0, 9.81
    thrust = kp * pos_err + ki * new_integral_error + kd * vel_err

    # Convert desired acceleration to desired force (u_d)
    u_d = m * (ddr + jnp.array([0.0, 0.0, g_acc])) - thrust

    # --- Attitude and Dynamics (same as before) ---
    f_d = jnp.linalg.norm(u_d)
    b_3d = u_d / (f_d + 1e-6)
    x_c = jnp.array([jnp.cos(yaw_d), jnp.sin(yaw_d), 0.0])
    b_2d_temp = jnp.cross(b_3d, x_c)
    b_2d = b_2d_temp / (jnp.linalg.norm(b_2d_temp) + 1e-6)
    b_1d = jnp.cross(b_2d, b_3d)
    R_d = jnp.column_stack((b_1d, b_2d, b_3d))

    e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
    
    # Feedforward (for yaw rate ONLY NOW)
    world_yaw_rate = jnp.array([0., 0., yaw_rate_d])
    Omega_ff = R.T @ world_yaw_rate
    
    # NOTE: For PID, we assume k_R is implicitly part of the PX4 inner loop
    # and not explicitly set in the same way. We use a placeholder here.
    k_R_pid = kr
    Omega_cmd = -k_R_pid * e_R + Omega_ff

    dR = R @ hat(Omega_cmd)
    u_applied = f_d * R @ jnp.array([0., 0., 1.])
    ddq = (u_applied - m * jnp.array([0., 0., g_acc])) / m
    dx = jnp.concatenate((dq, ddq))
    dOmega = (Omega_cmd - Omega_state) / dt

    # The derivative of the integral error is just the position error
    dIntegral_error = pos_err

    return dx, dR.flatten(), dOmega, dIntegral_error


@partial(jax.jit, static_argnums=(7, 9))
def jax_flatten_wrapper_pid(z_flat, t, kr, kp, ki, kd, integral_limit, reference_func, dt, z_unravel_func, ts_ref, r_ref, dr_ref, ddr_ref, yaw_ref, yaw_rate_ref):
    z_tree = z_unravel_func(z_flat)
    ref_func_partial = partial(reference_func, ts_ref=ts_ref, r_ref=r_ref, dr_ref=dr_ref, ddr_ref=ddr_ref, yaw_ref=yaw_ref, yaw_rate_ref=yaw_rate_ref)
    dz_tree = simulation_ode_pid(z_tree, t, kr, kp, ki, kd, integral_limit, ref_func_partial, dt)
    return jnp.concatenate(jax.tree_util.tree_leaves(dz_tree))

def run_jax_simulation_pid(gt_data, initial_pose_msg, gains):
    """
    Runs the JAX simulation using the PID controller.
    """
    ts_ref, r_ref, dr_ref, ddr_ref, yaw_ref, yaw_rate_ref = gt_data
    kr, kp, ki, kd, integral_limit = gains

    T_FINAL = ts_ref[-1]
    DT = ts_ref[1] - ts_ref[0]

    # --- Initial Conditions ---
    if initial_pose_msg:
        p = initial_pose_msg.pose.position
        o = initial_pose_msg.pose.orientation
        r0_jax = jnp.array([p.x, p.y, p.z])
        q_init = np.array([o.x, o.y, o.z, o.w])
        R0_jax = jnp.array(Rotation.from_quat(q_init).as_matrix())
    else:
        print("Warning: No initial pose from rosbag. Using trajectory file for initial conditions.")
        r0_jax, R0_jax = jnp.array(r_ref[0]), jnp.eye(3)

    # --- Add integral error to the initial state tree ---
    dr0_jax = jnp.array(dr_ref[0])
    x0_jax = jnp.concatenate([r0_jax, dr0_jax])
    z0_tree = (x0_jax, R0_jax.flatten(), jnp.zeros(3), jnp.zeros(3)) # (x, R, Omega, integral_error)
    z0_flat, z_unravel_func = jax.flatten_util.ravel_pytree(z0_tree)

    print("\n--- Starting JAX Simulation (PID Controller) ---")
    ode_for_solver = partial(jax_flatten_wrapper_pid, kr=kr,
                             kp=kp, ki=ki, kd=kd, integral_limit=integral_limit,
                             reference_func=npy_reference_func, 
                             dt=DT, 
                             z_unravel_func=z_unravel_func, 
                             ts_ref=jnp.array(ts_ref), 
                             r_ref=jnp.array(r_ref), 
                             dr_ref=jnp.array(dr_ref), 
                             ddr_ref=jnp.array(ddr_ref),
                             yaw_ref=jnp.array(yaw_ref),
                             yaw_rate_ref=jnp.array(yaw_rate_ref)
                             )

    z_history_flat, ts_jax = odeint_fixed_step(ode_for_solver, z0_flat, 0.0, T_FINAL, DT)

    z_history = jax.vmap(z_unravel_func)(z_history_flat)
    x_hist, R_flat_hist, w_jax_sim, _ = z_history # Unpack and ignore integral error history
    q_jax = x_hist[:, :3]
    R_jax = R_flat_hist.reshape(-1, 3, 3)
    euler_jax = Rotation.from_matrix(np.asarray(R_jax)).as_euler('xyz', degrees=True)
    print("--- JAX Simulation Complete ---")

    return ts_jax, q_jax, euler_jax, w_jax_sim

def get_gt_reference_attitude(gt_data):
    """
    Calculates the ground truth attitude (Euler angles) from the trajectory file.
    """
    ts_ref, r_ref, dr_ref, ddr_ref, yaw_ref, _ = gt_data

    @jax.jit
    def get_desired_attitude(r, dr, ddr, yaw_d):
        H, C, g, B = prior(r, dr) 
        tau = H @ ddr + C @ dr + g
        u_d = jnp.linalg.solve(B, tau)
        f_d = jnp.linalg.norm(u_d)
        b_3d = u_d / (f_d + 1e-6)
        x_c = jnp.array([jnp.cos(yaw_d), jnp.sin(yaw_d), 0.0])
        b_2d_temp = jnp.cross(b_3d, x_c)
        b_2d = b_2d_temp / (jnp.linalg.norm(b_2d_temp) + 1e-6)
        b_1d = jnp.cross(b_2d, b_3d)
        return jnp.column_stack((b_1d, b_2d, b_3d))

    R_d_hist = jax.vmap(get_desired_attitude)(r_ref, dr_ref, ddr_ref, yaw_ref)
    euler_gt = Rotation.from_matrix(np.asarray(R_d_hist)).as_euler('xyz', degrees=True)
    return euler_gt

# --- Plotting Function ---
def plot_and_save_comparison(gt_data, gazebo_data, jax_data, cmd_data, output_dir="bodyrate_figs"):
    """
    Plots all trajectories and angular states, and saves the figures.
    """
    # --- Unpack the data tuples ---
    ts_ref, r_ref, euler_gt = gt_data
    (t_gazebo_pose, q_gazebo, euler_gazebo), (t_gazebo_vel, w_gazebo) = gazebo_data
    ts_jax, q_jax, euler_jax, w_jax = jax_data
    (t_cmd_att, euler_cmd), (t_cmd_vel, w_cmd) = cmd_data

    # --- Create Output Directory ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- 3D Trajectory Plot ---
    fig_3d = plt.figure(figsize=(10, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot(r_ref[:, 0], r_ref[:, 1], r_ref[:, 2], 'r--', label='Ground Truth')
    ax_3d.plot(q_jax[:, 0], q_jax[:, 1], q_jax[:, 2], 'b-', label='JAX Sim', lw=2)
    ax_3d.plot(q_gazebo[:, 0], q_gazebo[:, 1], q_gazebo[:, 2], 'g:', label='Gazebo (Rosbag)', lw=2)
    ax_3d.set_title('3D Trajectory Comparison')
    ax_3d.set_xlabel('X (m)'); ax_3d.set_ylabel('Y (m)'); ax_3d.set_zlabel('Z (m)')
    ax_3d.legend(); ax_3d.axis('equal')
    fig_3d_path = os.path.join(output_dir, "comparison_3d_trajectory.png")
    fig_3d.savefig(fig_3d_path)
    print(f"Saved 3D comparison plot to: {fig_3d_path}")

    # --- Position Components Plot ---
    fig_pos, axs_pos = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    pos_labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs_pos[i].plot(ts_ref, r_ref[:, i], 'r--', label=f'Ground Truth {pos_labels[i]}')
        axs_pos[i].plot(ts_jax, q_jax[:, i], 'b-', label=f'JAX {pos_labels[i]}')
        axs_pos[i].plot(t_gazebo_pose, q_gazebo[:, i], 'g:', label=f'Gazebo {pos_labels[i]}')
        axs_pos[i].set_ylabel(f'{pos_labels[i]} (m)'); axs_pos[i].legend(); axs_pos[i].grid(True)
    axs_pos[2].set_xlabel('Time (s)')
    fig_pos.suptitle('Position Tracking Comparison', fontsize=16)
    fig_pos_path = os.path.join(output_dir, "comparison_position_vs_time.png")
    fig_pos.savefig(fig_pos_path)
    print(f"Saved position comparison plot to: {fig_pos_path}")

    # --- Attitude (Euler Angles) Plot ---
    fig_att, axs_att = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    angle_labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        axs_att[i].plot(ts_ref, euler_gt[:, i], 'k-.', label=f'Ground Truth Ref {angle_labels[i]}') # <-- ADD THIS LINE
        axs_att[i].plot(t_cmd_att, euler_cmd[:, i], 'r--', label=f'Command (Rosbag) {angle_labels[i]}') # <-- NEW
        axs_att[i].plot(ts_jax, euler_jax[:, i], 'b-', label=f'JAX {angle_labels[i]}')
        axs_att[i].plot(t_gazebo_pose, euler_gazebo[:, i], 'g:', label=f'Gazebo {angle_labels[i]}')
        axs_att[i].set_ylabel(f'{angle_labels[i]} (deg)'); axs_att[i].legend(); axs_att[i].grid(True)
    axs_att[2].set_xlabel('Time (s)')
    fig_att.suptitle('Attitude (Euler Angles) Comparison', fontsize=16)
    fig_att_path = os.path.join(output_dir, "comparison_attitude.png")
    fig_att.savefig(fig_att_path)
    print(f"Saved attitude comparison plot to: {fig_att_path}")

    # --- Angular Velocity Plot ---
    fig_vel, axs_vel = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    rate_labels = ['Roll Rate (P)', 'Pitch Rate (Q)', 'Yaw Rate (R)']
    for i in range(3):
        axs_vel[i].plot(t_cmd_vel, np.rad2deg(w_cmd[:, i]), 'r--', label=f'Command (Rosbag) {rate_labels[i]}')
        axs_vel[i].plot(ts_jax, np.rad2deg(w_jax[:, i]), 'b-', label=f'JAX {rate_labels[i]}')
        axs_vel[i].plot(t_gazebo_vel, np.rad2deg(w_gazebo[:, i]), 'g:', label=f'Gazebo {rate_labels[i]}')
        axs_vel[i].set_ylabel(f'{rate_labels[i]} (deg/s)'); axs_vel[i].legend(); axs_vel[i].grid(True)
    axs_vel[2].set_xlabel('Time (s)')
    fig_vel.suptitle('Angular Velocity Comparison', fontsize=16)
    fig_vel_path = os.path.join(output_dir, "comparison_angular_velocity.png")
    fig_vel.savefig(fig_vel_path)
    print(f"Saved angular velocity comparison plot to: {fig_vel_path}")
    
    # plt.show()

# def pid_to_sc(k_p, k_d, m):
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract, Simulate, and Compare Gazebo and JAX trajectories.")
    parser.add_argument('--rosbag', type=str, help="Path to the rosbag directory.", default="bodyrate_bag")
    parser.add_argument('--traj_file', type=str, help="Path to the .npy trajectory file.", default="circle_r2.0_t20s_alt1.5_initpsi0deg_pointToCenter_50hz_11col_2laps.npy")
    parser.add_argument('--velocity_topic', type=str, default="/mavros/local_position/velocity_body", help="Body velocity topic for angular rates.")
    parser.add_argument('--control_log_topic', type=str, default="/mlac_mission_node/control_log", help="Control log topic for FSM state.")
    parser.add_argument('--pose_topic', type=str, default="/mavros/local_position/pose", help="Pose topic for position tracking.")
    parser.add_argument('--attitude_setpoint_topic', type=str, default="/mavros/setpoint_raw/attitude", help="Attitude/rate command topic.")
    
    parser.add_argument('--controller_type', type=str, default="smc", choices=['smc', 'pid'], help="Type of controller to use in JAX sim ('smc' or 'pid').")
    parser.add_argument('--feedforward', action='store_true', help="Enable feedforward in the controller (if applicable).")

    parser.add_argument('--mass', type=float, help="Mass of the quadrotor in kg.", default=2.0)
    # Add other arguments as needed (pose_topic, gains, etc.)
    
    args = parser.parse_args()

    # --- Step 1: Extract Data from Rosbag ---
    print(f"--- Running Data Extraction on: {args.rosbag} ---")
    
    pose_data, vel_data, cmd_data, init_pose = extract_filtered_rosbag_data(
        args.rosbag, args.pose_topic, args.velocity_topic, args.control_log_topic, args.attitude_setpoint_topic
    )

    if pose_data[0] is not None and pose_data[0].size > 0:
        # --- Step 2: Run JAX Simulation ---
        traj_data = np.load(args.traj_file)

        # Unpack the initial data
        ts_ref = traj_data[:, 0]
        r_ref = traj_data[:, 1:4]
        dr_ref = traj_data[:, 4:7]
        yaw_ref = traj_data[:, 7]
        ddr_ref = traj_data[:, 8:11]

        if args.feedforward:
            yaw_rate_ref = np.gradient(np.unwrap(yaw_ref), ts_ref)
        else:
            yaw_rate_ref = np.zeros_like(ts_ref)
            
        gt_data_traj = (ts_ref, r_ref, dr_ref, ddr_ref, yaw_ref, yaw_rate_ref)
        
        euler_gt = get_gt_reference_attitude(gt_data_traj) 
                   
        # NOTE: Using default gains from the script for this example
        gains = (jnp.array([0.3, 0.3, 0.3]), 
                 jnp.diag(jnp.array([0.5, 0.5, 0.5])), 
                 jnp.diag(jnp.array([0.25, 0.25, 0.25])))
        
        # Gains that Kai used
        # vehicle mass
        m = args.mass

        if args.controller_type == 'smc':
            # Run the original Sliding Mode Controller simulation
            gains_smc = (jnp.array([0.3, 0.3, 0.3]),                # kr
                        jnp.diag(jnp.array([0.5, 0.5, 0.5])),       # K_mat
                        jnp.diag(jnp.array([0.25, 0.25, 0.25])))    # Lambda_mat
            # gains_smc = (jnp.array([1.0, 1.0, 1.0]),                # kr
            #             jnp.diag(jnp.array([1.0, 1.0, 1.0])),       # K_mat
            #             jnp.diag(jnp.array([1.0, 1.0, 2.0])))    # Lambda_mat

            ts_jax, q_jax, euler_jax, w_jax_sim = run_jax_simulation(gt_data_traj, init_pose, gains_smc)
            
            print(ts_jax)

        elif args.controller_type == 'pid':
            # Run the new PID Controller simulation
            gains_pid = (jnp.array([0.3, 0.3, 0.3]),        # kr
                        jnp.array([0.125, 0.125, 0.125]),   # kp
                        jnp.array([0.0, 0.0, 0.0]),         # ki
                        jnp.array([1.0, 1.0, 1.0]),         # kd
                        1.0)                                 # integral limit
            ts_jax, q_jax, euler_jax, w_jax_sim = run_jax_simulation_pid(gt_data_traj, init_pose, gains_pid)

        # --- Step 3: Plot Everything ---
        euler_gazebo = Rotation.from_quat(pose_data[2]).as_euler('xyz', degrees=True)

        output_dir = "bodyrate_figs/" + args.controller_type
        plot_and_save_comparison(
            (gt_data_traj[0], gt_data_traj[1], euler_gt), # Ground truth position
            ((pose_data[0], pose_data[1], euler_gazebo), (vel_data[0], vel_data[1])), # Gazebo states
            (ts_jax, q_jax, euler_jax, w_jax_sim), # JAX states
            cmd_data, # Commanded states from rosbag
            output_dir=output_dir
        )
        print("\n--- Comparison Complete ---")
    else:
        print("\n--- Script Finished: No data extracted from rosbag. ---")
#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
from functools import partial
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    from geometry_msgs.msg import PoseStamped
    from mavros_msgs.msg import AttitudeTarget
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import ROS 2 components: {e}")
    print("Please ensure your ROS 2 workspace is built and sourced.")
    sys.exit(1)

# --- Dynamics and Utils Imports ---
sys.path.append(os.path.abspath('..'))
try:
    from utils import hat, odeint_fixed_step
    from dynamics import prior
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import 'utils' or 'dynamics' modules: {e}")
    sys.exit(1)

# New Data Extraction Function
def get_rosbag_options(path, storage_id='sqlite3'):
    storage_options = StorageOptions(uri=path, storage_id=storage_id)
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    return storage_options, converter_options

def extract_open_loop_data_att_only_same_timing(rosbag_path, pose_topic, velocity_topic, control_log_topic, att_sp_topic):
    """
    Build a single consistent timeline using ONLY /mavros/setpoint_raw/attitude,
    assuming it's equivalent to loopback. Gate by control_log window if present.
    """
    reader = SequentialReader()
    storage_options, converter_options = get_rosbag_options(rosbag_path)
    reader.open(storage_options, converter_options)
    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}

    # Resolve message classes
    PoseStampedMsgClass = get_message(topic_types[pose_topic])
    VelocityMsgClass = get_message(topic_types[velocity_topic])
    ControllerLogMsgClass = get_message(topic_types[control_log_topic])
    AttitudeTargetMsgClass = get_message(topic_types[att_sp_topic])

    def header_ns(hdr):
        return int(hdr.stamp.sec) * 1_000_000_000 + int(hdr.stamp.nanosec)

    # ---------- Pass 1: find window from control_log (if available) ----------
    traj_exec_start_ns = -1
    traj_exec_end_ns   = -1
    reader.set_filter(StorageFilter(topics=[control_log_topic]))
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, ControllerLogMsgClass)
        if traj_exec_start_ns == -1 and msg.trajectory_execution_start_ros_time.sec > 0:
            traj_exec_start_ns = msg.trajectory_execution_start_ros_time.sec * 1e9 + msg.trajectory_execution_start_ros_time.nanosec
        if msg.trajectory_execution_end_ros_time.sec > 0:
            traj_exec_end_ns = msg.trajectory_execution_end_ros_time.sec * 1e9 + msg.trajectory_execution_end_ros_time.nanosec
            break

    # ---------- Pass 2: collect setpoints + poses (header time only) ----------
    reader.set_filter(StorageFilter(topics=[pose_topic, att_sp_topic, velocity_topic]))
    reader.seek(0)

    t_cmd_ns, thrust_raw, w_raw, mask_raw = [], [], [], []
    t_pose_ns, q_pose, quat_pose = [], [], []
    t_vel_ns, vel_body = [], []
    initial_pose_msg = None
    initial_velocity_msg = None


    while reader.has_next():
        topic, data, _ = reader.read_next()

        if topic == att_sp_topic:
            msg = deserialize_message(data, AttitudeTargetMsgClass)
            t_ns = header_ns(msg.header)

            # If control window is known, keep only inside it
            if traj_exec_start_ns != -1 and t_ns < traj_exec_start_ns: 
                continue
            if traj_exec_end_ns   != -1 and t_ns > traj_exec_end_ns: 
                continue

            t_cmd_ns.append(t_ns)
            thrust_raw.append(float(msg.thrust))
            w_raw.append([float(msg.body_rate.x), float(msg.body_rate.y), float(msg.body_rate.z)])
            mask_raw.append(int(msg.type_mask))

        elif topic == pose_topic:
            msg = deserialize_message(data, PoseStampedMsgClass)
            t_ns = header_ns(msg.header)

            if traj_exec_start_ns != -1 and t_ns < traj_exec_start_ns: 
                continue
            if traj_exec_end_ns   != -1 and t_ns > traj_exec_end_ns: 
                continue

            if initial_pose_msg is None:
                initial_pose_msg = msg
            t_pose_ns.append(t_ns)
            q_pose.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            quat_pose.append([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        
        elif topic == velocity_topic:
            # Currently not used, but could be for initial velocity
            msg = deserialize_message(data, VelocityMsgClass)
            t_ns = header_ns(msg.header)
            
            if traj_exec_start_ns != -1 and t_ns < traj_exec_start_ns: 
                continue
            if traj_exec_end_ns   != -1 and t_ns > traj_exec_end_ns: 
                continue
            
            if initial_velocity_msg is None:
                initial_velocity_msg = msg
            t_vel_ns.append(t_ns)
            vel_body.append([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])


    if len(t_cmd_ns) == 0:
        print("ERROR: No /mavros/setpoint_raw/attitude messages found in the selected window.")
        return (None, None, None), (None, None, None), None

    # Sort by time to be safe
    order_cmd = np.argsort(t_cmd_ns)
    t_cmd_ns  = np.array(t_cmd_ns)[order_cmd]
    thrust_raw = np.array(thrust_raw)[order_cmd]        # Fraction of thrust
    w_raw      = np.array(w_raw)[order_cmd]
    mask_raw   = np.array(mask_raw)[order_cmd]

    if len(t_pose_ns) > 0:
        order_pose = np.argsort(t_pose_ns)
        t_pose_ns  = np.array(t_pose_ns)[order_pose]
        q_pose     = np.array(q_pose)[order_pose]
        quat_pose  = np.array(quat_pose)[order_pose]
    else:
        t_pose_ns = np.array([])
        q_pose    = np.empty((0,3))
        quat_pose = np.empty((0,4))

    if len(t_vel_ns) > 0:
        order_pose = np.argsort(t_vel_ns)
        t_vel_ns  = np.array(t_vel_ns)[order_pose]
        vel_body     = np.array(vel_body)[order_pose]
    else:   
        t_vel_ns = np.array([])
        vel_body    = np.empty((0,3))

    # If control window was not found, use span of setpoints as window origin
    t0_ns = t_cmd_ns[0]
    t_cmd = (t_cmd_ns - t0_ns) * 1e-9
    t_pose = (t_pose_ns - t0_ns) * 1e-9 if t_pose_ns.size > 0 else np.array([])
    t_vel = (t_vel_ns - t0_ns) * 1e-9 if t_vel_ns.size > 0 else np.array([])

    # ---------- Honor type_mask with last-value-held + clamp thrust ----------
    def lvh_forward(arr, invalid_rows):
        arr = np.asarray(arr, dtype=float)
        valid = ~invalid_rows
        for i in range(arr.shape[0]):
            if not valid[i]:
                if i == 0:
                    j = np.argmax(valid)
                    arr[i] = arr[j] if valid[j] else 0.0
                else:
                    arr[i] = arr[i-1]
        return arr

    thrust = thrust_raw.copy()
    w = np.asarray(w_raw, dtype=float)

    thrust_invalid = (mask_raw & 64) != 0   # IGNORE_THRUST
    thrust = lvh_forward(thrust, thrust_invalid)
    thrust = np.clip(thrust, 0.0, 1.0)

    rx_inv = (mask_raw & 1) != 0   # IGNORE_ROLL_RATE
    ry_inv = (mask_raw & 2) != 0   # IGNORE_PITCH_RATE
    rz_inv = (mask_raw & 4) != 0   # IGNORE_YAW_RATE
    w[:,0] = lvh_forward(w[:,0], rx_inv)
    w[:,1] = lvh_forward(w[:,1], ry_inv)
    w[:,2] = lvh_forward(w[:,2], rz_inv)

    gazebo_states = (t_pose, q_pose, quat_pose, t_vel, vel_body)
    commanded_inputs = (t_cmd, thrust, w)

    # jax.debug.print(initial_velocity_msg)
    # jax.debug.print(type(initial_velocity_msg))

    return gazebo_states, commanded_inputs, initial_pose_msg, initial_velocity_msg


# ---------------------------------------------------------------------------- #
#                             CONVERT JAX TO NUMPY                             #
# ---------------------------------------------------------------------------- #
def euler_rates_to_rotation_matrix_derivative(yaw_pitch_roll, yaw_pitch_roll_rates):
    """
    Transforms Euler angle derivatives to the time derivative of a rotation matrix.
    This function uses the ZYX extrinsic convention for Euler angles. This corresponds
    to a sequence of rotations:
    1. Yaw (psi) about the world Z-axis.
    2. Pitch (theta) about the intermediate Y-axis.
    3. Roll (phi) about the final X-axis.
    Args:
        yaw_pitch_roll (np.ndarray): A 1x3 array of the current Euler angles
                                     [yaw, pitch, roll] in radians.
        yaw_pitch_roll_rates (np.ndarray): A 1x3 array of the rates of change
                                           [yaw_dot, pitch_dot, roll_dot]
                                           in radians per second.
    Returns:
        np.ndarray: The 3x3 time derivative of the rotation matrix (R_dot).
    """
    # Unpack angles: psi (yaw, Z), theta (pitch, Y), phi (roll, X)
    psi, theta, phi = yaw_pitch_roll
    # Unpack their rates of change
    psi_dot, theta_dot, phi_dot = yaw_pitch_roll_rates
    # Step 1: Define the transformation matrix that maps Euler rates to the
    # world-frame angular velocity vector (omega) for an extrinsic ZYX sequence.
    # The columns of this matrix are the axes of rotation (roll, pitch, yaw)
    # expressed in the world frame.
    # omega_world = T @ [roll_rate, pitch_rate, yaw_rate]^T
    T = np.array([
        [np.cos(psi) * np.cos(theta), -np.sin(psi), 0],
        [np.sin(psi) * np.cos(theta),  np.cos(psi), 0],
        [-np.sin(theta),                         0, 1]
    ])
    # The transformation matrix T must be multiplied by the rates in the
    # order corresponding to the columns: roll, then pitch, then yaw.
    rates_ordered = np.array([phi_dot, theta_dot, psi_dot])
    # Step 2: Calculate the angular velocity vector in the world frame
    omega_world = T @ rates_ordered
    # Step 3: Form the skew-symmetric matrix S(omega) from the world angular velocity
    omega_x, omega_y, omega_z = omega_world
    S_omega = np.array([
        [0, -omega_z, omega_y],
        [omega_z, 0, -omega_x],
        [-omega_y, omega_x, 0]
    ])
    # Step 4: Get the current rotation matrix R using SciPy
    # The sequence 'zyx' corresponds to our yaw-pitch-roll convention.
    rot = Rotation.from_euler('zyx', yaw_pitch_roll)
    R = rot.as_matrix()
    # Step 5: Calculate the derivative of the rotation matrix using the formula
    # R_dot = S(omega_world) * R
    R_dot = S_omega @ R
    return R_dot


# --- Helper function to reconstruct R from Euler angles ---
def euler_to_rotation_matrix(rpy):
    """Converts a roll, pitch, yaw vector to a 3x3 rotation matrix."""
    roll, pitch, yaw = rpy
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    # ZYX convention: R = Rz * Ry * Rx
    return R_z @ R_y @ R_x

def simulation_ode_open_loop_zoh(t, z, commands, mass, g_acc=9.81, hov_thrust=0.726):
    """
    Dynamics-only ODE. It takes a constant command for the integration step.
    State 'z' is now just (x, R_flatten).
    """
    # Unpack the constant commands for this step
    thrust_norm, Omega_cmd = commands

    # Unpack the current state
    x = z[:6]
    R_flatten = z[6:]
    q, dq = x[:3], x[3:]
    R = R_flatten.reshape((3, 3))

    # ---------------------------------------------------------------------------- #
    #                          Original Bodyrate Dynamics                          #
    # ---------------------------------------------------------------------------- #
    # # Convert normalized thrust to force magnitude
    # f_d = thrust_norm * mass * g_acc / hov_thrust

    # # --- Dynamics (using the provided constant commands) ---
    # dR = R @ hat(Omega_cmd)

    # u_applied = f_d * R @ jnp.array([0., 0., 1.])
    # H, C, g, _ = prior(q, dq)
    # ddq = jnp.linalg.solve(H, u_applied - C @ dq - g)
    # dx = jnp.concatenate((dq, ddq))

    # return dx, dR.flatten()

    # ---------------------------------------------------------------------------- #
    #                       Flying Inverted Pendulum Dynamics                      #
    # ---------------------------------------------------------------------------- #
    f_d = thrust_norm * g_acc / hov_thrust
    rpy = sp.spatial.transform.Rotation.from_matrix(R).as_euler('zyx', degrees=False)
    gamma, beta, alpha = rpy  # roll, pitch, yaw

    inverse_matrix = np.linalg.inv(np.array([
        [np.cos(beta)*np.cos(gamma), -np.sin(gamma), 0],
        [np.cos(beta)*np.sin(gamma),  np.cos(gamma), 0],
        [-np.sin(beta),               0,              1]
    ]))
    d_rpy = inverse_matrix @ Omega_cmd
    dgamma, dbeta, dalpha = d_rpy

    ddq = R @ np.array([0, 0, f_d]) - np.array([0, 0, g_acc])
    dx = np.concatenate((dq, ddq))
    dR = euler_rates_to_rotation_matrix_derivative(np.array([alpha, beta, gamma]), np.array([dalpha, dbeta, dgamma]))

    return np.concatenate([dx, dR.flatten()])

# Dynamics function using Euler angles in the state for NumPy
def simulation_ode_euler(t, z, commands, mass, g_acc=9.81, hov_thrust=0.727):
    """
    Your specific "Flying Inverted Pendulum" dynamics model adapted to use an
    Euler angle state vector z = [pos, vel, rpy].
    """
    # 1. Unpack commands and the 9-element state vector
    thrust_norm, Omega_cmd = commands
    pos, vel, rpy = z[0:3], z[3:6], z[6:9]
    roll, pitch, yaw = rpy  # Unpack for clarity, matching your original gamma, beta, alpha

    # 2. Reconstruct the rotation matrix 'R' from the current Euler angles
    R = euler_to_rotation_matrix(rpy)
    
    # 3. Calculate Translational Dynamics (copied from your original function)
    f_d = thrust_norm * g_acc / hov_thrust
    ddq = R @ np.array([0, 0, f_d]) - np.array([0, 0, g_acc])
    d_pos = vel
    d_vel = ddq

    # 4. Calculate Rotational Dynamics (copied from your original function)
    # This computes d(rpy)/dt from the body rates Omega_cmd
    # Note: Using aliases beta and gamma to match your original matrix exactly
    beta = pitch
    gamma = roll
    
    # This matrix becomes singular at pitch (beta) = +/- 90 degrees (Gimbal Lock)
    inverse_matrix = np.linalg.inv(np.array([
        [np.cos(beta)*np.cos(gamma), -np.sin(gamma), 0],
        [np.cos(beta)*np.sin(gamma),  np.cos(gamma), 0],
        [-np.sin(beta),               0,             1]
    ]))
    d_rpy = inverse_matrix @ Omega_cmd
    
    # 5. Return the unified 9-element derivative vector
    return np.concatenate([d_pos, d_vel, d_rpy]), f_d

def rk4_step_numpy(f, t, y, dt, *args):
    """
    A simple, one-step Runge-Kutta 4 (RK4) integrator in NumPy.
    """
    k1, f_d = f(t, y, *args)
    k2, _ = f(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
    k3, _ = f(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
    k4, _ = f(t + dt, y + dt * k3, *args)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), f_d

def run_jax_open_loop_simulation_synced(initial_pose_msg, commanded_inputs, mass):
    """
    Runs the open-loop simulation using pure NumPy, synchronized with rosbag timestamps.
    """
    t_cmd, thrust_cmd, w_cmd = commanded_inputs

    if not initial_pose_msg:
        print("Error: Could not get initial pose from rosbag.")
        return None, None

    # --- Initial Conditions from Rosbag (using NumPy) ---
    p = initial_pose_msg.pose.position
    o = initial_pose_msg.pose.orientation
    pos0 = np.array([p.x, p.y, p.z])
    vel0 = np.zeros(3)  # Assume starting from hover
    q_init = np.array([o.x, o.y, o.z, o.w])
    R0 = np.array(Rotation.from_quat(q_init).as_matrix())

    x0 = np.concatenate([pos0, vel0])
    # The state is now a single, flat 15-element vector
    z0_numpy = np.concatenate([x0, R0.flatten()])

    # --- Run the Time-Synchronized ZOH Simulation Loop ---
    print("\n--- Starting NumPy Open-Loop Simulation (Time-Synchronized ZOH) ---")

    ts_numpy = [t_cmd[0]]
    z_history = [z0_numpy]
    current_z = z0_numpy

    # Loop through each command and the corresponding time interval
    for i in range(len(t_cmd) - 1):
        t_start = t_cmd[i]
        t_end = t_cmd[i+1]
        dt = t_end - t_start

        if dt <= 0:  # Skip steps with zero or negative duration
            continue

        # The command is held constant for this entire interval
        held_commands = (thrust_cmd[i], w_cmd[i])

        # Integrate the dynamics over this non-uniform time step using RK4
        current_z = rk4_step_numpy(
            simulation_ode_open_loop_zoh,
            t_start,
            current_z,
            dt,
            held_commands,  # Pass extra args to the dynamics function
            mass
        )

        # Store the result
        ts_numpy.append(t_end)
        z_history.append(current_z)

    # Convert history list to a single NumPy array for easy slicing
    z_history_array = np.array(z_history)

    # --- Post-process results (now a simple array slice) ---
    q_numpy = z_history_array[:, 0:3]  # Extract the position columns
    
    print("--- NumPy Open-Loop Simulation Complete ---")

    return np.array(ts_numpy), q_numpy

def run_simulation_with_euler_synced(initial_pose_msg, initial_vel_msg, commanded_inputs, mass):
    """
    Runs the open-loop simulation using NumPy with an Euler angle state representation.
    """
    t_cmd, thrust_cmd, w_cmd = commanded_inputs

    if not initial_pose_msg:
        # ... (error handling) ...
        return None, None

    # --- Initial Conditions ---
    p = initial_pose_msg.pose.position
    v = initial_vel_msg.twist.linear
    o = initial_pose_msg.pose.orientation
    pos0 = np.array([p.x, p.y, p.z])
    vel0_body = np.array([v.x, v.y, v.z])
    q_init = np.array([o.x, o.y, o.z, o.w])
    
    # CHANGED: Convert initial orientation to Euler angles ('zyx' -> yaw, pitch, roll)
    rpy0 = Rotation.from_quat(q_init).as_euler('zyx', degrees=False)
    # Scipy returns [yaw, pitch, roll], but our state is [roll, pitch, yaw]
    # Let's re-order to be explicit: [roll, pitch, yaw]
    initial_roll, initial_pitch, initial_yaw = rpy0[2], rpy0[1], rpy0[0]
    rpy0_state_order = np.array([initial_roll, initial_pitch, initial_yaw])

    vel0 = np.linalg.inv(Rotation.from_euler('zyx', [initial_yaw, initial_pitch, initial_roll]).as_matrix()) @ vel0_body

    # CHANGED: The state is now a single, flat 9-element vector
    z0_numpy = np.concatenate([pos0, vel0, rpy0_state_order])

    # --- Run the Time-Synchronized ZOH Simulation Loop ---
    print("\n--- Starting NumPy Open-Loop Simulation (Euler State) ---")

    ts_numpy = [t_cmd[0]]
    z_history = [z0_numpy]
    current_z = z0_numpy
    f_d_history = []

    for i in range(len(t_cmd) - 1):
        t_start, t_end = t_cmd[i], t_cmd[i+1]
        dt = t_end - t_start

        if dt <= 0: continue

        held_commands = (thrust_cmd[i], w_cmd[i])

        # CHANGED: Call the new ODE function
        current_z, current_f_d = rk4_step_numpy(
            simulation_ode_euler,
            t_start,
            current_z,
            dt,
            held_commands,
            mass
        )
        z_history.append(current_z)
        ts_numpy.append(t_end)
        f_d_history.append(current_f_d)

    z_history_array = np.array(z_history)
    q_numpy = z_history_array
    f_d_history = np.array(f_d_history)
    
    print("--- NumPy Open-Loop Simulation Complete ---")

    return np.array(ts_numpy), q_numpy, f_d_history

# --- Plotting Function ---
def plot_open_loop_comparison(gazebo_states, jax_states, output_dir="open_loop_figs"):
    (t_pose, q_pose, quat_pose, t_vel, vel_body) = gazebo_states
    (ts_jax, q_jax, f_d_jax) = jax_states

    # Convert quaternions to RPY for Gazebo
    if quat_pose.shape[0] > 0:
        rpy_gaz = Rotation.from_quat(quat_pose).as_euler('zyx', degrees=False)  # yaw, pitch, roll
        rpy_gaz = rpy_gaz[:, ::-1]  # Reorder to roll, pitch, yaw
        rpy_gaz_on_cmd = np.column_stack([
            np.interp(ts_jax, t_pose, rpy_gaz[:, i]) for i in range(3)
        ])
    else:
        rpy_gaz_on_cmd = np.full((ts_jax.shape[0], 3), np.nan)

    # Interp Gazebo pose to command timeline for apples-to-apples x-axis
    if t_pose.size > 1 and q_pose.shape[0] > 1:
        q_gaz_on_cmd = np.column_stack([
            np.interp(ts_jax, t_pose, q_pose[:, i]) for i in range(3)
        ])
    else:
        q_gaz_on_cmd = np.full_like(q_jax, np.nan)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3D
    fig_3d = plt.figure(figsize=(10, 10))
    ax = fig_3d.add_subplot(111, projection='3d')
    ax.plot(q_jax[:,0], q_jax[:,1], q_jax[:,2], '-', label='JAX (setpoint timeline)', lw=2)
    ax.plot(q_gaz_on_cmd[:,0], q_gaz_on_cmd[:,1], q_gaz_on_cmd[:,2], ':', label='Gazebo (interp@setpoint)', lw=2)
    ax.set_title('Open-Loop 3D Trajectory (Common Setpoint Timeline)')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend(); ax.axis('equal')
    fig_3d.savefig(os.path.join(output_dir, "open_loop_3d_trajectory.png"))

    # XYZ vs time
    fig_pos, axs_pos = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs_pos[i].plot(ts_jax, q_jax[:, i], '-', label='JAX')
        axs_pos[i].plot(ts_jax, q_gaz_on_cmd[:, i], ':', label='Gazebo (interp)')
        axs_pos[i].set_ylabel(f'{labels[i]} (m)'); axs_pos[i].grid(True); axs_pos[i].legend()
        axs_pos[i].set_xlim([ts_jax[0], 15.0])
    axs_pos[2].set_xlabel('Time (s)')
    fig_pos.suptitle('Open-Loop Position on Common (Setpoint) Timeline', fontsize=16)
    fig_pos.savefig(os.path.join(output_dir, "open_loop_position_vs_time.png"))
    # plt.show()

    # rpy vs time
    fig_rpy, axs_rpy = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        axs_rpy[i].plot(ts_jax, q_jax[:, i + 6], '-', label='JAX')
        axs_rpy[i].plot(ts_jax, q_gaz_on_cmd[:, i + 6], ':', label='Gazebo (interp)')
        axs_rpy[i].set_ylabel(f'{labels[i]} (rad)'); axs_rpy[i].grid(True); axs_rpy[i].legend()
        axs_rpy[i].set_xlim([ts_jax[0], 15.0])
    axs_rpy[2].set_xlabel('Time (s)')
    fig_rpy.suptitle('Open-Loop RPY on Common (Setpoint) Timeline', fontsize=16)
    fig_rpy.savefig(os.path.join(output_dir, "open_loop_rpy_vs_time.png"))
    # plt.show()

    # Plot f_d_history in a single plot
    if f_d_jax is not None and len(f_d_jax) > 0:
        fig_fd, ax_fd = plt.subplots(figsize=(12, 5))
        # f_d_jax is one element shorter than ts_jax, so use ts_jax[1:]
        ax_fd.plot(ts_jax[1:], f_d_jax, label='f_d_history')
        ax_fd.set_xlabel('Time (s)')
        ax_fd.set_ylabel('f_d (N/kg)')
        ax_fd.set_title('f_d_history vs Time')
        ax_fd.grid(True)
        ax_fd.legend()
        fig_fd.savefig(os.path.join(output_dir, "open_loop_f_d_history.png"))
    plt.show()


# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Open-loop comparison of Gazebo and JAX dynamics.")
    parser.add_argument('--rosbag', type=str, required=True, help="Path to the rosbag directory.")
    parser.add_argument('--mass', type=float, default=2.0, help="Vehicle mass (kg).")
    parser.add_argument('--pose_topic', type=str, default="/mavros/local_position/pose")
    parser.add_argument('--control_log_topic', type=str, default="/mlac_mission_node/control_log")
    parser.add_argument('--attitude_setpoint_topic', type=str, default="/mavros/setpoint_raw/attitude")
    parser.add_argument('--velocity_topic', type=str, default="/mavros/local_position/velocity_body")

    args = parser.parse_args()

    # gazebo_states, commanded_inputs, init_pose = extract_open_loop_data(
    #     args.rosbag, args.pose_topic, args.control_log_topic, args.attitude_setpoint_topic
    # )

    gazebo_states, commanded_inputs, init_pose, initial_vel_body = extract_open_loop_data_att_only_same_timing(
    args.rosbag, args.pose_topic, args.velocity_topic, args.control_log_topic, args.attitude_setpoint_topic
)
    
    if gazebo_states[0] is not None and gazebo_states[0].size > 0:
        # --- Step 2: Run the Time-Synchronized JAX Sim ---
        # No need to calculate an average dt anymore
        # ts_jax, q_jax = run_jax_open_loop_simulation_synced(
        #     init_pose, commanded_inputs, args.mass
        # )
        t_numpy, q_numpy, f_d_numpy = run_simulation_with_euler_synced(
            init_pose, initial_vel_body, commanded_inputs, args.mass
        )

        # --- Step 3: Plot ---
        if t_numpy is not None:
            plot_open_loop_comparison(gazebo_states, (t_numpy, q_numpy, f_d_numpy))
            print("\n--- Open-Loop Comparison Complete ---")
        else:
            print("\n--- JAX Simulation Failed ---")
    else:
        print("\n--- Script Finished: No data extracted. ---")
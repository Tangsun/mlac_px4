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

from jax_rotation import rotation_matrix_to_euler_jax

# --- Data Extraction Function ---
def get_rosbag_options(path, storage_id='sqlite3'):
    storage_options = StorageOptions(uri=path, storage_id=storage_id)
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    return storage_options, converter_options

def extract_open_loop_data(rosbag_path, pose_topic, control_log_topic, att_sp_topic):
    reader = SequentialReader()
    storage_options, converter_options = get_rosbag_options(rosbag_path)
    reader.open(storage_options, converter_options)
    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    
    PoseStampedMsgClass = get_message(topic_types[pose_topic])
    ControllerLogMsgClass = get_message(topic_types[control_log_topic])
    AttitudeTargetMsgClass = get_message(topic_types[att_sp_topic])

    # --- First Pass: Find Tracking Window ---
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
            
    # --- Second Pass: Extract All Necessary Data ---
    reader.set_filter(StorageFilter(topics=[pose_topic, att_sp_topic]))
    reader.seek(0)
    
    t_pose, q_pose, quat_pose = [], [], []
    initial_pose_msg = None
    
    t_cmd, thrust_cmd, w_cmd = [], [], []

    print("Extracting data from tracking window...")
    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()
        if traj_exec_start_ns != -1 and t_ns >= traj_exec_start_ns:
            if traj_exec_end_ns != -1 and t_ns > traj_exec_end_ns: break
            relative_time = (t_ns - traj_exec_start_ns) / 1e9

            if topic == pose_topic:
                msg = deserialize_message(data, PoseStampedMsgClass)
                if initial_pose_msg is None: initial_pose_msg = msg
                t_pose.append(relative_time)
                q_pose.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
                quat_pose.append([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            
            elif topic == att_sp_topic:
                msg = deserialize_message(data, AttitudeTargetMsgClass)
                t_cmd.append(relative_time)
                thrust_cmd.append(msg.thrust)
                w_cmd.append([msg.body_rate.x, msg.body_rate.y, msg.body_rate.z])

    gazebo_states = (np.array(t_pose), np.array(q_pose), np.array(quat_pose))
    commanded_inputs = (np.array(t_cmd), np.array(thrust_cmd), np.array(w_cmd))
    
    return gazebo_states, commanded_inputs, initial_pose_msg

# New Data Extraction Function
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


def simulation_ode_open_loop_zoh(z, t, commands, mass, g_acc, hov_thrust=0.726):
    """
    Dynamics-only ODE. It takes a constant command for the integration step.
    State 'z' is now just (x, R_flatten).
    """
    # Unpack the constant commands for this step
    thrust_norm, Omega_cmd = commands

    # Unpack the current state
    x, R_flatten = z
    q, dq = x[:3], x[3:]
    R = R_flatten.reshape((3, 3))

    # ---------------------------------------------------------------------------- #
    #                          Original Bodyrate Dynamics                          #
    # ---------------------------------------------------------------------------- #
    # Convert normalized thrust to force magnitude
    f_d = thrust_norm * mass * g_acc / hov_thrust

    # --- Dynamics (using the provided constant commands) ---
    dR = R @ hat(Omega_cmd)

    u_applied = f_d * R @ jnp.array([0., 0., 1.])
    H, C, g, _ = prior(q, dq)
    ddq = jnp.linalg.solve(H, u_applied - C @ dq - g)
    dx = jnp.concatenate((dq, ddq))

    return dx, dR.flatten()


def run_jax_open_loop_simulation_synced(initial_pose_msg, initial_velocity_msg, commanded_inputs, mass):
    """
    Runs the JAX open-loop simulation using a ZOH on commands, perfectly synchronized
    with the timestamps from the rosbag.
    """
    t_cmd, thrust_cmd, w_cmd = commanded_inputs
    
    print(t_cmd)

    if not initial_pose_msg:
        print("Error: Could not get initial pose from rosbag.")
        return None, None

    # --- Initial Conditions from Rosbag ---
    p = initial_pose_msg.pose.position
    o = initial_pose_msg.pose.orientation
    r0_jax = jnp.array([p.x, p.y, p.z])
    q_init = np.array([o.x, o.y, o.z, o.w])
    R0_jax = jnp.array(Rotation.from_quat(q_init).as_matrix())

    v = initial_velocity_msg.twist.linear
    vel0 = np.array([v.x, v.y, v.z])
    # dr0_jax = jnp.linalg.inv(R0_jax) @ jnp.array(vel0)
    dr0_jax = R0_jax @ jnp.array(vel0)
    x0_jax = jnp.concatenate([r0_jax, dr0_jax])
    
    # --- Simplified initial state tree ---
    z0_tree = (x0_jax, R0_jax.flatten())
    z0_flat, z_unravel_func = jax.flatten_util.ravel_pytree(z0_tree)

    # --- Define the single-step integration function for the ZOH loop ---
    @jax.jit
    def step_func(z_flat, t_start, dt, held_commands):

        # Define the dynamics function for this specific step
        ode_func_partial = partial(simulation_ode_open_loop_zoh, commands=held_commands, mass=mass, g_acc=9.81)

        # Import the specific integrator step from your utils
        from utils import rk38_step 

        # Wrapper for the integrator step to handle flattened state
        def flat_dynamics(z_flat_inner, t_inner):
            z_tree_inner = z_unravel_func(z_flat_inner)
            dz_tree_inner = ode_func_partial(z_tree_inner, t_inner)
            return jnp.concatenate(jax.tree_util.tree_leaves(dz_tree_inner))

        z_next_flat = rk38_step(flat_dynamics, dt, z_flat, t_start)
        return z_next_flat

    # --- Run the Time-Synchronized ZOH Simulation Loop ---
    print(f"\n--- Starting JAX Open-Loop Simulation (Time-Synchronized ZOH) ---")

    ts_jax = [t_cmd[0]]
    z_history_flat = [z0_flat]
    current_z_flat = z0_flat

    # Loop through each command and the corresponding time interval
    for i in range(len(t_cmd) - 1):
        t_start = t_cmd[i]
        t_end = t_cmd[i+1]
        dt = t_end - t_start

        # The command is held constant for this entire interval
        held_commands = (thrust_cmd[i], w_cmd[i])

        # Integrate the dynamics over this non-uniform time step
        current_z_flat = step_func(current_z_flat, t_start, dt, held_commands)

        # Store the result
        ts_jax.append(t_end)
        z_history_flat.append(current_z_flat)

    z_history_flat = jnp.array(z_history_flat)

    # --- Post-process results ---
    z_history = jax.vmap(z_unravel_func)(z_history_flat)
    x_hist, _ = z_history
    q_jax = x_hist[:, :3]
    print("--- JAX Open-Loop Simulation Complete ---")

    return np.array(ts_jax), q_jax


# --- Plotting Function ---
def plot_open_loop_comparison(gazebo_states, jax_states, output_dir="open_loop_figs"):
    (t_pose, q_gazebo, _, _, _) = gazebo_states
    (ts_jax, q_jax) = jax_states

    # Interp Gazebo pose to command timeline for apples-to-apples x-axis
    if t_pose.size > 1 and q_gazebo.shape[0] > 1:
        q_gaz_on_cmd = np.column_stack([
            np.interp(ts_jax, t_pose, q_gazebo[:, i]) for i in range(3)
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
    plt.show()


# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Open-loop comparison of Gazebo and JAX dynamics.")
    parser.add_argument('--rosbag', type=str, required=True, help="Path to the rosbag directory.")
    parser.add_argument('--mass', type=float, default=2.0, help="Vehicle mass (kg).")
    parser.add_argument('--pose_topic', type=str, default="/mavros/local_position/pose")
    parser.add_argument('--velocity_topic', type=str, default="/mavros/local_position/velocity_body")
    parser.add_argument('--control_log_topic', type=str, default="/mlac_mission_node/control_log")
    parser.add_argument('--attitude_setpoint_topic', type=str, default="/mavros/setpoint_raw/attitude")
    
    args = parser.parse_args()

    # gazebo_states, commanded_inputs, init_pose = extract_open_loop_data(
    #     args.rosbag, args.pose_topic, args.control_log_topic, args.attitude_setpoint_topic
    # )

    gazebo_states, commanded_inputs, init_pose, init_velocity = extract_open_loop_data_att_only_same_timing(
        args.rosbag, args.pose_topic, args.velocity_topic, args.control_log_topic, args.attitude_setpoint_topic
    )
    
    if gazebo_states[0] is not None and gazebo_states[0].size > 0:
        # --- Step 2: Run the Time-Synchronized JAX Sim ---
        # No need to calculate an average dt anymore
        ts_jax, q_jax = run_jax_open_loop_simulation_synced(
            init_pose, init_velocity, commanded_inputs, args.mass
        )

        # --- Step 3: Plot ---
        if ts_jax is not None:
            plot_open_loop_comparison(gazebo_states, (ts_jax, q_jax))
            print("\n--- Open-Loop Comparison Complete ---")
        else:
            print("\n--- JAX Simulation Failed ---")
    else:
        print("\n--- Script Finished: No data extracted. ---")
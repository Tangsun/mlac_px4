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


# --- JAX Open-Loop Simulation Functions ---
def commanded_inputs_func(t, t_cmd, thrust_cmd, w_cmd):
    """Interpolates both normalized thrust and angular velocity commands."""
    thrust_norm = jnp.interp(t, t_cmd, thrust_cmd)
    omega_cmd_x = jnp.interp(t, t_cmd, w_cmd[:, 0])
    omega_cmd_y = jnp.interp(t, t_cmd, w_cmd[:, 1])
    omega_cmd_z = jnp.interp(t, t_cmd, w_cmd[:, 2])
    omega_cmd = jnp.array([omega_cmd_x, omega_cmd_y, omega_cmd_z])
    return thrust_norm, omega_cmd

def simulation_ode_open_loop_zoh(z, t, commands, dt, mass, g_acc):
    """
    Dynamics-only ODE. State 'z' is now just (x, R_flatten).
    """
    # Unpack the constant commands for this step
    thrust_norm, Omega_cmd = commands

    # Unpack the current state
    x, R_flatten = z
    q, dq = x[:3], x[3:]
    R = R_flatten.reshape((3, 3))

    # Convert normalized thrust to force magnitude
    f_d = thrust_norm * mass * g_acc / 0.726

    # --- Dynamics (using the provided constant commands) ---
    # Rotational dynamics are driven directly by the command
    dR = R @ hat(Omega_cmd)

    # Translational dynamics are driven by thrust and current attitude
    u_applied = f_d * R @ jnp.array([0., 0., 1.])
    H, C, g, _ = prior(q, dq)
    ddq = jnp.linalg.solve(H, u_applied - C @ dq - g)
    dx = jnp.concatenate((dq, ddq))

    return dx, dR.flatten()

@partial(jax.jit, static_argnums=(2, 4, 5, 6))
def jax_flatten_wrapper_open_loop_zoh(z_flat, t, command_func, dt, mass, g_acc, z_unravel_func, t_cmd, thrust_cmd, w_cmd):
    """
    Wrapper that enforces zero-order-hold on the commanded inputs.
    """
    z_tree = z_unravel_func(z_flat)

    # 1. Determine the time at the beginning of the current discrete step
    t_hold = (t // dt) * dt

    # 2. Sample the command ONCE at the beginning of the step
    command_func_partial = partial(command_func, t_cmd=t_cmd, thrust_cmd=thrust_cmd, w_cmd=w_cmd)
    held_commands = command_func_partial(t_hold)

    # 3. Call the dynamics function with the HELD command
    dz_tree = simulation_ode_open_loop_zoh(z_tree, t, held_commands, dt, mass, g_acc)

    return jnp.concatenate(jax.tree_util.tree_leaves(dz_tree))

def run_jax_open_loop_simulation(initial_pose_msg, commanded_inputs, dt, mass):
    """
    Runs the JAX open-loop simulation using a ZOH on commands and a simplified state.
    """
    t_cmd, thrust_cmd, w_cmd = commanded_inputs
    T_FINAL = t_cmd[-1] if len(t_cmd) > 0 else 0

    if not initial_pose_msg:
        print("Error: Could not get initial pose from rosbag.")
        return None, None

    # --- Initial Conditions ---
    p = initial_pose_msg.pose.position
    o = initial_pose_msg.pose.orientation
    r0_jax = jnp.array([p.x, p.y, p.z])
    dr0_jax = jnp.zeros(3) # Assume starting from hover
    x0_jax = jnp.concatenate([r0_jax, dr0_jax])
    q_init = np.array([o.x, o.y, o.z, o.w])
    R0_jax = jnp.array(Rotation.from_quat(q_init).as_matrix())

    # --- MODIFIED: Simplified initial state tree ---
    z0_tree = (x0_jax, R0_jax.flatten())
    z0_flat, z_unravel_func = jax.flatten_util.ravel_pytree(z0_tree)

    # --- Define the single-step integration function for ZOH ---
    @jax.jit
    def step_func(z_flat, t):
        # 1. Sample the command ONCE at the start of the step (t)
        command_func_partial = partial(commanded_inputs_func, t_cmd=t_cmd, thrust_cmd=thrust_cmd, w_cmd=w_cmd)
        held_commands = command_func_partial(t)

        # 2. Define the dynamics function for the integrator
        ode_func_partial = partial(simulation_ode_open_loop_zoh, commands=held_commands, dt=dt, mass=mass, g_acc=9.81)

        # 3. Integrate dynamics over one time step
        from utils import rk38_step 

        # --- MODIFIED: Simplified flat_dynamics for the new state ---
        def flat_dynamics(z_flat_inner, t_inner):
            z_tree_inner = z_unravel_func(z_flat_inner)
            dz_tree_inner = ode_func_partial(z_tree_inner, t_inner)
            return jnp.concatenate(jax.tree_util.tree_leaves(dz_tree_inner))

        z_next_flat = rk38_step(flat_dynamics, dt, z_flat, t)
        return z_next_flat

    # --- Run the ZOH Simulation Loop ---
    print(f"\n--- Starting JAX Open-Loop Simulation (ZOH, Simplified State, DT={dt:.4f}s) ---")
    ts_sim = jnp.arange(0.0, T_FINAL, dt)
    z_history_flat = [z0_flat]
    current_z_flat = z0_flat

    for t in ts_sim[:-1]:
        current_z_flat = step_func(current_z_flat, t)
        z_history_flat.append(current_z_flat)

    z_history_flat = jnp.array(z_history_flat)

    # --- MODIFIED: Post-process results for the new state ---
    z_history = jax.vmap(z_unravel_func)(z_history_flat)
    x_hist, _ = z_history # Unpack only x and R, ignore Omega
    q_jax = x_hist[:, :3]
    print("--- JAX Open-Loop Simulation Complete ---")

    return ts_sim, q_jax


# --- Plotting Function ---
def plot_open_loop_comparison(gazebo_states, jax_states, output_dir="open_loop_figs"):
    (t_gazebo, q_gazebo, _) = gazebo_states
    
    # examine gazebo time
    print(f"Gazebo time: start {t_gazebo[0]:.3f}s, end {t_gazebo[-1]:.3f}s, duration {t_gazebo[-1]-t_gazebo[0]:.3f}s")
    print(f"Gazebo time stamps: {t_gazebo}")
    
    
    (ts_jax, q_jax) = jax_states
    print(f"JAX time: {ts_jax}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3D Trajectory Plot
    fig_3d = plt.figure(figsize=(10, 10))
    ax = fig_3d.add_subplot(111, projection='3d')
    ax.plot(q_jax[:, 0], q_jax[:, 1], q_jax[:, 2], 'b-', label='JAX Open-Loop Sim', lw=2)
    ax.plot(q_gazebo[:, 0], q_gazebo[:, 1], q_gazebo[:, 2], 'g:', label='Gazebo (Rosbag)', lw=2)
    ax.set_title('Open-Loop 3D Trajectory Comparison'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend(); ax.axis('equal')
    fig_3d.savefig(os.path.join(output_dir, "open_loop_3d_trajectory.png"))
    print(f"Saved 3D open-loop plot to: {os.path.join(output_dir, 'open_loop_3d_trajectory.png')}")
    
    # Position Components Plot
    fig_pos, axs_pos = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    pos_labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs_pos[i].plot(ts_jax, q_jax[:, i], 'b-', label=f'JAX {pos_labels[i]}')
        axs_pos[i].plot(t_gazebo, q_gazebo[:, i], 'g:', label=f'Gazebo {pos_labels[i]}')
        axs_pos[i].set_ylabel(f'{pos_labels[i]} (m)'); axs_pos[i].legend(); axs_pos[i].grid(True)
    axs_pos[2].set_xlabel('Time (s)'); fig_pos.suptitle('Open-Loop Position Comparison', fontsize=16)
    fig_pos.savefig(os.path.join(output_dir, "open_loop_position_vs_time.png"))
    print(f"Saved position open-loop plot to: {os.path.join(output_dir, 'open_loop_position_vs_time.png')}")

    plt.show()


# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Open-loop comparison of Gazebo and JAX dynamics.")
    parser.add_argument('--rosbag', type=str, required=True, help="Path to the rosbag directory.")
    parser.add_argument('--mass', type=float, default=2.0, help="Vehicle mass (kg).")
    parser.add_argument('--pose_topic', type=str, default="/mavros/local_position/pose")
    parser.add_argument('--control_log_topic', type=str, default="/mlac_mission_node/control_log")
    parser.add_argument('--attitude_setpoint_topic', type=str, default="/mavros/setpoint_raw/attitude")
    
    args = parser.parse_args()

    gazebo_states, commanded_inputs, init_pose = extract_open_loop_data(
        args.rosbag, args.pose_topic, args.control_log_topic, args.attitude_setpoint_topic
    )
    
    if gazebo_states[0] is not None and gazebo_states[0].size > 0:
        t_cmd = commanded_inputs[0]
        if len(t_cmd) > 1:
            command_dt = np.mean(np.diff(t_cmd))
        else:
            command_dt = 0.02
            
        ts_jax, q_jax = run_jax_open_loop_simulation(init_pose, commanded_inputs, command_dt, args.mass)

        if ts_jax is not None:
            plot_open_loop_comparison(gazebo_states, (ts_jax, q_jax))
            print("\n--- Open-Loop Comparison Complete ---")
        else:
            print("\n--- JAX Simulation Failed ---")
    else:
        print("\n--- Script Finished: No data extracted. ---")
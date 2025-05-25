# generate_traj_data.py
import numpy as np
import math
import os
import argparse # For selecting trajectory type

def generate_circle_data(T, dt, radius, center_x, center_y, alt, initial_psi=0.0):
    """
    Generates position, velocity, and yaw data for a circular trajectory
    with yaw tangent to the path.

    Args:
        T (float): Total duration of one circle revolution (seconds).
        dt (float): Time step for sampling (seconds).
        radius (float): Radius of the circle (meters).
        center_x (float): X-coordinate of the circle's center.
        center_y (float): Y-coordinate of the circle's center.
        alt (float): Constant altitude (z-coordinate).
        initial_psi (float): This argument is ignored; yaw is dynamically calculated.

    Returns:
        numpy.ndarray: An array of shape (N, 8) with columns
                       [time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, psi].
                       Returns None if T or dt is invalid.
    """
    if T <= 0 or dt <= 0:
        print("Error: Duration (T) and time step (dt) must be positive.")
        return None
    if radius <= 0:
        print("Error: Radius must be positive.")
        return None

    # Time vector
    times = np.arange(0, T + dt/2, dt) # Use dt/2 in endpoint to include T reliably
    num_steps = len(times)

    # Pre-allocate array for 8 columns
    # Columns: time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, psi
    trajectory_data = np.zeros((num_steps, 8))

    # Angular velocity (rad/s)
    omega = 2.0 * math.pi / T

    # Calculate trajectory components at each time step
    trajectory_data[:, 0] = times  # time
    theta = omega * times

    # Position (px, py, pz) - Cols 1, 2, 3
    trajectory_data[:, 1] = center_x + radius * np.cos(theta)
    trajectory_data[:, 2] = center_y + radius * np.sin(theta)
    trajectory_data[:, 3] = alt

    # Velocity (vx, vy, vz) - Cols 4, 5, 6
    trajectory_data[:, 4] = -radius * omega * np.sin(theta)
    trajectory_data[:, 5] = radius * omega * np.cos(theta)
    trajectory_data[:, 6] = 0.0

    # Yaw (psi) - tangent to path - Col 7
    # psi = atan2(vy, vx) which simplifies to theta + pi/2
    trajectory_data[:, 7] = theta + np.pi / 2.0
    # Optional: Normalize psi to [-pi, pi] for consistency if desired.
    # trajectory_data[:, 7] = (trajectory_data[:, 7] + np.pi) % (2 * np.pi) - np.pi

    return trajectory_data


def generate_setpoint_hold_trajectory(dt, setpoint_x, setpoint_y, setpoint_z, duration, initial_psi=0.0):
    """
    Generates a trajectory file for holding a single setpoint for a specified duration.
    Velocities and accelerations are zero. Yaw is constant.

    Args:
        dt (float): Time step for sampling (seconds).
        setpoint_x (float): Target X-coordinate.
        setpoint_y (float): Target Y-coordinate.
        setpoint_z (float): Target Z-coordinate.
        duration (float): Total duration to hold the setpoint (seconds).
        initial_psi (float): Constant yaw angle (radians).

    Returns:
        numpy.ndarray: An array of shape (N, 8) with columns
                       [time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, psi].
                       Returns None if duration or dt is invalid.
    """
    if duration <= 0 or dt <= 0:
        print("Error: Duration and time step (dt) must be positive.")
        return None

    times = np.arange(0, duration + dt/2, dt) # Use dt/2 in endpoint to include duration reliably
    num_steps = len(times)

    # Columns: time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, psi
    trajectory_data = np.zeros((num_steps, 8))

    trajectory_data[:, 0] = times          # time
    trajectory_data[:, 1] = setpoint_x     # pos_x
    trajectory_data[:, 2] = setpoint_y     # pos_y
    trajectory_data[:, 3] = setpoint_z     # pos_z
    # Columns 4, 5, 6 (vel_x, vel_y, vel_z) remain zero by default
    trajectory_data[:, 7] = initial_psi    # psi

    return trajectory_data


# --- Main execution part ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate different types of trajectories.")
    parser.add_argument(
        '--type', type=str, default='circle',
        choices=['circle', 'setpoint_hold'], # Added new type
        help='Type of trajectory to generate.'
    )
    # Add arguments for setpoint_hold if needed, or use hardcoded values for now
    parser.add_argument('--pos_x', type=float, default=1.0, help="Setpoint X for setpoint_hold")
    parser.add_argument('--pos_y', type=float, default=1.0, help="Setpoint Y for setpoint_hold")
    parser.add_argument('--pos_z', type=float, default=4.0, help="Setpoint Z for setpoint_hold")
    parser.add_argument('--duration', type=float, default=20.0, help="Duration for setpoint_hold (seconds)")

    args = parser.parse_args()

    TARGET_RATE = 50.0  # Hz
    DT = 1.0 / TARGET_RATE
    INITIAL_YAW_CONSTANT = 0.0 # Used by takeoff_hover_land and setpoint_hold

    script_dir = os.path.dirname(os.path.realpath(__file__))
    package_traj_data_dir = os.path.abspath(os.path.join(script_dir, "..", 'src/mlac_sim', "traj_data"))
    
    print(f"Attempting to save trajectories to: {package_traj_data_dir}")
    os.makedirs(package_traj_data_dir, exist_ok=True)
    absolute_output_dir = package_traj_data_dir


    if args.type == 'circle':
        DURATION_ONE_LAP = 20.0 
        RADIUS = 2.0
        CENTER_X = 0.0
        CENTER_Y = 0.0
        ALTITUDE = 2.0
        OUTPUT_FILENAME = f"circle_tangential_yaw_r{RADIUS}_t{DURATION_ONE_LAP}s_alt{ALTITUDE}_{TARGET_RATE:.0f}hz_8col.npy"

        print(f"Generating 8-column circle trajectory with tangential yaw: R={RADIUS}m, T_lap={DURATION_ONE_LAP}s, Alt={ALTITUDE}m at {TARGET_RATE}Hz")
        trajectory_array = generate_circle_data(
            T=DURATION_ONE_LAP, dt=DT, radius=RADIUS,
            center_x=CENTER_X, center_y=CENTER_Y, alt=ALTITUDE
        )
    
    elif args.type == 'setpoint_hold':
        SETPOINT_X = args.pos_x
        SETPOINT_Y = args.pos_y
        SETPOINT_Z = args.pos_z
        DURATION = args.duration # Use the parsed duration
        OUTPUT_FILENAME = f"setpoint_hold_x{SETPOINT_X}_y{SETPOINT_Y}_z{SETPOINT_Z}_t{DURATION}s_{TARGET_RATE:.0f}hz_8col.npy"
        
        print(f"Generating setpoint hold trajectory: Target=({SETPOINT_X},{SETPOINT_Y},{SETPOINT_Z})m, Duration={DURATION}s, Psi={INITIAL_YAW_CONSTANT}rad at {TARGET_RATE}Hz")
        trajectory_array = generate_setpoint_hold_trajectory(
            dt=DT,
            setpoint_x=SETPOINT_X, setpoint_y=SETPOINT_Y, setpoint_z=SETPOINT_Z,
            duration=DURATION, 
            initial_psi=INITIAL_YAW_CONSTANT
        )
    else:
        print(f"Unknown trajectory type: {args.type}")
        trajectory_array = None

    # --- Save ---
    if trajectory_array is not None and trajectory_array.shape[0] > 0 :
        filepath = os.path.join(absolute_output_dir, OUTPUT_FILENAME)
        print(f"Saving trajectory data to {filepath}...")
        np.save(filepath, trajectory_array)
        print("Done.")
        print(f"Data shape: {trajectory_array.shape}") 
        print(f"Trajectory duration: {trajectory_array[-1, 0]:.3f}s")
        print(f"Columns: [t, px, py, pz, vx, vy, vz, psi]")
    elif trajectory_array is not None and trajectory_array.shape[0] == 0:
        print("Generated trajectory is empty, not saving.")
    else:
        print("Failed to generate trajectory.")
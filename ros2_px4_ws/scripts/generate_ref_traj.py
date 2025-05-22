# generate_traj_data.py
import numpy as np
import math
import os
import argparse # For selecting trajectory type

def generate_circle_data(T, dt, radius, center_x, center_y, alt, initial_psi=0.0):
    """
    Generates position, velocity, and yaw data for a circular trajectory.

    Args:
        T (float): Total duration of one circle revolution (seconds).
        dt (float): Time step for sampling (seconds).
        radius (float): Radius of the circle (meters).
        center_x (float): X-coordinate of the circle's center.
        center_y (float): Y-coordinate of the circle's center.
        alt (float): Constant altitude (z-coordinate).
        initial_psi (float): Desired constant yaw angle (radians).

    Returns:
        numpy.ndarray: An array of shape (N, 8) with columns
                       [time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, psi].
                       Returns None if T or dt is invalid.
    """
    if T <= 0 or dt <= 0:
        print("Error: Duration (T) and time step (dt) must be positive.")
        return None

    # Time vector
    times = np.arange(0, T + dt/2, dt) # Use dt/2 in endpoint to include T reliably
    num_steps = len(times)

    # Pre-allocate array
    # Columns: time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, psi
    trajectory_data = np.zeros((num_steps, 8))

    # Angular velocity (rad/s)
    omega = 2.0 * math.pi / T

    # Calculate position and velocity at each time step
    trajectory_data[:, 0] = times                                 # time
    theta = omega * times
    trajectory_data[:, 1] = center_x + radius * np.cos(theta)     # pos_x
    trajectory_data[:, 2] = center_y + radius * np.sin(theta)     # pos_y
    trajectory_data[:, 3] = alt                                   # pos_z (constant)
    trajectory_data[:, 4] = -radius * omega * np.sin(theta)       # vel_x
    trajectory_data[:, 5] = radius * omega * np.cos(theta)        # vel_y
    trajectory_data[:, 6] = 0.0                                   # vel_z (constant)
    trajectory_data[:, 7] = initial_psi                           # psi (constant yaw)

    return trajectory_data

def generate_takeoff_hover_land_data(dt, hover_x, hover_y, hover_z, hover_duration,
                                      takeoff_duration, land_duration,
                                      start_x=0.0, start_y=0.0, start_z=0.0, initial_psi=0.0):
    """
    Generates a trajectory for takeoff, hover at a specified point, and then land.
    Includes a constant yaw angle.

    Args:
        dt (float): Time step for sampling (seconds).
        hover_x, hover_y, hover_z (float): Target hover coordinates.
        hover_duration (float): Duration to hover at the target point (seconds).
        takeoff_duration (float): Duration for the takeoff phase (seconds).
        land_duration (float): Duration for the landing phase (seconds).
        start_x, start_y, start_z (float): Starting position (typically 0,0,0).
        initial_psi (float): Desired constant yaw angle (radians).

    Returns:
        numpy.ndarray: An array of shape (N, 8) with columns
                       [time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, psi].
    """
    if dt <= 0 or takeoff_duration <= 0 or land_duration <= 0 or hover_duration < 0:
        print("Error: Durations and dt must be positive (hover_duration can be zero).")
        return None

    segments = []
    current_time = 0.0
    num_cols = 8 # time, p, v, psi

    # --- 1. Takeoff Phase (from start_pos to hover_pos) ---
    takeoff_steps = int(takeoff_duration / dt)
    if takeoff_steps > 0:
        takeoff_times = np.linspace(0, takeoff_duration, takeoff_steps, endpoint=False)
        pos_x_takeoff = np.linspace(start_x, hover_x, takeoff_steps)
        pos_y_takeoff = np.linspace(start_y, hover_y, takeoff_steps)
        pos_z_takeoff = np.linspace(start_z, hover_z, takeoff_steps)

        vel_x_takeoff = np.full(takeoff_steps, (hover_x - start_x) / takeoff_duration)
        vel_y_takeoff = np.full(takeoff_steps, (hover_y - start_y) / takeoff_duration)
        vel_z_takeoff = np.full(takeoff_steps, (hover_z - start_z) / takeoff_duration)
        psi_takeoff = np.full(takeoff_steps, initial_psi)

        segment_takeoff = np.vstack([
            takeoff_times + current_time,
            pos_x_takeoff, pos_y_takeoff, pos_z_takeoff,
            vel_x_takeoff, vel_y_takeoff, vel_z_takeoff,
            psi_takeoff
        ]).T
        segments.append(segment_takeoff)
        current_time += takeoff_duration

    # --- 2. Hover Phase ---
    hover_steps = int(hover_duration / dt)
    if hover_steps > 0:
        hover_times = np.linspace(0, hover_duration, hover_steps, endpoint=False)
        segment_hover = np.zeros((hover_steps, num_cols))
        segment_hover[:, 0] = hover_times + current_time
        segment_hover[:, 1] = hover_x
        segment_hover[:, 2] = hover_y
        segment_hover[:, 3] = hover_z
        # Velocities are zero during hover (cols 4,5,6)
        segment_hover[:, 7] = initial_psi # Constant psi
        segments.append(segment_hover)
        current_time += hover_duration

    # --- 3. Land Phase (from hover_pos to land_pos, assumed to be (hover_x, hover_y, start_z)) ---
    land_steps = int(land_duration / dt)
    if land_steps > 0:
        # Ensure last point is included for landing
        land_times = np.linspace(0, land_duration, land_steps, endpoint=True)
        pos_x_land = np.full(land_steps, hover_x)
        pos_y_land = np.full(land_steps, hover_y)
        pos_z_land = np.linspace(hover_z, start_z, land_steps)

        vel_x_land = np.zeros(land_steps)
        vel_y_land = np.zeros(land_steps)
        vel_z_land = np.full(land_steps, (start_z - hover_z) / land_duration)
        psi_land = np.full(land_steps, initial_psi)

        # Ensure final velocity is zero if it's the absolute end
        if land_steps > 1 : vel_z_land[-1] = 0.0

        segment_land = np.vstack([
            land_times + current_time,
            pos_x_land, pos_y_land, pos_z_land,
            vel_x_land, vel_y_land, vel_z_land,
            psi_land
        ]).T
        segments.append(segment_land)

    if not segments:
        return np.zeros((0, num_cols))

    trajectory_data = np.concatenate(segments, axis=0)
    
    # Ensure time is monotonic and starts from 0, recalculating based on dt
    # This handles potential floating point issues from concatenating linspace results
    total_steps = trajectory_data.shape[0]
    trajectory_data[:,0] = np.arange(total_steps) * dt

    return trajectory_data

# --- Main execution part ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate different types of trajectories.")
    parser.add_argument(
        '--type', type=str, default='takeoff_hover_land',
        choices=['circle', 'takeoff_hover_land'],
        help='Type of trajectory to generate.'
    )
    args = parser.parse_args()

    TARGET_RATE = 50.0  # Hz
    DT = 1.0 / TARGET_RATE
    INITIAL_YAW = 0.0 # Default initial yaw for trajectories

    # Common output directory (save to source space for easy inclusion by setup.py)
    # Adjust this path if your script is not in the same directory level as mlac_sim package
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Assuming mlac_sim is a sibling directory to where this script might be (e.g. if script is in a 'scripts' folder)
    # For direct use within basic_offboard, or if you move this script into mlac_sim:
    # For saving into mlac_sim's source:
    # Modify this to point to your mlac_sim/traj_data source directory correctly
    # Example: if this script is in <ws>/src/basic_offboard/basic_offboard/
    # and mlac_sim is in <ws>/src/mlac_sim/
    ws_root = os.path.abspath(os.path.join(script_dir, "..")) # Adjust based on actual location
    relative_output_dir_to_ws_src = os.path.join("src", "mlac_sim", "traj_data")
    absolute_output_dir = os.path.join(ws_root, relative_output_dir_to_ws_src)
    
    # Fallback if ws_root logic is complex or script is run from arbitrary location:
    # You might need to hardcode or pass the target directory as an argument
    # For simplicity, let's assume a direct path for now if the above is tricky:
    # relative_output_dir = os.path.join("~", "mlac_px4", "ros2_px4_ws", "src", "mlac_sim", "traj_data")
    # absolute_output_dir = os.path.expanduser(relative_output_dir)

    print(f"Attempting to save trajectories to: {absolute_output_dir}")
    os.makedirs(absolute_output_dir, exist_ok=True)


    if args.type == 'circle':
        # Circle parameters
        DURATION = 20.0
        RADIUS = 2.0
        CENTER_X = 0.0
        CENTER_Y = 0.0
        ALTITUDE = 1.5
        OUTPUT_FILENAME = "circle_trajectory_8col_50hz.npy" # New filename

        print(f"Generating circle trajectory: R={RADIUS}m, T={DURATION}s, Alt={ALTITUDE}m, Psi={INITIAL_YAW}rad at {TARGET_RATE}Hz (dt={DT:.3f}s)")
        trajectory_array = generate_circle_data(
            T=DURATION, dt=DT, radius=RADIUS,
            center_x=CENTER_X, center_y=CENTER_Y, alt=ALTITUDE, initial_psi=INITIAL_YAW
        )
    elif args.type == 'takeoff_hover_land':
        # Takeoff, Hover, Land parameters
        HOVER_X = 0.0
        HOVER_Y = 0.0
        HOVER_Z = 2.0
        HOVER_DURATION = 10.0
        TAKEOFF_DURATION = 4.0
        LAND_DURATION = 5.0
        OUTPUT_FILENAME = "takeoff_hover_land_8col_50hz.npy" # New filename

        print(f"Generating takeoff-hover-land trajectory: Target=({HOVER_X},{HOVER_Y},{HOVER_Z})m, HoverTime={HOVER_DURATION}s, Psi={INITIAL_YAW}rad at {TARGET_RATE}Hz")
        trajectory_array = generate_takeoff_hover_land_data(
            dt=DT,
            hover_x=HOVER_X, hover_y=HOVER_Y, hover_z=HOVER_Z,
            hover_duration=HOVER_DURATION,
            takeoff_duration=TAKEOFF_DURATION,
            land_duration=LAND_DURATION,
            initial_psi=INITIAL_YAW
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
        print(f"Data shape: {trajectory_array.shape}") # Should be (N, 8)
        print(f"Trajectory duration: {trajectory_array[-1, 0]:.3f}s")
    elif trajectory_array is not None and trajectory_array.shape[0] == 0:
        print("Generated trajectory is empty, not saving.")
    else:
        print("Failed to generate trajectory.")

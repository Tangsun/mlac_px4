# generate_traj_data.py
import numpy as np
import math
import os
import argparse # For selecting trajectory type

def generate_circle_data(T, dt, radius, center_x, center_y, alt, initial_psi=0.0): # initial_psi is ignored for tangential yaw
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

def generate_takeoff_hover_land_data(dt, hover_x, hover_y, hover_z, hover_duration,
                                      takeoff_duration, land_duration,
                                      start_x=0.0, start_y=0.0, start_z=0.0, initial_psi=0.0):
    """
    Generates a trajectory for takeoff, hover at a specified point, and then land.
    Includes a constant yaw angle. Output is 8 columns.
    (This function remains unchanged from your provided version)
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
    
    total_steps = trajectory_data.shape[0]
    if total_steps > 0:
        trajectory_data[:,0] = np.arange(total_steps) * dt


    return trajectory_data

# --- Main execution part ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate different types of trajectories.")
    parser.add_argument(
        '--type', type=str, default='circle',
        choices=['circle', 'takeoff_hover_land'],
        help='Type of trajectory to generate.'
    )
    args = parser.parse_args()

    TARGET_RATE = 50.0  # Hz
    DT = 1.0 / TARGET_RATE
    INITIAL_YAW_FOR_TAKEOFF_LAND = 0.0 # Used by takeoff_hover_land

    # Output directory logic (from your script)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Simplified path assumption: script is in a 'scripts' folder, and 'mlac_sim' is a sibling to that 'scripts' folder's parent.
    # Or, if script is in 'mlac_sim/scripts/', then 'traj_data' is '../traj_data'.
    # Assuming the script is in a directory like `ros2_ws/src/your_scripts_pkg/scripts`
    # and you want to save to `ros2_ws/src/mlac_sim/traj_data`
    # This requires knowing the relative structure or making `absolute_output_dir` configurable.
    # For now, using a path relative to the script, assuming it's in `mlac_sim/scripts/`
    package_traj_data_dir = os.path.abspath(os.path.join(script_dir, "..", 'src/mlac_sim', "traj_data"))
    # if not os.path.exists(os.path.join(script_dir, "..", "package.xml")): # Simple check if script is in pkg/scripts
    #     # Fallback: If not in pkg/scripts, try to find mlac_sim relative to a presumed workspace structure
    #     ws_root_candidate = os.path.abspath(os.path.join(script_dir, "..")) # Go up if script is deep
    #     potential_dir = os.path.join(ws_root_candidate, "src", "mlac_sim", "traj_data")
    #     if os.path.exists(os.path.join(ws_root_candidate, "src", "mlac_sim")):
    #         package_traj_data_dir = potential_dir
    #     else: # Default to a local 'traj_data' folder if structure is not as expected
    #         package_traj_data_dir = os.path.join(script_dir, "traj_data_output")


    print(f"Attempting to save trajectories to: {package_traj_data_dir}")
    os.makedirs(package_traj_data_dir, exist_ok=True)
    absolute_output_dir = package_traj_data_dir


    if args.type == 'circle':
        # Circle parameters
        DURATION_ONE_LAP = 20.0 # Time for one lap
        RADIUS = 2.0
        CENTER_X = 0.0
        CENTER_Y = 0.0
        ALTITUDE = 2.0
        # Updated filename for 8-column tangential yaw circle
        OUTPUT_FILENAME = f"circle_tangential_yaw_r{RADIUS}_t{DURATION_ONE_LAP}s_alt{ALTITUDE}_{TARGET_RATE:.0f}hz_8col.npy"

        print(f"Generating 8-column circle trajectory with tangential yaw: R={RADIUS}m, T_lap={DURATION_ONE_LAP}s, Alt={ALTITUDE}m at {TARGET_RATE}Hz")
        trajectory_array = generate_circle_data(
            T=DURATION_ONE_LAP, dt=DT, radius=RADIUS,
            center_x=CENTER_X, center_y=CENTER_Y, alt=ALTITUDE
        ) # initial_psi is ignored by the new function logic
    elif args.type == 'takeoff_hover_land':
        # Takeoff, Hover, Land parameters
        HOVER_X = 0.0
        HOVER_Y = 0.0
        HOVER_Z = 2.0
        HOVER_DURATION = 10.0
        TAKEOFF_DURATION = 4.0
        LAND_DURATION = 5.0
        OUTPUT_FILENAME = f"takeoff_hover_land_h{HOVER_Z}m_t{HOVER_DURATION}s_{TARGET_RATE:.0f}hz_8col.npy" # Original filename

        print(f"Generating takeoff-hover-land trajectory: Target=({HOVER_X},{HOVER_Y},{HOVER_Z})m, HoverTime={HOVER_DURATION}s, Psi={INITIAL_YAW_FOR_TAKEOFF_LAND}rad at {TARGET_RATE}Hz")
        trajectory_array = generate_takeoff_hover_land_data(
            dt=DT,
            hover_x=HOVER_X, hover_y=HOVER_Y, hover_z=HOVER_Z,
            hover_duration=HOVER_DURATION,
            takeoff_duration=TAKEOFF_DURATION,
            land_duration=LAND_DURATION,
            initial_psi=INITIAL_YAW_FOR_TAKEOFF_LAND
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
        print(f"Columns: [t, px, py, pz, vx, vy, vz, psi]") # Correct for 8 columns
    elif trajectory_array is not None and trajectory_array.shape[0] == 0:
        print("Generated trajectory is empty, not saving.")
    else:
        print("Failed to generate trajectory.")
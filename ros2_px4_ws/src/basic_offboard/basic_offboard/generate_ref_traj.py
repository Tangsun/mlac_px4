# generate_traj_data.py
import numpy as np
import math
import os

def generate_circle_data(T, dt, radius, center_x, center_y, alt):
    """
    Generates position and velocity data for a circular trajectory.

    Args:
        T (float): Total duration of one circle revolution (seconds).
        dt (float): Time step for sampling (seconds).
        radius (float): Radius of the circle (meters).
        center_x (float): X-coordinate of the circle's center.
        center_y (float): Y-coordinate of the circle's center.
        alt (float): Constant altitude (z-coordinate).

    Returns:
        numpy.ndarray: An array of shape (N, 7) with columns
                       [time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z].
                       Returns None if T or dt is invalid.
    """
    if T <= 0 or dt <= 0:
        print("Error: Duration (T) and time step (dt) must be positive.")
        return None

    # Time vector
    times = np.arange(0, T + dt/2, dt) # Use dt/2 in endpoint to include T reliably
    num_steps = len(times)

    # Pre-allocate array
    # Columns: time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
    trajectory_data = np.zeros((num_steps, 7))

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

    return trajectory_data

# --- Main execution part ---
if __name__ == "__main__":
    # --- Parameters ---
    TARGET_RATE = 50.0  # Hz (matches desired controller rate)
    DT = 1.0 / TARGET_RATE
    DURATION = 20.0     # Total duration of trajectory (seconds)
    RADIUS = 2.0        # Radius of the circle (meters)
    CENTER_X = 0.0      # Center X (relative to initial hover/origin)
    CENTER_Y = 0.0      # Center Y (relative to initial hover/origin)
    ALTITUDE = 1.5      # Hover/trajectory altitude (meters)

    OUTPUT_FILENAME = "circle_trajectory_50hz.npy"
    OUTPUT_DIR = "./traj_data" # Save in the current directory, or specify full path

    # --- Generate ---
    print(f"Generating circle trajectory: R={RADIUS}m, T={DURATION}s, Alt={ALTITUDE}m at {TARGET_RATE}Hz (dt={DT:.3f}s)")
    trajectory_array = generate_circle_data(
        T=DURATION, dt=DT, radius=RADIUS,
        center_x=CENTER_X, center_y=CENTER_Y, alt=ALTITUDE
    )

    # --- Save ---
    if trajectory_array is not None:
        filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        # Ensure output directory exists if specified
        # os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Saving trajectory data to {filepath}...")
        np.save(filepath, trajectory_array)
        print("Done.")
        print(f"Data shape: {trajectory_array.shape}")
        print(f"Final time in data: {trajectory_array[-1, 0]:.3f}s")
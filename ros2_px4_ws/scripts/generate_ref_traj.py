import numpy as np
import math
import os
import argparse 

def euler_from_quaternion(w, x, y, z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def generate_circle_data(T, dt, radius, center_x, center_y, alt, initial_yaw_rad=0.0, num_cols_output=8):
    if T <= 0 or dt <= 0:
        print("Error: Duration (T) and time step (dt) must be positive.")
        return None
    if radius <= 0:
        print("Error: Radius must be positive.")
        return None

    times = np.arange(0, T + dt/2, dt) 
    num_steps = len(times)
    trajectory_data = np.zeros((num_steps, num_cols_output))
    trajectory_data[:, 0] = times

    omega = 2.0 * math.pi / T 
    theta_pos_start = initial_yaw_rad - np.pi / 2.0
    theta_for_pos = omega * times + theta_pos_start

    # Position (px, py, pz) - Cols 1, 2, 3
    trajectory_data[:, 1] = center_x + radius * np.cos(theta_for_pos)
    trajectory_data[:, 2] = center_y + radius * np.sin(theta_for_pos)
    trajectory_data[:, 3] = alt

    # Velocity (vx, vy, vz) - Cols 4, 5, 6
    trajectory_data[:, 4] = -radius * omega * np.sin(theta_for_pos) 
    trajectory_data[:, 5] = radius * omega * np.cos(theta_for_pos)  
    trajectory_data[:, 6] = 0.0 
    
    # Yaw (psi) - tangent to path - Col 7
    # calculated_psi = theta_for_pos + np.pi / 2.0
    # trajectory_data[:, 7] = (calculated_psi + np.pi) % (2 * np.pi) - np.pi
    trajectory_data[:, 7] = initial_yaw_rad
    
    # Acceleration (ax, ay, az) - Cols 8, 9, 10
    if num_cols_output >= 11:
        trajectory_data[:, 8] = -radius * omega**2 * np.cos(theta_for_pos)  
        trajectory_data[:, 9] = -radius * omega**2 * np.sin(theta_for_pos)  
        trajectory_data[:, 10] = 0.0 
    
    # Jerk (jx, jy, jz) - Cols 11, 12, 13
    if num_cols_output >= 14:
        trajectory_data[:, 11] = radius * omega**3 * np.sin(theta_for_pos)
        trajectory_data[:, 12] = -radius * omega**3 * np.cos(theta_for_pos)
        trajectory_data[:, 13] = 0.0
        
    # Desired Yaw Rate (dpsi) - Col 14 (if num_cols_output >= 15) or Col 8 (if num_cols_output == 9)
    if num_cols_output >= 15:
        trajectory_data[:, 14] = omega
    elif num_cols_output == 9 : 
         trajectory_data[:, 8] = omega # dpsi in 9th column (index 8)
    return trajectory_data

def generate_setpoint_hold_trajectory(dt, setpoint_x, setpoint_y, setpoint_z, duration, initial_psi_rad=0.0, num_cols_output=8):
    if duration <= 0 or dt <= 0:
        print("Error: Duration and time step (dt) must be positive.")
        return None
    times = np.arange(0, duration + dt/2, dt) 
    num_steps = len(times)
    trajectory_data = np.zeros((num_steps, num_cols_output))

    trajectory_data[:, 0] = times        
    trajectory_data[:, 1] = setpoint_x   
    trajectory_data[:, 2] = setpoint_y   
    trajectory_data[:, 3] = setpoint_z   
    # Velocities (cols 4,5,6) are already zero
    trajectory_data[:, 7] = initial_psi_rad 
    # Accelerations (cols 8,9,10 if num_cols_output >= 11) are already zero
    # Jerks (cols 11,12,13 if num_cols_output >= 14) are already zero
    # dPsi (col 14 if num_cols_output >= 15, or col 8 if num_cols_output == 9) is already zero
    return trajectory_data

def generate_setpoint_rotating_yaw_trajectory(dt, setpoint_x, setpoint_y, setpoint_z, duration, 
                                             initial_yaw_rad=0.0, yaw_rate_rps=0.0, num_cols_output=8):
    if duration <= 0 or dt <= 0:
        print("Error: Duration and time step (dt) must be positive.")
        return None
    times = np.arange(0, duration + dt/2, dt)
    num_steps = len(times)
    trajectory_data = np.zeros((num_steps, num_cols_output))

    trajectory_data[:, 0] = times          
    trajectory_data[:, 1] = setpoint_x     
    trajectory_data[:, 2] = setpoint_y     
    trajectory_data[:, 3] = setpoint_z     
    # Velocities (cols 4,5,6) are zero
    # Accelerations (cols 8,9,10 if num_cols_output >= 11) are zero
    # Jerks (cols 11,12,13 if num_cols_output >= 14) are zero
    
    psi_values = initial_yaw_rad + yaw_rate_rps * times
    trajectory_data[:, 7] = (psi_values + np.pi) % (2 * np.pi) - np.pi # psi in col 7

    if num_cols_output >= 15:
        trajectory_data[:, 14] = yaw_rate_rps # dpsi in col 14
    elif num_cols_output == 9: 
        trajectory_data[:, 8] = yaw_rate_rps # dpsi in col 8
        
    return trajectory_data

# --- Main execution part ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate different types of trajectories.")
    parser.add_argument(
        '--type', type=str, default='circle',
        choices=['circle', 'setpoint_hold', 'setpoint_rotating_yaw'], 
        help='Type of trajectory to generate.'
    )
    parser.add_argument('--pos_x', type=float, default=0.0, help="Setpoint X or Center X for circle")
    parser.add_argument('--pos_y', type=float, default=0.0, help="Setpoint Y or Center Y for circle")
    parser.add_argument('--pos_z', type=float, default=2.0, help="Setpoint Z or Altitude for circle")
    parser.add_argument('--duration', type=float, default=20.0, help="Duration for trajectory (seconds)")
    parser.add_argument('--radius', type=float, default=2.0, help="Radius for circle trajectory (meters)")
    parser.add_argument('--dt_step', type=float, default=0.02, help="Time step for sampling (seconds), e.g., 0.02 for 50Hz.")
    parser.add_argument('--initial_yaw_deg', type=float, default=0.0, 
                        help="Initial yaw in degrees.")
    parser.add_argument('--yaw_rate_dps', type=float, default=15.0, 
                        help="Yaw rotation speed in degrees per second (for setpoint_rotating_yaw).")
    parser.add_argument('--num_cols', type=int, default=11, choices=[8, 9, 11, 15], # Defaulting to 11 to include accelerations
                        help="Number of columns for the output .npy file. "
                             "8=(t,p,v,psi); "
                             "9=(t,p,v,psi,dpsi_col8); "
                             "11=(t,p,v,psi,a); "
                             "15=(t,p,v,psi,a,j,dpsi_col14).")

    args = parser.parse_args()

    TARGET_RATE = 1.0 / args.dt_step if args.dt_step > 0 else 50.0
    DT = args.dt_step
    
    initial_yaw_rad_from_args = np.deg2rad(args.initial_yaw_deg)
    yaw_rate_rps_from_args = np.deg2rad(args.yaw_rate_dps)

    print(f"Using initial_yaw_rad: {initial_yaw_rad_from_args:.3f} (from {args.initial_yaw_deg:.1f} deg)")
    if args.type == 'setpoint_rotating_yaw':
        print(f"Using yaw_rate_rps: {yaw_rate_rps_from_args:.3f} (from {args.yaw_rate_dps:.1f} deg/s)")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    package_traj_data_dir = os.path.abspath(os.path.join(script_dir, '..', 'src/mlac_sim', "traj_data"))
    
    print(f"Attempting to save trajectories to: {package_traj_data_dir}")
    os.makedirs(package_traj_data_dir, exist_ok=True)
    absolute_output_dir = package_traj_data_dir

    trajectory_array = None
    OUTPUT_FILENAME = "default_trajectory.npy"

    if args.type == 'circle':
        DURATION_ONE_LAP = args.duration
        RADIUS = args.radius
        CENTER_X = args.pos_x 
        CENTER_Y = args.pos_y
        ALTITUDE = args.pos_z
        
        OUTPUT_FILENAME = f"circle_r{RADIUS}_t{DURATION_ONE_LAP}s_alt{ALTITUDE}_initpsi{args.initial_yaw_deg:.0f}deg_{TARGET_RATE:.0f}hz_{args.num_cols}col.npy"
        print(f"Generating {args.num_cols}-column circle trajectory...")
        trajectory_array = generate_circle_data(
            T=DURATION_ONE_LAP, dt=DT, radius=RADIUS,
            center_x=CENTER_X, center_y=CENTER_Y, alt=ALTITUDE,
            initial_yaw_rad=initial_yaw_rad_from_args,
            num_cols_output=args.num_cols
        )
    
    elif args.type == 'setpoint_hold':
        SETPOINT_X = args.pos_x
        SETPOINT_Y = args.pos_y
        SETPOINT_Z = args.pos_z
        DURATION = args.duration
        OUTPUT_FILENAME = f"setpoint_hold_x{SETPOINT_X}_y{SETPOINT_Y}_z{SETPOINT_Z}_t{DURATION}s_psi{args.initial_yaw_deg:.0f}deg_{TARGET_RATE:.0f}hz_{args.num_cols}col.npy"
        print(f"Generating {args.num_cols}-column setpoint hold trajectory...")
        trajectory_array = generate_setpoint_hold_trajectory(
            dt=DT,
            setpoint_x=SETPOINT_X, setpoint_y=SETPOINT_Y, setpoint_z=SETPOINT_Z,
            duration=DURATION, 
            initial_psi_rad=initial_yaw_rad_from_args,
            num_cols_output=args.num_cols
        )
    elif args.type == 'setpoint_rotating_yaw':
        SETPOINT_X = args.pos_x
        SETPOINT_Y = args.pos_y
        SETPOINT_Z = args.pos_z 
        DURATION = args.duration
        OUTPUT_FILENAME = (f"setpoint_rot_yaw_x{SETPOINT_X}_y{SETPOINT_Y}_z{SETPOINT_Z}_t{DURATION}s_"
                           f"initpsi{args.initial_yaw_deg:.0f}deg_rate{args.yaw_rate_dps:.0f}dps_"
                           f"{TARGET_RATE:.0f}hz_{args.num_cols}col.npy")
        print(f"Generating {args.num_cols}-column setpoint_rotating_yaw trajectory...")
        trajectory_array = generate_setpoint_rotating_yaw_trajectory(
            dt=DT,
            setpoint_x=SETPOINT_X, setpoint_y=SETPOINT_Y, setpoint_z=SETPOINT_Z,
            duration=DURATION, 
            initial_yaw_rad=initial_yaw_rad_from_args,
            yaw_rate_rps=yaw_rate_rps_from_args,
            num_cols_output=args.num_cols
        )
    else:
        print(f"Unknown trajectory type: {args.type}")

    if trajectory_array is not None and trajectory_array.shape[0] > 0 :
        filepath = os.path.join(absolute_output_dir, OUTPUT_FILENAME)
        print(f"Saving trajectory data to {filepath}...")
        np.save(filepath, trajectory_array)
        print("Done.")
        print(f"Data shape: {trajectory_array.shape}") 
        print(f"Trajectory duration: {trajectory_array[-1, 0]:.3f}s")
        
        col_headers = ["[t", "px", "py", "pz", "vx", "vy", "vz", "psi"]
        if args.num_cols == 9: col_headers.append("dpsi_col8]")
        elif args.num_cols == 11: col_headers.extend(["ax", "ay", "az]"])
        elif args.num_cols == 15: col_headers.extend(["ax", "ay", "az", "jx", "jy", "jz", "dpsi_col14]"])
        else: col_headers[-1] += "]" 
        print(f"Columns: {' '.join(col_headers)}")

        if trajectory_array.shape[0] > 0:
            print(f"First point: {trajectory_array[0, :]}")
            print(f"Initial Px={trajectory_array[0,1]:.2f}, Py={trajectory_array[0,2]:.2f}, Pz={trajectory_array[0,3]:.2f}")
            print(f"Initial Vx={trajectory_array[0,4]:.2f}, Vy={trajectory_array[0,5]:.2f}, Vz={trajectory_array[0,6]:.2f}")
            print(f"Initial Psi={trajectory_array[0,7]:.3f} rad ({np.rad2deg(trajectory_array[0,7]):.1f} deg)")
            if args.num_cols == 9 and trajectory_array.shape[1] > 8: print(f"Initial dPsi (col 8)={trajectory_array[0,8]:.3f} rad/s")
            if args.num_cols >= 11 and trajectory_array.shape[1] > 10: print(f"Initial Ax={trajectory_array[0,8]:.2f}, Ay={trajectory_array[0,9]:.2f}, Az={trajectory_array[0,10]:.2f}")
            if args.num_cols >= 15 and trajectory_array.shape[1] > 14: print(f"Initial dPsi (col 14)={trajectory_array[0,14]:.3f} rad/s")

    elif trajectory_array is not None and trajectory_array.shape[0] == 0:
        print("Generated trajectory is empty, not saving.")
    else:
        print("Failed to generate trajectory.")
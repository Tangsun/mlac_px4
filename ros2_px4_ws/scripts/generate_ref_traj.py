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

def generate_circle_data(T, dt, radius, center_x, center_y, alt, 
                         initial_yaw_rad=0.0, num_cols_output=8, 
                         zero_dpsi_flag=False, force_zero_yaw_angle_flag=False): # Added flags
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

    trajectory_data[:, 1] = center_x + radius * np.cos(theta_for_pos)
    trajectory_data[:, 2] = center_y + radius * np.sin(theta_for_pos)
    trajectory_data[:, 3] = alt

    trajectory_data[:, 4] = -radius * omega * np.sin(theta_for_pos) 
    trajectory_data[:, 5] = radius * omega * np.cos(theta_for_pos)  
    trajectory_data[:, 6] = 0.0 
    
    # Yaw (psi) and Yaw Rate (dpsi)
    final_psi_val = 0.0
    final_dpsi_val = 0.0

    if force_zero_yaw_angle_flag:
        final_psi_val = 0.0
        final_dpsi_val = 0.0
    else:
        # Current circle implementation uses fixed yaw from initial_yaw_rad
        final_psi_val = initial_yaw_rad 
        final_dpsi_val = 0.0 # As initial_yaw_rad is constant
        # zero_dpsi_flag doesn't change this fixed yaw logic for dpsi unless tangential yaw was an option

    trajectory_data[:, 7] = final_psi_val
    
    if num_cols_output >= 11:
        trajectory_data[:, 8] = -radius * omega**2 * np.cos(theta_for_pos)  
        trajectory_data[:, 9] = -radius * omega**2 * np.sin(theta_for_pos)  
        trajectory_data[:, 10] = 0.0 
    
    if num_cols_output >= 14:
        trajectory_data[:, 11] = radius * omega**3 * np.sin(theta_for_pos)
        trajectory_data[:, 12] = -radius * omega**3 * np.cos(theta_for_pos)
        trajectory_data[:, 13] = 0.0
        
    if num_cols_output >= 15:
        trajectory_data[:, 14] = final_dpsi_val
    elif num_cols_output == 9 : 
         trajectory_data[:, 8] = final_dpsi_val
    return trajectory_data

def generate_setpoint_hold_trajectory(dt, setpoint_x, setpoint_y, setpoint_z, duration, 
                                      initial_psi_rad=0.0, num_cols_output=8, 
                                      zero_dpsi_flag=False, force_zero_yaw_angle_flag=False): # Added flags
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

    final_psi_val = 0.0
    final_dpsi_val = 0.0

    if force_zero_yaw_angle_flag:
        final_psi_val = 0.0
        # dpsi is already 0 for hold
    else:
        final_psi_val = initial_psi_rad
        # dpsi is already 0 for hold, zero_dpsi_flag is redundant here.
    
    trajectory_data[:, 7] = final_psi_val
    
    if num_cols_output >= 15:
        trajectory_data[:, 14] = final_dpsi_val 
    elif num_cols_output == 9: 
        trajectory_data[:, 8] = final_dpsi_val
    return trajectory_data

def generate_setpoint_rotating_yaw_trajectory(dt, setpoint_x, setpoint_y, setpoint_z, duration, 
                                             initial_yaw_rad=0.0, yaw_rate_rps=0.0, num_cols_output=8, 
                                             zero_dpsi_flag=False, force_zero_yaw_angle_flag=False): # Added flags
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
    
    effective_yaw_rate = yaw_rate_rps
    if zero_dpsi_flag: # This flag forces yaw rate to zero
        effective_yaw_rate = 0.0
    
    final_psi_values_array = np.zeros(num_steps)
    final_dpsi_val = 0.0

    if force_zero_yaw_angle_flag: # This flag forces yaw ANGLE to zero (and thus rate to zero)
        final_psi_values_array.fill(0.0)
        final_dpsi_val = 0.0
    else:
        psi_calc_array = initial_yaw_rad + effective_yaw_rate * times
        final_psi_values_array = (psi_calc_array + np.pi) % (2 * np.pi) - np.pi
        final_dpsi_val = effective_yaw_rate
    
    trajectory_data[:, 7] = final_psi_values_array

    if num_cols_output >= 15:
        trajectory_data[:, 14] = final_dpsi_val
    elif num_cols_output == 9: 
        trajectory_data[:, 8] = final_dpsi_val
        
    return trajectory_data

def generate_figure8_data(T_cycle, dt, width_L, height_W, center_x, center_y, alt, 
                          initial_yaw_rad=0.0, num_cols_output=8, 
                          zero_dpsi_flag=False, force_zero_yaw_angle_flag=False): # Added flags
    if T_cycle <= 0 or dt <= 0:
        print("Error: Cycle duration (T_cycle) and time step (dt) must be positive.")
        return None
    if width_L <= 0 or height_W <= 0:
        print("Error: Figure-8 width (L) and height (W) must be positive.")
        return None

    times = np.arange(0, T_cycle + dt/2, dt)
    num_steps = len(times)
    trajectory_data = np.zeros((num_steps, num_cols_output))
    trajectory_data[:, 0] = times

    omega = 2.0 * math.pi / T_cycle

    trajectory_data[:, 1] = center_x + width_L * np.sin(omega * times)
    trajectory_data[:, 2] = center_y + (height_W / 2.0) * np.sin(2 * omega * times)
    trajectory_data[:, 3] = alt

    trajectory_data[:, 4] = width_L * omega * np.cos(omega * times)
    trajectory_data[:, 5] = (height_W / 2.0) * 2 * omega * np.cos(2 * omega * times) 
    trajectory_data[:, 6] = 0.0

    final_psi_val = 0.0
    final_dpsi_val = 0.0

    if force_zero_yaw_angle_flag:
        final_psi_val = 0.0
        final_dpsi_val = 0.0
    else:
        # Current figure8 implementation uses fixed yaw from initial_yaw_rad
        final_psi_val = initial_yaw_rad
        final_dpsi_val = 0.0 # As initial_yaw_rad is constant
        # zero_dpsi_flag doesn't change this for dpsi unless tangential yaw was an option

    trajectory_data[:, 7] = final_psi_val
    
    if num_cols_output >= 11:
        trajectory_data[:, 8] = -width_L * omega**2 * np.sin(omega * times)
        trajectory_data[:, 9] = -(height_W / 2.0) * (2 * omega)**2 * np.sin(2 * omega * times) 
        trajectory_data[:, 10] = 0.0

    if num_cols_output >= 14:
        trajectory_data[:, 11] = -width_L * omega**3 * np.cos(omega * times)
        trajectory_data[:, 12] = -(height_W / 2.0) * (2 * omega)**3 * np.cos(2 * omega * times) 
        trajectory_data[:, 13] = 0.0
    
    if num_cols_output >= 15:
        trajectory_data[:, 14] = final_dpsi_val
    elif num_cols_output == 9:
        trajectory_data[:, 8] = final_dpsi_val

    return trajectory_data


# --- Main execution part ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate different types of trajectories.")
    parser.add_argument(
        '--type', type=str, default='circle',
        choices=['circle', 'setpoint_hold', 'setpoint_rotating_yaw', 'figure8'], 
        help='Type of trajectory to generate.'
    )
    # Common args
    parser.add_argument('--pos_x', type=float, default=0.0, help="Center X for circle/figure8, Setpoint X for hold/rotating_yaw")
    parser.add_argument('--pos_y', type=float, default=0.0, help="Center Y for circle/figure8, Setpoint Y for hold/rotating_yaw")
    parser.add_argument('--pos_z', type=float, default=2.0, help="Altitude for circle/figure8, Setpoint Z for hold/rotating_yaw")
    parser.add_argument('--duration', type=float, default=20.0, help="Duration for trajectory (seconds). For circle/figure8, this is T_cycle.")
    parser.add_argument('--dt_step', type=float, default=0.02, help="Time step for sampling (seconds), e.g., 0.02 for 50Hz.")
    parser.add_argument('--initial_yaw_deg', type=float, default=0.0, 
                        help="Initial yaw in degrees. Overridden if --force_zero_yaw_angle is used.")
    parser.add_argument('--num_cols', type=int, default=11, choices=[8, 9, 11, 15],
                        help="Number of columns for the output .npy file. "
                             "8=(t,p,v,psi); "
                             "9=(t,p,v,psi,dpsi_col8); "
                             "11=(t,p,v,psi,a); "
                             "15=(t,p,v,psi,a,j,dpsi_col14).")
    parser.add_argument('--zero_dpsi', action='store_true', 
                        help="Force dpsi (yaw RATE) to be zero. For rotating_yaw, this makes yaw fixed to initial_yaw_deg.")
    parser.add_argument('--force_zero_yaw_angle', action='store_true',
                        help="Force desired yaw ANGLE (psi) to be zero throughout the trajectory. This also makes dpsi zero.")

    # Circle specific
    parser.add_argument('--radius', type=float, default=2.0, help="Radius for circle trajectory (meters)")
    # Rotating yaw specific
    parser.add_argument('--yaw_rate_dps', type=float, default=15.0, 
                        help="Yaw rotation speed in degrees per second (for setpoint_rotating_yaw). Not used if --zero_dpsi or --force_zero_yaw_angle is active.")
    # Figure8 specific
    parser.add_argument('--fig8_width_L', type=float, default=4.0, help="Width (L, max x-extent from center) for figure-8 trajectory (meters)")
    parser.add_argument('--fig8_height_W', type=float, default=2.0, help="Total Height (W) for figure-8 trajectory (meters)")


    args = parser.parse_args()

    TARGET_RATE = 1.0 / args.dt_step if args.dt_step > 0 else 50.0
    DT = args.dt_step
    
    initial_yaw_rad_from_args = np.deg2rad(args.initial_yaw_deg)
    yaw_rate_rps_from_args = np.deg2rad(args.yaw_rate_dps)

    # Handle precedence of yaw flags
    if args.force_zero_yaw_angle:
        print("Info: --force_zero_yaw_angle is active. Yaw angle will be 0 rad, dpsi will be 0 rad/s.")
        print(f"       (--initial_yaw_deg '{args.initial_yaw_deg}' and relevant parts of --yaw_rate_dps are ignored for psi output).")
        if args.zero_dpsi:
            print("       (--zero_dpsi is also active but is superseded by --force_zero_yaw_angle for dpsi output).")
    elif args.zero_dpsi:
        print(f"Info: --zero_dpsi is active. dpsi will be 0 rad/s.")
        if args.type == 'setpoint_rotating_yaw':
             print(f"       For setpoint_rotating_yaw, yaw will be fixed at initial_yaw_deg: {args.initial_yaw_deg:.1f} deg.")
             print(f"       (--yaw_rate_dps '{args.yaw_rate_dps}' is ignored).")

    if not args.force_zero_yaw_angle: # Only print these if yaw angle is not forced to zero
        print(f"Using initial_yaw_rad: {initial_yaw_rad_from_args:.3f} (from {args.initial_yaw_deg:.1f} deg)")
        if args.type == 'setpoint_rotating_yaw' and not args.zero_dpsi:
            print(f"Using yaw_rate_rps: {yaw_rate_rps_from_args:.3f} (from {args.yaw_rate_dps:.1f} deg/s)")


    script_dir = os.path.dirname(os.path.realpath(__file__))
    package_traj_data_dir = os.path.abspath(os.path.join(script_dir, '..', 'src/mlac_sim', "traj_data"))
    
    if not os.path.exists(os.path.join(script_dir, '..', 'src/mlac_sim')):
        package_traj_data_dir = os.path.join(script_dir, "traj_data_output")
        print(f"Warning: Standard mlac_sim/traj_data path not found relative to script." 
              f"Using fallback output directory: {package_traj_data_dir}")

    print(f"Attempting to save trajectories to: {package_traj_data_dir}")
    os.makedirs(package_traj_data_dir, exist_ok=True)
    absolute_output_dir = package_traj_data_dir

    trajectory_array = None
    OUTPUT_FILENAME = "default_trajectory.npy"
    
    common_args_for_gen = {
        "dt": DT,
        "initial_yaw_rad": initial_yaw_rad_from_args,
        "num_cols_output": args.num_cols,
        "zero_dpsi_flag": args.zero_dpsi,
        "force_zero_yaw_angle_flag": args.force_zero_yaw_angle
    }

    if args.type == 'circle':
        DURATION_ONE_LAP = args.duration
        RADIUS = args.radius
        CENTER_X = args.pos_x 
        CENTER_Y = args.pos_y
        ALTITUDE = args.pos_z
        
        OUTPUT_FILENAME = f"circle_r{RADIUS}_t{DURATION_ONE_LAP:.0f}s_alt{ALTITUDE}_initpsi{args.initial_yaw_deg:.0f}deg"
        if args.force_zero_yaw_angle: OUTPUT_FILENAME += "_FORCEZEROPsi"
        elif args.zero_dpsi: OUTPUT_FILENAME += "_zeroDPsi"
        OUTPUT_FILENAME += f"_{TARGET_RATE:.0f}hz_{args.num_cols}col.npy"

        print(f"Generating {args.num_cols}-column circle trajectory...")
        trajectory_array = generate_circle_data(
            T=DURATION_ONE_LAP, radius=RADIUS,
            center_x=CENTER_X, center_y=CENTER_Y, alt=ALTITUDE,
            **common_args_for_gen
        )
    
    elif args.type == 'setpoint_hold':
        SETPOINT_X = args.pos_x
        SETPOINT_Y = args.pos_y
        SETPOINT_Z = args.pos_z
        DURATION = args.duration
        OUTPUT_FILENAME = f"setpoint_hold_x{SETPOINT_X}_y{SETPOINT_Y}_z{SETPOINT_Z}_t{DURATION:.0f}s_psi{args.initial_yaw_deg:.0f}deg"
        if args.force_zero_yaw_angle: OUTPUT_FILENAME += "_FORCEZEROPsi"
        # zero_dpsi is implicit for hold
        OUTPUT_FILENAME += f"_{TARGET_RATE:.0f}hz_{args.num_cols}col.npy"

        print(f"Generating {args.num_cols}-column setpoint hold trajectory...")
        trajectory_array = generate_setpoint_hold_trajectory(
            setpoint_x=SETPOINT_X, setpoint_y=SETPOINT_Y, setpoint_z=SETPOINT_Z,
            duration=DURATION, 
            **common_args_for_gen
        )

    elif args.type == 'setpoint_rotating_yaw':
        SETPOINT_X = args.pos_x
        SETPOINT_Y = args.pos_y
        SETPOINT_Z = args.pos_z 
        DURATION = args.duration
        OUTPUT_FILENAME = (f"setpoint_rot_yaw_x{SETPOINT_X}_y{SETPOINT_Y}_z{SETPOINT_Z}_t{DURATION:.0f}s_"
                           f"initpsi{args.initial_yaw_deg:.0f}deg_rate{args.yaw_rate_dps:.0f}dps")
        if args.force_zero_yaw_angle: OUTPUT_FILENAME += "_FORCEZEROPsi"
        elif args.zero_dpsi: OUTPUT_FILENAME += "_zeroDPsi"
        OUTPUT_FILENAME += f"_{TARGET_RATE:.0f}hz_{args.num_cols}col.npy"
        
        print(f"Generating {args.num_cols}-column setpoint_rotating_yaw trajectory...")
        trajectory_array = generate_setpoint_rotating_yaw_trajectory(
            setpoint_x=SETPOINT_X, setpoint_y=SETPOINT_Y, setpoint_z=SETPOINT_Z,
            duration=DURATION, 
            yaw_rate_rps=yaw_rate_rps_from_args,
            **common_args_for_gen
        )

    elif args.type == 'figure8':
        T_CYCLE = args.duration 
        FIG8_L = args.fig8_width_L
        FIG8_W = args.fig8_height_W
        CENTER_X = args.pos_x
        CENTER_Y = args.pos_y
        ALTITUDE = args.pos_z
        OUTPUT_FILENAME = (f"figure8_L{FIG8_L}_W{FIG8_W}_t{T_CYCLE:.0f}s_alt{ALTITUDE}_"
                           f"initpsi{args.initial_yaw_deg:.0f}deg")
        if args.force_zero_yaw_angle: OUTPUT_FILENAME += "_FORCEZEROPsi"
        elif args.zero_dpsi: OUTPUT_FILENAME += "_zeroDPsi" # Though for current figure8 dpsi is already 0
        OUTPUT_FILENAME += f"_{TARGET_RATE:.0f}hz_{args.num_cols}col.npy"

        print(f"Generating {args.num_cols}-column figure-8 trajectory...")
        trajectory_array = generate_figure8_data(
            T_cycle=T_CYCLE, width_L=FIG8_L, height_W=FIG8_W,
            center_x=CENTER_X, center_y=CENTER_Y, alt=ALTITUDE,
            **common_args_for_gen
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
            print(f"First point data (example):")
            print(f"  t={trajectory_array[0,0]:.2f}, Px={trajectory_array[0,1]:.2f}, Py={trajectory_array[0,2]:.2f}, Pz={trajectory_array[0,3]:.2f}")
            print(f"  Vx={trajectory_array[0,4]:.2f}, Vy={trajectory_array[0,5]:.2f}, Vz={trajectory_array[0,6]:.2f}")
            print(f"  Psi={trajectory_array[0,7]:.3f} rad ({np.rad2deg(trajectory_array[0,7]):.1f} deg)")
            if args.num_cols == 9 and trajectory_array.shape[1] > 8: print(f"  dPsi (col 8)={trajectory_array[0,8]:.3f} rad/s")
            if args.num_cols >= 11 and trajectory_array.shape[1] > 10: print(f"  Ax={trajectory_array[0,8]:.2f}, Ay={trajectory_array[0,9]:.2f}, Az={trajectory_array[0,10]:.2f}")
            if args.num_cols >= 15 and trajectory_array.shape[1] > 14: print(f"  dPsi (col 14)={trajectory_array[0,14]:.3f} rad/s")

    elif trajectory_array is not None and trajectory_array.shape[0] == 0:
        print("Generated trajectory is empty, not saving.")
    else:
        print("Failed to generate trajectory.")
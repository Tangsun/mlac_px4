#!/usr/bin/env python3

import subprocess
import os
import argparse
import time
import sys
import datetime # For logging in this batch script

def log_batch_run_details(log_file_path, trial_num, mission_description, world_name, traj_idx, traj_file, sim_duration):
    """Appends details of a specific trial within the batch to a log file."""
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"[{current_time_str}] Batch Trial: {trial_num} | "
            f"Mission Desc Passed to Launch: \"{mission_description}\" | "
            f"World: {world_name} | Traj File: {traj_file} | Traj Idx: {traj_idx} | "
            f"Sim Duration: {sim_duration}s\n"
        )
        with open(log_file_path, "a") as f:
            f.write(log_entry)
        print(f"Logged batch trial {trial_num} details to: {log_file_path}")
    except Exception as e:
        print(f"Error writing to batch log file {log_file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Repeatedly runs PX4 SITL simulations with varying worlds and trajectory indices."
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=100, # Default number of simulations
        help="The total number (N) of simulation runs to perform."
    )
    parser.add_argument(
        "--base_mission_desc",
        type=str,
        default="BatchSim", # Changed default
        help="Base description for the missions. '_Trial_X_World_Y_Traj_Z' will be appended."
    )
    parser.add_argument(
        "--trajectory_file",
        type=str,
        default="N100_T30.0_spline_11col_zero_yaw.npy",
        help="Name of the .npy trajectory file (must contain at least N trajectories for N runs if indexing up to N-1)."
    )
    parser.add_argument(
        "--launch_script_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "launch_px4sim_windy_tmux.py"),
        help="Path to your tmux launch script (e.g., launch_px4sim_windy_tmux.py)."
    )
    parser.add_argument(
        "--sim_duration_sec",
        type=int,
        default=90, # Default from your launch script
        help="Duration in seconds for each simulation run before auto-killing the TMUX session."
    )

    parser.add_argument(
        "--delay_between_runs_sec",
        type=int,
        default=10, 
        help="Delay in seconds between the end of one simulation and the start of the next."
    )
    parser.add_argument(
        "--start_trial_index",
        type=int,
        default=0,
        help="Starting index for trials (e.g., if you want to resume or run a subset like 0-99, then 100-199)."
    )
    parser.add_argument(
        "--batch_log_file",
        type=str,
        default=os.path.expanduser("~/mlac_px4/ros2_px4_ws/rosbag_windy/batch_runs_summary_log.txt"),
        help="Log file to record the parameters of each batch run."
    )


    args = parser.parse_args()

    if args.num_simulations <= 0:
        print("Error: Number of simulations (N) must be positive.")
        sys.exit(1)

    launch_script_abs_path = os.path.expanduser(args.launch_script_path)
    if not os.path.isfile(launch_script_abs_path):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        potential_path = os.path.join(script_dir, args.launch_script_path.replace("./scripts/", "")) # Adjust if needed
        if os.path.isfile(potential_path):
            launch_script_abs_path = potential_path
        else:
            print(f"Error: Launch script not found at {launch_script_abs_path} or {potential_path}")
            print("Please provide the correct path using --launch_script_path.")
            sys.exit(1)
    
    print(f"Starting batch of {args.num_simulations} simulation runs...")
    print(f"Launch script: {launch_script_abs_path}")
    print(f"Batch summary log: {args.batch_log_file}")


    for i in range(args.start_trial_index, args.start_trial_index + args.num_simulations):
        world_name = f"windy_{i}" 
        trajectory_idx = i
        # This mission_description is passed to launch_px4sim_windy_tmux.py
        # which then logs it to its own windy_missions_log.txt along with its timestamped bag name.
        current_mission_description = f"{args.base_mission_desc}_Trial_{i}_World_{world_name}_TrajIdx_{trajectory_idx}"

        print(f"\n--- Starting Batch Trial {i} (Simulation {i - args.start_trial_index + 1}/{args.num_simulations}) ---")
        print(f"  Mission Description for this run: {current_mission_description}")
        print(f"  World: {world_name}.sdf")
        print(f"  Trajectory File: {args.trajectory_file}")
        print(f"  Trajectory Index: {trajectory_idx}")
        print(f"  Simulation Duration: {args.sim_duration_sec}s")
        
        # Log details for this specific batch trial *before* launching
        log_batch_run_details(
            args.batch_log_file, 
            trial_num=i,
            mission_description=current_mission_description,
            world_name=world_name,
            traj_idx=trajectory_idx,
            traj_file=args.trajectory_file,
            sim_duration=args.sim_duration_sec
        )

        command = [
            "python3", # Or just "python" if that's your python 3 executable
            launch_script_abs_path,
            current_mission_description, # This is the positional 'mission_description' arg for the launch script
            "--trajectory_file", args.trajectory_file,
            "--trajectory_index", str(trajectory_idx),
            "--world_name", world_name,
            "--auto_kill_duration_sec", str(args.sim_duration_sec)
        ]
            
        try:
            print(f"Executing command: {' '.join(command)}")
            # The launch_script_path itself handles TMUX creation and auto-kill.
            # This script just needs to call it and wait for it to complete.
            # subprocess.run will block until launch_px4sim_windy_tmux.py finishes (i.e., after TMUX is killed).
            process = subprocess.run(command, check=True) 
            print(f"Finished simulation for Trial {i}.")

        except subprocess.CalledProcessError as e:
            print(f"Error during simulation for Trial {i} (World: {world_name}, Traj Idx: {trajectory_idx}):")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            print("Continuing to the next trial if any, or stopping if this was the last.")
        except Exception as e_gen:
            print(f"An unexpected error occurred while trying to run simulation for Trial {i}: {e_gen}")
            print("Continuing to the next trial if any, or stopping if this was the last.")
        
        if i < (args.start_trial_index + args.num_simulations - 1):
            print(f"Waiting {args.delay_between_runs_sec}s before starting next trial...")
            time.sleep(args.delay_between_runs_sec)

    print(f"\nBatch simulation of {args.num_simulations} runs complete.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import subprocess
import os
import time
import datetime
import argparse

def log_mission_details(log_file_path, mission_description, bag_directory_name, wind_params, trajectory_file):
    """
    Appends mission details to the specified log file.
    """
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"[{current_time_str}] Mission: \"{mission_description}\" | "
            f"Bag Directory: {bag_directory_name} | "
            f"Wind (X,Y,Z): ({wind_params.get('x',0)}, {wind_params.get('y',0)}, {wind_params.get('z',0)}) | "
            f"Trajectory File: {trajectory_file}\n"
        )
        with open(log_file_path, "a") as f:
            f.write(log_entry)
        print(f"Logged mission details to: {log_file_path}")
    except Exception as e:
        print(f"Error writing to log file {log_file_path}: {e}")

def run_tmux_commands(session_name, commands, auto_kill_duration_sec=None):
    """
    Sets up a TMUX session with multiple panes, running specific commands.
    Optionally kills the session after a duration.
    """
    num_panes = len(commands)
    if num_panes == 0:
        print("No commands provided.")
        return

    try:
        print(f"Checking for existing TMUX session: {session_name}")
        check_session_cmd = ["tmux", "has-session", "-t", session_name]
        session_exists = subprocess.run(check_session_cmd, capture_output=True, text=True).returncode == 0

        if session_exists:
            print(f"TMUX session '{session_name}' already exists. Killing it and starting fresh.")
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
            time.sleep(1)

        print(f"Starting new TMUX session: {session_name}")
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "WindyTest"], check=True)
        print(f"Session '{session_name}' created.")
        time.sleep(1)

        # Pane splitting logic - ensure it handles the number of panes correctly
        if num_panes > 1: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2) # Pane 1 (0.1)
        if num_panes > 2: subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2) # Pane 2 (0.2) - splits 0.0
        if num_panes > 3: subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.1"], check=True); time.sleep(0.2) # Pane 3 (0.3) - splits 0.1
        if num_panes > 4: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2) # Pane 4 (0.4) - splits 0.0
        if num_panes > 5: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.2"], check=True); time.sleep(0.2) # Pane 5 (0.5) - splits 0.2
        if num_panes > 6: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.1"], check=True); time.sleep(0.2) # Pane 6 (0.6) - splits 0.1
        # If you add more panes beyond 7, you'll need to extend this logic or use a more dynamic splitting approach.

        print("Sending commands to panes...")
        for i, command_string in enumerate(commands):
            target_pane = f"{session_name}:0.{i}"
            print(f"  Sending to pane {i} (TMUX target {target_pane}): {command_string[:100]}...")
            subprocess.run(["tmux", "send-keys", "-t", target_pane, command_string, "C-m"], check=True)
            time.sleep(0.5)

        if num_panes > 1:
            print("Arranging panes using 'tiled' layout...")
            subprocess.run(["tmux", "select-layout", "-t", f"{session_name}:0", "tiled"], check=True)

        print(f"\nTMUX session '{session_name}' is set up.")
        
        if auto_kill_duration_sec and auto_kill_duration_sec > 0 :
            print(f"Session will automatically be killed in {auto_kill_duration_sec} seconds.")
            time.sleep(auto_kill_duration_sec)
            print(f"Killing TMUX session '{session_name}'...")
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
            print("Session killed.")
        else:
            print("To attach, run: tmux attach-session -t", session_name)
            print("The script will now attempt to attach automatically.")
            os.execvp("tmux", ["tmux", "attach-session", "-t", session_name])

    except subprocess.CalledProcessError as e:
        print(f"Error setting up TMUX session: {e}")
        print(f"  To clean up a failed new session, try: tmux kill-session -t {session_name}")
    except FileNotFoundError:
        print("Error: 'tmux' command not found. Is tmux installed and in your PATH?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated windy trajectory testing.")
    parser.add_argument("mission_desc", type=str, help="Brief description of the test.")
    parser.add_argument("--wind_x", type=float, default=0.0, help="Wind velocity X component (m/s).")
    parser.add_argument("--wind_y", type=float, default=0.0, help="Wind velocity Y component (m/s).")
    parser.add_argument("--wind_z", type=float, default=0.0, help="Wind velocity Z component (m/s, typically 0).")
    parser.add_argument("--traj_file", type=str, 
                        default="setpoint_hold_x1.0_y1.0_z4.0_t20.0s_50hz_8col.npy", 
                        help="Name of the trajectory .npy file in mlac_sim/traj_data/ to be loaded by mlac_mission_node.")
    parser.add_argument("--sim_duration_sec", type=int, default=0, 
                        help="Total duration for the simulation run before auto-killing tmux (0 to disable auto-kill and attach).")

    args = parser.parse_args()

    session_name = f"windy_test_{args.mission_desc.replace(' ', '_').lower()}_{int(time.time())}"
    
    ros2_ws_path = os.path.expanduser("~/mlac_px4/ros2_px4_ws")
    px4_src_path = os.path.expanduser("~/mlac_px4/px4_src/PX4-Autopilot")
    qgc_path = os.path.expanduser("~/mlac_px4/px4_src/")
    

    venv_path = os.path.expanduser("~/mlac_px4/mlac_env")

    mission_trajectory_filename = args.traj_file 
    
    expected_traj_path = os.path.join(ros2_ws_path, "src/mlac_sim/traj_data", mission_trajectory_filename)
    if not os.path.exists(expected_traj_path):
        print(f"ERROR: Trajectory file not found: {expected_traj_path}")
        print(f"Please ensure the file '{mission_trajectory_filename}' exists in '~/mlac_px4/ros2_px4_ws/src/mlac_sim/traj_data/'.")
        print(f"You can generate it using 'generate_ref_traj.py'.")
        exit(1)

    now = datetime.datetime.now()
    timestamp_for_bag_dir = now.strftime("%Y%m%d_%H%M%S")
    bag_directory_name = f"rosbag_windX{args.wind_x}_windY{args.wind_y}_windZ{args.wind_z}_{timestamp_for_bag_dir}"
    
    base_bag_and_log_path = os.path.join(ros2_ws_path, "rosbag_data")
    full_bag_output_path = os.path.join(base_bag_and_log_path, bag_directory_name)
    info_log_file_path = os.path.join(base_bag_and_log_path, "automated_tests_log.txt")

    log_mission_details(info_log_file_path, args.mission_desc, bag_directory_name, 
                        {'x': args.wind_x, 'y': args.wind_y, 'z': args.wind_z}, 
                        mission_trajectory_filename)

    px4_pane_cmd = (
        f"echo '>>> Starting PX4 SITL (gz_x500 with windy world)...'; "
        f"cd {px4_src_path} && PX4_GZ_WORLD=windy make px4_sitl gz_x500; "
        f"echo 'PX4 pane exited.'; exec bash"
    )
    
    mavros_pane_cmd = (
        f"sleep 15; echo '>>> Launching MAVROS...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557; "
        f"echo 'MAVROS pane exited.'; exec bash"
    )
    
    set_wind_cmd = (
        f"sleep 25; "
        f"echo '>>> Waiting for Gazebo and setting wind to X={args.wind_x}, Y={args.wind_y}, Z={args.wind_z}...'; "
        f"gz topic -t '/world/windy/wind' -m gz.msgs.Wind -p 'linear_velocity: {{x: {args.wind_x}, y: {args.wind_y}, z: {args.wind_z}}}'; "
        f"echo 'Wind set command sent (X:{args.wind_x}, Y:{args.wind_y}, Z:{args.wind_z}). Pane will stay open.'; exec bash"
    )
    
    mlac_node_cmd = (
        f"sleep 30; echo '>>> Preparing mlac_mission_node...'; "
        f"source {venv_path}/bin/activate && source {ros2_ws_path}/install/setup.bash && "
        f"export PYTHONPATH=\"{venv_path}/lib/python3.10/site-packages${{PYTHONPATH:+:$PYTHONPATH}}\" && "
        f"echo 'Running mlac_mission_node with trajectory: {mission_trajectory_filename}' && "
        f"ros2 run mlac_sim mlac_mission_node --ros-args -p trajectory_file_name:='{mission_trajectory_filename}'; "
        f"echo 'mlac_mission_node pane exited.'; exec bash"
    )

    topics_to_record = [
        "/mavros/state", "/mavros/local_position/pose", "/mavros/local_position/velocity_body",
        "/mavros/attitude", "/mavros/setpoint_raw/attitude",
        f"/mlac_mission_node/control_log", f"/mlac_mission_node/trajectory_complete_status",
        "/mission_control/command", "/world/windy/wind"
    ]
    rosbag_cmd = (
        f"sleep 35; echo '>>> Recording ROS bag to {full_bag_output_path}...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"ros2 bag record -o {full_bag_output_path} {' '.join(topics_to_record)}; "
        f"echo 'rosbag record pane exited.'; exec bash"
    )

    mission_start_cmd = (
        f"sleep 45; echo '>>> Triggering mission start...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"ros2 topic pub --once /mission_control/command std_msgs/msg/String '{{data: \"START_TRAJECTORY\"}}'; "
        f"echo 'Mission start command sent.'; exec bash"
    )

    qgc_cmd = (
        f"sleep 20; echo '>>> Launching QGroundControl from {qgc_path}...'; "
        f"cd {qgc_path} && ./QGroundControl.AppImage; "
        f"echo 'QGC pane exited.'; exec bash"
    )

    commands = [
        px4_pane_cmd,       # Pane 0
        mavros_pane_cmd,    # Pane 1
        set_wind_cmd,       # Pane 2
        mlac_node_cmd,      # Pane 3
        rosbag_cmd,         # Pane 4
        mission_start_cmd,  # Pane 5
        qgc_cmd             # Pane 6 
    ]

    run_tmux_commands(session_name, commands, auto_kill_duration_sec=args.sim_duration_sec)
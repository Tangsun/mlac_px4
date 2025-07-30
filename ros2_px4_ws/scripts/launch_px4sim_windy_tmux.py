#!/usr/bin/env python3

import subprocess
import os
import time
import datetime
import argparse # For command-line arguments

def log_mission_details(log_file_path, mission_description, bag_directory_name):
    """
    Appends mission details to the specified log file.
    """
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"[{current_time_str}] Mission: \"{mission_description}\" | "
            f"Bag Directory: {bag_directory_name}\n"
        )
        with open(log_file_path, "a") as f:
            f.write(log_entry)
        print(f"Logged mission details to: {log_file_path}")
    except Exception as e:
        print(f"Error writing to log file {log_file_path}: {e}")

def run_tmux_commands(session_name, commands, auto_kill_duration_sec=0):
    """
    Sets up a TMUX session with multiple panes, running specific commands.
    Optionally sends Ctrl+C to panes and kills the session after a duration.
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
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "ROS_Sim"], check=True)
        print(f"Session '{session_name}' created.")
        time.sleep(1)

        # Pane creation logic
        if num_panes > 1: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2)
        if num_panes > 2: subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2) 
        if num_panes > 3: subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.1"], check=True); time.sleep(0.2) 
        if num_panes > 4: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2)
        if num_panes > 5: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.2"], check=True); time.sleep(0.2)
        if num_panes > 6: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.1"], check=True); time.sleep(0.2)
        if num_panes > 7: 
            subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.3"], check=True); time.sleep(0.2)
        
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
        
        if auto_kill_duration_sec and auto_kill_duration_sec > 0:
            print(f"Simulation will run for {auto_kill_duration_sec} seconds.")
            time.sleep(auto_kill_duration_sec)
            
            print(f"Attempting to send Ctrl+C to panes in session '{session_name}' before killing...")
            for i in range(num_panes): # num_panes is the number of commands/panes
                target_pane = f"{session_name}:0.{i}"
                try:
                    print(f"  Sending Ctrl+C to pane {target_pane}...")
                    # Send Ctrl+C twice for better chance of interrupting
                    subprocess.run(["tmux", "send-keys", "-t", target_pane, "C-c"], check=False)
                    time.sleep(0.1) # Small delay between Ctrl+C attempts
                    subprocess.run(["tmux", "send-keys", "-t", target_pane, "C-c"], check=False)
                except Exception as e_ctrlc:
                    print(f"    Warning: Could not send Ctrl+C to pane {target_pane}: {e_ctrlc}")
            
            print("Waiting briefly (e.g., 2 seconds) for processes to respond to Ctrl+C...")
            time.sleep(2) # Give processes a moment to attempt graceful shutdown

            print(f"Killing TMUX session '{session_name}'...")
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
            print("Session killed.")
        else:
            print("Auto-kill disabled. To attach, run: tmux attach-session -t", session_name)
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
    parser = argparse.ArgumentParser(
        description="Run PX4 SITL, MAVROS, mlac_node, record ROS bag, log mission details, and optionally auto-arm/offboard and auto-kill."
    )
    parser.add_argument(
        "mission_description",
        type=str,
        help="A brief description of the mission/test being recorded."
    )
    parser.add_argument(
        "--trajectory_file",
        type=str,
        default = "N100_T30.0_spline_11col_zero_yaw.npy",
        help="Name of the .npy trajectory file in 'mlac_sim/traj_data/' folder to be used by mlac_mission_node."
    )
    parser.add_argument(
        "--trajectory_index",
        type=int,
        default = 22,
        help="index of the trajectory in the .npy file (if multiple) to be used by mlac_mission_node."
    )
    parser.add_argument(
        "--world_name",
        type=str,
        default="windy_test", 
        help="Name of the Gazebo world file (e.g., windy_test, default) to be used by PX4 SITL."
    )
    parser.add_argument(
        "--auto_kill_duration_sec",
        type=int,
        default=90, 
        help="Duration in seconds before automatically killing the TMUX session. Set to 0 to disable auto-kill and attach instead."
    )

    args = parser.parse_args()
    mission_desc_from_arg = args.mission_description
    trajectory_file_name = args.trajectory_file
    trajectory_index = args.trajectory_index
    world_name_arg = args.world_name
    auto_kill_sec = args.auto_kill_duration_sec

    session = "mlac_sim_main" 
    
    ros2_ws_path = os.path.expanduser("~/mlac_ijrr/mlac_px4/ros2_px4_ws") # Kai's path to the ROS 2 workspace
    px4_src_path = os.path.expanduser("~/PX4-Autopilot")        # Kai's PX4 source path
    venv_path = os.path.expanduser("~/mlac_ijrr/mlac_px4/mlac_env") # Kai's virtual environment path

    now = datetime.datetime.now()
    timestamp_for_bag_dir = now.strftime("%m%d_%H%M%S")
    bag_directory_name = f"windy_bag_{timestamp_for_bag_dir}" 
    
    base_bag_and_log_path = os.path.join(ros2_ws_path, "raw_traj_data")
    full_bag_output_path = os.path.join(base_bag_and_log_path, bag_directory_name) 
    info_log_file_path = os.path.join(base_bag_and_log_path, "windy_data_traj.txt") 

    print(f"Generated bag path for this run: {full_bag_output_path}")
    print(f"Mission log file: {info_log_file_path}")

    log_mission_details(info_log_file_path, mission_desc_from_arg, bag_directory_name)

    mlac_node_command = (
        f"sleep 10; "
        f"echo '>>> Preparing to run mlac_mission_node...'; "
        f"echo 'Activating virtual environment ({venv_path})...' && "
        f"source {venv_path}/bin/activate && "
        f"echo 'Sourcing ROS 2 workspace ({ros2_ws_path})...' && "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"echo 'Exporting PYTHONPATH with venv site-packages...' && "
        f"export PYTHONPATH=\"{venv_path}/lib/python3.10/site-packages${{PYTHONPATH:+:$PYTHONPATH}}\" && "
        f"echo 'Running mlac_mission_node...' && "
        f"ros2 run mlac_sim mlac_mission_node --ros-args \
            -p trajectory_file_name:='{trajectory_file_name}' \
            -p trajectory_index:={trajectory_index} ; "
        f"echo 'mlac_mission_node pane exited.'; exec bash"
    )

    set_stream_rates_command = (
        f"sleep 10; "
        f"echo '>>> Attempting to set MAVLink stream rates...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"echo 'Setting LOCAL_POSITION_NED (ID 32) to 50Hz...' && "
        f"ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval '{{message_id: 32, message_rate: 50.0}}' && "
        f"sleep 0.5 && "
        f"echo 'Setting ATTITUDE (ID 30) to 50Hz...' && "
        f"ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval '{{message_id: 30, message_rate: 50.0}}' && "
        f"echo 'Finished setting stream rates.'; "
        f"exec bash"
    )
    
    topics_to_record = [
        "/mavros/state", "/mavros/local_position/pose", "/mavros/local_position/velocity_body", 
        "/mavros/attitude", "/mavros/setpoint_raw/attitude", 
        f"/mlac_mission_node/control_log",  f"/mlac_mission_node/trajectory_complete_status", 
        "/mission_control/command", "/rosout",
    ]
    rosbag_command = (
        f"sleep 30; " 
        f"echo '>>> Preparing to record ROS bag data to {full_bag_output_path}...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"echo 'Starting ros2 bag record...' && "
        f"ros2 bag record -o {full_bag_output_path} {' '.join(topics_to_record)}; "
        f"echo 'ros2 bag record pane exited.'; exec bash"
    )

    px4_pane_cmd = (
        f"echo '>>> Starting PX4 SITL (gz_x500 with world: {world_name_arg})...'; "
        f"cd {px4_src_path} && PX4_GZ_WORLD={world_name_arg} make px4_sitl gz_x500; "
        f"echo 'PX4 SITL pane exited.'; exec bash"
    )

    flight_initiation_command = (
        f"sleep 20; " 
        f"echo '>>> Initiating flight sequence...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"echo 'Sending START_MISSION command...' && "
        f"ros2 topic pub --once /mission_control/command std_msgs/msg/String '{{data: \"START_MISSION\"}}' && "
        f"sleep 5 && " 
        f"echo 'Attempting to ARM drone...' && "
        f"ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool '{{value: true}}' && "
        f"sleep 2 && "
        f"echo 'Attempting to set OFFBOARD mode...' && "
        f"ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode '{{custom_mode: \"OFFBOARD\"}}'; "
        f"echo 'Flight initiation commands sent.'; exec bash" 
    )
    
    general_command_pane = (
        f"sleep 5; "
        f"echo '>>> Sourcing workspace & venv for general commands...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"source {venv_path}/bin/activate && "
        f"echo 'General command pane ready.'; "
        f"exec bash"
    )

    commands_to_run = [
        px4_pane_cmd,      
        f"sleep 5; echo '>>> Launching MAVROS...'; source {ros2_ws_path}/install/setup.bash && ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557; echo 'MAVROS pane exited.'; exec bash", 
        mlac_node_command,  
        set_stream_rates_command, 
        rosbag_command,     
        f"sleep 15; echo '>>> Launching QGroundControl...'; cd {px4_src_path} && ./QGroundControl-x86_64.AppImage; echo 'QGC pane exited.'; exec bash", 
        flight_initiation_command, 
        general_command_pane 
    ]

    commands_to_run = [cmd for cmd in commands_to_run if cmd]
    
    if len(commands_to_run) > 8: 
        print(f"Warning: Pane splitting logic is explicitly defined for up to 8 panes. You have {len(commands_to_run)} commands.")

    run_tmux_commands(session, commands_to_run, auto_kill_duration_sec=auto_kill_sec)

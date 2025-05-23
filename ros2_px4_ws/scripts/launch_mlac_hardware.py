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

def run_tmux_commands(session_name, commands):
    """
    Sets up a TMUX session with multiple panes, running specific commands.
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
            time.sleep(1) # Give it a moment to die

        print(f"Starting new TMUX session: {session_name}")
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "ROS_Sim"], check=True)
        print(f"Session '{session_name}' created.")
        time.sleep(1)

        # Pane creation logic (adjust layout as preferred)
        # This creates a sequence of panes, relying on `select-layout tiled`
        if num_panes > 1: # For 2nd command/pane (index 1)
            subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2)
        if num_panes > 2: # For 3rd command/pane (index 2)
            subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2) # Splits the top-left pane
        if num_panes > 3: # For 4th command/pane (index 3)
            subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.1"], check=True); time.sleep(0.2) # Splits the bottom-left pane
        if num_panes > 4: # For 5th command/pane (index 4)
             # This will split the top-left (0.0) pane vertically again.
            subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2)
        if num_panes > 5: # For 6th command/pane (index 5)
            # This will split the (original) top-right (0.2) pane vertically.
            subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.2"], check=True); time.sleep(0.2)
        if num_panes > 6: # For 7th command/pane (index 6)
            # This will split the (original) bottom-left (0.1) pane vertically.
            subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.1"], check=True); time.sleep(0.2)
        # Add more splits here if you have > 7 commands, adjusting targets carefully or using a more programmatic approach.

        print("Sending commands to panes...")
        for i, command_string in enumerate(commands):
            target_pane = f"{session_name}:0.{i}" # TMUX numbers panes 0, 1, 2... after splits in order of creation
            print(f"  Sending to pane {i} (TMUX target {target_pane}): {command_string[:100]}...")
            subprocess.run(["tmux", "send-keys", "-t", target_pane, command_string, "C-m"], check=True)
            time.sleep(0.5) # Increased sleep slightly

        if num_panes > 1:
            print("Arranging panes using 'tiled' layout...")
            subprocess.run(["tmux", "select-layout", "-t", f"{session_name}:0", "tiled"], check=True)

        print(f"\nTMUX session '{session_name}' is set up.")
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
    parser = argparse.ArgumentParser(
        description="Run PX4 SITL, MAVROS, mlac_node, record ROS bag, and log mission details."
    )
    parser.add_argument(
        "mission_description",
        type=str,
        help="A brief description of the mission/test being recorded."
    )
    args = parser.parse_args()
    mission_desc_from_arg = args.mission_description

    session = "mlac_sim_main" # Changed session name slightly
    
    ros2_ws_path = os.path.expanduser("~/mlac_px4/ros2_px4_ws")
    px4_src_path = os.path.expanduser("~/mlac_px4/px4_src/PX4-Autopilot")
    venv_path = os.path.expanduser("~/mlac_px4/mlac_env")

    # --- Bagging and Logging Setup ---
    now = datetime.datetime.now()
    timestamp_for_bag_dir = now.strftime("%Y%m%d_%H%M%S")
    bag_directory_name = f"rosbag_{mission_desc_from_arg.replace(' ', '_')}_{timestamp_for_bag_dir}" # Include description in folder name
    
    base_bag_and_log_path = os.path.join(ros2_ws_path, "rosbag_data")
    full_bag_output_path = os.path.join(base_bag_and_log_path, bag_directory_name) # This is the argument for ros2 bag record -o
    info_log_file_path = os.path.join(base_bag_and_log_path, "missions_log.txt") # General log file

    print(f"Generated bag path for this run: {full_bag_output_path}")
    print(f"Mission log file: {info_log_file_path}")

    log_mission_details(info_log_file_path, mission_desc_from_arg, bag_directory_name)

    # --- Define Commands for TMUX Panes ---

    # Command for Pane running mlac_mission_node
    mlac_node_command = (
        f"sleep 20; "
        f"echo '>>> Preparing to run mlac_mission_node...'; "
        f"echo 'Activating virtual environment ({venv_path})...' && "
        f"source {venv_path}/bin/activate && "
        f"echo 'Sourcing ROS 2 workspace ({ros2_ws_path})...' && "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"echo 'Exporting PYTHONPATH with venv site-packages...' && "
        f"export PYTHONPATH=\"{venv_path}/lib/python3.10/site-packages${{PYTHONPATH:+:$PYTHONPATH}}\" && "
        f"echo 'Running mlac_mission_node...' && "
        f"ros2 run mlac_sim mlac_mission_node; " # Add any specific params if needed, e.g., --ros-args -p trajectory_file_name:="your_traj.npy"
        f"echo 'mlac_mission_node pane exited.'; exec bash"
    )

    # Command to set MAVLink stream rates
    set_stream_rates_command = (
        f"sleep 25; "
        f"echo '>>> Attempting to set MAVLink stream rates...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"echo 'Setting LOCAL_POSITION_NED (ID 32) to 50Hz...' && "
        f"ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval '{{message_id: 32, message_rate: 50.0}}' && "
        f"sleep 0.5 && "
        # AttitudeTarget (83) is what we send, MAVROS should handle its rate. Let's monitor actual attitude.
        f"echo 'Setting ATTITUDE (ID 30) to 50Hz...' && "
        f"ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval '{{message_id: 30, message_rate: 50.0}}' && "
        f"echo 'Finished setting stream rates.'; "
        f"exec bash"
    )
    
    # Command for a general-purpose command pane
    command_pane_setup = (
        f"sleep 5; "
        f"echo '>>> Sourcing workspace & venv for mission commands...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"source {venv_path}/bin/activate && "
        f"echo 'Workspace sourced. Ready for mission commands (e.g., ros2 topic pub /mission_control/command ...).'; "
        f"exec bash"
    )

    # Command for ROS Bag Recording
    # Key topics: MAVROS state, local position, our attitude setpoints, our controller log, TF transforms
    topics_to_record = [
        "/mavros/state",
        "/mavros/local_position/pose",
        "/mavros/local_position/velocity_body", # Or velocity_local_ned if preferred
        "/mavros/attitude", # Actual vehicle attitude from PX4
        "/mavros/setpoint_raw/attitude", # What mlac_node sends to MAVROS
        f"/mlac_mission_node/control_log", # Custom log from your mlac_node
        f"/mlac_mission_node/trajectory_complete_status", # FSM status
        "/mission_control/command", # Commands sent to the FSM
        # "/tf",
        # "/tf_static"
    ]
    rosbag_command = (
        f"sleep 30; " # Ensure other nodes are up
        f"echo '>>> Preparing to record ROS bag data to {full_bag_output_path}...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"echo 'Starting ros2 bag record...' && "
        f"ros2 bag record -o {full_bag_output_path} {' '.join(topics_to_record)}; "
        f"echo 'ros2 bag record pane exited.'; exec bash"
    )

    commands_to_run = [
        # Pane 0: Logging rosbag data
        # f"echo '>>> Starting PX4 SITL (gz_x500)...'; cd {px4_src_path} && make px4_sitl gazebo-classic; echo 'PX4 SITL pane exited.'; exec bash",
        rosbag_command,

        # Pane 1: Launch MAVROS
        f"sleep 15; echo '>>> Launching MAVROS...'; source {ros2_ws_path}/install/setup.bash && ros2 launch mavros px4.launch tgt_system:=11; echo 'MAVROS pane exited.'; exec bash",

        # Pane 2: Run your mlac_mission_node
        mlac_node_command,
        
        # Pane 3: Set MAVLink Stream Rates
        set_stream_rates_command,

        # Pane 4: Record ROS Bag Data
        # rosbag_command,
        f"sleep 20; echo '>>> Launching mocap'; source {ros2_ws_path}/install/setup.bash && ros2 run mlac_sim repub_odom_node; exec bash",

        # Pane 5: General command pane for sending mission commands etc.
        command_pane_setup,
        
        # Pane 6: Launch QGroundControl
        f"sleep 35; echo '>>> Launching QGroundControl...'; cd {os.path.dirname(px4_src_path)} && ./QGroundControl.AppImage; echo 'QGC pane exited.'; exec bash", # Assuming QGC is in ~/mlac_px4/
    ]

    # Filter out any None commands if you conditionally add them (not strictly needed here)
    commands_to_run = [cmd for cmd in commands_to_run if cmd]
    
    if len(commands_to_run) > 7: # Current splitting logic is for up to 7
        print(f"Warning: Pane splitting logic is explicitly defined for up to 7 panes. You have {len(commands_to_run)} commands.")

    run_tmux_commands(session, commands_to_run)
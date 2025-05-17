#!/usr/bin/env python3

import subprocess
import os
import time
import datetime
import argparse # For command-line arguments

def log_mission_details(log_file_path, mission_description, input_bag_path_str, output_bag_directory_name):
    """
    Appends mission details, including input and output bag info, to the specified log file.

    Args:
        log_file_path (str): The full path to the info.txt log file.
        mission_description (str): The mission description from the command line.
        input_bag_path_str (str): Path of the bag file used for replaying.
        output_bag_directory_name (str): The name of the new rosbag directory created for this run.
    """
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"[{current_time_str}] Mission: \"{mission_description}\" | "
            f"Input Bag (Replayed): {input_bag_path_str} | "
            f"Output Bag (Recorded): {output_bag_directory_name}\n"
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
        print(f"Starting new TMUX session: {session_name}")
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "ROS_Replay_Sim"], check=True)
        print(f"Session '{session_name}' created.")
        time.sleep(1)

        # Pane splitting logic (adjust if more/less than 5 panes are defined in command_list)
        if num_panes > 1: # Pane 0 exists, split for Pane 1
            subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True)
            time.sleep(0.5)
        if num_panes > 2: # Split Pane 0 for Pane 2
            subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.0"], check=True)
            time.sleep(0.5)
        if num_panes > 3: # Split Pane 1 for Pane 3
            subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.1"], check=True)
            time.sleep(0.5)
        if num_panes > 4: # Split Pane 1 (now 0.1, the top part of bottom-left) for Pane 4
            subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.1"], check=True)
            time.sleep(0.5)
        # If you have exactly 5 commands, pane indices will be 0.0, 0.2, 0.4, 0.1, 0.3 (approx, depends on split order)
        # The default targetting 0.0, 0.1, 0.2, 0.3, 0.4 works if splits are done carefully.
        # For 5 panes as laid out:
        # 0.0 (top-left), 0.2 (top-right)
        # 0.1 (bottom-left), 0.3 (bottom-center), 0.4 (bottom-right) - this needs more complex splitting.
        # Let's keep the simpler split for now, assuming user can rearrange.
        # A common 5-pane layout: main-vertical, then split right, then split bottom-left, then split bottom-right.
        # The current split logic is:
        # Pane 0: 0.0
        # Pane 1: 0.1 (below 0.0)
        # Pane 2: 0.2 (right of 0.0)
        # Pane 3: 0.3 (below 0.1)
        # Pane 4: 0.4 (right of 0.1) - This might not be the most intuitive layout for 5.
        # For simplicity, I'll use the direct 0.i indexing and let tmux's default tiling handle it.


        print("Sending commands to panes...")
        for i, command_string in enumerate(commands):
            target_pane = f"{session_name}:0.{i}"
            print(f"  Sending to pane {i} ({target_pane}): {command_string[:100]}...")
            subprocess.run(["tmux", "send-keys", "-t", target_pane, command_string, "C-m"], check=True)
            time.sleep(0.2)

        print("Arranging panes (tiled layout)...")
        subprocess.run(["tmux", "select-layout", "-t", f"{session_name}:0", "tiled"], check=True)

        print(f"Attaching to TMUX session '{session_name}'...")
        os.execvp("tmux", ["tmux", "attach-session", "-t", session_name])

    except subprocess.CalledProcessError as e:
        print(f"Error setting up TMUX session: {e}")
        print(f"  tmux kill-session -t {session_name}")
    except FileNotFoundError:
        print("Error: 'tmux' command not found. Is tmux installed and in your PATH?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PX4 SITL, MAVROS, replay an attitude bag, record a new bag, and log details."
    )
    parser.add_argument(
        "input_bag_path",
        type=str,
        help="Full path to the existing rosbag directory to be replayed by attitude_replay_node."
    )
    parser.add_argument(
        "mission_description",
        type=str,
        help="A brief description of the mission/test being recorded."
    )
    args = parser.parse_args()
    
    input_bag_to_replay = os.path.expanduser(args.input_bag_path)
    mission_desc_from_arg = args.mission_description

    session = "px4_ros2_replay_rec" 
    now = datetime.datetime.now()
    timestamp_for_output_bag = now.strftime("%Y%m%d_%H%M%S")
    
    output_bag_directory_name = f"replay_rosbag_{timestamp_for_output_bag}"
    base_bag_and_log_path = os.path.expanduser("~/mlac_px4/ros2_px4_ws/rosbag_data") # Common base for logs and bags
    full_output_bag_path = os.path.join(base_bag_and_log_path, output_bag_directory_name)
    info_log_file_path = os.path.join(base_bag_and_log_path, "info.txt")

    print(f"Input bag for replay: {input_bag_to_replay}")
    print(f"Output bag will be recorded to: {full_output_bag_path}")
    print(f"Mission log file: {info_log_file_path}")

    # Log mission details before starting TMUX
    log_mission_details(info_log_file_path, mission_desc_from_arg, input_bag_to_replay, output_bag_directory_name)

    # --- Define TMUX Commands ---
    # Ensure your attitude_replay_node executable is correctly named in your package's setup.py
    # and the package is sourced.
    # The attitude_replay_node should be the one that takes bag_file_path parameter.
    
    # Delays:
    # MAVROS needs PX4 SITL (sleep 15 for MAVROS)
    # Replay node and New Bag Recording need MAVROS (sleep 5-10 after MAVROS launch, so total sleep 20-25)
    # QGC needs MAVROS (sleep 5 after MAVROS, so total sleep 20)

    command_list = [
        # Pane 0: Start PX4 SITL
        "echo '>>> Starting PX4 SITL (gz_x500)...'; cd ~/mlac_px4/px4_src/PX4-Autopilot && make px4_sitl gz_x500; exec bash",

        # Pane 1: Launch MAVROS
        "sleep 15; echo '>>> Launching MAVROS...'; source /opt/ros/humble/setup.bash && ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557; exec bash",

        # Pane 2: Run the attitude_replay_node (reading the INPUT bag)
        # This node publishes to /mavros/setpoint_raw/attitude
        # Make sure 'attitude_replay_node' is the correct executable name from your setup.py
        # and it's the version that reads directly from a bag file.
        f"sleep 25; echo '>>> Running Attitude Replay Node (reading {input_bag_to_replay})...'; "
        f"cd ~/mlac_px4/ros2_px4_ws && source install/setup.bash && "
        f"ros2 run basic_offboard attitude_replay --ros-args " # Ensure 'basic_offboard' is your package name
        f"-p bag_file_path:='{input_bag_to_replay}' "
        f"-p recorded_topic_name:='/mavros/setpoint_raw/target_attitude' " # Or whatever topic is in your input bag
        f"-p publish_topic_name:='/mavros/setpoint_raw/attitude' "
        f"-p replay_rate_hz:=50.0; exec bash",

        # Pane 3: Record a NEW ROS bag (the OUTPUT bag)
        # This records the result of the replay, including the commands sent by attitude_replay_node
        f"sleep 25; echo '>>> Recording NEW ROS bag to {full_output_bag_path}...'; "
        f"cd ~/mlac_px4/ros2_px4_ws && source install/setup.bash && " # Source workspace if ros2 bag is part of it or for alias
        f"ros2 bag record -o {full_output_bag_path} "
        "/mavros/state "
        "/mavros/local_position/pose "
        "/mavros/setpoint_raw/attitude "             # Commands published by your replay node
        "/mavros/setpoint_raw/target_attitude "      # What MAVROS targets based on above
        "/tf /tf_static; " # Add other topics as needed
        "exec bash", 

        # Pane 4: Launch QGroundControl (optional monitoring)
        "sleep 20; echo '>>> Launching QGroundControl...'; cd ~/mlac_px4/px4_src && ./QGroundControl.AppImage; exec bash",
    ]
    # Adjust number of panes if QGC is not needed, or add other monitoring tools.
    # If using 4 panes, remove the 5th command and adjust split logic if needed.

    run_tmux_commands(session, command_list)

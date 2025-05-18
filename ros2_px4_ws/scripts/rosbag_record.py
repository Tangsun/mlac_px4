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
        print(f"Starting new TMUX session: {session_name}")
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "ROS_Sim"], check=True)
        print(f"Session '{session_name}' created.")
        time.sleep(1)

        # Adjust pane splitting for potentially 6 panes (e.g., 2x3 or 3x2 like layout)
        # This creates a series of splits. 'select-layout tiled' will attempt to arrange them.
        if num_panes > 1: # For 2nd command/pane
            subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True)
        if num_panes > 2: # For 3rd command/pane
            subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.0"], check=True)
        if num_panes > 3: # For 4th command/pane (creates a 2x2 base with 0.0, 0.1, 0.2, 0.3)
            subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.1"], check=True)
        if num_panes > 4: # For 5th command/pane
            subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True) # Splits top-left
        if num_panes > 5: # For 6th command/pane
            subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.2"], check=True) # Splits top-right (original 0.2)

        print("Sending commands to panes...")
        for i, command_string in enumerate(commands):
            target_pane = f"{session_name}:0.{i}" # TMUX numbers panes 0, 1, 2... after splits
            print(f"  Sending to pane {i} (TMUX target {target_pane}): {command_string[:100]}...")
            subprocess.run(["tmux", "send-keys", "-t", target_pane, command_string, "C-m"], check=True)
            time.sleep(0.2)

        print("Arranging panes using 'tiled' layout...")
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
        description="Run PX4 SITL, MAVROS, set stream rates, record ROS bag, and log mission details."
    )
    parser.add_argument(
        "mission_description",
        type=str,
        help="A brief description of the mission/test being recorded."
    )
    args = parser.parse_args()
    mission_desc_from_arg = args.mission_description

    session = "px4_ros2_bagrec_rates" 
    now = datetime.datetime.now()
    timestamp_for_bag_dir = now.strftime("%Y%m%d_%H%M%S")
    bag_directory_name = f"rosbag_{timestamp_for_bag_dir}"
    
    base_bag_and_log_path = os.path.expanduser("~/mlac_px4/ros2_px4_ws/rosbag_data")
    full_bag_path_for_this_run = os.path.join(base_bag_and_log_path, bag_directory_name)
    info_log_file_path = os.path.join(base_bag_and_log_path, "info.txt")

    print(f"Generated bag path for this run: {full_bag_path_for_this_run}")
    print(f"Mission log file: {info_log_file_path}")

    log_mission_details(info_log_file_path, mission_desc_from_arg, bag_directory_name)

    # Define the command for setting stream rates
    # MAVLink IDs: ATTITUDE_TARGET=83, LOCAL_POSITION_NED=32
    # Desired rate: 50Hz
    service_call_commands = (
        "sleep 22; " # Wait for MAVROS to be fully up (MAVROS starts at sleep 15)
        "echo '>>> Setting MAVLink stream rates...'; "
        "source /opt/ros/humble/setup.bash && " # Source ROS if not already in pane's environment
        "echo 'Setting ATTITUDE_TARGET (ID 83) to 100Hz...' && "
        "ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval '{message_id: 83, message_rate: 100.0}' && "
        "sleep 1 && echo 'Setting LOCAL_POSITION_NED (ID 32) to 100Hz...' && "
        "ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval '{message_id: 32, message_rate: 100.0}'; " # Removed HIGHRES_IMU call
        "echo 'Finished setting stream rates.'; exec bash" # Keep pane open
    )

    # --- Define TMUX Commands (still 6 commands, one pane for service calls) ---
    command_list = [
        # Pane 0: Start PX4 SITL
        "echo '>>> Starting PX4 SITL (gz_x500)...'; cd ~/mlac_px4/px4_src/PX4-Autopilot && make px4_sitl gz_x500; exec bash",
        
        # Pane 1: Launch MAVROS
        "sleep 15; echo '>>> Launching MAVROS...'; source /opt/ros/humble/setup.bash && ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557; exec bash",
        
        # Pane 2: Set MAVLink Stream Rates via Service Calls
        service_call_commands,
        
        # Pane 3: Launch QGroundControl (delay adjusted)
        "sleep 28; echo '>>> Launching QGroundControl...'; cd ~/mlac_px4/px4_src && ./QGroundControl.AppImage; exec bash",
        
        # Pane 4: Run ROS 2 package (mission_controller, delay adjusted)
        # "sleep 33; echo '>>> Running ROS 2 mission_controller...'; cd ~/mlac_px4/ros2_px4_ws && source install/setup.bash && ros2 run basic_offboard setpoint_publisher; exec bash",
        "sleep 33; echo '>>> Running ROS 2 mission_controller...'; cd ~/mlac_px4/ros2_px4_ws && source install/setup.bash && ros2 run basic_offboard mission_controller; exec bash",
        
        # Pane 5: Record ROS bag data (delay adjusted)
        f"sleep 38; echo '>>> Recording ROS bag data to {full_bag_path_for_this_run}...'; "
        f"ros2 bag record -o {full_bag_path_for_this_run} "
        # Original set of topics, /mavros/imu/data removed
        "/mavros/state /mavros/local_position/pose /mavros/setpoint_position/local /mavros/setpoint_raw/target_attitude; " 
        "exec bash", 
    ]

    run_tmux_commands(session, command_list)

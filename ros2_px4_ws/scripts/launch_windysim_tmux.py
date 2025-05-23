#!/usr/bin/env python3

import subprocess
import os
import time
# import shlex # No longer needed for send-keys

def run_tmux_commands(session_name, commands):
    """
    Sets up a TMUX session with multiple panes based on the provided example structure,
    running specific commands with delays. Corrected quoting for send-keys.

    Args:
        session_name (str): Name for the TMUX session.
        commands (list): A list of command strings to execute in each pane.
                         Delays should be included within the command strings using 'sleep'.
                         Append '; exec bash' to commands that might exit quickly
                         if you want the pane to remain open.
    """
    num_panes = len(commands)
    if num_panes == 0:
        print("No commands provided.")
        return

    try:
        print(f"Starting new TMUX session: {session_name}")
        # Start a new detached TMUX session (without running the first command yet)
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "ROS_Sim"], check=True)
        print(f"Session '{session_name}' created.")
        time.sleep(1) # Small delay for session creation

        # --- Pane Splitting ---
        # Create the required number of panes (N-1 splits for N panes)
        # Pane 0 already exists.
        if num_panes > 1:
            # Split 0 vertically -> 0 (top), 1 (bottom)
            subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True)
            time.sleep(0.5)
        if num_panes > 2:
             # Split 0 horizontally -> 0 (top-left), 2 (top-right)
            subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.0"], check=True)
            time.sleep(0.5)
        if num_panes > 3:
            # Split 1 vertically -> 1 (bottom-left-top), 3 (bottom-left-bottom)
            subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.1"], check=True)
            time.sleep(0.5)
        if num_panes > 4:
             # Split 1 horizontally -> 1 (bottom-left-top-left), 4 (bottom-left-top-right)
            subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.1"], check=True)
            time.sleep(0.5)
        # Add more splits here if num_panes > 5 following a similar pattern

        # --- Command Sending ---
        print("Sending commands to panes...")
        for i, command_string in enumerate(commands):
            target_pane = f"{session_name}:0.{i}"
            print(f"  Sending to pane {i}: {command_string[:60]}...") # Print truncated command

            # *** FIX: Send the command string directly without shlex.quote ***
            # This ensures the shell interprets the string as multiple commands separated by ; or &&
            subprocess.run(["tmux", "send-keys", "-t", target_pane, command_string, "C-m"], check=True)
            time.sleep(0.2) # Small delay between sending commands

        # --- Layout & Attach ---
        print("Arranging panes...")
        # Select a layout (tiled arranges evenly)
        subprocess.run(["tmux", "select-layout", "-t", f"{session_name}:0", "tiled"], check=True)

        # Attach to the session
        print(f"Attaching to TMUX session '{session_name}'...")
        # Use execvp to replace the python script process with tmux attach
        os.execvp("tmux", ["tmux", "attach-session", "-t", session_name])

    except subprocess.CalledProcessError as e:
        print(f"Error setting up TMUX session: {e}")
        print("Make sure tmux is installed and running correctly.")
        print("You might need to kill the session if it was partially created:")
        print(f"  tmux kill-session -t {session_name}")
    except FileNotFoundError:
        print("Error: 'tmux' command not found. Is tmux installed and in your PATH?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Define the session name
    session = "px4_ros2_sim" # Changed name slightly to avoid conflict

    # Define the commands as a list of strings.
    # Include delays ('sleep X;') and '; exec bash' directly in the strings.
    # Adjust delays based on how long each step typically takes on your system.
    command_list = [
        # Pane 0: Start PX4 SITL
        "echo '>>> Starting PX4 SITL (gz_x500)...'; cd ~/mlac_px4/px4_src/PX4-Autopilot && PX4_GZ_WORLD=windy make px4_sitl gz_x500; exec bash",

        # Pane 1: Launch MAVROS (with delay)
        "sleep 15; echo '>>> Launching MAVROS...'; source /opt/ros/humble/setup.bash && ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557; exec bash",

        # Pane 2: Launch QGroundControl (with delay)
        "sleep 20; echo '>>> Launching QGroundControl...'; cd ~/mlac_px4/px4_src && ./QGroundControl.AppImage; exec bash",

        # Pane 3: Run ROS 2 package (with delay)
        # "sleep 25; echo '>>> Running ROS 2 setpoint_publisher...'; cd ~/mlac_px4/ros2_px4_ws && source install/setup.bash && ros2 run basic_offboard mission_controller; exec bash",
        # "sleep 25; echo '>>> Running ROS 2 setpoint_publisher...'; cd ~/mlac_px4/ros2_px4_ws && source install/setup.bash && ros2 run basic_offboard attitude_setpoint; exec bash",

        # Pane 4: Check vehicle state (with delay)
        "sleep 30; echo '>>> Checking vehicle state...'; ros2 topic echo /mavros/local_position/pose"
        # No 'exec bash' needed here as 'topic echo' runs continuously
    ]

    # Run the setup
    run_tmux_commands(session, command_list)
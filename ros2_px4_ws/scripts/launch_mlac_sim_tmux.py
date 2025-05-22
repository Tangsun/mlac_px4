#!/usr/bin/env python3

import subprocess
import os
import time

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
            print(f"TMUX session '{session_name}' already exists. Attaching.")
        else:
            print(f"Starting new TMUX session: {session_name}")
            subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "ROS_Sim"], check=True)
            print(f"Session '{session_name}' created.")
            time.sleep(1)

            # Create panes
            if num_panes > 1:
                print("Creating pane for command 1...")
                subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True)
                time.sleep(0.2)
            if num_panes > 2:
                print("Creating pane for command 2...")
                subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.0"], check=True)
                time.sleep(0.2)
            if num_panes > 3:
                print("Creating pane for command 3...")
                subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.1"], check=True)
                time.sleep(0.2)
            if num_panes > 4:
                print("Creating pane for command 4...")
                subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True)
                time.sleep(0.2)
            if num_panes > 5:
                print("Creating pane for command 5...")
                subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.2"], check=True)
                time.sleep(0.2)

            print("Sending commands to panes...")
            for i, command_string in enumerate(commands):
                target_pane_index = i
                target_pane = f"{session_name}:0.{target_pane_index}"
                print(f"  Sending to pane {target_pane_index} (TMUX target {target_pane}): {command_string[:100]}...")
                subprocess.run(["tmux", "send-keys", "-t", target_pane, command_string, "C-m"], check=True)
                time.sleep(0.5)

            if num_panes > 1:
                print("Arranging panes using 'tiled' layout...")
                subprocess.run(["tmux", "select-layout", "-t", f"{session_name}:0", "tiled"], check=True)

        print(f"\nTMUX session '{session_name}' is set up.")
        print("To attach, run: tmux attach-session -t", session_name)
        print("The script will now attempt to attach automatically.")
        os.execvp("tmux", ["tmux", "attach-session", "-t", session_name])

    except subprocess.CalledProcessError as e:
        print(f"Error setting up TMUX session: {e}")
        if not session_exists:
             print(f"  To clean up a failed new session, try: tmux kill-session -t {session_name}")
    except FileNotFoundError:
        print("Error: 'tmux' command not found. Is tmux installed and in your PATH?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    session = "mlac_sim_run"

    # Command for the pane running mlac_mission_node
    # This now includes: activate venv, source workspace, then export modified PYTHONPATH
    mlac_node_command = (
        "sleep 20; " # Adjust sleep as needed
        "echo '>>> Preparing to run mlac_mission_node...'; "
        "echo 'Activating virtual environment (/home/sunbochen/mlac_px4/mlac_env)...' && "
        "source /home/sunbochen/mlac_px4/mlac_env/bin/activate && "
        "echo 'Sourcing ROS 2 workspace (~/mlac_px4/ros2_px4_ws)...' && "
        "source ~/mlac_px4/ros2_px4_ws/install/setup.bash && "
        "echo 'Exporting PYTHONPATH with venv site-packages...' && "
        "export PYTHONPATH=\"/home/sunbochen/mlac_px4/mlac_env/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}\" && "
        "echo '--- Current PYTHONPATH for mlac_mission_node pane ---' && " # Debug print
        "echo $PYTHONPATH && " # Debug print
        "echo '--- End PYTHONPATH ---' && " # Debug print
        "echo 'Running mlac_mission_node...' && "
        "ros2 run mlac_sim mlac_mission_node; "
        "exec bash" # Keeps the pane alive
    )

    # Command to set MAVLink stream rates (optional)
    set_stream_rates_command = (
        "sleep 25; " # Increased sleep to ensure mlac_node_command has sourced etc.
        "echo '>>> Attempting to set MAVLink stream rates...'; "
        "source ~/mlac_px4/ros2_px4_ws/install/setup.bash && " # Source workspace
        "echo 'Setting LOCAL_POSITION_NED (ID 32) to 100Hz...' && "
        "ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval '{message_id: 32, message_rate: 100.0}' && "
        "sleep 0.5 && "
        "echo 'Setting ATTITUDE_TARGET (ID 83) to 100Hz...' && "
        "ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval '{message_id: 83, message_rate: 100.0}' && "
        "echo 'Finished setting stream rates.'; "
        "exec bash"
    )
    
    # Command for a general-purpose command pane (e.g., for sending mission commands)
    command_pane_setup = (
        "sleep 5; " # Short sleep
        "echo '>>> Sourcing workspace for sending mission commands...'; "
        "source ~/mlac_px4/ros2_px4_ws/install/setup.bash && "
         # Optionally activate venv here too if you run python scripts from this pane
        "echo 'source /home/sunbochen/mlac_px4/mlac_env/bin/activate' >> ~/.bashrc_tmux_pane_temp && " # For convenience if pane respawns
        "source /home/sunbochen/mlac_px4/mlac_env/bin/activate && "
        "echo 'Workspace sourced. Ready for mission commands (e.g., ros2 topic pub /mission_control/command ...).'; "
        "exec bash"
    )


    commands_to_run = [
        # Pane 0: Start PX4 SITL
        "echo '>>> Starting PX4 SITL (gz_x500)...'; cd ~/mlac_px4/px4_src/PX4-Autopilot && make px4_sitl gz_x500; exec bash",

        # Pane 1: Launch MAVROS
        "sleep 15; echo '>>> Launching MAVROS...'; source ~/mlac_px4/ros2_px4_ws/install/setup.bash && ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557; exec bash",

        # Pane 2: Run your mlac_mission_node (with venv activation and PYTHONPATH export)
        mlac_node_command,

        # Pane 3: (Optional) Launch QGroundControl
        "sleep 30; echo '>>> Launching QGroundControl...'; cd ~/mlac_px4/px4_src && ./QGroundControl.AppImage; exec bash",
        
        # Pane 4: (Optional) Set MAVLink Stream Rates
        set_stream_rates_command,

        # Pane 5: (Optional but Recommended) General command pane
        command_pane_setup,
    ]

    # Filter out any None commands if you conditionally add them
    commands_to_run = [cmd for cmd in commands_to_run if cmd]
    
    if len(commands_to_run) > 6: # Current splitting logic is for up to 6
        print(f"Warning: Pane splitting logic is explicitly defined for up to 6 panes. You have {len(commands_to_run)} commands.")

    run_tmux_commands(session, commands_to_run)

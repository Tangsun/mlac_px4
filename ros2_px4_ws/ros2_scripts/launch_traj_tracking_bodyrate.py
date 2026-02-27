#!/usr/bin/env python3
"""
Launch PX4 SITL + MAVROS + mlac_mission_node for bodyrate controller testing.

Typical usage (diagnostic trajectory with default world):
  python3 launch_traj_tracking_bodyrate.py "phase2 rotation fix test"

With circle trajectory and auto-kill:
  python3 launch_traj_tracking_bodyrate.py "circle tracking" \
      --trajectory_file circle_r2.0_t20s_alt1.5_initpsi0deg_pointToCenter_50hz_11col.npy \
      --auto_kill_duration_sec 120

For Kai's machine:
  python3 launch_traj_tracking_bodyrate.py "test" --directory Kai
"""

import subprocess
import os
import time
import datetime
import argparse


def log_mission_details(log_file_path, mission_description, bag_directory_name):
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
    num_panes = len(commands)
    if num_panes == 0:
        print("No commands provided.")
        return

    try:
        print(f"Checking for existing TMUX session: {session_name}")
        session_exists = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True, text=True
        ).returncode == 0

        if session_exists:
            print(f"TMUX session '{session_name}' already exists. Killing it.")
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
            time.sleep(1)

        print(f"Starting new TMUX session: {session_name}")
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "bodyrate_test"], check=True)
        time.sleep(1)

        if num_panes > 1: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2)
        if num_panes > 2: subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2)
        if num_panes > 3: subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.1"], check=True); time.sleep(0.2)
        if num_panes > 4: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.0"], check=True); time.sleep(0.2)
        if num_panes > 5: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.2"], check=True); time.sleep(0.2)
        if num_panes > 6: subprocess.run(["tmux", "split-window", "-v", "-t", f"{session_name}:0.1"], check=True); time.sleep(0.2)
        if num_panes > 7: subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0.3"], check=True); time.sleep(0.2)

        print("Sending commands to panes...")
        for i, command_string in enumerate(commands):
            target_pane = f"{session_name}:0.{i}"
            print(f"  Pane {i}: {command_string[:100]}...")
            subprocess.run(["tmux", "send-keys", "-t", target_pane, command_string, "C-m"], check=True)
            time.sleep(0.5)

        if num_panes > 1:
            subprocess.run(["tmux", "select-layout", "-t", f"{session_name}:0", "tiled"], check=True)

        print(f"\nTMUX session '{session_name}' is set up.")

        if auto_kill_duration_sec and auto_kill_duration_sec > 0:
            print(f"Simulation will run for {auto_kill_duration_sec} seconds.")
            time.sleep(auto_kill_duration_sec)

            print(f"Sending Ctrl+C to all panes...")
            for i in range(num_panes):
                target_pane = f"{session_name}:0.{i}"
                try:
                    subprocess.run(["tmux", "send-keys", "-t", target_pane, "C-c"], check=False)
                    time.sleep(0.1)
                    subprocess.run(["tmux", "send-keys", "-t", target_pane, "C-c"], check=False)
                except Exception:
                    pass

            time.sleep(2)
            print(f"Killing TMUX session '{session_name}'...")
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
            print("Session killed.")
        else:
            print("To attach: tmux attach-session -t", session_name)
            os.execvp("tmux", ["tmux", "attach-session", "-t", session_name])

    except KeyboardInterrupt:
        print("\nCtrl+C detected! Shutting down.")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up TMUX: {e}")
    except FileNotFoundError:
        print("Error: 'tmux' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print(f"\n--- Cleaning up TMUX session '{session_name}' ---")
        session_exists = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True
        ).returncode == 0
        if session_exists:
            try:
                for i in range(len(commands)):
                    subprocess.run(["tmux", "send-keys", "-t", f"{session_name}:0.{i}", "C-c"], check=False)
                time.sleep(1)
                subprocess.run(["tmux", "kill-session", "-t", session_name], check=True, capture_output=True)
                print(f"Session '{session_name}' killed.")
            except subprocess.CalledProcessError:
                print(f"Session '{session_name}' may have already closed.")
        else:
            print(f"Session '{session_name}' already gone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch PX4 SITL + MAVROS + mlac_mission_node for bodyrate controller testing."
    )
    parser.add_argument("mission_description", type=str,
                        help="Brief description of the test (logged with rosbag).")
    parser.add_argument("--directory", type=str, default="ST",
                        choices=["ST", "Kai"],
                        help="Directory preset: ST (Sunbochen) or Kai.")
    parser.add_argument("--trajectory_file", type=str,
                        default="diagnostic_50hz_11col.npy",
                        help="Trajectory .npy filename in mlac_sim/traj_data/.")
    parser.add_argument("--trajectory_index", type=int, default=0,
                        help="Trajectory index within the .npy file.")
    parser.add_argument("--controller_type", type=str, default="pid",
                        help="Controller type (pid, coml, coml_debug).")
    parser.add_argument("--control_level", type=str, default="bodyrate",
                        help="Control level (attitude, bodyrate).")
    parser.add_argument("--bodyrate_kp", nargs='+', type=float,
                        default=[0.3, 0.3, 0.3],
                        help="Bodyrate proportional gains (e.g., --bodyrate_kp 0.3 0.3 0.3).")
    parser.add_argument("--world_name", type=str, default="default",
                        help="Gazebo world name (e.g., default, windy_test).")
    parser.add_argument("--auto_kill_duration_sec", type=int, default=0,
                        help="Auto-kill after N seconds. 0 = attach to tmux instead.")
    args = parser.parse_args()

    # --- Directory presets ---
    if args.directory == "Kai":
        ros2_ws_path = os.path.expanduser("~/mlac_ijrr/mlac_px4/ros2_px4_ws")
        px4_src_path = os.path.expanduser("~/PX4-Autopilot")
        venv_path = os.path.expanduser("~/mlac_ijrr/mlac_px4/mlac_env")
        QGC_name = "QGroundControl-x86_64.AppImage"
    else:  # ST
        ros2_ws_path = os.path.expanduser("~/mlac_px4/ros2_px4_ws")
        px4_src_path = os.path.expanduser("~/mlac_px4/px4_src/PX4-Autopilot")
        venv_path = os.path.expanduser("~/mlac_px4/mlac_env")
        QGC_name = "../QGroundControl.AppImage"

    # --- Rosbag setup ---
    now = datetime.datetime.now()
    timestamp = now.strftime("%m%d_%H%M%S")
    bag_dir_name = f"bodyrate_diag_{timestamp}"

    base_bag_path = os.path.join(ros2_ws_path, "rosbag_data")
    full_bag_path = os.path.join(base_bag_path, bag_dir_name)
    log_file = os.path.join(base_bag_path, "bodyrate_diag_log.txt")

    print(f"Rosbag output: {full_bag_path}")
    log_mission_details(log_file, args.mission_description, bag_dir_name)

    bodyrate_kp_str = str(args.bodyrate_kp)

    # --- Pane commands ---

    # Pane 0: PX4 SITL
    px4_cmd = (
        f"echo '>>> Starting PX4 SITL (gz_x500, world: {args.world_name})...'; "
        f"cd {px4_src_path} && PX4_GZ_WORLD={args.world_name} make px4_sitl gz_x500; "
        f"echo 'PX4 SITL exited.'; exec bash"
    )

    # Pane 1: MAVROS
    mavros_cmd = (
        f"sleep 5; "
        f"echo '>>> Launching MAVROS...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557; "
        f"echo 'MAVROS exited.'; exec bash"
    )

    # Pane 2: mlac_mission_node
    mlac_cmd = (
        f"sleep 10; "
        f"echo '>>> Launching mlac_mission_node...'; "
        f"source {venv_path}/bin/activate && "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"export PYTHONPATH=\"{venv_path}/lib/python3.10/site-packages${{PYTHONPATH:+:$PYTHONPATH}}\" && "
        f"ros2 run mlac_sim mlac_mission_node --ros-args "
        f"-p controller_type:='{args.controller_type}' "
        f"-p control_level:='{args.control_level}' "
        f"-p bodyrate_kp:='{bodyrate_kp_str}' "
        f"-p trajectory_file_name:='{args.trajectory_file}' "
        f"-p trajectory_index:={args.trajectory_index}; "
        f"echo 'mlac_mission_node exited.'; exec bash"
    )

    # Pane 3: Set MAVLink stream rates
    stream_cmd = (
        f"sleep 10; "
        f"echo '>>> Setting MAVLink stream rates...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"ros2 service call /mavros/set_message_interval "
        f"mavros_msgs/srv/MessageInterval '{{message_id: 32, message_rate: 50.0}}' && "
        f"sleep 0.5 && "
        f"ros2 service call /mavros/set_message_interval "
        f"mavros_msgs/srv/MessageInterval '{{message_id: 30, message_rate: 50.0}}' && "
        f"echo 'Stream rates set.'; exec bash"
    )

    # Pane 4: Rosbag recording
    topics = [
        "/mavros/state",
        "/mavros/local_position/pose",
        "/mavros/local_position/velocity_body",
        "/mavros/attitude",
        "/mavros/setpoint_raw/attitude",
        "/mavros/setpoint_raw/target_attitude",
        "/mlac_mission_node/control_log",
        "/mlac_mission_node/trajectory_complete_status",
        "/mission_control/command",
    ]
    rosbag_cmd = (
        f"sleep 25; "
        f"echo '>>> Recording rosbag to {full_bag_path}...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"ros2 bag record -o {full_bag_path} {' '.join(topics)}; "
        f"echo 'Rosbag recording exited.'; exec bash"
    )

    # Pane 5: QGroundControl
    qgc_cmd = (
        f"sleep 15; "
        f"echo '>>> Launching QGroundControl...'; "
        f"cd {px4_src_path} && ./{QGC_name}; "
        f"echo 'QGC exited.'; exec bash"
    )

    # Pane 6: Auto flight initiation (START_MISSION → ARM → OFFBOARD)
    flight_cmd = (
        f"sleep 20; "
        f"echo '>>> Initiating flight sequence...'; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"echo 'Sending START_MISSION...' && "
        f"ros2 topic pub --once /mission_control/command std_msgs/msg/String '{{data: \"START_MISSION\"}}' && "
        f"sleep 5 && "
        f"echo 'Arming...' && "
        f"ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool '{{value: true}}' && "
        f"sleep 2 && "
        f"echo 'Setting OFFBOARD mode...' && "
        f"ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode '{{custom_mode: \"OFFBOARD\"}}'; "
        f"echo 'Flight initiation done.'; exec bash"
    )

    # Pane 7: General command pane
    general_cmd = (
        f"sleep 5; "
        f"source {ros2_ws_path}/install/setup.bash && "
        f"source {venv_path}/bin/activate && "
        f"echo 'Ready for commands.'; exec bash"
    )

    commands = [
        px4_cmd,        # 0: PX4 SITL
        mavros_cmd,     # 1: MAVROS
        mlac_cmd,       # 2: mlac_mission_node
        stream_cmd,     # 3: MAVLink stream rates
        rosbag_cmd,     # 4: Rosbag
        qgc_cmd,        # 5: QGroundControl
        flight_cmd,     # 6: Auto flight init
        general_cmd,    # 7: General commands
    ]

    session = "bodyrate_diag"
    run_tmux_commands(session, commands, auto_kill_duration_sec=args.auto_kill_duration_sec)

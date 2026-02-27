#!/usr/bin/env python3
"""
Quick visualization of tracking performance from a rosbag.
Plots measured vs reference position/yaw and command streams.

Usage:
    python3 plot_tracking.py /path/to/rosbag [--output-dir ./results/diag]
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from rosbag_utils import (
    extract_attitude_data,
    plot_pose_vs_reference,
    plot_command_streams,
)


def main():
    parser = argparse.ArgumentParser(description="Plot tracking performance from rosbag.")
    parser.add_argument("rosbag", help="Path to rosbag directory.")
    parser.add_argument("--output-dir", default=None,
                        help="Save plots here. If not set, displays interactively.")
    parser.add_argument("--pose-topic", default="/mavros/local_position/pose")
    parser.add_argument("--velocity-topic", default="/mavros/local_position/velocity_body")
    parser.add_argument("--control-log-topic", default="/mlac_mission_node/control_log")
    parser.add_argument("--attitude-setpoint-topic", default="/mavros/setpoint_raw/attitude")
    args = parser.parse_args()

    print(f"Extracting data from: {args.rosbag}")
    data = extract_attitude_data(
        args.rosbag,
        pose_topic=args.pose_topic,
        velocity_topic=args.velocity_topic,
        att_setpoint_topic=args.attitude_setpoint_topic,
        control_log_topic=args.control_log_topic,
    )

    pose_data = data["pose"]
    reference_data = data["reference"]
    attitude_cmd_data = data["attitude_cmd"]
    bodyrate_cmd_data = data["bodyrate_cmd"]

    t_pose = pose_data[0]
    t_ref = reference_data[0]
    print(f"Pose samples: {len(t_pose)}, Reference samples: {len(t_ref)}")
    if len(t_pose) > 0:
        print(f"Pose time range: [{t_pose[0]:.2f}, {t_pose[-1]:.2f}] s")
    if len(t_ref) > 0:
        print(f"Reference time range: [{t_ref[0]:.2f}, {t_ref[-1]:.2f}] s")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        tracking_path = os.path.join(args.output_dir, "tracking_pose_vs_ref.png")
        commands_path = os.path.join(args.output_dir, "tracking_commands.png")
    else:
        tracking_path = None
        commands_path = None

    print("Plotting pose vs reference...")
    plot_pose_vs_reference(pose_data, reference_data, output_path=tracking_path)

    print("Plotting command streams...")
    plot_command_streams(attitude_cmd_data, bodyrate_cmd_data, output_path=commands_path)

    if args.output_dir:
        print(f"Plots saved to {args.output_dir}/")
    else:
        print("Displaying plots...")


if __name__ == "__main__":
    main()

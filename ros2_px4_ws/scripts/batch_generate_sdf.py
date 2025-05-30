#!/usr/bin/env python3

import subprocess
import os
import argparse
import sys
import time

def main():
    parser = argparse.ArgumentParser(
        description="Repeatedly calls generate_sdf.py to create N different SDF world files with varying seeds."
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=100,
        help="The number (N) of SDF files to generate."
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=12,
        help="The starting seed. Each generated file will use base_seed + iteration_number as its seed."
    )
    parser.add_argument(
        "--generate_script_path",
        type=str,
        default="./scripts/generate_sdf.py", # Assumes this script is run from ros2_ws_path
        help="Path to the generate_sdf.py script."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='~/mlac_px4/px4_src/PX4-Autopilot/Tools/simulation/gz/worlds/',
        help="Output directory for the generated SDF files (will be passed to generate_sdf.py)."
    )
    # Add other arguments from generate_sdf.py if you want to control them from this batch script
    parser.add_argument('--min_angle', type=float, help="Minimum wind angle (degrees). Overrides generate_sdf.py default.")
    parser.add_argument('--max_angle', type=float, help="Maximum wind angle (degrees). Overrides generate_sdf.py default.")
    parser.add_argument('--max_speed', type=float, help="Maximum wind speed (m/s). Overrides generate_sdf.py default.")
    parser.add_argument('--beta_a', type=float, help="Shape parameter 'a' for beta distribution. Overrides generate_sdf.py default.")
    parser.add_argument('--beta_b', type=float, help="Shape parameter 'b' for beta distribution. Overrides generate_sdf.py default.")


    args = parser.parse_args()

    if args.num_files <= 0:
        print("Error: Number of files (N) must be positive.")
        sys.exit(1)

    generate_script_abs_path = os.path.expanduser(args.generate_script_path)
    if not os.path.isfile(generate_script_abs_path):
        print(f"Error: generate_sdf.py script not found at {generate_script_abs_path}")
        print("Please provide the correct path using --generate_script_path or ensure it's in the default location.")
        sys.exit(1)
        
    output_dir_abs_path = os.path.expanduser(args.output_dir)
    # The generate_sdf.py script handles os.makedirs for its output_dir, so we don't need it here.

    print(f"Starting batch generation of {args.num_files} SDF world files...")
    print(f"Output directory: {output_dir_abs_path}")

    for i in range(args.num_files):
        current_seed = args.base_seed + i
        world_name = f"windy_{i}"
        # output_filename = f"{world_name}.sdf" # generate_sdf.py will form this if --output_filename is not given

        print(f"\nGenerating file {i+1}/{args.num_files}: world_name='{world_name}', seed={current_seed}")

        command = [
            "python", # Or "python3" if that's how you run it
            generate_script_abs_path,
            "--world_name", world_name,
            # We let generate_sdf.py create the filename from world_name
            # So, we don't pass --output_filename here.
            "--seed", str(current_seed),
            "--output_dir", output_dir_abs_path # Pass the output directory
        ]

        # Conditionally add other wind parameters if provided
        if args.min_angle is not None:
            command.extend(["--min_angle", str(args.min_angle)])
        if args.max_angle is not None:
            command.extend(["--max_angle", str(args.max_angle)])
        if args.max_speed is not None:
            command.extend(["--max_speed", str(args.max_speed)])
        if args.beta_a is not None:
            command.extend(["--beta_a", str(args.beta_a)])
        if args.beta_b is not None:
            command.extend(["--beta_b", str(args.beta_b)])
            
        try:
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Successfully generated {world_name}.sdf")
            if process.stdout:
                print("Output from generate_sdf.py:")
                print(process.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error generating {world_name}.sdf:")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            print("Stopping batch generation due to error.")
            sys.exit(1)
        except Exception as e_gen:
            print(f"An unexpected error occurred while trying to run generate_sdf.py for {world_name}.sdf: {e_gen}")
            print("Stopping batch generation due to error.")
            sys.exit(1)
            
        time.sleep(0.1) # Small delay between generations, if needed

    print(f"\nBatch generation of {args.num_files} SDF files complete.")

if __name__ == "__main__":
    main()

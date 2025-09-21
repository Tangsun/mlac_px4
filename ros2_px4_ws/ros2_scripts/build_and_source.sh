#!/bin/bash

# Exit the script immediately if any command fails
set -e

# --- Script Start ---

echo "==> Moving to the workspace root directory..."
cd ..

echo "==> Building the workspace with colcon..."
colcon build

echo "==> Sourcing the main ROS 2 Humble environment..."
source /opt/ros/humble/setup.bash

echo "==> Sourcing the local workspace environment..."
source install/setup.bash

echo "==> Returning to the scripts directory..."
cd ros2_scripts

echo "✅ Done! Your environment is ready."
# mlac_msgs/msg/GoalControl.msg
std_msgs/Header header

# Define constants for mode for clarity in receiving node
uint8 MODE_POS_CTRL=0
uint8 MODE_VEL_CTRL=1
uint8 MODE_ACC_CTRL=2

uint8 mode_xy   # Use constants above, e.g., GoalControl.MODE_POS_CTRL
uint8 mode_z    # Use constants above

geometry_msgs/Point position       # Desired position (p)
geometry_msgs/Vector3 velocity   # Desired velocity (v)
geometry_msgs/Vector3 acceleration   # Desired acceleration feedforward (a)
geometry_msgs/Vector3 jerk           # Desired jerk feedforward (j)
float64 yaw                        # Desired yaw angle (psi)
float64 yaw_rate                   # Desired yaw rate (dpsi)
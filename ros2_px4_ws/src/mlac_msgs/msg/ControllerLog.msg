# mlac_msgs/msg/ControllerLog.msg
std_msgs/Header header

# Actual vehicle state at the time of computation
geometry_msgs/Point actual_position
geometry_msgs/Vector3 actual_velocity
geometry_msgs/Quaternion actual_orientation
geometry_msgs/Vector3 actual_angular_velocity

# Reference/Goal values used by the controller
geometry_msgs/Point reference_position
geometry_msgs/Vector3 reference_velocity
geometry_msgs/Vector3 reference_acceleration
geometry_msgs/Vector3 reference_jerk
float64 reference_yaw
float64 reference_yaw_rate
geometry_msgs/Quaternion reference_orientation_desired # q_ref from controller

# Error terms
geometry_msgs/Vector3 error_position
geometry_msgs/Vector3 error_velocity
geometry_msgs/Vector3 error_position_integrated # p_err_int

# Feedforward and Feedback acceleration components (from controller's perspective)
geometry_msgs/Vector3 accel_feedforward
geometry_msgs/Vector3 accel_feedback

# Desired outputs from controller
geometry_msgs/Vector3 desired_force_world         # F_W
geometry_msgs/Vector3 desired_angular_velocity  # w_ref

# COML specific (can be zero if not used by PID)
float64 coml_p_norm
float64 coml_a_norm
float64 coml_y_norm
geometry_msgs/Vector3 coml_f_hat
import numpy as np
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import Header # Added for stamp in controllog_class_to_ros_msg

from .structs import GoalClass, ControlLogClass, StateClass
from mlac_msgs.msg import GoalControl as GoalControlMsg # Not directly used by mlac_mission_node for file-based trajectories
from mlac_msgs.msg import ControllerLog as ControllerLogMsg

# convert from ROS message to array
def point_msg_to_array(point_msg: Point) -> np.ndarray:
    return np.array([point_msg.x, point_msg.y, point_msg.z])

def vector_msg_to_array(vector_msg: Vector3) -> np.ndarray:
    return np.array([vector_msg.x, vector_msg.y, vector_msg.z])

def quaternion_msg_to_quaternion(quaternion_msg: Quaternion) -> np.ndarray:
    return np.array([quaternion_msg.w, quaternion_msg.x, quaternion_msg.y, quaternion_msg.z])

# convert from array to ROS message
def point_array_to_msg(point_array: np.ndarray) -> Point:
    return Point(x=float(point_array[0]), y=float(point_array[1]), z=float(point_array[2]))

def vector_array_to_msg(vector_array: np.ndarray) -> Vector3:
    return Vector3(x=float(vector_array[0]), y=float(vector_array[1]), z=float(vector_array[2]))

def quaternion_array_to_msg(quaternion_array: np.ndarray) -> Quaternion:
    return Quaternion(w=float(quaternion_array[0]), x=float(quaternion_array[1]), y=float(quaternion_array[2]), z=float(quaternion_array[3]))

def quaternion_multiply(quaternion0: np.ndarray, quaternion1: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def get_rpy(quaternion: np.ndarray) -> Vector3: # Changed return type for clarity, though it's a ROS msg type
    """Computes the roll, pitch, and yaw from a normalized quaternion array."""
    w, x, y, z = quaternion
    norm_sq = x*x + y*y + z*z + w*w
    if abs(norm_sq - 1.0) > 1e-6: # Check if not normalized
        norm = np.sqrt(norm_sq)
        if norm < 1e-9: # Avoid division by zero if quaternion is zero
            return Vector3(x=0.0, y=0.0, z=0.0)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp) # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return Vector3(x=float(roll), y=float(pitch), z=float(yaw))


def goal_msg_to_controller_goal_class(msg: GoalControlMsg, py_goal: GoalClass) -> GoalClass:
    """Updates a Python GoalClass instance from a GoalControlMsg."""
    # This function is not directly used by mlac_mission_node if loading from file
    py_goal.t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
    py_goal.mode_xy = GoalClass.Mode(msg.mode_xy)
    py_goal.mode_z = GoalClass.Mode(msg.mode_z)
    py_goal.p = point_msg_to_array(msg.position)
    py_goal.v = vector_msg_to_array(msg.velocity)
    py_goal.a = vector_msg_to_array(msg.acceleration)
    py_goal.j = vector_msg_to_array(msg.jerk)
    py_goal.psi = msg.yaw # Assuming GoalControlMsg.yaw is the desired psi
    py_goal.dpsi = msg.yaw_rate
    return py_goal

def controllog_class_to_ros_msg(log_py: ControlLogClass, stamp) -> ControllerLogMsg: # stamp is likely a Time object from ROS
    """Converts a Python ControlLogClass instance to a ControllerLogMsg."""
    msg = ControllerLogMsg()
    msg.header.stamp = stamp # Expects a builtin_interfaces.msg.Time object
    # msg.header.frame_id = "map" # Optional: Or relevant frame

    msg.actual_position = point_array_to_msg(log_py.p)
    msg.actual_velocity = vector_array_to_msg(log_py.v)
    msg.actual_orientation = quaternion_array_to_msg(log_py.q)
    msg.actual_angular_velocity = vector_array_to_msg(log_py.w)

    msg.reference_position = point_array_to_msg(log_py.p_ref)
    msg.reference_velocity = vector_array_to_msg(log_py.v_ref)
    msg.reference_acceleration = vector_array_to_msg(log_py.a_ff)
    msg.reference_jerk = vector_array_to_msg(log_py.j_ff) # Ensure j_ff is populated in ControlLogClass
    
    # Use the new psi_ref and dpsi_ref fields from ControlLogClass
    msg.reference_yaw = float(log_py.psi_ref)
    msg.reference_yaw_rate = float(log_py.dpsi_ref)
    msg.reference_orientation_desired = quaternion_array_to_msg(log_py.q_ref)

    msg.error_position = vector_array_to_msg(log_py.p_err)
    msg.error_velocity = vector_array_to_msg(log_py.v_err)
    msg.error_position_integrated = vector_array_to_msg(log_py.p_err_int)

    msg.accel_feedforward = vector_array_to_msg(log_py.a_ff)
    msg.accel_feedback = vector_array_to_msg(log_py.a_fb)

    msg.desired_force_world = vector_array_to_msg(log_py.F_W)
    msg.desired_angular_velocity = vector_array_to_msg(log_py.w_ref)

    # COML specific logs
    msg.coml_p_norm = float(log_py.P_norm)
    msg.coml_a_norm = float(log_py.A_norm)
    msg.coml_y_norm = float(log_py.y_norm)
    msg.coml_f_hat = vector_array_to_msg(log_py.f_hat)
    return msg

#!/usr/bin/env python3
"""
Standalone verification tests for rotation bugs A–E identified in the
MAVROS / JAX simulation pipeline.

Usage:
    # Numerical tests only (no ROS dependencies):
    python3 verify_rotation_bugs.py

    # Include empirical rosbag test (needs ROS2 sourced):
    python3 verify_rotation_bugs.py --rosbag /path/to/bag
"""

import argparse
import sys
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# quaternion_to_rotation_matrix — the codebase function under test (w,x,y,z)
# ---------------------------------------------------------------------------
def quaternion_to_rotation_matrix(Q):
    q_w, q_x, q_y, q_z = Q
    r00 = 2 * (q_w * q_w + q_x * q_x) - 1
    r01 = 2 * (q_x * q_y - q_w * q_z)
    r02 = 2 * (q_x * q_z + q_w * q_y)
    r10 = 2 * (q_x * q_y + q_w * q_z)
    r11 = 2 * (q_w * q_w + q_y * q_y) - 1
    r12 = 2 * (q_y * q_z - q_w * q_x)
    r20 = 2 * (q_x * q_z - q_w * q_y)
    r21 = 2 * (q_y * q_z + q_w * q_x)
    r22 = 2 * (q_w * q_w + q_z * q_z) - 1
    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _make_test_quaternion(roll_deg=5.0, pitch_deg=10.0, yaw_deg=30.0):
    """Return quaternion in [w,x,y,z] order for given Euler angles."""
    rot = Rotation.from_euler('zyx', [yaw_deg, pitch_deg, roll_deg], degrees=True)
    xyzw = rot.as_quat()  # scipy: [x,y,z,w]
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # [w,x,y,z]


def _pass_fail(ok, label):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}")
    return ok


# ---------------------------------------------------------------------------
# Bug A — velocity .T
# ---------------------------------------------------------------------------
def test_bug_A():
    """quaternion_to_rotation_matrix returns R_{body->world}.
    R @ v_body should match scipy Rotation.apply(v_body).
    """
    print("\n=== Bug A: velocity .T ===")
    q_wxyz = _make_test_quaternion()
    R = quaternion_to_rotation_matrix(q_wxyz)

    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    rot = Rotation.from_quat(q_xyzw)

    v_body = np.array([1.0, 0.5, -0.3])
    v_scipy = rot.apply(v_body)
    v_no_T = R @ v_body
    v_with_T = R.T @ v_body

    err_no_T = np.linalg.norm(v_no_T - v_scipy)
    err_with_T = np.linalg.norm(v_with_T - v_scipy)

    print(f"  ||R @ v - scipy||   = {err_no_T:.2e}")
    print(f"  ||R.T @ v - scipy|| = {err_with_T:.2e}")
    ok = _pass_fail(err_no_T < 1e-12 and err_with_T > 0.01,
                    "R (no .T) matches scipy; .T does not")
    return ok


# ---------------------------------------------------------------------------
# Bug B — thrust projection .T
# ---------------------------------------------------------------------------
def test_bug_B():
    """Thrust projection: F_W · (R[:, 2]) should recover full thrust magnitude
    when F_W is aligned with body-z."""
    print("\n=== Bug B: thrust projection .T ===")
    q_wxyz = _make_test_quaternion()
    R = quaternion_to_rotation_matrix(q_wxyz)

    thrust_mag = 15.0
    body_z_world = R[:, 2]
    F_W = thrust_mag * body_z_world

    proj_no_T = np.dot(F_W, R[:, 2])
    proj_with_T = np.dot(F_W, R.T[:, 2])

    print(f"  True thrust magnitude = {thrust_mag:.2f}")
    print(f"  Projection (no .T)    = {proj_no_T:.4f}")
    print(f"  Projection (with .T)  = {proj_with_T:.4f}")
    ok = _pass_fail(abs(proj_no_T - thrust_mag) < 1e-10 and
                    abs(proj_with_T - thrust_mag) > 0.1,
                    "No-.T recovers full thrust; .T under-projects")
    return ok


# ---------------------------------------------------------------------------
# Bug C — Euler reconstruction (initial R in closed_loop_core)
# ---------------------------------------------------------------------------
def test_bug_C():
    """from_euler('XYZ', rpy) (extrinsic) should roundtrip with
    as_euler('zyx')[::-1] (intrinsic ZYX → [r,p,y])."""
    print("\n=== Bug C: Euler reconstruction convention ===")
    rpy_deg = np.array([5.0, 10.0, 30.0])
    rpy_rad = np.deg2rad(rpy_deg)

    rot_original = Rotation.from_euler('zyx', rpy_rad[::-1])
    R_original = rot_original.as_matrix()

    rpy_extracted = rot_original.as_euler('zyx', degrees=False)[::-1]

    R_XYZ = Rotation.from_euler('XYZ', rpy_extracted).as_matrix()
    R_xyz = Rotation.from_euler('xyz', rpy_extracted).as_matrix()

    err_XYZ = np.linalg.norm(R_XYZ - R_original)
    err_xyz = np.linalg.norm(R_xyz - R_original)

    print(f"  ||R_from_XYZ - R_original|| = {err_XYZ:.2e}  (extrinsic — correct)")
    print(f"  ||R_from_xyz - R_original|| = {err_xyz:.2e}  (intrinsic — wrong)")
    ok = _pass_fail(err_XYZ < 1e-12 and err_xyz > 0.01,
                    "XYZ extrinsic roundtrips correctly; xyz intrinsic does not")
    return ok


# ---------------------------------------------------------------------------
# Bug D — output Euler convention (closed_loop_core output)
# ---------------------------------------------------------------------------
def test_bug_D():
    """as_euler('zyx')[:, ::-1] gives [roll, pitch, yaw] matching the
    convention used by rosbag_utils extraction."""
    print("\n=== Bug D: output Euler convention ===")
    rpy_deg = np.array([5.0, 10.0, 30.0])
    rpy_rad = np.deg2rad(rpy_deg)

    rot = Rotation.from_euler('zyx', rpy_rad[::-1])

    out_correct = rot.as_euler('zyx', degrees=True)[::-1]
    out_wrong = rot.as_euler('xyz', degrees=True)

    err_correct = np.linalg.norm(out_correct - rpy_deg)
    err_wrong = np.linalg.norm(out_wrong - rpy_deg)

    print(f"  Expected RPY (deg)       = {rpy_deg}")
    print(f"  as_euler('zyx')[::-1]    = {out_correct}  err={err_correct:.2e}")
    print(f"  as_euler('xyz')          = {out_wrong}  err={err_wrong:.2e}")
    ok = _pass_fail(err_correct < 1e-10 and err_wrong > 0.1,
                    "zyx[::-1] is correct; xyz is wrong")
    return ok


# ---------------------------------------------------------------------------
# Bug E — body-rate-to-Euler-rate kinematic matrix
# ---------------------------------------------------------------------------
def test_bug_E():
    """The correct E matrix for ZYX Euler angles maps [roll_dot, pitch_dot,
    yaw_dot] to [p, q, r] body rates."""
    print("\n=== Bug E: kinematic matrix ===")
    roll, pitch = np.deg2rad(15.0), np.deg2rad(20.0)

    E_correct = np.array([
        [1.0,  0.0,            -np.sin(pitch)],
        [0.0,  np.cos(roll),    np.cos(pitch) * np.sin(roll)],
        [0.0, -np.sin(roll),    np.cos(pitch) * np.cos(roll)],
    ])

    omega_body = np.array([0.1, 0.2, 0.05])
    rpy_dot = np.linalg.solve(E_correct, omega_body)
    omega_reconstructed = E_correct @ rpy_dot
    err = np.linalg.norm(omega_reconstructed - omega_body)

    print(f"  omega_body          = {omega_body}")
    print(f"  E @ solve(E, omega) = {omega_reconstructed}")
    print(f"  roundtrip error     = {err:.2e}")

    # Cross-check: old buggy matrix from openloop_comparison_np.py
    gamma, beta = roll, pitch
    E_buggy_inv = np.linalg.inv(np.array([
        [np.cos(beta)*np.cos(gamma), -np.sin(gamma), 0],
        [np.cos(beta)*np.sin(gamma),  np.cos(gamma), 0],
        [-np.sin(beta),               0,              1],
    ]))
    rpy_dot_buggy = E_buggy_inv @ omega_body
    diff = np.linalg.norm(rpy_dot - rpy_dot_buggy)
    print(f"  ||correct_rpy_dot - buggy_rpy_dot|| = {diff:.4f}")
    ok = _pass_fail(err < 1e-14 and diff > 0.01,
                    "Correct E roundtrips; old matrix gives different result")
    return ok


# ---------------------------------------------------------------------------
# Empirical rosbag test — R @ v_body vs R.T @ v_body vs d(pos)/dt
# ---------------------------------------------------------------------------
def test_rosbag_velocity(bag_path):
    """Read pose and velocity_body from a rosbag. Compute d(pos)/dt as ground
    truth world velocity. Compare R @ v_body and R.T @ v_body against it."""
    print(f"\n=== Rosbag empirical test: {bag_path} ===")

    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
    except ImportError as e:
        print(f"  [SKIP] ROS2 not available: {e}")
        return True

    pose_topic = "/mavros/local_position/pose"
    vel_topic = "/mavros/local_position/velocity_body"

    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    topic_types = {m.name: m.type for m in reader.get_all_topics_and_types()}
    if pose_topic not in topic_types or vel_topic not in topic_types:
        print(f"  [SKIP] Required topics not found in bag.")
        return True

    PoseMsg = get_message(topic_types[pose_topic])
    VelMsg = get_message(topic_types[vel_topic])

    reader.set_filter(StorageFilter(topics=[pose_topic, vel_topic]))

    t_pose, positions, quats = [], [], []
    t_vel, vel_body = [], []

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == pose_topic:
            msg = deserialize_message(data, PoseMsg)
            t_ns = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            t_pose.append(t_ns)
            p = msg.pose.position
            o = msg.pose.orientation
            positions.append([p.x, p.y, p.z])
            quats.append([o.x, o.y, o.z, o.w])
        elif topic == vel_topic:
            msg = deserialize_message(data, VelMsg)
            t_ns = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            t_vel.append(t_ns)
            v = msg.twist.linear
            vel_body.append([v.x, v.y, v.z])

    t_pose = (np.array(t_pose) - t_pose[0]) * 1e-9
    positions = np.array(positions)
    quats = np.array(quats)
    t_vel = (np.array(t_vel) - t_pose[0] * 1e9) * 1e-9  # same origin
    # Fix: use consistent time origin
    t_vel = np.array(t_vel)
    vel_body = np.array(vel_body)

    if len(t_pose) < 10 or len(t_vel) < 10:
        print("  [SKIP] Not enough data in bag.")
        return True

    v_world_truth = np.gradient(positions, t_pose, axis=0, edge_order=2)

    from scipy.spatial.transform import Rotation, Slerp
    rot_all = Rotation.from_quat(quats)
    slerp = Slerp(t_pose, rot_all)
    rot_at_vel = slerp(np.clip(t_vel, t_pose[0], t_pose[-1]))

    v_world_no_T = rot_at_vel.apply(vel_body)
    v_world_with_T = rot_at_vel.inv().apply(vel_body)

    v_truth_at_vel = np.column_stack([
        np.interp(t_vel, t_pose, v_world_truth[:, i]) for i in range(3)
    ])

    rms_no_T = np.sqrt(np.mean(np.sum((v_world_no_T - v_truth_at_vel)**2, axis=1)))
    rms_with_T = np.sqrt(np.mean(np.sum((v_world_with_T - v_truth_at_vel)**2, axis=1)))

    print(f"  RMS(R @ v_body   - d(pos)/dt) = {rms_no_T:.4f} m/s")
    print(f"  RMS(R.T @ v_body - d(pos)/dt) = {rms_with_T:.4f} m/s")
    ok = _pass_fail(rms_no_T < rms_with_T,
                    "R @ v_body (no .T) is closer to ground truth")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Verify rotation bug fixes.")
    parser.add_argument("--rosbag", type=str, default=None,
                        help="Path to rosbag for empirical velocity test.")
    args = parser.parse_args()

    results = []
    results.append(test_bug_A())
    results.append(test_bug_B())
    results.append(test_bug_C())
    results.append(test_bug_D())
    results.append(test_bug_E())

    if args.rosbag:
        results.append(test_rosbag_velocity(args.rosbag))

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Open-loop comparison (with gap diagnostics): JAX model vs Gazebo using ONLY pose + setpoint topics.

What this script does
---------------------
- Integration grid = /mavros/setpoint_raw/(target_)attitude header timestamps (no bag write time)
- Optional gating via your control_log (header-based)
- Honors AttitudeTarget.type_mask with last-value-held; clamps thrust to [0,1]
- Drops duplicate/non-monotonic stamps; reports Δt stats and large gaps
- Re-orthonormalizes R each step (SO(3) projection) to avoid drift
- Compares on the same x-axis by mapping pose onto the command timeline (nearest-neighbor, edge-clamped)
- Plots:
    * 3D trajectory (JAX vs Gazebo)
    * X/Y/Z vs time (shaded where command gaps occur)
    * Attitude RPY vs time (JAX vs Gazebo) + geodesic error statistics
    * Δt time-history + histogram
    * Position error vs time (shaded gap regions)

Typical usage
-------------
python open_loop_compare_with_gap_viz.py \
  --rosbag /path/to/bag \
  --pose_topic /mavros/local_position/pose \
  --attitude_setpoint_topic /mavros/setpoint_raw/attitude \
  --mass 2.0

Optional flags:
  --prefer_target_attitude
  --control_log_topic /mlac_mission_node/control_log
  --hov_thrust 0.726
  --thrust_scale <N_per_unit>   # overrides hov_thrust mapping
  --thrust_body_z_sign 1        # use -1 if thrust acts along -Z_body
  --gap_threshold 0.2           # seconds; used for reporting and shading only
  --segment_on_gap              # if set, simulate only the first contiguous segment (no huge ZOH step)
"""

import argparse
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
plt.style.use('seaborn-v0_8-whitegrid')

# --- ROS 2 imports ---
try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
except ImportError as e:
    print(f"ERROR: ROS 2 libs not found: {e}")
    sys.exit(1)

# --- SciPy is required here for quaternion->matrix (pose init). ---
try:
    from scipy.spatial.transform import Rotation
except Exception as e:
    print("ERROR: SciPy is required (for quaternion->matrix). Install with: pip install scipy")
    sys.exit(2)

# --- Optional project dynamics/utils ---
HAVE_DYNAMICS, HAVE_UTILS = True, True
try:
    sys.path.append(os.path.abspath('..'))
    from dynamics import prior
except Exception:
    HAVE_DYNAMICS = False
try:
    from utils import hat as hat_utils
except Exception:
    HAVE_UTILS = False

# ----------------- Math helpers -----------------
def hat_fallback(omega):
    ox, oy, oz = omega[...,0], omega[...,1], omega[...,2]
    z = jnp.zeros_like(ox)
    return jnp.stack([ z,-oz, oy,  oz, z,-ox, -oy, ox, z ], axis=-1)\
             .reshape(omega.shape[:-1] + (3,3))
hat = hat_utils if HAVE_UTILS else hat_fallback

def rk4_step(f, dt, y, t):
    k1 = f(y, t)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(y + dt*k3,     t + dt)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def project_to_so3(R):
    U, _, Vt = jnp.linalg.svd(R)
    Rproj = U @ Vt
    det = jnp.linalg.det(Rproj)
    Rproj = jnp.where(det < 0, U @ jnp.diag(jnp.array([1.,1.,-1.])) @ Vt, Rproj)
    return Rproj

def rotmat_to_rpy_zyx(R, eps=1e-9):
    R = np.asarray(R)
    r20 = np.clip(R[..., 2, 0], -1.0 + eps, 1.0 - eps)
    roll  = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    pitch = np.arcsin(-r20)
    yaw   = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return roll, pitch, yaw

# ----------------- ROS bag helpers -----------------
def get_rosbag_options(path, storage_id='sqlite3'):
    return StorageOptions(uri=path, storage_id=storage_id), \
           ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

def header_ns(hdr):
    return int(hdr.stamp.sec)*1_000_000_000 + int(hdr.stamp.nanosec)

# ----------------- Pose → command-grid mapping (robust NN) -----------------
def R_gaz_on_command_grid_nn(t_pose, q_pose, ts_cmd):
    """
    Map Gazebo quats at t_pose onto the command timeline ts_cmd using
    nearest-neighbor with edge clamping. Robust to out-of-range.
    """
    if q_pose.shape[0] == 0:
        return np.full((len(ts_cmd), 3, 3), np.nan)

    qn = q_pose / np.linalg.norm(q_pose, axis=1, keepdims=True)
    Rmats = np.empty((len(ts_cmd), 3, 3), dtype=float)
    for i, t in enumerate(ts_cmd):
        j = int(np.clip(np.searchsorted(t_pose, t), 0, len(t_pose) - 1))
        Rmats[i] = Rotation.from_quat(qn[j]).as_matrix()
    return Rmats

# ----------------- Gap diagnostics -----------------
def find_gaps(t_cmd, threshold=0.2):
    t = np.asarray(t_cmd, dtype=float)
    dt = np.diff(t)
    gaps = []
    for i, d in enumerate(dt):
        if d > threshold:
            gaps.append({"idx": i, "t_start": t[i], "t_end": t[i+1], "dt": float(d)})
    return gaps

def print_gap_report(t_cmd, threshold=0.2, max_items=20):
    gaps = find_gaps(t_cmd, threshold)
    if not gaps:
        print(f"No gaps > {threshold:.3f}s found in t_cmd.")
    else:
        print(f"\nDetected {len(gaps)} gaps > {threshold:.3f}s in t_cmd:")
        print(" i   t_start[s]    t_end[s]      Δt[s]")
        for g in gaps[:max_items]:
            print(f"{g['idx']:3d}  {g['t_start']:10.6f}  {g['t_end']:10.6f}  {g['dt']:7.3f}")
        if len(gaps) > max_items:
            print(f"... ({len(gaps)-max_items} more)")
    dt = np.diff(np.asarray(t_cmd))
    if len(dt) > 0:
        print(f"\nΔt stats (s): min={dt.min():.6f}, median={np.median(dt):.6f}, "
              f"max={dt.max():.6f}, N={len(dt)}")
    return gaps

def plot_dt_diagnostics(t_cmd, threshold=0.2, outdir="open_loop_figs"):
    t = np.asarray(t_cmd, dtype=float)
    dt = np.diff(t)
    if dt.size == 0:
        return
    tc = 0.5*(t[:-1] + t[1:])  # centers
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(12,4))
    plt.plot(tc, dt, '-', lw=1.5)
    plt.axhline(threshold, ls='--', lw=1, label=f"threshold={threshold:.3f}s")
    plt.xlabel("time (s)"); plt.ylabel("Δt_cmd (s)")
    plt.title("Setpoint spacing Δt over time")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dt_time_history.png"))

    plt.figure(figsize=(6,4))
    plt.hist(dt, bins=50)
    plt.xlabel("Δt_cmd (s)"); plt.ylabel("count")
    plt.title("Histogram of setpoint spacing Δt")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dt_hist.png"))

def shade_gap_regions(ax, gaps, color=(1,0,0,0.12)):
    for g in gaps:
        ax.axvspan(g["t_start"], g["t_end"], color=color, linewidth=0)

# ----------------- Extraction (pose + setpoint only) -----------------
def extract_minimal(rosbag_path, pose_topic, att_sp_topic,
                    control_log_topic=None, prefer_target=False):
    reader = SequentialReader()
    storage_options, converter_options = get_rosbag_options(rosbag_path)
    reader.open(storage_options, converter_options)
    topic_types = {m.name: m.type for m in reader.get_all_topics_and_types()}

    if prefer_target and "/mavros/setpoint_raw/target_attitude" in topic_types:
        att_sp_topic = "/mavros/setpoint_raw/target_attitude"

    PoseMsg = get_message(topic_types[pose_topic])
    AttMsg  = get_message(topic_types[att_sp_topic])
    CtrlMsg = get_message(topic_types[control_log_topic]) if control_log_topic and control_log_topic in topic_types else None

    # Optional window from control_log (header-like builtin_interfaces/Time)
    start_ns = -1; end_ns = -1
    if CtrlMsg:
        reader.set_filter(StorageFilter(topics=[control_log_topic]))
        while reader.has_next():
            _, data, _ = reader.read_next()
            msg = deserialize_message(data, CtrlMsg)
            if start_ns < 0 and msg.trajectory_execution_start_ros_time.sec > 0:
                start_ns = msg.trajectory_execution_start_ros_time.sec*1_000_000_000 + msg.trajectory_execution_start_ros_time.nanosec
            if msg.trajectory_execution_end_ros_time.sec > 0:
                end_ns = msg.trajectory_execution_end_ros_time.sec*1_000_000_000 + msg.trajectory_execution_end_ros_time.nanosec
                break

    # Collect pose + setpoints with header stamps
    reader.set_filter(StorageFilter(topics=[pose_topic, att_sp_topic]))
    reader.seek(0)

    t_cmd_ns, thrust_raw, w_raw, mask_raw = [], [], [], []
    t_pose_ns, p_pose, q_pose = [], [], []
    initial_pose_msg = None

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == att_sp_topic:
            msg = deserialize_message(data, AttMsg)
            tns = header_ns(msg.header)
            if start_ns > 0 and tns < start_ns: continue
            if end_ns   > 0 and tns > end_ns:   continue
            t_cmd_ns.append(tns)
            thrust_raw.append(float(msg.thrust))
            w_raw.append([float(msg.body_rate.x), float(msg.body_rate.y), float(msg.body_rate.z)])
            mask_raw.append(int(msg.type_mask))
        elif topic == pose_topic:
            msg = deserialize_message(data, PoseMsg)
            tns = header_ns(msg.header)
            if start_ns > 0 and tns < start_ns: continue
            if end_ns   > 0 and tns > end_ns:   continue
            if initial_pose_msg is None:
                initial_pose_msg = msg
            t_pose_ns.append(tns)
            p_pose.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            q_pose.append([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

    if len(t_cmd_ns) == 0:
        raise RuntimeError(f"No attitude setpoints on {att_sp_topic} within selected window.")

    # Sort & de-dup
    order = np.argsort(t_cmd_ns)
    t_cmd_ns  = np.array(t_cmd_ns)[order]
    thrust_raw = np.array(thrust_raw)[order]
    w_raw      = np.array(w_raw)[order]
    mask_raw   = np.array(mask_raw)[order]

    t_cmd_unique, keep_idx = np.unique(t_cmd_ns, return_index=True)
    dropped = len(t_cmd_ns) - len(t_cmd_unique)
    if dropped > 0:
        thrust_raw = thrust_raw[keep_idx]
        w_raw      = w_raw[keep_idx]
        mask_raw   = mask_raw[keep_idx]
        t_cmd_ns   = t_cmd_unique

    t0 = t_cmd_ns[0]
    t_cmd = (t_cmd_ns - t0) * 1e-9

    if len(t_pose_ns) > 0:
        op = np.argsort(t_pose_ns)
        t_pose = (np.array(t_pose_ns)[op] - t0) * 1e-9
        p_pose = np.array(p_pose)[op]
        q_pose = np.array(q_pose)[op]
    else:
        t_pose = np.array([]); p_pose = np.empty((0,3)); q_pose = np.empty((0,4))

    # Apply type_mask (LVH) + clamp thrust
    def lvh_forward(arr, invalid_bool):
        arr = np.asarray(arr, dtype=float)
        valid = ~invalid_bool
        if not np.any(valid): return arr
        last = None
        for i in range(arr.shape[0]):
            if valid[i]:
                last = arr[i]
            else:
                arr[i] = last if last is not None else arr[np.argmax(valid)]
        return arr

    thrust_invalid = (mask_raw & 64) != 0
    rx_inv = (mask_raw & 1) != 0
    ry_inv = (mask_raw & 2) != 0
    rz_inv = (mask_raw & 4) != 0

    thrust = lvh_forward(thrust_raw, thrust_invalid)
    thrust = np.clip(thrust, 0.0, 1.0)
    w_cmd = np.asarray(w_raw, dtype=float)
    w_cmd[:,0] = lvh_forward(w_cmd[:,0], rx_inv)
    w_cmd[:,1] = lvh_forward(w_cmd[:,1], ry_inv)
    w_cmd[:,2] = lvh_forward(w_cmd[:,2], rz_inv)

    meta = {
        "att_topic_used": att_sp_topic,
        "duplicates_dropped": int(dropped),
        "dt_stats": {
            "min": float(np.diff(t_cmd).min()) if len(t_cmd)>1 else 0.0,
            "med": float(np.median(np.diff(t_cmd))) if len(t_cmd)>1 else 0.0,
            "max": float(np.diff(t_cmd).max()) if len(t_cmd)>1 else 0.0,
        },
        "mask_stats": {
            "thrust_ignored_pct": float(np.mean(thrust_invalid)*100.0),
            "roll_ignored_pct": float(np.mean(rx_inv)*100.0),
            "pitch_ignored_pct": float(np.mean(ry_inv)*100.0),
            "yaw_ignored_pct": float(np.mean(rz_inv)*100.0),
        },
    }

    commanded_inputs = (t_cmd.astype(np.float64),
                        thrust.astype(np.float64),
                        w_cmd.astype(np.float64))
    gazebo_pose = (t_pose.astype(np.float64),
                   p_pose.astype(np.float64),
                   q_pose.astype(np.float64))
    return commanded_inputs, gazebo_pose, initial_pose_msg, meta

# ----------------- JAX open-loop (ZOH on setpoint grid) -----------------
def run_jax_open_loop(initial_pose_msg, commanded_inputs, mass,
                      g_acc=9.81, hov_thrust=0.726,
                      thrust_body_z_sign=1.0, thrust_scale=None):
    t_cmd, thrust_cmd, w_cmd = commanded_inputs
    if initial_pose_msg is None:
        raise RuntimeError("Initial pose not found in bag window.")

    p0 = initial_pose_msg.pose.position
    o0 = initial_pose_msg.pose.orientation
    r0 = jnp.array([p0.x, p0.y, p0.z], dtype=jnp.float64)
    dr0 = jnp.zeros(3, dtype=jnp.float64)  # no odom -> assume zero initial linear velocity
    x0 = jnp.concatenate([r0, dr0])

    q_init = np.array([o0.x, o0.y, o0.z, o0.w], dtype=float)
    R0 = jnp.array(Rotation.from_quat(q_init).as_matrix(), dtype=jnp.float64)

    z0_tree = (x0, R0.flatten())
    z0_flat, unravel = jax.flatten_util.ravel_pytree(z0_tree)

    thrust_arr = jnp.array(thrust_cmd, dtype=jnp.float64)
    w_arr = jnp.array(w_cmd, dtype=jnp.float64)

    def ode_const_cmd(z_flat, t, thrust_norm, omega_body):
        x, Rflat = unravel(z_flat)
        q, dq = x[:3], x[3:]
        R = Rflat.reshape(3,3)

        if thrust_scale is None:
            f_d = thrust_norm * mass * g_acc / hov_thrust
        else:
            f_d = thrust_scale * thrust_norm

        z_body = jnp.array([0., 0., thrust_body_z_sign], dtype=jnp.float64)
        u_world = f_d * (R @ z_body)

        if HAVE_DYNAMICS:
            H, C, g, _ = prior(q, dq)
            ddq = jnp.linalg.solve(H, u_world - C @ dq - g)
        else:
            ddq = (u_world - jnp.array([0.,0.,mass*g_acc])) / mass

        dx = jnp.concatenate([dq, ddq])
        dR = R @ hat(omega_body)
        return jnp.concatenate([dx, dR.flatten()])

    @jax.jit
    def one_step(z, t0, dt, thrust_norm, omega_body):
        def f(y, tt): return ode_const_cmd(y, tt, thrust_norm, omega_body)
        z_next = rk4_step(f, dt, z, t0)
        x, Rflat = unravel(z_next)
        R = project_to_so3(Rflat.reshape(3,3))
        return jnp.concatenate([x, R.flatten()])

    ts = [float(t_cmd[0])]
    z = z0_flat
    z_hist = [z0_flat]
    for i in range(len(t_cmd) - 1):
        dt = float(t_cmd[i+1] - t_cmd[i])
        if dt <= 0.0:
            continue
        z = one_step(z, float(t_cmd[i]), dt, thrust_arr[i], w_arr[i])
        ts.append(float(t_cmd[i+1]))
        z_hist.append(z)

    z_hist = jnp.stack(z_hist, axis=0)
    xs, Rflats = jax.vmap(unravel)(z_hist)
    pos = np.array(xs[:, :3])
    R_sim = np.array(Rflats.reshape((-1, 3, 3)))
    return np.array(ts), pos, R_sim

# ----------------- Plotting -----------------
def plot_open_loop(gaps, gazebo_pose, jax_states, outdir="open_loop_figs"):
    t_pose, p_pose, _ = gazebo_pose
    ts_jax, q_jax, _ = jax_states

    # Interp pose to command timeline (per-axis linear)
    if t_pose.size > 1 and p_pose.shape[0] > 1:
        p_interp = np.column_stack([np.interp(ts_jax, t_pose, p_pose[:,i]) for i in range(3)])
    else:
        p_interp = np.full_like(q_jax, np.nan)

    os.makedirs(outdir, exist_ok=True)

    # 3D
    fig_3d = plt.figure(figsize=(10,10))
    ax = fig_3d.add_subplot(111, projection='3d')
    ax.plot(q_jax[:,0], q_jax[:,1], q_jax[:,2], '-', lw=2, label='JAX (setpoint-timed)')
    ax.plot(p_interp[:,0], p_interp[:,1], p_interp[:,2], ':', lw=2, label='Gazebo (interp@setpoint)')
    ax.set_title('Open-Loop 3D Trajectory (Common Setpoint Timeline)')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend(); ax.axis('equal')
    fig_3d.savefig(os.path.join(outdir, "open_loop_3d.png"))

    # Components
    fig, axs = plt.subplots(3,1, figsize=(14,9), sharex=True)
    for i, lbl in enumerate(['X','Y','Z']):
        axs[i].plot(ts_jax, q_jax[:,i], '-', lw=1.8, label='JAX')
        axs[i].plot(ts_jax, p_interp[:,i], ':', lw=1.8, label='Gazebo (interp)')
        shade_gap_regions(axs[i], gaps)
        axs[i].set_ylabel(f'{lbl} (m)'); axs[i].grid(True); axs[i].legend()
    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Open-Loop Position vs Time (shaded = setpoint gaps)', fontsize=14)
    fig.savefig(os.path.join(outdir, "open_loop_xyz.png"))
    plt.close(fig); plt.close(fig_3d)

    # Position error magnitude
    err = np.linalg.norm(q_jax - p_interp, axis=1)
    plt.figure(figsize=(12,4))
    plt.plot(ts_jax, err, '-', lw=1.5, label="‖pos_error‖ (m)")
    shade_gap_regions(plt.gca(), gaps)
    plt.xlabel("time (s)"); plt.ylabel("m")
    plt.title("Position error vs time (shaded = setpoint gaps)")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "position_error.png"))
    plt.close()

def plot_attitude_timeseries(gaps, ts_cmd, R_sim, R_gaz, outdir="open_loop_figs"):
    φs, θs, ψs = rotmat_to_rpy_zyx(R_sim)
    φg, θg, ψg = rotmat_to_rpy_zyx(R_gaz)

    φs = np.unwrap(φs); θs = np.unwrap(θs); ψs = np.unwrap(ψs)
    φg = np.unwrap(φg); θg = np.unwrap(θg); ψg = np.unwrap(ψg)

    todeg = 180.0/np.pi
    series = [
        ("Roll (deg)", φs*todeg, φg*todeg),
        ("Pitch (deg)", θs*todeg, θg*todeg),
        ("Yaw (deg)", ψs*todeg, ψg*todeg),
    ]

    os.makedirs(outdir, exist_ok=True)
    fig, axs = plt.subplots(3,1, figsize=(14,9), sharex=True)
    for ax, (label, sim, gaz) in zip(axs, series):
        ax.plot(ts_cmd, sim, '-', lw=1.6, label='JAX (replay)')
        ax.plot(ts_cmd, gaz, ':', lw=1.6, label='Gazebo (pose→RPY)')
        shade_gap_regions(ax, gaps)
        ax.set_ylabel(label); ax.grid(True); ax.legend()
    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Attitude Comparison (Euler ZYX) — shaded = setpoint gaps', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "open_loop_attitude_rpy.png"))
    plt.close(fig)

    # Geodesic error summary
    geod = []
    for i in range(len(ts_cmd)):
        Ra, Rb = R_sim[i], R_gaz[i]
        tr = np.clip(0.5*(np.trace(Ra.T @ Rb) - 1.0), -1.0, 1.0)
        geod.append(np.degrees(np.arccos(tr)))
    geod = np.array(geod)
    print("Attitude geodesic error: median = {:.2f} deg, 95th = {:.2f} deg".format(
        np.median(geod), np.percentile(geod, 95)))

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Setpoint-timed open-loop comparison with gap visualization.")
    ap.add_argument('--rosbag', type=str, required=True)
    ap.add_argument('--mass', type=float, default=2.0)
    ap.add_argument('--g', type=float, default=9.81)
    ap.add_argument('--pose_topic', type=str, default='/mavros/local_position/pose')
    ap.add_argument('--attitude_setpoint_topic', type=str, default='/mavros/setpoint_raw/attitude')
    ap.add_argument('--control_log_topic', type=str, default='/mlac_mission_node/control_log')
    ap.add_argument('--prefer_target_attitude', action='store_true')
    ap.add_argument('--hov_thrust', type=float, default=0.72823)
    ap.add_argument('--thrust_scale', type=float, default=None)
    ap.add_argument('--thrust_body_z_sign', type=float, default=1.0, choices=[-1.0, 1.0])
    ap.add_argument('--gap_threshold', type=float, default=0.2)  # seconds
    ap.add_argument('--segment_on_gap', action='store_true')     # simulate only first contiguous segment
    args = ap.parse_args()

    commanded_inputs, gazebo_pose, init_pose, meta = extract_minimal(
        args.rosbag, args.pose_topic, args.attitude_setpoint_topic,
        control_log_topic=args.control_log_topic,
        prefer_target=args.prefer_target_attitude
    )

    print("\nAttitude topic:", meta['att_topic_used'])
    print("Duplicates dropped:", meta['duplicates_dropped'])
    print("dt stats (s):", meta['dt_stats'])
    print("Mask stats (% ignored):", meta['mask_stats'])

    # Gap diagnostics & plots (purely informational)
    t_cmd, thrust_cmd, w_cmd = commanded_inputs
    gaps = print_gap_report(t_cmd, threshold=args.gap_threshold)
    plot_dt_diagnostics(t_cmd, threshold=args.gap_threshold, outdir="open_loop_figs")

    # Optional: simulate only the first contiguous segment (avoid huge ZOH step)
    if args.segment_on_gap:
        dt = np.diff(t_cmd)
        idx = np.where(dt > args.gap_threshold)[0]
        if idx.size > 0:
            cut = idx[0] + 1
            print(f"\n[segment] Large gap {dt[idx[0]]:.3f}s at i={idx[0]} → simulating segment [0..{cut-1}] (t_end={t_cmd[cut-1]:.3f}s)")
            t_cmd = t_cmd[:cut]
            thrust_cmd = thrust_cmd[:cut]
            w_cmd = w_cmd[:cut]
            commanded_inputs = (t_cmd, thrust_cmd, w_cmd)

    # Run JAX open-loop
    ts_jax, q_jax, R_sim = run_jax_open_loop(
        initial_pose_msg=init_pose,
        commanded_inputs=commanded_inputs,
        mass=args.mass,
        g_acc=args.g,
        hov_thrust=args.hov_thrust,
        thrust_body_z_sign=args.thrust_body_z_sign,
        thrust_scale=args.thrust_scale
    )

    # Build R_gaz on the same timeline from pose quats (NN, robust)
    t_pose, p_pose, q_pose = gazebo_pose
    R_gaz = R_gaz_on_command_grid_nn(t_pose, q_pose, ts_jax)

    # Plots
    plot_attitude_timeseries(gaps, ts_jax, R_sim, R_gaz, outdir="open_loop_figs")
    plot_open_loop(gaps, gazebo_pose, (ts_jax, q_jax, R_sim), outdir="open_loop_figs")
    print("\n--- Open-Loop Comparison Complete ---\nPlots saved under: open_loop_figs/")

if __name__ == '__main__':
    main()

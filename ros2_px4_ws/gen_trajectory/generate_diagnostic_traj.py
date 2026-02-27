#!/usr/bin/env python3
"""
Generate a multi-phase diagnostic trajectory for controller validation.

Each phase isolates a single degree of freedom so bugs can be pinpointed.
Transitions use cosine ramps for smooth velocity/acceleration profiles.

Phases:
  1. Hover          — hold position, validates thrust model
  2. Z step         — climb 1 m, tests vertical dynamics
  3. X step         — move 2 m forward, tests pitch coupling
  4. Y step         — move 2 m left, tests roll coupling
  5. Yaw 90°        — rotate in place, tests yaw dynamics
  6. Combined return — return to start while yawing back

Output: 11-column .npy  [t, px, py, pz, vx, vy, vz, psi, ax, ay, az]
"""

import argparse
import os
import numpy as np


def cosine_ramp(t, t_start, t_end, val_start, val_end):
    """Cosine interpolation: smooth start and stop between two values."""
    tau = np.clip((t - t_start) / (t_end - t_start), 0.0, 1.0)
    alpha = 0.5 * (1.0 - np.cos(np.pi * tau))
    return val_start + (val_end - val_start) * alpha


def cosine_ramp_dot(t, t_start, t_end, val_start, val_end):
    """First derivative of cosine ramp."""
    dur = t_end - t_start
    tau = np.clip((t - t_start) / dur, 0.0, 1.0)
    in_range = (t >= t_start) & (t <= t_end)
    return np.where(in_range,
                    (val_end - val_start) * np.pi / (2.0 * dur) * np.sin(np.pi * tau),
                    0.0)


def cosine_ramp_ddot(t, t_start, t_end, val_start, val_end):
    """Second derivative of cosine ramp."""
    dur = t_end - t_start
    tau = np.clip((t - t_start) / dur, 0.0, 1.0)
    in_range = (t >= t_start) & (t <= t_end)
    return np.where(in_range,
                    (val_end - val_start) * (np.pi / (2.0 * dur))**2
                    * np.pi * np.cos(np.pi * tau) / np.pi,  # simplifies
                    0.0)


def generate_diagnostic(dt=0.02, hover_pos=None, hold_sec=3.0, ramp_sec=3.0):
    """Build the diagnostic trajectory phase by phase."""
    if hover_pos is None:
        hover_pos = np.array([0.0, -2.0, 1.5])

    x0, y0, z0 = hover_pos
    psi0 = 0.0

    # Phase timeline: [description, duration, end_pos, end_psi]
    waypoints = [
        ("hover",    hold_sec, [x0,       y0,       z0      ], psi0),
        ("z_up",     ramp_sec, [x0,       y0,       z0 + 1.0], psi0),
        ("z_hold",   hold_sec, [x0,       y0,       z0 + 1.0], psi0),
        ("x_fwd",    ramp_sec, [x0 + 2.0, y0,       z0 + 1.0], psi0),
        ("x_hold",   hold_sec, [x0 + 2.0, y0,       z0 + 1.0], psi0),
        ("y_left",   ramp_sec, [x0 + 2.0, y0 + 2.0, z0 + 1.0], psi0),
        ("y_hold",   hold_sec, [x0 + 2.0, y0 + 2.0, z0 + 1.0], psi0),
        ("yaw_90",   ramp_sec, [x0 + 2.0, y0 + 2.0, z0 + 1.0], np.pi / 2),
        ("yaw_hold", hold_sec, [x0 + 2.0, y0 + 2.0, z0 + 1.0], np.pi / 2),
        ("return",   ramp_sec + 1.0, [x0, y0, z0], psi0),
        ("final",    hold_sec, [x0, y0, z0], psi0),
    ]

    # Build phase boundaries
    phase_times = [0.0]
    for _, dur, _, _ in waypoints:
        phase_times.append(phase_times[-1] + dur)
    total_T = phase_times[-1]

    times = np.arange(0, total_T + dt / 2, dt)
    N = len(times)

    pos = np.zeros((N, 3))
    vel = np.zeros((N, 3))
    acc = np.zeros((N, 3))
    psi = np.zeros(N)

    # Current state at the start of each phase
    cur_pos = np.array([x0, y0, z0], dtype=float)
    cur_psi = psi0

    for phase_idx, (name, dur, end_pos_list, end_psi) in enumerate(waypoints):
        t_start = phase_times[phase_idx]
        t_end = phase_times[phase_idx + 1]
        end_pos = np.array(end_pos_list, dtype=float)

        mask = (times >= t_start) & (times < t_end)
        if phase_idx == len(waypoints) - 1:
            mask = (times >= t_start) & (times <= t_end)

        t_seg = times[mask]

        for ax_i in range(3):
            pos[mask, ax_i] = cosine_ramp(t_seg, t_start, t_end, cur_pos[ax_i], end_pos[ax_i])
            vel[mask, ax_i] = cosine_ramp_dot(t_seg, t_start, t_end, cur_pos[ax_i], end_pos[ax_i])
            acc[mask, ax_i] = cosine_ramp_ddot(t_seg, t_start, t_end, cur_pos[ax_i], end_pos[ax_i])

        psi[mask] = cosine_ramp(t_seg, t_start, t_end, cur_psi, end_psi)

        cur_pos = end_pos.copy()
        cur_psi = end_psi

    traj = np.column_stack([times, pos, vel, psi, acc])
    assert traj.shape[1] == 11, f"Expected 11 columns, got {traj.shape[1]}"

    return traj, waypoints, phase_times


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic trajectory for controller validation.")
    parser.add_argument("--dt", type=float, default=0.02, help="Time step (s), default 50Hz")
    parser.add_argument("--hover-x", type=float, default=0.0)
    parser.add_argument("--hover-y", type=float, default=-2.0)
    parser.add_argument("--hover-z", type=float, default=1.5)
    parser.add_argument("--hold-sec", type=float, default=3.0, help="Hold duration per phase (s)")
    parser.add_argument("--ramp-sec", type=float, default=3.0, help="Ramp transition duration (s)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Defaults to mlac_sim/traj_data/")
    args = parser.parse_args()

    hover_pos = np.array([args.hover_x, args.hover_y, args.hover_z])

    traj, waypoints, phase_times = generate_diagnostic(
        dt=args.dt, hover_pos=hover_pos,
        hold_sec=args.hold_sec, ramp_sec=args.ramp_sec,
    )

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        out_dir = os.path.abspath(os.path.join(script_dir, '..', 'src', 'mlac_sim', 'traj_data'))
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    hz = int(1.0 / args.dt)
    fname = f"diagnostic_{hz}hz_11col.npy"
    fpath = os.path.join(out_dir, fname)
    np.save(fpath, traj)

    print(f"Saved: {fpath}")
    print(f"Shape: {traj.shape}  Duration: {traj[-1, 0]:.1f}s")
    print(f"\nPhase timeline:")
    for i, (name, dur, end_pos, end_psi) in enumerate(waypoints):
        t0 = phase_times[i]
        t1 = phase_times[i + 1]
        print(f"  [{t0:5.1f} - {t1:5.1f}s]  {name:12s}  -> pos={end_pos}, psi={np.rad2deg(end_psi):.0f}°")


if __name__ == "__main__":
    main()

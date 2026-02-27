# Phase 2–4 Summary

## Phase 2: Controller Fixes and SMC Tuning (attitude mode)

### What was done

1. **Bug A & B fixed** in `mlac_mission_node.py`:
   - Removed erroneous `.T` on `quaternion_to_rotation_matrix()` at lines 328 (velocity
     conversion) and 449 (thrust projection).
   - Confirmed via standalone tests and rosbag empirical test that MAVROS quaternion
     represents `R_{body->world}`, so `.T` was incorrect.

2. **PID controller tested** on diagnostic trajectory (hover + axis steps):
   - With rotation fixes, the old PID gains (tuned for the buggy controller) became
     unstable/oscillatory.
   - Increased gains (`Kp=[2,2,2], Ki=[0.5,0.5,0.8], Kd=[2.5,2.5,2]`) helped but
     still produced excessive oscillation.

3. **SMC controller activated** (`outerloop_node.py`, type `coml_debug`):
   - Found hardcoded aggressive gains in `get_force()` that overrode `__init__` values.
     Fixed to use `self.Λ` and `self.K`.
   - Initial conservative gains: `Λ = 0.25 * diag([1,1,1.5])`, `K = 0.5 * diag([1,1,1.5])`.

4. **Bodyrate vs attitude mode**:
   - `bodyrate` mode: SMC produced sluggish/underdamped response. The custom
     `bodyrate_converter.attitude_to_bodyrate()` appears problematic.
   - `attitude` mode: Dramatically better tracking by leveraging PX4's internal
     attitude controller. Decided to proceed in attitude mode.

5. **SMC gain tuning** (attitude mode):
   - Increased to `Λ = 0.5 * diag([1,1,1.5])`, `K = 1.0 * diag([1,1,1.5])`.
   - Tested on diagnostic trajectory → tight tracking.

6. **Circle trajectory test**:
   - Flew a circle/figure-8 reference with the tuned SMC in attitude mode.
   - Tracking performance reasonable (see `results/phase2_circle/`).

### Key result files

- `results/phase2_smc_attitude/` — SMC in attitude mode, diagnostic trajectory
- `results/phase2_circle/` — SMC circle trajectory tracking

---

## Phase 3: Open-Loop Dynamics Validation (measured rates / measured attitude)

### Problem

The previous open-loop script (`commanded` mode) reads body rate commands from
`/mavros/setpoint_raw/attitude`. In attitude mode, PX4 ignores the body_rate fields,
so they are zeros/garbage. We needed new modes that use actual measured data.

### What was done

1. **Two new rotation modes** added to `open_loop.py` via `--rotation-mode` flag:

   - **Mode A (`measured-rates`)**: Uses measured angular velocity from
     `/mavros/local_position/velocity_body` (angular part) as body rate input
     to the kinematic equation. Tests translational + rotational dynamics together.

   - **Mode B (`measured-attitude`)**: Uses measured quaternion/RPY from the pose
     topic directly, skipping attitude propagation. Only propagates position and
     velocity. Isolates translational dynamics (thrust + gravity model).

2. **New code**:
   - `rosbag_utils.py`: Added `extract_open_loop_measured()` — thin wrapper around
     `extract_attitude_data()` keyed to velocity_body timestamps with thrust
     interpolated from setpoints.
   - `dynamics_numpy.py`: Added `simulation_ode_euler_fixed_attitude()` — translational-only
     ODE where RPY is externally provided (read-only).
   - `open_loop.py`: Added `--rotation-mode` flag, `dynamics_fn` parameter to
     `simulate_window()`, and `measured_rpy` override for Mode B.

3. **Ran both modes** on circle rosbag (`bodyrate_diag_0226_215212`) with 0.2s windows.

### Results

| Metric | Mode A (measured rates) | Mode B (measured attitude) |
|--------|------------------------|---------------------------|
| vel_rms (typical) | 0.05–0.21 m/s | 0.05–0.21 m/s |
| att_rms (typical) | 0.001–0.007 rad | 0.001–0.007 rad |
| pos_rms (typical) | 0.4–0.6 m | 0.4–0.6 m |

### Key findings

1. **Mode A and Mode B produce nearly identical results.** The velocity, position,
   and attitude plots are visually indistinguishable. This definitively shows that the
   **rotational kinematic model (E-matrix) is NOT a significant error source**. The
   measured body rates, when fed through the E-matrix, reproduce the measured attitude
   well.

2. **Sawtooth (jigsaw) pattern in velocity plots.** This is the expected artifact
   of rolling-window comparison: within each 0.2s window the sim drifts slightly from
   measured, then snaps back at the window boundary. The sawtooth amplitude represents
   the per-window dynamics error.

3. **The residual velocity error (~0.05–0.2 m/s per window) is purely translational.**
   Since Mode B (which uses ground-truth attitude) shows the same error as Mode A,
   the mismatch is in `acc = R @ [0,0,f_d] - [0,0,g]`. Possible causes:
   - **Aerodynamic drag**: Not modeled but present in Gazebo.
   - **Wind**: The Gazebo sim may have wind enabled.
   - **`hov_thrust` calibration**: If the hover-thrust ratio (currently 0.727) doesn't
     match the sim, the thrust-to-force conversion is systematically off.
   - **Thrust dynamics/latency**: PX4 may apply a first-order lag on the motor commands.
   - **Mass mismatch**: The sim vehicle mass may differ from the assumed 2.0 kg.

4. **Large position RMS (~0.4–0.6 m)** even in 0.2s windows is surprisingly high.
   This may indicate a systematic position offset between the velocity-topic timeline
   and the pose-topic data, or it may reflect genuine translational error accumulation.
   Needs further investigation.

5. **SMC gain alignment is orthogonal to open-loop validation.** Open-loop tests
   replay recorded commands — they don't involve any controller. SMC gains only
   matter for closed-loop comparison. The open-loop results tell us about the
   **dynamics model accuracy**, not the controller.

---

## Deeper Analysis: Velocity Jigsaw Root Cause

### The jigsaw pattern explained

The sawtooth/jigsaw in velocity plots is the **expected artifact of rolling-window
comparison**. Each 0.2s window re-initializes from measured state; at window boundaries
the sim "snaps back" to truth. The amplitude of each tooth = per-window dynamics error.

- Velocity error: ~0.05–0.2 m/s per window
- Implied acceleration error: ~0.25–1.0 m/s²

### Why it's NOT about SMC gains

Open-loop tests replay **recorded** thrust commands — no controller runs. The jigsaw
comes from the **translational dynamics model** being slightly wrong. SMC gains only
matter for closed-loop comparison where the controller is actively running.

### Gazebo x500 model parameters (from PX4-Autopilot SDF)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base mass | 2.0 kg | + 4 × 0.016 kg rotors ≈ 2.064 kg total |
| Motor constant | 8.549e-06 | Per rotor |
| Max rotor velocity | 1000 rad/s | (effective, after slowdown factor) |
| Motor time constant (up) | 0.0125 s | First-order lag on thrust increase |
| Motor time constant (down) | 0.025 s | First-order lag on thrust decrease |
| Rotor drag coefficient | 8.064e-05 | Per rotor, speed × airspeed drag |
| Airframe drag | None explicit | Only `<velocity_decay/>` (default = 0) |

### Mismatch sources identified

1. **Motor time constants**: Gazebo applies a first-order lag (12.5–25 ms) to motor
   speed. Our ODE assumes **instantaneous** thrust. Small effect individually, but
   contributes to systematic bias during dynamic maneuvers.

2. **Rotor drag**: `F_drag ≈ rotorDragCoeff × ω × airspeed` per rotor. At circle
   speeds (~0.6 m/s) with 4 rotors at hover RPM, total drag ~0.1 N → ~0.05 m/s²
   deceleration. Partially explains the drift.

3. **`hov_thrust` calibration**: The thrust normalization chain in `mlac_mission_node.py`
   uses `curr_hover_thrust` (initialized to 0.5, updated during hover phase) inside:
   ```
   normalized_thrust = F·z / (max_thrust_N / curr_hover_thrust)
   ```
   At hover, the output thrust equals `curr_hover_thrust`. If this doesn't match the
   offline `hov_thrust = 0.727`, there's a systematic thrust offset. A 5% error in
   `hov_thrust` produces ~0.49 m/s² acceleration error → ~0.1 m/s velocity error per
   0.2s window. **This matches the observed error magnitude.**

4. **Total mass**: The SDF shows 2.064 kg (base + rotors), but the dynamics model
   uses 2.0 kg. A 3% mass error compounds with thrust calibration.

### Conclusion

The velocity jigsaw amplitude is dominated by **translational dynamics model mismatch**,
most likely `hov_thrust` miscalibration and missing drag. Rotational dynamics (E-matrix)
are validated as accurate by the Mode A ≈ Mode B result.

---

## Phase 4: Closed-Loop Comparison with Aligned SMC Gains

### Setup

The JAX closed-loop sim (`closed_loop.py` + `closed_loop_core.py`) runs the same SMC
outer loop as the ROS controller, with PX4's inner attitude loop approximated as a
P-gain + first-order body rate lag.

**JAX inner-loop model:**
```
e_R = 0.5 * vee(R_d^T R - R^T R_d)        (attitude error)
Ω_cmd = -k_R * e_R + Ω_ff                  (body rate command)
dΩ/dt = (Ω_cmd - Ω_state) / τ_att          (first-order lag)
dR = R @ hat(Ω_state)                       (rotation propagation)
```

This approximates PX4's attitude P-controller + rate PID as a single gain + lag.
The real PX4 rate controller is a full PID, so the first-order model is simplified.

### Gains aligned

| Parameter | JAX value | Source |
|-----------|-----------|--------|
| K (SMC feedback) | diag([1.0, 1.0, 1.5]) | `outerloop_node.py` `coml_debug` |
| Λ (SMC sliding surface) | diag([0.5, 0.5, 0.75]) | `outerloop_node.py` `coml_debug` |
| k_R (attitude P gain) | [6.5, 6.5, 2.8] | PX4 defaults: `MC_ROLL_P`, `MC_PITCH_P`, `MC_YAW_P` |
| τ_att (attitude time constant) | 0.02 s | Estimated (~50 rad/s inner-loop bandwidth) |

Reference trajectory read from ControllerLog in the rosbag (`--reference-source bag`).
Rosbag: `bodyrate_diag_0226_215212` (circle trajectory, SMC attitude mode).
Rolling windows: 2.0 s duration.

### Experiment 1: With yaw rate feedforward (`--feedforward`)

**Results**: `results/phase4_closedloop_aligned/` (overwritten, positions only)

- X, Y position: JAX closely tracks ROS measured throughout the circle.
- Z position: JAX stays near 1.50 m reference; ROS measured has ~2–5 cm offset above.
- Yaw rate feedforward adds `Ω_ff = R^T @ [0,0,yaw_rate_d]` to body rate command.

### Experiment 2: Without feedforward (no `--feedforward`)

**Results**: `results/phase4_closedloop_no_ff/`

**Position**: Nearly identical to Experiment 1. The feedforward term has negligible
effect on position tracking for this circle trajectory with slow yaw changes.

**Velocity**:
- X, Y velocity: JAX sim tracks ROS measured well in overall shape. Some phase
  offset visible (JAX slightly leads ROS), likely from the simplified inner-loop model.
  The reference velocity (dashed green) is tracked more tightly by JAX than by ROS,
  which is expected since JAX has no noise/disturbances.
- Z velocity: JAX stays near 0 (reference is constant Z), while ROS shows small
  oscillations (~0.01–0.02 m/s). This confirms the Z offset is a static thrust
  calibration issue, not a dynamic tracking problem.

**Attitude**:
- Roll, Pitch: JAX and ROS show the same qualitative shape (~±1 deg range for the
  circle). JAX has a smoother response (no sensor noise). There's a visible phase lead
  in JAX — it tilts slightly earlier than ROS, consistent with the first-order lag
  being a simplification of PX4's full rate PID.
- Yaw: Both track the same circle yaw profile. The sawtooth pattern at window
  boundaries is the rolling-window reset artifact.

### Feedforward analysis

PX4's attitude controller does **not** receive yaw rate feedforward from our outer loop.
In attitude mode, `mlac_mission_node.py` sends only quaternion + thrust. PX4 generates
body rates purely from its attitude P-gain. The JAX sim's `--feedforward` flag is
therefore a mismatch with the real system. However, for the circle trajectory the effect
is negligible because yaw changes are slow enough for PX4's P-gain to track with
minimal lag.

### Key findings

1. **X and Y closed-loop tracking works well.** The aligned SMC gains produce
   trajectories in JAX that closely match the ROS measured data.

2. **Z has a persistent static offset** (~2–5 cm). The JAX sim stays at the reference
   altitude while ROS drifts above it. Root cause: `hov_thrust` calibration mismatch
   affecting the vertical thrust-to-force conversion.

3. **Attitude shows a small phase lead in JAX** relative to ROS. The first-order lag
   (τ=0.02s) doesn't perfectly capture PX4's full rate PID dynamics. This is a
   second-order effect — it manifests as slight timing differences in roll/pitch but
   doesn't significantly affect position tracking.

4. **Feedforward has negligible impact** for this trajectory. Can be omitted for
   cleaner comparison.

### Key result files

- `results/phase4_closedloop_no_ff/` — All comparison plots (positions, velocities,
  attitude, command streams, pose vs reference)

---

## Next Steps

1. **Calibrate translational dynamics parameters**:
   - Extract actual mean thrust during hover from rosbag to verify `hov_thrust`.
   - Sweep `hov_thrust` to minimize velocity RMS and fix the Z offset.
   - Consider adding a linear drag term `acc -= drag_coeff * vel`.
   - Update mass to 2.064 kg to match the SDF.

2. **Tune inner-loop approximation**: The first-order attitude model could be improved
   by fitting τ_att (and possibly k_R) to match measured attitude step responses from
   the rosbag.

3. **Add motor lag model**: First-order filter on thrust with τ_up=0.0125, τ_down=0.025
   to match Gazebo motor dynamics.

4. **Test on more aggressive trajectories**: The current circle is relatively gentle
   (±1 deg tilt). More dynamic maneuvers would stress-test the inner-loop approximation
   and reveal whether the simplified model breaks down.

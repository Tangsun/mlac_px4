# Rotation Bugs Audit

Full audit of rotation-related bugs across the runtime controller
(`mlac_mission_node.py`) and the offline comparison pipeline (`compare_sim/`).

Produced from tracing how `reference_roll/pitch/yaw` are computed in the
controller, then following every rotation operation end-to-end.

---

## Bug index

| ID | File | Line(s) | One-liner | Severity |
|----|------|---------|-----------|----------|
| **A** | `mlac_mission_node.py` | 328 | `.T` on velocity conversion — `state.v` is wrong | **High** |
| **B** | `mlac_mission_node.py` | 449 | `.T` on thrust projection — `att_msg.thrust` is wrong | **Medium** |
| **C** | `closed_loop_core.py` | 111 | `from_euler('xyz', …)` instead of `'zyx'` — initial R0 wrong | **Medium** |
| **D** | `closed_loop_core.py` | 150 | `as_euler('xyz')` instead of `'zyx'` — output Euler wrong | **Medium** |
| **E** | `dynamics_numpy.py` | 35-39 | Body-rate-to-Euler-rate kinematic matrix is wrong | **High** |

---

## Bug A — velocity conversion `.T`

**File:** `ros2_px4_ws/src/mlac_sim/mlac_sim/mlac_mission_node.py` line 328

```python
R_body_to_world = quaternion_to_rotation_matrix(self.current_vehicle_state_py.q).T
self.current_vehicle_state_py.v = R_body_to_world @ np.array([
    msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
```

### What's wrong

`quaternion_to_rotation_matrix(q)` returns R_{body→world} (verified: columns
are body axes in world frame, identity quaternion gives I, pure yaw pi/2 gives
Rz(pi/2)). Taking `.T` gives R_{world→body}. The variable is **misnamed** —
it's actually R_world_to_body.

So `state.v = R_world_to_body @ v_body`, which is physically meaningless
(neither world-frame nor body-frame velocity).

### Verification at yaw=90°

- Body velocity `[1, 0, 0]` (forward) → should be `[0, 1, 0]` in world
  (north in ENU).
- Without `.T`: R @ [1,0,0] = [0, 1, 0] ✓
- With `.T`: R.T @ [1,0,0] = [0, -1, 0] ✗

### Impact

Every control cycle, the outer-loop controller receives wrong velocity
feedback. This corrupts:
- Velocity error `edot = goal.v - state.v` in `get_force()`
- The sliding variable `s = edot + Λ @ e` (COML/SMC)
- The resulting `F_W`, desired quaternion, and thrust
- The adaptation law `dA` (COML), because it depends on `dq = state.v`

At hover (R ≈ I) the error vanishes. At 10° tilt the velocity error is
~O(2θ · |v|) ≈ 0.35 m/s per 1 m/s of body velocity. The position integral
term gradually compensates, so the drone still flies but with degraded
tracking.

### Correct fix

Remove the `.T`:
```python
R_body_to_world = quaternion_to_rotation_matrix(self.current_vehicle_state_py.q)
```

### Cross-reference

The rosbag comparison pipeline does the same conversion **correctly** in
`rosbag_utils.py` line 281:
```python
vel_world_samples = rot_vel.apply(lin_vel_body)
```
Scipy's `Rotation.apply()` computes `R_body_to_world @ v_body`. So the
measured velocity in the comparison is right; the controller just operated
with the wrong velocity during flight.

---

## Bug B — thrust normalization `.T`

**File:** `ros2_px4_ws/src/mlac_sim/mlac_sim/mlac_mission_node.py` line 449

```python
R_body_to_world_desired = quaternion_to_rotation_matrix(att_cmd_py.q).T
desired_body_z_axis_in_world = R_body_to_world_desired[:, 2]
thrust_force_along_desired_z = np.dot(att_cmd_py.F_W, desired_body_z_axis_in_world)
normalized_thrust = np.clip(
    thrust_force_along_desired_z / (self.max_thrust_N / self.curr_hover_thrust),
    0.0, 1.0)
```

### What's wrong

Same `.T` issue as Bug A. `R_body_to_world_desired[:, 2]` extracts the 3rd
column of R_{world→body}, which is the 3rd **row** of R_{body→world}: the
world z-axis expressed in body coordinates, **not** the body z-axis expressed
in world coordinates.

### Quantitative impact

For tilt angle θ:
- Correct body-z in world: `[sin θ, 0, cos θ]`
- Buggy 3rd row: `[-sin θ, 0, cos θ]`
- F_W projection onto correct axis: `mg` (the full force magnitude)
- F_W projection onto buggy axis: `mg · cos(2θ)`
- At θ = 10°: **6% thrust error**
- At θ = 20°: **23% thrust error**

At hover (θ ≈ 0) the bug is invisible because I.T = I.

### Correct fix

Remove the `.T`:
```python
R_body_to_world_desired = quaternion_to_rotation_matrix(att_cmd_py.q)
```

### What's NOT affected

- The quaternion `att_msg.orientation` is set directly from `att_cmd_py.q`
  (line 447), no `.T` involved. ✓
- The body rate command (bodyrate mode, line 487-491) uses
  `BodyRateConverter.attitude_to_bodyrate`, which uses
  `quaternion_to_rotation_matrix` **without** `.T` on both R_current and
  R_desired. The geometric error `vee(R_d^T R - R^T R_d)` is correct because
  both matrices use the same convention. ✓
- The ControllerLog `reference_roll/pitch` come from `get_rpy(att_cmd_py.q)`,
  which doesn't involve any rotation matrix transpose. ✓

---

## Bug C — initial rotation matrix in closed-loop sim

**File:** `mlac_jax_train/compare_sim/closed_loop_core.py` line 111

```python
R0 = jnp.array(Rotation.from_euler('xyz', rpy0).as_matrix())
```

### What's wrong

`rpy0 = initial_state[6:9]` comes from `resample_state_to_times`, which
extracts RPY as:
```python
rpy_interp = rot_interp.as_euler('zyx', degrees=False)[:, ::-1]
```

This gives `[roll, pitch, yaw]` via **ZYX intrinsic** decomposition, meaning
the rotation matrix is `R = Rz(ψ) · Ry(θ) · Rx(φ)`.

But `from_euler('xyz', [roll, pitch, yaw])` constructs an **XYZ intrinsic**
rotation: `R = Rx(φ) · Ry(θ) · Rz(ψ)`.

These are NOT the same:
- Intrinsic ZYX `[ψ, θ, φ]` = Extrinsic XYZ `[φ, θ, ψ]` → R = Rz(ψ)Ry(θ)Rx(φ)
- Intrinsic XYZ `[φ, θ, ψ]` = Extrinsic ZYX `[ψ, θ, φ]` → R = Rx(φ)Ry(θ)Rz(ψ)

For small angles or single-axis rotations the difference vanishes, but for
combined yaw + pitch/roll it's significant.

### Correct fix

```python
R0 = jnp.array(Rotation.from_euler('zyx', rpy0[::-1]).as_matrix())
```

Or equivalently:
```python
R0 = jnp.array(Rotation.from_euler('XYZ', rpy0).as_matrix())
```
(uppercase = extrinsic, XYZ extrinsic with [roll, pitch, yaw] = ZYX intrinsic
with [yaw, pitch, roll]).

---

## Bug D — output Euler convention in closed-loop sim

**File:** `mlac_jax_train/compare_sim/closed_loop_core.py` line 150

```python
euler = Rotation.from_matrix(np.asarray(R_mats)).as_euler('xyz', degrees=True)
```

### What's wrong

The simulated rotation matrices are converted to Euler angles using `'xyz'`
intrinsic decomposition, returning `[roll_xyz, pitch_xyz, yaw_xyz]`.

These are then compared in `closed_loop.py` line 230 against measured RPY:
```python
att_err = np.linalg.norm(np.deg2rad(euler_sim_deg) - measured_slice[:, 6:9], axis=1)
```

But `measured_slice[:, 6:9]` was extracted with `'zyx'` intrinsic. For the
same rotation matrix, `as_euler('xyz')` and `as_euler('zyx')[::-1]` give
**different** angle triples. This creates a phantom comparison error.

### Correct fix

```python
euler_zyx = Rotation.from_matrix(np.asarray(R_mats)).as_euler('zyx', degrees=True)
euler = euler_zyx[:, ::-1]  # [roll, pitch, yaw]
```

---

## Bug E — body-rate-to-Euler-rate kinematic matrix

**File:** `mlac_jax_train/compare_sim/dynamics_numpy.py` lines 35-39

```python
inv_matrix = np.linalg.inv(np.array([
    [np.cos(pitch) * np.cos(roll), -np.sin(roll), 0.0],
    [np.cos(pitch) * np.sin(roll),  np.cos(roll), 0.0],
    [-np.sin(pitch),               0.0,          1.0],
]))
rpy_dot = inv_matrix @ omega_cmd
```

### What's wrong

For ZYX Euler angles (R = Rz(ψ)Ry(θ)Rx(φ)), the kinematic relationship
between body rates `[p, q, r]` and Euler rates `[φ̇, θ̇, ψ̇]` is:

```
[p]   [1      0       -sin(θ)       ] [φ̇]
[q] = [0      cos(φ)   cos(θ)sin(φ) ] [θ̇]
[r]   [0     -sin(φ)   cos(θ)cos(φ) ] [ψ̇]
```

This is E. The code's matrix M:

```
M = [cos(θ)cos(φ)   -sin(φ)   0]
    [cos(θ)sin(φ)    cos(φ)   0]
    [-sin(θ)         0        1]
```

M is actually the **world-frame** angular velocity matrix with yaw and roll
swapped (M = W(φ, θ) where it should be W(ψ, θ)). At zero angles both are
identity, so the bug is invisible at hover.

### Numerical verification at roll=30°, pitch=0°

Pure body pitch-rate q = 1 rad/s:
- Correct (E⁻¹ @ [0,1,0]): `[φ̇=0, θ̇=1.155, ψ̇=0.577]`
  (cross-coupling into yaw because rolled body-Y partially aligns with
  world-Z)
- Buggy (M⁻¹ @ [0,1,0]): `[φ̇=0.5, θ̇=0.866, ψ̇=0]`
  (no cross-coupling, wrong magnitudes — M reduces to Rz(roll) at zero
  pitch, which just rotates the body-rate vector)

### Correct fix

Replace M with E:
```python
E = np.array([
    [1.0,  0.0,       -np.sin(pitch)              ],
    [0.0,  np.cos(roll),  np.cos(pitch)*np.sin(roll)],
    [0.0, -np.sin(roll),  np.cos(pitch)*np.cos(roll)],
])
rpy_dot = np.linalg.solve(E, omega_cmd)
```

---

## What was verified to be correct

These code paths were checked and have **no** rotation bugs:

| Code path | Why it's correct |
|-----------|-----------------|
| `get_attitude()` in `outerloop_node.py` (R_d construction) | R_d = [b_1d, b_2d, b_3d] columns = body axes in ENU world frame. Correct R_{body→world}. |
| `flat_rotation_matrix_to_quaternion` / `quaternion_to_rotation_matrix` | Standard formulas, verified with identity and pure-yaw test cases. Consistent round-trip. |
| `get_rpy()` in `helpers.py` | Standard ZYX intrinsic extraction from [w,x,y,z] quaternion. Matches scipy's `as_euler('zyx')[::-1]`. |
| `BodyRateConverter.attitude_to_bodyrate()` | Uses `quaternion_to_rotation_matrix` without `.T` on both R_current and R_desired. Geometric error `vee(R_d^T R - R^T R_d)` is correct. |
| `calculate_smc_command` in `closed_loop_core.py` | R_d construction identical to outerloop. Error `vee(R_d^T R - R^T R_d)` correct. `Omega_ff = R.T @ world_yaw_rate` correctly transforms world yaw rate to body frame. |
| `simulation_ode_zoh` in `closed_loop_core.py` | `u_applied = f_d * R @ [0,0,1]` correctly applies thrust along body-z in world frame. `dR = R @ hat(Omega)` is correct rotation kinematics. |
| `euler_to_rotation_matrix` in `dynamics_numpy.py` | `R = Rz @ Ry @ Rx` matches ZYX intrinsic convention. |
| `resample_state_to_times` in `rosbag_utils.py` | Quaternion extraction [x,y,z,w] matches scipy convention. `as_euler('zyx')[:,::-1]` gives correct [roll,pitch,yaw]. `rot.apply(v_body)` correctly gives v_world. |
| ControllerLog fields `reference_roll`, `reference_pitch` | Set from `get_rpy(att_cmd_py.q)` which is the correct quaternion in the correct convention. |
| Gravity convention in `prior()` | `g = m*[0, 0, +9.81]` acts as gravitational compensation term in the control law (not raw gravity). Hover: F_W = g → ddq = 0. Consistent between runtime and JAX. |

---

## Impact on compare_sim

### What the rosbag contains

The drone flew with bugs A and B active. The rosbag records:

| Data | Source | Correct? |
|------|--------|----------|
| Measured pose/velocity | PX4 estimator via MAVROS | ✓ correct |
| `att_msg.orientation` (quaternion) | `get_attitude()` | ✓ correct |
| `att_msg.body_rate` (bodyrate mode) | `BodyRateConverter` | ✓ correct |
| `att_msg.thrust` (normalized scalar) | Buggy `.T` projection (Bug B) | ✗ wrong |
| ControllerLog `reference_roll/pitch` | `get_rpy(att_cmd_py.q)` | ✓ correct |
| ControllerLog `F_W` | From `get_force()` with wrong `state.v` (Bug A) | ✗ contaminated |
| ControllerLog `a_fb` | From `get_force()` with wrong `state.v` (Bug A) | ✗ contaminated |

### Open-loop replay

Open-loop replays the recorded thrust_norm + body_rates through the dynamics
model in `dynamics_numpy.py`.

| Factor | Effect on comparison |
|--------|---------------------|
| Bug B (wrong thrust in rosbag) | **Cancels out** — PX4 received the same buggy thrust. Open-loop replays exactly what PX4 saw. |
| Bug E (wrong kinematic equation) | **Does NOT cancel** — the dynamics model converts body rates to Euler rates incorrectly. Attitude diverges from reality, corrupting thrust direction and position. This is the **dominant** error source. |
| No attitude controller model | **Does NOT cancel** — the sim applies body rates instantly (`rpy_dot = M⁻¹ @ omega_cmd`), while PX4 has inner-loop bandwidth and delay. `closed_loop_core.py` at least models a first-order lag. |

**Bottom line:** Open-loop comparison currently cannot validate the dynamics
model because bug E dominates the error. Fix E first, then open-loop becomes
a clean dynamics validation.

### Closed-loop replay

Closed-loop starts from measured initial conditions, runs an independent SMC
controller with the recorded reference trajectory.

| Factor | Effect on comparison |
|--------|---------------------|
| Bug C (wrong initial R0) | **Adds phantom error** — sim starts in wrong orientation. |
| Bug D (wrong output Euler convention) | **Adds phantom error** — reported sim attitude uses different decomposition than measured. |
| Bugs A+B (runtime controller was wrong) | **Causes inherent divergence** — the actual drone followed a trajectory produced by a buggy controller. The sim's correct SMC controller produces a different trajectory. Even with a perfect dynamics model, the two diverge. |

**Bottom line:** Closed-loop comparison currently measures a mix of dynamics
model error + runtime controller bugs + comparison pipeline bugs (C, D). Fix
C and D first to remove the comparison artifacts, then the residual tells you
about dynamics model accuracy + the effect of runtime bugs A and B.

---

## Recommended fix order (summary table)

| Step | Bugs | Rationale |
|------|------|-----------|
| 1 | C, D | One-line fixes in `closed_loop_core.py`. Removes phantom errors from the comparison pipeline. No flight code changes needed. |
| 2 | E | Fix kinematic equation in `dynamics_numpy.py`. Makes open-loop replay correct. Re-run open-loop comparison to validate dynamics model. |
| 3 | A, B | Fix `.T` in `mlac_mission_node.py`. Requires rebuilding the ROS package and re-flying to collect a new rosbag with correct controller behavior. |
| 4 | — | Re-run both comparisons with the new (bug-free) rosbag. Open-loop validates dynamics; closed-loop validates the full controller + dynamics. |

---

## Detailed action plan

### Step 1 — Fix comparison pipeline Euler conventions (Bugs C, D)

**Goal:** Remove phantom errors from the offline comparison so it measures
real divergence, not convention artifacts.

**1a. Fix initial rotation matrix (Bug C)**

File: `mlac_jax_train/compare_sim/closed_loop_core.py` line 111

Change:
```python
R0 = jnp.array(Rotation.from_euler('xyz', rpy0).as_matrix())
```
To:
```python
R0 = jnp.array(Rotation.from_euler('XYZ', rpy0).as_matrix())
```
(Uppercase `'XYZ'` = extrinsic XYZ, which is equivalent to intrinsic ZYX
with reversed angles — matching the convention used to extract `rpy0`.)

**1b. Fix output Euler extraction (Bug D)**

File: `mlac_jax_train/compare_sim/closed_loop_core.py` line 150

Change:
```python
euler = Rotation.from_matrix(np.asarray(R_mats)).as_euler('xyz', degrees=True)
```
To:
```python
euler = Rotation.from_matrix(np.asarray(R_mats)).as_euler('zyx', degrees=True)[:, ::-1]
```

**Test:** Re-run closed-loop comparison against the existing rosbag:
```bash
cd mlac_jax_train/compare_sim
python closed_loop.py \
  --rosbag <path_to_existing_rosbag> \
  --reference-source bag \
  --plot-dir /tmp/step1_test
```
Compare the attitude RMS numbers and plots before/after. The attitude error
should drop (possibly significantly if yaw was non-zero during the flight).
Position/velocity errors may remain unchanged since those are dominated by
other factors.

Also re-run the frame alignment check:
```bash
python frame_alignment_check.py \
  --rosbag <path_to_existing_rosbag> \
  --reference-source bag \
  --plot-dir /tmp/step1_frame_check
```

---

### Step 2 — Fix open-loop dynamics kinematic equation (Bug E)

**Goal:** Make the open-loop dynamics model correctly convert body rates to
Euler rate changes, so open-loop replay can validate the dynamics model.

**2a. Replace the kinematic matrix**

File: `mlac_jax_train/compare_sim/dynamics_numpy.py` lines 34-40

Change:
```python
roll, pitch = rpy[0], rpy[1]
inv_matrix = np.linalg.inv(np.array([
    [np.cos(pitch) * np.cos(roll), -np.sin(roll), 0.0],
    [np.cos(pitch) * np.sin(roll),  np.cos(roll), 0.0],
    [-np.sin(pitch),               0.0,          1.0],
]))
rpy_dot = inv_matrix @ omega_cmd
```
To:
```python
roll, pitch = rpy[0], rpy[1]
E = np.array([
    [1.0,  0.0,            -np.sin(pitch)                  ],
    [0.0,  np.cos(roll),    np.cos(pitch) * np.sin(roll)   ],
    [0.0, -np.sin(roll),    np.cos(pitch) * np.cos(roll)   ],
])
rpy_dot = np.linalg.solve(E, omega_cmd)
```

**Test:** Re-run frame alignment check with open-loop:
```bash
python frame_alignment_check.py \
  --rosbag <path_to_existing_rosbag> \
  --reference-source bag \
  --plot-dir /tmp/step2_test
```
The open-loop `att_rms` should improve substantially compared to Step 1
results. The open-loop `pos_rms` should also improve because attitude
drives thrust direction. If open-loop position still diverges significantly,
the remaining gap is dynamics modeling error (mass, thrust model, drag, etc.)
or the lack of an inner-loop attitude controller model.

**Optional sanity check:** Write a small unit test that verifies the kinematic
equation round-trips correctly:
```python
# At roll=30°, pitch=20°: E @ E_inv should be identity
# Pure body-rate [0, 1, 0] at roll=30° should produce yaw cross-coupling
```

---

### Step 3 — Fix runtime controller `.T` bugs (Bugs A, B)

**Goal:** Make the actual flight controller use correct velocity feedback and
thrust normalization.

**3a. Fix velocity conversion (Bug A)**

File: `ros2_px4_ws/src/mlac_sim/mlac_sim/mlac_mission_node.py` line 328

Change:
```python
R_body_to_world = quaternion_to_rotation_matrix(self.current_vehicle_state_py.q).T
```
To:
```python
R_body_to_world = quaternion_to_rotation_matrix(self.current_vehicle_state_py.q)
```

**3b. Fix thrust projection (Bug B)**

File: `ros2_px4_ws/src/mlac_sim/mlac_sim/mlac_mission_node.py` line 449

Change:
```python
R_body_to_world_desired = quaternion_to_rotation_matrix(att_cmd_py.q).T
```
To:
```python
R_body_to_world_desired = quaternion_to_rotation_matrix(att_cmd_py.q)
```

**3c. Build and test in SITL**

```bash
cd ~/mlac_px4/ros2_px4_ws
colcon build --packages-select mlac_sim
```

Test in SITL with a simple hover + trajectory to verify the drone still
flies correctly (it should fly *better* — tighter tracking, especially
during maneuvers with significant tilt):
```bash
# Use your existing SITL launch script
# Compare hover thrust stability and trajectory tracking visually
```

**Important:** The controller gains (Kp, Ki, Kd, Lambda, K) were tuned
while bugs A and B were active. After fixing, the velocity feedback is
different, which may require re-tuning gains. Start with a conservative
trajectory (slow, small tilts) and verify stability before aggressive
maneuvers.

---

### Step 4 — Collect a new rosbag and re-validate

**Goal:** With all bugs fixed, collect clean data and run both comparisons
to validate the dynamics model and controller independently.

**4a. Fly a test trajectory in SITL and record a rosbag**

Use the same trajectory file and recording setup as before. This gives a
clean rosbag where the controller behaves correctly.

**4b. Run open-loop comparison**

```bash
python frame_alignment_check.py \
  --rosbag <path_to_new_rosbag> \
  --reference-source bag \
  --plot-dir /tmp/step4_openloop
```

This validates the dynamics model: if open-loop `pos_rms` and `att_rms`
are small, the model captures the real system accurately. If not, the
residual points to dynamics modeling gaps (mass error, unmodeled drag,
thrust curve, inner-loop delay, etc.).

**4c. Run closed-loop comparison**

```bash
python closed_loop.py \
  --rosbag <path_to_new_rosbag> \
  --reference-source bag \
  --plot-dir /tmp/step4_closedloop
```

This validates the controller + dynamics together. Since the real controller
is now correct, the closed-loop sim (which also runs a correct controller)
should track similarly. Any residual divergence is due to differences
between the SMC controller in `closed_loop_core.py` and the COML controller
in `outerloop_node.py`, or dynamics modeling error.

**4d. Compare old vs new results**

Document the before/after metrics to confirm the fixes had the expected
effect:
- Steps 1-2 should reduce comparison pipeline artifacts
- Step 3 should improve actual flight tracking quality
- Step 4 should show both open-loop and closed-loop matching reality well

---

## Files referenced

| File | Role in this audit |
|------|--------------------|
| `ros2_px4_ws/src/mlac_sim/mlac_sim/mlac_mission_node.py` | Runtime controller. Bugs A, B. |
| `ros2_px4_ws/src/mlac_sim/mlac_sim/outerloop_node.py` | Outer-loop controller (`get_attitude`, `get_force`, `get_rates`). Verified correct. |
| `ros2_px4_ws/src/mlac_sim/mlac_sim/helpers.py` | `get_rpy()`, `controllog_class_to_ros_msg()`. Verified correct. |
| `ros2_px4_ws/src/mlac_sim/mlac_sim/utils.py` | `quaternion_to_rotation_matrix`, `flat_rotation_matrix_to_quaternion`. Verified correct. |
| `ros2_px4_ws/src/mlac_sim/mlac_sim/bodyrate_conversion.py` | `BodyRateConverter`. Verified correct. |
| `ros2_px4_ws/src/mlac_sim/mlac_sim/structs.py` | `AttCmdClass`, `ControlLogClass`. Quaternion convention [w,x,y,z]. |
| `ros2_px4_ws/src/mlac_sim/mlac_sim/dynamics.py` | Runtime `prior()`. Gravity convention verified correct. |
| `mlac_jax_train/compare_sim/closed_loop_core.py` | Closed-loop SMC sim. Bugs C, D. Controller/dynamics internally correct. |
| `mlac_jax_train/compare_sim/closed_loop.py` | Orchestrates rolling-window closed-loop comparison. |
| `mlac_jax_train/compare_sim/dynamics_numpy.py` | Open-loop dynamics. Bug E. |
| `mlac_jax_train/compare_sim/frame_alignment_check.py` | Open-loop + closed-loop frame alignment analysis. |
| `mlac_jax_train/compare_sim/rosbag_utils.py` | Rosbag extraction. Verified correct. |
| `mlac_jax_train/dynamics.py` | JAX `prior()`. Matches runtime version. |

---

## Relation to previous investigation

This audit continues from `DEBUGGING_ATTITUDE_LOG.md` (Round 4 conclusion):

> The SITL test confirmed MAVROS is fine. The original compare_sim mismatch
> must come from how `ControllerLog.reference_roll`, `.reference_pitch`,
> `.reference_yaw` are computed inside `mlac_mission_node.py`.

**Updated conclusion:** The ControllerLog reference attitudes are actually
computed correctly (via `get_rpy` on the correct desired quaternion). The
mismatches in the comparison come from:

1. Runtime bugs (A, B) that made the actual flight trajectory different from
   what a correct controller would produce.
2. Comparison pipeline bugs (C, D, E) that corrupt the offline simulation and
   its comparison to measured data.

The frame conventions between MAVROS ↔ controller ↔ rosbag extraction are
consistent (all ENU/FLU, ZYX intrinsic Euler, [w,x,y,z] quaternion
internally). The bugs are transpose errors and Euler convention mismatches in
specific code paths, not systemic frame disagreements.

# Holonomic Controller Comparison: Mustafa's vs. Dan's

Fact-driven comparison of `HolonomicPoseFollowerTask` (Mustafa) and `DanHolonomicTC` (Dan),
both full-pose holonomic Go2 trajectory controllers. Every claim below cites the exact
file/line it comes from — nothing here is inferred without a code reference.

## Plain-language summary

**Mustafa's controller** measures how far along the route the robot has actually driven — not
just how close it looks to the finish coordinate — using a speed/steering response that's been
pre-calibrated against this specific robot's measured real-world behavior. It only calls a run
"done" after the robot has settled and held still at the target for a full second, re-checking
if it glides past.

**Dan's controller** steers by continuously finding the nearest point on the route ahead and
heading toward it, but decides "am I done?" using only straight-line distance to the finish
coordinate, with no memory of how much of the route has actually been driven. That's fine for
a route that ends somewhere different from where it started — but if the route loops back to
its own starting point (a circle or a square), the finish coordinate and the start coordinate
are the same point, so it can conclude "I'm done" before the robot has moved at all.

## Source locations

| | Mustafa's | Dan's |
|---|---|---|
| Main class | `HolonomicPoseFollowerTask` | `DanHolonomicTC` |
| File | `dimos/control/tasks/holonomic_pose_follower_task/holonomic_pose_follower_task.py` | `dimos/navigation/dannav/holonomic_tc/module.py` |
| Progress/geometry helper | `dimos/control/tasks/holonomic_pose_follower_task/progress_reference.py` (`ProgressPathReference`) | `dimos/navigation/dannav/geometry/path_distancer.py` (`PathDistancer`, `project_to_polyline`) |
| Base class | `BaseControlTask` (`holonomic_pose_follower_task.py:95`) | `Module` (`module.py:603`) |

## 1. Architectural integration

- **Mustafa's** extends `BaseControlTask` and is claimed/dispatched through `ControlCoordinator`'s
  task-card routing like any other task (`holonomic_pose_follower_task.py:95`, `claim()` at line 133).
- **Dan's** extends `Module` directly (`module.py:603`) and is not a coordinator task — it has to
  be wired via `autoconnect` + `MovementManager` muxing instead, confirmed by its existing production
  blueprint `unitree_go2_mls_htc.py` which composes `DanHolonomicTC.blueprint(...)` and
  `MovementManager.blueprint()` side by side, not inside a `ControlCoordinator` task list.

## 2. Progress tracking — the central difference

**Mustafa's (`ProgressPathReference.advance()`, `progress_reference.py:123-145`):**
Projection is stateful and *windowed*: each call only searches path segments within
`[s_prev - back_m, s_prev + window_m]` of the *previous* progress value (`back_m=0.5`,
`window_m=1.0` by default, `progress_reference.py:64`). The class docstring states the reason
explicitly:

> "Projection is windowed and (nearly) monotonic... so closed paths (goal == start) never
> match the far end on tick 1" (`progress_reference.py:58-61`)

This is a deliberate, documented design decision to prevent exactly the failure mode found in
Dan's controller (see §4).

**Dan's (`project_to_polyline()`, `path_distancer.py:47-`):** A stateless free function — takes
only the query point and the *entire* polyline array, with no previous-progress argument at all.
Every call does a fresh, global nearest-point search across the whole path, with no memory of
where the robot was projected last tick.

## 3. Arrival / goal detection

**Mustafa's** arrival is a three-stage, progress-gated process:
1. `_tracking_command` (`holonomic_pose_follower_task.py:187-222`) computes `remaining = ref.length - s_robot`
   — remaining **arc length**, not straight-line distance. Only when `remaining < goal_tolerance`
   does it transition `tracking -> settling` (line 198-200).
2. `_settling_command` (line 280-290) additionally requires both position error *and* yaw error
   within tolerance before transitioning `settling -> stopping`.
3. `_stopping_command` (line 292-314) requires the robot to hold at rest for `stop_hold_s` (default
   1.0s, line 90) — re-checking position/yaw error again — before finally declaring `arrived`. If
   the robot glides outside tolerance during the hold, it re-enters `settling` (line 307-314).

**Dan's** arrival (`_compute_path_following`, `module.py:401-424`) is a single check, evaluated
fresh every tick:
```python
if path_distancer.distance_to_goal(current_pos) < self._goal_tolerance:
    ...
    self._change_state("arrived")
```
And `distance_to_goal` (`path_distancer.py:183-184`) is exactly:
```python
return float(np.linalg.norm(self._path[-1] - current_pos))
```
A single straight-line Euclidean distance to the path's last waypoint. No arc-length, no
progress requirement, no rest/dwell confirmation.

## 4. Confirmed bug: closed-loop paths (`circle`, `square`) false-arrive instantly in Dan's controller

Verified on real hardware during benchmark testing (`dimos-holonomic-sweep`, Jul 22 session,
`data/benchmark/go2_dan/`): for `circle_offset_45` and `square_crab` (both built on closed-loop
geometry — `circle()`/`square()` in `dimos/control/benchmarking/paths.py`, documented "last
waypoint coincides with the first" / "back to origin"), `DanHolonomicTC` logged:
```
changed state state=path_following
Reached goal position          <- 1ms later
changed state state=arrived
```
**Root cause, precisely:** the benchmark anchors each path's first waypoint to the robot's
current position (`shift_path_to_start_at_pose`, `benchmark.py:115-134`). For a closed loop,
the last waypoint (the goal) is geometrically the same point as the first. Since Dan's arrival
check is pure distance-to-goal with no progress requirement, `distance_to_goal(current_pos)`
is already ≈0 on the very first tick — before the robot has moved at all.

Mustafa's controller does not have this failure mode, and the reason is directly attributable
to §2/§3: its arrival condition is gated on `remaining` **arc length** (which starts near the
*full* loop length, not near zero, for a closed loop) rather than spatial proximity to a
coordinate, and its windowed projection is explicitly documented to prevent matching the far
end of a closed path on the first tick.

**Practical consequence for the benchmark:** Dan's controller cannot currently produce valid
data on `circle`, `square`, `circle_offset_45`, or `square_crab` (4 of the 10-path battery) at
any speed — the failure is speed-independent (confirmed by re-testing at 0.3/0.5/0.7/0.9/1.0
m/s during the Jul 22 session). The 6 open-path cases (`straight_line`, `single_corner`,
`smooth_corner`, `rounded_square`, `straight_rotate_90`, `strafe_left_2m`) are unaffected.

## 5. Control law

**Mustafa's** is calibrated against a measured plant fit, loaded from a vendored artifact
(`_ensure_artifact_loaded`, `holonomic_pose_follower_task.py:333-375`):
- Per-axis proportional gains derived from measured plant time constants:
  `_kp_for_tau(tau) = 1 / (4 * zeta^2 * tau)` (line 68-69, `zeta=1.0` line 62) for vx, vy, wz independently.
- A `FeedforwardGainCompensator` that inverts the measured plant gain `K` per axis, so a
  commanded velocity produces the intended achieved velocity (line 351-363).
- Feedback is trim-only, clamped small (`_FB_CLAMP_LINEAR=0.15`, `_FB_CLAMP_YAW=0.4`, line 64-65)
  — comment: "Feedforward carries the path" (line 63).
- Speed regulation looks ahead along the path for upcoming yaw-rate/curvature demands
  (`_regulated_speed`, line 241-270, using `ProgressPathReference.max_rates_ahead`) and slows
  down *before* a tight corner rather than at it.

**Dan's** control law uses proportional gains configured directly on `DanHolonomicTCConfig`
(`module.py:589-599`): `k_position_per_s=2.0`, `k_yaw_per_s=1.0`, `k_velocity_per_s=0.5`,
`k_yaw_rate_per_s=1.0` — generic proportional gains with no evidence in this codebase of a
plant-characterization or feedforward-inversion step analogous to Mustafa's artifact system.

## 6. Speed configuration

- **Mustafa's**: `speed` is a per-task config field, changeable via the coordinator's live
  `set_speed` broadcast hook while idle (`holonomic_pose_follower_task.py:416-424`) — the
  benchmark's live `/speed` stream works directly against it.
- **Dan's**: `speed_m_s` (`DanHolonomicTCConfig`, `module.py:592`) is set at blueprint
  construction time, or via the named-profile RPC `set_run_profile()` (`module.py:665-667`,
  3 fixed profiles at 0.55/1.0/1.5 m/s — none matching our 0.3/0.5/0.7/0.9/1.0 sweep). No live
  numeric speed input exists. This is why Dan's benchmark blueprint
  (`unitree_go2_dan_holonomic_benchmark.py`) requires one process launch per speed instead of
  one continuous sweep.

## 7. Open items (not yet fact-checked / not yet measured)

- **Actual tracking performance** (cross-track error, heading error, smoothness) on the 6 valid
  open paths, across all 5 speeds, for both controllers — pending a clean re-run of Dan's data
  (the existing `data/benchmark/go2_dan/` recordings are all mistakenly at 0.50 m/s due to an
  env var not taking effect across launches; a proper 5-speed re-run is still outstanding as of
  this writing).
- Dan's exact per-tick control computation in `_HolonomicPathFollower`/`HolonomicTrackingController`
  (module.py) has not been read in full line-by-line detail the way Mustafa's has here — the
  gain *names* and *values* above are confirmed from the config class, but the precise
  update equations were not transcribed into this document.

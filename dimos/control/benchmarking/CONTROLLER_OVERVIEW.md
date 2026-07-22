# Holonomic Trajectory Controller Comparison: Mustafa vs Dan

Both controllers take a full pose path (position and independently commanded yaw) and drive
the Go2 to follow it, holonomically. Both are being benchmarked on the same path battery at
the same speeds. This doc walks through how each one actually works.

## Data flow

**Mustafa's controller**

```
odom (x, y, yaw)
      |
      v
project onto path, searching only near the last known progress point
(windowed search, not the whole path)
      |
      v
progress s_robot -----------------> remaining = path_length - s_robot
      |                                       |
      v                                       v
lookahead point at s_robot + L        arrival gate: remaining < tolerance?
      |
      v
feedforward velocity (from a plant model fit to this robot)
   + feedback trim (small correction toward the lookahead point)
      |
      v
twist (vx, vy, wz)
```

**Dan's controller**

```
odom (x, y, yaw)
      |
      v
project onto path, searching the entire path fresh every tick
      |
      v
nearest point + arc length s                straight line distance
      |                                      to the final waypoint (x, y)
      v                                             |
lookahead point at s + L                             v
      |                                       arrival gate: distance < tolerance?
      v
proportional control
(gains on position error, yaw error, velocity error, yaw rate error)
      |
      v
twist (vx, vy, wz)
```

The steering side of both looks similar in shape: project onto the path, take a lookahead
point, compute a velocity command toward it. The arrival side is where they genuinely differ,
and that difference has real consequences.

## Start

Dan's controller checks its heading against the path's initial direction before it starts
driving. If the error is above `orientation_tolerance`, it rotates in place first, and only
switches to `path_following` once roughly aligned.

Mustafa's controller has no separate rotate-first phase. Since it tracks position and yaw
independently and simultaneously, it starts driving immediately and corrects heading as part
of the same continuous feedback loop, no discrete alignment step needed.

## Progress

This is the core difference between the two.

Mustafa's projection is stateful. Each tick, it only searches for the nearest point on the
path within a small window around where it was last found (about half a meter behind to a
meter ahead of the last known position on the path). It cannot suddenly jump to a distant part
of the path. This is a deliberate design choice: for a path that loops back to its own start
(a circle or a square), a naive global search would find the goal and the start equally close
together right from the first tick, since they are literally the same point. The windowed
search rules that out.

Dan's projection is stateless. Every tick, it searches the entire path fresh, with no memory
of where it was last matched. For an open path this is fine, since there is only one place on
the path close to the robot at any given time. For a closed loop, the start and the goal are
the same coordinate, so there is nothing forcing the match to prefer "still near the start"
over "already at the goal."

## Arrival

Mustafa's arrival is gated on how much of the path is left to drive, not on spatial distance.
It tracks `remaining`, the arc length from the robot's current progress to the end of the
path. Only once `remaining` drops below tolerance does it move from `tracking` to `settling`.
From there it still needs position and yaw error both within tolerance, and then it has to
hold still at that pose for a full second before it finally declares `arrived`. If it drifts
out of tolerance during that hold, it goes back to `settling` and tries again.

Dan's arrival is a single check: straight line distance from the current position to the
path's last waypoint, evaluated fresh every tick. No arc length, no progress requirement, no
settle and hold.

For an open path, this works out the same in practice, since the robot can only get close to
the final waypoint by actually having driven there. For a closed loop, the final waypoint is
the same coordinate as the start. So `distance to goal` is already near zero the instant the
path is armed, before the robot moves at all. Confirmed on hardware: Dan's controller
declares `arrived` on `circle`, `square`, and their full pose variants within a tick of
starting, at every speed tested. The other six paths in the battery, which do not loop, are
unaffected.

## Control law

Mustafa's gains are derived from a measured plant fit for this specific robot: a first order
time constant and gain per axis (vx, vy, wz), loaded from a vendored calibration artifact. On
top of that sits a feedforward compensator that inverts the measured gain, so a commanded
velocity produces the intended achieved velocity, with feedback doing only small trim
corrections. It also looks ahead along the path for upcoming curvature or yaw rate demand and
slows down before a tight corner instead of at it.

Dan's gains are four proportional constants, one each for position, yaw, velocity, and yaw
rate. There is no evidence in the code of an equivalent plant characterization or feedforward
step. This does not mean it tracks worse, that is an open, measured question, just that the
two controllers get to their commanded velocity through different means.

## Speed

Mustafa's speed can change live, mid session, since it is read from a broadcast the
coordinator forwards to the task. Dan's speed is fixed at launch (or picked from one of three
named profiles), with no live input. Sweeping Dan's controller across multiple speeds means
relaunching it once per speed rather than one continuous run.

## Summary table

| | Mustafa's | Dan's |
|---|---|---|
| Start | Drives immediately, corrects heading while moving | Rotates in place first if not roughly aligned |
| Progress | Windowed search near last known position | Fresh global search every tick |
| Arrival | Remaining path distance, then settle and hold for 1s | Straight line distance to final point, checked once |
| Control law | Plant-fit gains plus feedforward, feedback is trim only | Fixed proportional gains, no known feedforward |
| Speed | Changeable live, mid session | Fixed at launch or one of 3 named profiles |
| Preemption | Native coordinator hook, aborts cleanly | No hook, needs an external mixer module |
| Closed loops | Handled correctly | Confirmed bug: arrives instantly, never drives |

## Where the code lives

- Mustafa: `dimos/control/tasks/holonomic_pose_follower_task/holonomic_pose_follower_task.py`,
  progress tracking in `progress_reference.py` in the same folder.
- Dan: `dimos/navigation/dannav/holonomic_tc/module.py`, projection and arrival check in
  `dimos/navigation/dannav/geometry/path_distancer.py`.

Full technical writeup with exact references: `CONTROLLER_COMPARISON.md`, same folder.

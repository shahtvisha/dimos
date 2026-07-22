# Go2 Trajectory Controllers — How They Work

Three controllers are being benchmarked against the same test: same paths, same speeds,
same scoring. This is what each one actually does, in plain terms.

## The shared test

Every controller gets fed the same thing: a path to follow and a target speed. A separate
recorder watches the robot's actual position the whole time and scores how well it tracked
the path — it doesn't know or care which controller is driving.

## P-controller

The simplest of the three. It picks a point a fixed distance ahead on the path, and steers
toward it: if it's pointing the wrong way, it turns in place first; otherwise it drives
forward while correcting its heading proportionally to how far off it is.

- **Can only face the direction it's driving.** No sideways movement, no holding a different
  heading than its travel direction.
- Works fine on normal paths (straight lines, corners, loops). Cannot meaningfully attempt
  paths that require facing a different way than it's traveling (e.g. driving sideways) —
  it's not built to do that at all.

## Mustafa's controller

Tracks both **where** it is and **which way it's facing** independently — it can drive
sideways while facing a completely different direction, like walking through a doorway
while looking down a hallway.

- Its speed and steering response are **calibrated to this specific robot** — the numbers
  come from measuring how the real Go2 actually responds to commands, not generic settings.
- It measures **how far along the route it has actually driven**, not just how close it
  looks to the finish line. It only calls a run "done" after settling in place at the
  target for a full second — if it drifts, it corrects and re-checks.

## Dan's controller

Also holonomic — can drive sideways and hold an independent heading, same as Mustafa's.

- It continuously finds the nearest point on the path ahead and steers there.
- **Known limitation:** it decides "am I done?" using straight-line distance to the finish
  point only — it doesn't track how much of the route it's actually driven. This works fine
  for a route that ends somewhere different from where it started. But for a route that
  loops back to its own starting point (a circle or a square), the start and the finish are
  the same spot — so it can conclude "I'm done" before actually driving anywhere. Confirmed
  on real hardware, at every speed tested; the other 6 (non-looping) paths are unaffected.
- Also: its target speed is set once when it starts, not adjustable mid-run — so testing it
  at 5 different speeds means restarting it 5 times, not one continuous session like the
  other two.

## At a glance

| | P-controller | Mustafa's | Dan's |
|---|---|---|---|
| Can face a different way than it's driving? | No | Yes | Yes |
| Calibrated to this specific robot? | No | Yes | Not confirmed |
| Can reliably finish a looping path? | Yes | Yes | **No — confirmed bug** |
| Speed adjustable mid-session? | Yes | Yes | No |

## Want more detail?

Full technical write-up with exact code references: `CONTROLLER_COMPARISON.md` (same folder).

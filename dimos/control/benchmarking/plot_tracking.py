#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Commanded-vs-actual tracking plots, per run: x, y, heading, each with its
own commanded-vs-actual panel and a separate error panel.

"Commanded" here means the reference path's own x/y/yaw at the nearest point
to the robot's actual position — the same projection score.py already uses
for cross-track/heading error, so the numbers agree with the existing scores.

    python -m dimos.control.benchmarking.plot_tracking <recordings_dir> [--path NAME] [--speed V]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path as FsPath

import numpy as np

from dimos.control.benchmarking.benchmark import RunRecording
from dimos.control.benchmarking.score import load_recordings
from dimos.control.benchmarking.scoring import _path_xy, _reference_yaw, nearest_segment
from dimos.utils.trigonometry import angle_diff


_WINDOW_BACK = 5
_WINDOW_FWD = 15


def _windowed_nearest_segment(
    pt: np.ndarray, ref_xy: np.ndarray, last_idx: int | None
) -> tuple[int, float, float]:
    """Like ``nearest_segment``, but searches only near the last match --
    a global search is ambiguous on loop paths, where start and end coincide."""
    n_segs = len(ref_xy) - 1
    if last_idx is None:
        lo, hi = 0, min(n_segs, _WINDOW_FWD)
    else:
        lo = max(0, last_idx - _WINDOW_BACK)
        hi = min(n_segs, last_idx + _WINDOW_FWD)

    seg_idx, dist, t_along = nearest_segment(pt, ref_xy[lo : hi + 1])
    return lo + seg_idx, dist, t_along


def _commanded_series(rec: RunRecording) -> dict[str, np.ndarray]:
    """For every tick, project the actual pose onto the reference path and
    return time series for actual/commanded x, y, heading (+ errors)."""
    ref_xy = np.array([[p[0], p[1]] for p in rec.reference], dtype=np.float64)
    ref_yaw = np.unwrap(np.array([p[2] for p in rec.reference], dtype=np.float64))

    t = np.array([tick[0] for tick in rec.ticks], dtype=np.float64)
    t = t - t[0]
    actual_x = np.array([tick[1] for tick in rec.ticks], dtype=np.float64)
    actual_y = np.array([tick[2] for tick in rec.ticks], dtype=np.float64)
    actual_yaw = np.unwrap(np.array([tick[3] for tick in rec.ticks], dtype=np.float64))

    cmd_x = np.empty_like(actual_x)
    cmd_y = np.empty_like(actual_y)
    cmd_yaw = np.empty_like(actual_yaw)

    last_idx: int | None = None
    for i in range(len(actual_x)):
        pt = np.array([actual_x[i], actual_y[i]])
        seg_idx, _dist, t_along = _windowed_nearest_segment(pt, ref_xy, last_idx)
        last_idx = seg_idx
        foot = ref_xy[seg_idx] + t_along * (ref_xy[seg_idx + 1] - ref_xy[seg_idx])
        cmd_x[i], cmd_y[i] = foot
        cmd_yaw[i] = _reference_yaw(ref_yaw, seg_idx, t_along)

    heading_err = np.array(
        [angle_diff(a, c) for a, c in zip(actual_yaw, cmd_yaw, strict=True)]
    )

    return {
        "t": t,
        "actual_x": actual_x,
        "cmd_x": cmd_x,
        "err_x": actual_x - cmd_x,
        "actual_y": actual_y,
        "cmd_y": cmd_y,
        "err_y": actual_y - cmd_y,
        "actual_yaw": np.degrees(actual_yaw),
        "cmd_yaw": np.degrees(cmd_yaw),
        "err_yaw": np.degrees(heading_err),
    }


def plot_run(rec: RunRecording, out_path: str | FsPath) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = _commanded_series(rec)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)

    channels = [
        ("x position", "x (m)", "actual_x", "cmd_x", "err_x", "error (m)"),
        ("y position", "y (m)", "actual_y", "cmd_y", "err_y", "error (m)"),
        ("heading", "heading (deg)", "actual_yaw", "cmd_yaw", "err_yaw", "error (deg)"),
    ]

    for row, (title, ylabel, actual_key, cmd_key, err_key, err_ylabel) in enumerate(channels):
        left, right = axes[row]

        left.plot(s["t"], s[cmd_key], color="black", lw=2.0, label="commanded")
        left.plot(s["t"], s[actual_key], color="tab:red", lw=1.3, label="actual")
        left.set_ylabel(ylabel)
        left.set_title(f"{title}: commanded vs actual")
        left.grid(True, alpha=0.3)
        left.legend(fontsize=8)

        right.plot(s["t"], s[err_key], color="tab:orange", lw=1.3)
        right.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        right.set_ylabel(err_ylabel)
        right.set_title(f"{title}: error (actual - commanded)")
        right.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("time (s)")
    axes[-1][1].set_xlabel("time (s)")

    arrived_str = "arrived" if rec.arrived else f"NOT arrived ({rec.reason})"
    fig.suptitle(
        f"{rec.robot} | path={rec.path} | speed={rec.speed:g} m/s | {arrived_str}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Commanded-vs-actual tracking plots per run")
    ap.add_argument("recordings_dir", help="directory of per-run *.json recordings")
    ap.add_argument("--path", default=None, help="only plot runs for this path name")
    ap.add_argument("--speed", type=float, default=None, help="only plot runs at this speed")
    ap.add_argument("--out-dir", default=None, help="output dir (default: <recordings_dir>/tracking_plots)")
    args = ap.parse_args()

    recs = load_recordings(args.recordings_dir)
    if args.path is not None:
        recs = [r for r in recs if r.path == args.path]
    if args.speed is not None:
        recs = [r for r in recs if math.isclose(r.speed, args.speed, abs_tol=1e-6)]
    if not recs:
        raise SystemExit("no matching recordings found")

    out_dir = FsPath(args.out_dir) if args.out_dir else FsPath(args.recordings_dir) / "tracking_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rec in recs:
        out_path = out_dir / f"{rec.robot}_{rec.path}_v{rec.speed:.2f}_tracking.png"
        plot_run(rec, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

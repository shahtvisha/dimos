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

"""One combined commanded-vs-actual + error figure per full-pose trajectory,
with all given controllers overlaid, at a single speed.

Each controller's run was anchored to wherever the robot happened to start
that session, so runs are transformed into a shared canonical frame first
(reference start -> origin, initial heading -> +x — same convention as
score.py's ``_canonicalize``, extended here to also rotate heading values,
not just position) so all controllers' commanded/actual lines line up.

    python -m dimos.control.benchmarking.plot_tracking_combined \\
        data/benchmark/go2_mustafa_final:Mustafa \\
        data/benchmark/go2_dan_final:Dan \\
        data/benchmark/go2_pcontroller:P-controller \\
        --speed 0.5 --out-dir data/benchmark/fullpose_combined
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path as FsPath

import numpy as np

from dimos.control.benchmarking.benchmark import RunRecording
from dimos.control.benchmarking.score import load_recordings
from dimos.control.benchmarking.scoring import _reference_yaw, nearest_segment
from dimos.utils.trigonometry import angle_diff

FULLPOSE_PATHS = ("straight_rotate_90", "strafe_left_2m", "circle_offset_45", "square_crab")

_COLORS = {
    0: "#0072B2",
    1: "#D55E00",
    2: "#009E73",
    3: "#CC79A7",
}


def _canonical_frame(reference: list[list[float]]) -> tuple[float, float, float]:
    """(ox, oy, th): reference start position + initial heading, same
    convention as score.py's _canonicalize."""
    ox, oy = reference[0][0], reference[0][1]
    th = 0.0
    for p in reference[1:]:
        px, py = p[0], p[1]
        if math.hypot(px - ox, py - oy) > 1e-6:
            th = math.atan2(py - oy, px - ox)
            break
    return ox, oy, th


def _to_canonical(x: np.ndarray, y: np.ndarray, yaw: np.ndarray, ox: float, oy: float, th: float):
    c, s = math.cos(-th), math.sin(-th)
    xc = (x - ox) * c - (y - oy) * s
    yc = (x - ox) * s + (y - oy) * c
    yawc = yaw - th
    return xc, yc, yawc


_WINDOW_BACK = 5
_WINDOW_FWD = 15


def _windowed_nearest_segment(
    pt: np.ndarray, ref_xy: np.ndarray, last_idx: int | None
) -> tuple[int, float, float]:
    """Like ``nearest_segment``, but searches only near the last match instead
    of the whole path -- a global search is ambiguous on loop paths, where the
    start and end are the same point. Seeded at index 0 on the first call,
    since every run is anchored so the path starts where the robot starts."""
    n_segs = len(ref_xy) - 1
    if last_idx is None:
        lo, hi = 0, min(n_segs, _WINDOW_FWD)
    else:
        lo = max(0, last_idx - _WINDOW_BACK)
        hi = min(n_segs, last_idx + _WINDOW_FWD)

    seg_idx, dist, t_along = nearest_segment(pt, ref_xy[lo : hi + 1])
    return lo + seg_idx, dist, t_along


def _series(rec: RunRecording) -> dict[str, np.ndarray]:
    """Commanded + actual x/y/heading (canonical frame), and their errors."""
    ox, oy, th = _canonical_frame(rec.reference)

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

    actual_xc, actual_yc, actual_yawc = _to_canonical(actual_x, actual_y, actual_yaw, ox, oy, th)
    cmd_xc, cmd_yc, cmd_yawc = _to_canonical(cmd_x, cmd_y, cmd_yaw, ox, oy, th)

    return {
        "t": t,
        "actual_x": actual_xc,
        "cmd_x": cmd_xc,
        "err_x": actual_x - cmd_x,
        "actual_y": actual_yc,
        "cmd_y": cmd_yc,
        "err_y": actual_y - cmd_y,
        "actual_yaw": np.degrees(actual_yawc),
        "cmd_yaw": np.degrees(cmd_yawc),
        "err_yaw": np.degrees(heading_err),
    }


POS_ERR_LIM = 0.30  # m
HEAD_ERR_LIM = 120.0  # deg

_CHANNELS = [
    ("x position", "x (m)", "actual_x", "cmd_x", "err_x", "error (m)", POS_ERR_LIM, "m"),
    ("y position", "y (m)", "actual_y", "cmd_y", "err_y", "error (m)", POS_ERR_LIM, "m"),
    ("heading", "heading (deg)", "actual_yaw", "cmd_yaw", "err_yaw", "error (deg)", HEAD_ERR_LIM, "deg"),
]


def _draw_row(
    left,
    right,
    row_title: str,
    ylabel: str,
    err_ylabel: str,
    actual_key: str,
    cmd_key: str,
    err_key: str,
    err_lim: float,
    unit: str,
    labeled_series: list[tuple[str, dict[str, np.ndarray]]],
) -> None:
    """Draw one channel's commanded-vs-actual + error pair into the given axes."""
    ref_drawn = False
    for i, (label, s) in enumerate(labeled_series):
        color = _COLORS.get(i, "gray")
        if not ref_drawn:
            left.plot(s["t"], s[cmd_key], color="black", lw=2.2, label="commanded", zorder=10)
            ref_drawn = True
        left.plot(s["t"], s[actual_key], color=color, lw=1.3, label=label, alpha=0.9)
        left.plot(s["t"][-1], s[actual_key][-1], "o", ms=4.5, color=color, zorder=11)
        right.plot(s["t"], s[err_key], color=color, lw=1.3, label=label, alpha=0.9)

        err = s[err_key]
        worst = int(np.nanargmax(np.abs(err)))
        right.annotate(
            f"{err[worst]:+.3f}{unit}",
            (s["t"][worst], err[worst]),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7.5, color=color,
        )

    right.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    right.set_ylim(-err_lim, err_lim)
    left.set_ylabel(ylabel)
    left.set_title(f"{row_title}: commanded vs actual")
    left.grid(True, alpha=0.3)
    left.legend(fontsize=8)
    right.set_ylabel(err_ylabel)
    right.set_title(f"{row_title}: error (actual - commanded)")
    right.grid(True, alpha=0.3)
    right.legend(fontsize=8)


def plot_combined(
    path_name: str,
    labeled_series: list[tuple[str, dict[str, np.ndarray]]],
    speed: float,
    out_path: str | FsPath,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=False)
    for row, (title, ylabel, actual_key, cmd_key, err_key, err_ylabel, err_lim, unit) in enumerate(_CHANNELS):
        _draw_row(
            axes[row][0], axes[row][1], title, ylabel, err_ylabel,
            actual_key, cmd_key, err_key, err_lim, unit, labeled_series,
        )

    axes[-1][0].set_xlabel("time (s)")
    axes[-1][1].set_xlabel("time (s)")

    fig.suptitle(f"{path_name} @ {speed:g} m/s -- controller comparison", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_all_combined(
    per_path_series: list[tuple[str, list[tuple[str, dict[str, np.ndarray]]]]],
    speed: float,
    out_path: str | FsPath,
) -> None:
    """One giant figure: every full-pose trajectory stacked, each with its own
    x/y/heading x (commanded-vs-actual | error) block, all controllers overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_paths = len(per_path_series)
    n_rows = n_paths * 3
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 3.6 * n_rows), sharex=False)
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for p_idx, (path_name, labeled_series) in enumerate(per_path_series):
        for c_idx, (title, ylabel, actual_key, cmd_key, err_key, err_ylabel, err_lim, unit) in enumerate(_CHANNELS):
            row = p_idx * 3 + c_idx
            _draw_row(
                axes[row][0], axes[row][1],
                f"[{path_name}] {title}", ylabel, err_ylabel,
                actual_key, cmd_key, err_key, err_lim, unit, labeled_series,
            )

    axes[-1][0].set_xlabel("time (s)")
    axes[-1][1].set_xlabel("time (s)")

    fig_height_in = 3.6 * n_rows
    fig.tight_layout()
    fig.subplots_adjust(top=1.0 - (1.4 / fig_height_in))
    fig.suptitle(
        f"Full-pose trajectory comparison @ {speed:g} m/s (all controllers, all trajectories)",
        fontsize=16,
        y=1.0 - (0.3 / fig_height_in),
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combined commanded-vs-actual + error plots, all controllers overlaid, full-pose paths only"
    )
    ap.add_argument("dirs", nargs="+", help="one or more <recordings_dir>:<label> pairs")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--out-dir", default="data/benchmark/fullpose_combined")
    args = ap.parse_args()

    labeled_dirs = []
    for entry in args.dirs:
        if ":" not in entry:
            raise SystemExit(f"expected <dir>:<label>, got {entry!r}")
        d, label = entry.rsplit(":", 1)
        labeled_dirs.append((d, label))

    out_dir = FsPath(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_path_series: list[tuple[str, list[tuple[str, dict[str, np.ndarray]]]]] = []

    for path_name in FULLPOSE_PATHS:
        labeled_series = []
        for d, label in labeled_dirs:
            recs = [
                r
                for r in load_recordings(d)
                if r.path == path_name and math.isclose(r.speed, args.speed, abs_tol=1e-6)
            ]
            if not recs:
                print(f"skip {label} for {path_name}: no matching recording at {args.speed:g} m/s")
                continue
            labeled_series.append((label, _series(recs[0])))

        if not labeled_series:
            print(f"skip {path_name}: no controllers have data")
            continue

        out_path = out_dir / f"{path_name}_v{args.speed:.2f}_combined.png"
        plot_combined(path_name, labeled_series, args.speed, out_path)
        print(f"wrote {out_path}")

        per_path_series.append((path_name, labeled_series))

    if per_path_series:
        all_out_path = out_dir / f"all_fullpose_v{args.speed:.2f}_combined.png"
        plot_all_combined(per_path_series, args.speed, all_out_path)
        print(f"wrote {all_out_path}")


if __name__ == "__main__":
    main()

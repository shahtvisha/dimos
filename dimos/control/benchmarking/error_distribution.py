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

"""Distribution of per-run error metrics across many recordings, per controller.

Reuses ``score.py``'s ``score_recordings`` (the same trusted per-run cte/heading
scoring already used by ``compare_controllers.py`` and the operating-point map),
so the numbers here always agree with the rest of the benchmark suite.

Point of this tool: a single run's CTE tells you nothing about repeatability.
Feed it either many repeated runs of one path+speed (the real distribution
question -- "how much does this vary run to run?") or the full path x speed
sweep (a looser proxy, pooled across conditions -- the title says which).

    python -m dimos.control.benchmarking.error_distribution \\
        data/benchmark/go2_mustafa_final:"Holonomic Pose Controller" \\
        data/benchmark/go2_dan_final:"Holonomic Velocity Controller" \\
        data/benchmark/go2_pcontroller:"Baseline P-Controller" \\
        [--path circle_offset_45] [--speed 0.5] [--out dist.png] [--json dist.json]

Note: ``score_run`` does a global nearest-point search, which is ambiguous on
loop paths (e.g. ``circle_offset_45``) right where start and end coincide --
same caveat as the rest of the suite. Worth a second look if a loop path's
cte_max looks like an outlier.
"""

from __future__ import annotations

import argparse
import json
import statistics

from dimos.control.benchmarking.score import load_recordings, score_recordings
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_COLORS = {
    0: "#0072B2",
    1: "#D55E00",
    2: "#009E73",
    3: "#CC79A7",
}

# (dict key, display label, unit conversion factor, unit label)
_METRICS = [
    ("cte_rms", "CTE RMS", 1.0, "m"),
    ("cte_max", "CTE max", 1.0, "m"),
    ("heading_err_rms", "Heading err RMS", 57.29577951308232, "deg"),  # rad -> deg
    ("heading_err_max", "Heading err max", 57.29577951308232, "deg"),
]


def _load_runs(recordings_dir: str, path: str | None, speed: float | None) -> list[dict]:
    recs = load_recordings(recordings_dir)
    if not recs:
        logger.warning(f"no recordings found in {recordings_dir}")
        return []
    _, runs = score_recordings(recs, tolerances_cm=[5, 10, 15])
    if path is not None:
        runs = [r for r in runs if r["path"] == path]
    if speed is not None:
        runs = [r for r in runs if abs(r["speed"] - speed) < 1e-6]
    return runs


def summarize(labeled_dirs: list[tuple[str, str]], path: str | None, speed: float | None) -> dict:
    """Per-controller summary stats for each metric, plus arrival rate."""
    summary: dict[str, dict] = {}
    for d, label in labeled_dirs:
        runs = _load_runs(d, path, speed)
        if not runs:
            summary[label] = {"n": 0}
            continue
        entry: dict = {"n": len(runs), "arrived_pct": 100.0 * sum(r["arrived"] for r in runs) / len(runs)}
        for key, _, factor, unit in _METRICS:
            vals = [r[key] * factor for r in runs]
            entry[key] = {
                "mean": statistics.fmean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "median": statistics.median(vals),
                "unit": unit,
            }
        summary[label] = entry
    return summary


def print_summary(summary: dict) -> None:
    for label, entry in summary.items():
        if entry.get("n", 0) == 0:
            print(f"{label}: no matching runs")
            continue
        print(f"\n{label}  (n={entry['n']}, arrived={entry['arrived_pct']:.0f}%)")
        for key, disp, _, unit in _METRICS:
            m = entry[key]
            print(
                f"  {disp:<18} mean={m['mean']:.4f}{unit}  std={m['std']:.4f}{unit}  "
                f"median={m['median']:.4f}{unit}  range=[{m['min']:.4f}, {m['max']:.4f}]{unit}"
            )


def plot_distribution(
    labeled_dirs: list[tuple[str, str]],
    path: str | None,
    speed: float | None,
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_label_runs = {label: _load_runs(d, path, speed) for d, label in labeled_dirs}
    labels = [label for label in per_label_runs if per_label_runs[label]]
    if not labels:
        raise SystemExit("no matching runs in any of the given directories")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (key, disp, factor, unit) in zip(axes, _METRICS, strict=True):
        data = [[r[key] * factor for r in per_label_runs[label]] for label in labels]
        bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        for i, box in enumerate(bp["boxes"]):
            color = _COLORS.get(i % len(_COLORS), "gray")
            box.set_facecolor(color)
            box.set_alpha(0.35)
            box.set_edgecolor(color)
        for i, vals in enumerate(data):
            color = _COLORS.get(i % len(_COLORS), "gray")
            jitter = [i + 1 + (0.08 * (((j * 2654435761) % 1000) / 1000.0 - 0.5)) for j in range(len(vals))]
            ax.scatter(jitter, vals, color=color, s=14, alpha=0.6, zorder=3)
        ax.set_ylabel(f"{disp} ({unit})")
        ax.set_title(disp)
        ax.grid(True, alpha=0.3, axis="y")

    n_runs_note = ", ".join(f"{label} n={len(per_label_runs[label])}" for label in labels)
    scope = []
    if path is not None:
        scope.append(f"path={path}")
    if speed is not None:
        scope.append(f"speed={speed:g} m/s")
    scope_str = ", ".join(scope) if scope else "pooled across all paths/speeds in each dir"
    fig.suptitle(f"Error distribution across runs -- {scope_str}\n({n_runs_note})", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def _parse_labeled_dir(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"expected <dir>:<label>, got {s!r}")
    d, label = s.split(":", 1)
    return d, label


def main() -> None:
    ap = argparse.ArgumentParser(description="Distribution of per-run error metrics, per controller")
    ap.add_argument("recordings", nargs="+", type=_parse_labeled_dir, help="<dir>:<label> ...")
    ap.add_argument("--path", default=None, help="restrict to one path name (recommended for a true repeated-run distribution)")
    ap.add_argument("--speed", type=float, default=None, help="restrict to one speed")
    ap.add_argument("--out", default="error_distribution.png", help="output plot path")
    ap.add_argument("--json", default=None, help="optional: also write summary stats as JSON")
    args = ap.parse_args()

    summary = summarize(args.recordings, args.path, args.speed)
    print_summary(summary)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.json}")

    plot_distribution(args.recordings, args.path, args.speed, args.out)


if __name__ == "__main__":
    main()

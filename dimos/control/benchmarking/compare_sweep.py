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

"""Combine several already-recorded benchmark sessions (one per swept
``lookahead``/``regulate_horizon`` value, each in its own directory) into ONE
trajectory-grid figure.

Same grid as ``score.py``'s per-session ``_plot_xy`` (one box per battery
path, x meters vs y meters, black reference path) but every box now overlays
every swept value at once instead of showing one session at a time: color
encodes the swept value (a gradient), line style encodes speed
(solid/dashed/...), and a run that timed out without arriving is marked with
a red X at wherever it stopped.

Run this AFTER the hardware sessions (one per value, each pointed at its own
folder via ``HOLO_OUT_DIR``), e.g.::

    HOLO_LOOKAHEAD=0.10 HOLO_OUT_DIR=data/benchmark/go2_sweep/la010 dimos run unitree-go2-holonomic-benchmark
    HOLO_LOOKAHEAD=0.15 HOLO_OUT_DIR=data/benchmark/go2_sweep/la015 dimos run unitree-go2-holonomic-benchmark
    ... (one hardware session per value)

    python -m dimos.control.benchmarking.compare_sweep --param lookahead --runs \\
        0.10=data/benchmark/go2_sweep/la010 \\
        0.15=data/benchmark/go2_sweep/la015 \\
        0.20=data/benchmark/go2_sweep/la020 \\
        0.25=data/benchmark/go2_sweep/la025 \\
        --out data/benchmark/go2_sweep/lookahead_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dimos.control.benchmarking.score import _canonicalize, load_recordings, score_recordings
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_LINESTYLES = ["-", "--", "-.", ":"]


def parse_runs(pairs: list[str]) -> dict[float, Path]:
    """Parse ``value=dir`` CLI pairs into ``{value: recordings_dir}``."""
    out: dict[float, Path] = {}
    for pair in pairs:
        value_str, _, dir_str = pair.partition("=")
        if not dir_str:
            raise SystemExit(f"--runs entries must look like VALUE=DIR, got {pair!r}")
        out[float(value_str)] = Path(dir_str)
    return out


def collect_runs(
    param: str, runs_by_value: dict[float, Path], tolerances_cm: list[float]
) -> list[dict[str, Any]]:
    """Load + score every value's recordings directory; tag each scored run
    with the swept value so the plotter can color by it."""
    all_runs: list[dict[str, Any]] = []
    for value, d in sorted(runs_by_value.items()):
        recs = load_recordings(d)
        if not recs:
            raise SystemExit(f"no run recordings found in {d} ({param}={value:g})")
        _, runs = score_recordings(recs, tolerances_cm)
        for r in runs:
            r[param] = value
        all_runs.extend(runs)
        logger.info(f"{param}={value:g}: loaded {len(runs)} scored run(s) from {d}")
    return all_runs


def plot_combined(param: str, all_runs: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm, colors as mcolors, pyplot as plt
    from matplotlib.lines import Line2D

    path_names = sorted({r["path"] for r in all_runs})
    speeds = sorted({r["speed"] for r in all_runs})
    values = sorted({r[param] for r in all_runs})
    if not path_names:
        raise SystemExit("no scored runs to plot")

    speed_style = {s: _LINESTYLES[i % len(_LINESTYLES)] for i, s in enumerate(speeds)}
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    cmap = cm.get_cmap("viridis")

    cols = min(len(path_names), 3)
    rows = -(-len(path_names) // cols)  # ceil
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.2 * rows), squeeze=False)
    flat = [ax for row in axes for ax in row]

    for ax, name in zip(flat, path_names, strict=False):
        runs_for_path = [r for r in all_runs if r["path"] == name]
        ref_drawn = False
        for r in runs_for_path:
            ref_c, ex_c = _canonicalize(r["ref"], r["exec"])
            if not ref_drawn:
                ax.plot(
                    [p[0] for p in ref_c],
                    [p[1] for p in ref_c],
                    "k-",
                    lw=2.2,
                    label="reference",
                    zorder=5,
                )
                ax.plot(0.0, 0.0, "ko", ms=5, zorder=5)
                ref_drawn = True
            if not ex_c:
                continue
            color = cmap(norm(r[param]))
            ax.plot(
                [p[0] for p in ex_c],
                [p[1] for p in ex_c],
                color=color,
                linestyle=speed_style[r["speed"]],
                lw=1.5,
                alpha=0.95 if r["arrived"] else 0.5,
            )
            if not r["arrived"]:
                ax.plot(ex_c[-1][0], ex_c[-1][1], marker="x", color="red", ms=9, mew=2, zorder=6)
        ax.set_title(name)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
    for ax in flat[len(path_names) :]:
        ax.set_visible(False)

    fig.suptitle(f"go2 holonomic: executed trajectory vs reference, swept by {param}")
    fig.tight_layout(rect=(0.0, 0.05, 0.90, 0.96))

    # Colorbar + speed-style legend in dedicated space reserved OUTSIDE the grid
    # (added after tight_layout so they can't be laid over a subplot).
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    fig.colorbar(sm, cax=cbar_ax, label=param)

    style_handles = [
        Line2D([0], [0], color="gray", linestyle=ls, lw=2, label=f"v={s:g}")
        for s, ls in speed_style.items()
    ]
    style_handles.append(
        Line2D([0], [0], marker="x", color="red", linestyle="none", label="did not arrive")
    )
    fig.legend(
        handles=style_handles,
        loc="lower center",
        ncol=len(style_handles),
        bbox_to_anchor=(0.45, 0.0),
    )

    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"wrote combined comparison plot -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--param", choices=["lookahead", "regulate_horizon"], required=True)
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="VALUE=DIR",
        help="e.g. 0.10=data/benchmark/go2_sweep/la010 0.15=data/benchmark/go2_sweep/la015 ...",
    )
    ap.add_argument("--tolerances", default="5,10,15", help="cm, comma-separated")
    ap.add_argument("--out", default=None, help="output PNG path (default: <param>_sweep_comparison.png)")
    args = ap.parse_args()

    runs_by_value = parse_runs(args.runs)
    tolerances = [float(t) for t in args.tolerances.split(",") if t.strip()]
    all_runs = collect_runs(args.param, runs_by_value, tolerances)

    out_path = Path(args.out) if args.out else Path(f"{args.param}_sweep_comparison.png")
    plot_combined(args.param, all_runs, out_path)


if __name__ == "__main__":
    main()

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

"""Side-by-side controller comparison: one row per path, one column per controller.

Reuses the same recording-loading and trajectory-canonicalization logic as
``score.py``, just laid out as a grid across controllers instead of a single
column per controller.

    python -m dimos.control.benchmarking.compare_controllers \\
        data/benchmark/go2:Mustafa \\
        data/benchmark/go2_dan_final:Dan \\
        data/benchmark/go2_pf:P-controller \\
        --out data/benchmark/controller_comparison.png
"""

from __future__ import annotations

import argparse

from dimos.control.benchmarking.score import _canonicalize, load_recordings, score_recordings
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def _load_runs(recordings_dir: str) -> list[dict]:
    recs = load_recordings(recordings_dir)
    if not recs:
        logger.warning(f"no recordings found in {recordings_dir}")
        return []
    _, runs = score_recordings(recs, tolerances_cm=[5, 10, 15])
    return runs


def compare(
    labeled_dirs: list[tuple[str, str]],
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_runs = {label: _load_runs(d) for d, label in labeled_dirs}
    paths = sorted({r["path"] for runs in all_runs.values() for r in runs})
    if not paths:
        raise SystemExit("no recordings found in any of the given directories")

    n_rows, n_cols = len(paths), len(labeled_dirs)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False
    )

    for row, path_name in enumerate(paths):
        for col, (_, label) in enumerate(labeled_dirs):
            ax = axes[row][col]
            prs = [r for r in all_runs[label] if r["path"] == path_name]
            if not prs:
                ax.set_title(f"{path_name} — {label} (no data)")
                ax.axis("off")
                continue
            ref_drawn = False
            for r in sorted(prs, key=lambda r: r["speed"]):
                ref_c, ex_c = _canonicalize(r["ref"], r["exec"])
                if not ref_drawn:
                    ax.plot(
                        [p[0] for p in ref_c], [p[1] for p in ref_c],
                        "k-", lw=2.0, label="reference"
                    )
                    ax.plot(0.0, 0.0, "ko", ms=5)
                    ref_drawn = True
                if not ex_c:
                    continue
                ax.plot(
                    [p[0] for p in ex_c],
                    [p[1] for p in ex_c],
                    lw=1.2,
                    label=f"v={r['speed']:g} (cte={r['cte_max'] * 100:.0f}cm"
                    f"{'' if r['arrived'] else ', NOT arrived'})",
                )
            ax.set_title(f"{path_name} — {label}")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6)

    fig.suptitle("Controller comparison: executed trajectory vs reference path")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side controller comparison plot")
    ap.add_argument(
        "dirs",
        nargs="+",
        help="one or more <recordings_dir>:<label> pairs, e.g. data/benchmark/go2:Mustafa",
    )
    ap.add_argument("--out", default="data/benchmark/controller_comparison.png")
    args = ap.parse_args()

    labeled_dirs = []
    for entry in args.dirs:
        if ":" not in entry:
            raise SystemExit(f"expected <dir>:<label>, got {entry!r}")
        d, label = entry.rsplit(":", 1)
        labeled_dirs.append((d, label))

    compare(labeled_dirs, args.out)


if __name__ == "__main__":
    main()

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

"""Converts benchmark RunRecordings (flat JSON) into memory2 SqliteStores, so
the existing, unmodified ``dimos mem rerun <out>.db`` replays them: actual +
commanded pose as moving 3D arrows and growing path trails, plus x/y/heading
actual-vs-commanded and error as scalar time series -- all through
render_store()'s ordinary generic-stream walk, nothing custom on the render
side.

Three modes:
  - Single recording -> single store, one controller (main/build_store).
  - Batch over <dir>:<label> pairs (same convention as plot_tracking_combined.py)
    -> one combined .rrd per (path, speed), all controllers overlaid, so a
    single file answers "how does each controller compare here" instead of
    making people flip between separate per-controller files
    (main_batch/build_combined_store).
  - Master store: every (controller, path, speed) run's streams in ONE file,
    checked in once (main_master/build_master_store), then queried afterwards
    for whichever run(s) you want -- one run, or several overlaid for
    comparison -- via render_selected()/main_view, without reconverting
    anything. This is the one to check into the repo for others to pull.

Commanded pose reuses the same windowed nearest-point search already trusted
in plot_tracking_combined.py (loop paths are ambiguous under a global search).
Unlike that script, this does NOT canonicalize the frame -- a replay should
sit in the real world frame it was recorded in, not a shared comparison frame
(canonicalization only mattered there for overlaying position plots; here the
reference path itself is the shared frame, so no rotation is needed).

Must be invoked via ``python -c "from ... import main; main()"``, NOT
``python -m``: memory2 resolves a stored stream's payload type by its dotted
module path, and a class defined in a script run via ``-m`` gets tagged with
module "__main__" instead, which breaks reopening the store later.

    # single recording
    python -c "from dimos.control.benchmarking.mem2_replay import main; main()" \\
        <recording.json> <out.db>
    dimos mem rerun <out.db>

    # batch, all controllers overlaid, one file per (path, speed)
    python -c "from dimos.control.benchmarking.mem2_replay import main_batch; main_batch()" \\
        "data/benchmark/go2_mustafa_final:Holonomic Pose Controller" \\
        "data/benchmark/go2_dan_final:Holonomic Velocity Controller" \\
        "data/benchmark/go2_pcontroller:Baseline P-Controller" \\
        --out-dir data/benchmark/rerun_replays
    rerun data/benchmark/rerun_replays/*.rrd

    # master store: everything in one file, query afterwards
    python -c "from dimos.control.benchmarking.mem2_replay import main_master; main_master()" \\
        "data/benchmark/go2_mustafa_final:Holonomic Pose Controller" \\
        "data/benchmark/go2_dan_final:Holonomic Velocity Controller" \\
        "data/benchmark/go2_pcontroller:Baseline P-Controller" \\
        data/benchmark/all_runs.db

    # pose vs velocity controller, same path/speed, overlaid (opens the viewer)
    python -c "from dimos.control.benchmarking.mem2_replay import main_view; main_view()" \\
        data/benchmark/all_runs.db compare.rrd \\
        --path circle_offset_45 --speed 0.9 --hpc --hvc

    # no --hpc/--hvc/--pc given -> overlays every controller available for that path/speed
    python -c "from dimos.control.benchmarking.mem2_replay import main_view; main_view()" \\
        data/benchmark/all_runs.db compare.rrd --path circle_offset_45 --speed 0.9

    # advanced: exact run-name prefixes (e.g. copied from main_master()'s printed listing)
    python -c "from dimos.control.benchmarking.mem2_replay import main_view; main_view()" \\
        data/benchmark/all_runs.db compare.rrd \\
        --run-prefix Holonomic_Pose_Controller_circle_offset_45_v0_90 Baseline_P_Controller_circle_offset_45_v0_90
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path as FsPath
from typing import TYPE_CHECKING, Any

import numpy as np

from dimos.control.benchmarking.benchmark import RunRecording
from dimos.control.benchmarking.plot_tracking_combined import _windowed_nearest_segment
from dimos.control.benchmarking.scoring import _reference_yaw
from dimos.memory2.cli.render import render_store
from dimos.memory2.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.trigonometry import angle_diff

if TYPE_CHECKING:
    from rerun._baseclasses import Archetype
    from rerun.blueprint import Blueprint

_ARROW_LEN = 0.3
_COMMANDED_COLOR = [0, 0, 0]  # black

# Same colorblind-safe palette used throughout the matplotlib tools in this
# package (plot_tracking_combined.py's _COLORS), so a controller is the same
# color everywhere it shows up, rerun included.
_PALETTE = [
    [0, 114, 178],  # blue
    [213, 94, 0],  # orange
    [0, 158, 115],  # green
    [204, 121, 167],  # purple
]

# Short, human-readable labels for the stream-name suffixes _write_run()
# writes (see _stream_name()) -- used only by render_selected() to keep
# per-run panel titles from being the entire "<label>_<path>_v<speed>_..."
# prefix; the underlying stream/identifier names are unaffected.
_DISPLAY_NAMES = {
    "actual_pose": "Actual Pose",
    "commanded_pose": "Commanded Pose",
    "commanded_path": "Reference Path",
    "actual_path": "Actual Path",
    "err_x_m": "Error X (m)",
    "err_y_m": "Error Y (m)",
    "err_yaw_deg": "Error Yaw (deg)",
    "actual_x_m": "Actual X (m)",
    "cmd_x_m": "Commanded X (m)",
    "actual_y_m": "Actual Y (m)",
    "cmd_y_m": "Commanded Y (m)",
    "actual_yaw_deg": "Actual Yaw (deg)",
    "cmd_yaw_deg": "Commanded Yaw (deg)",
}

# The 3 controllers this benchmark suite compares are a small, stable set, so
# main_view() exposes them as short boolean flags (--hpc/--hvc/--pc) instead
# of the full "--controller <label>" a growing, open-ended set (like path
# names) would need.
_CONTROLLER_LABELS = {
    "hpc": "Holonomic Pose Controller",
    "hvc": "Holonomic Velocity Controller",
    "pc": "Baseline P-Controller",
}

# Fixed color per known controller, so a controller is always the same color
# regardless of what order its directory happens to be passed in -- e.g.
# --hpc is always red, --hvc is always blue. The shared reference path stays
# black (_COMMANDED_COLOR) either way, since it's the same for every
# controller on a given (path, speed).
_CONTROLLER_COLORS: dict[str, list[int]] = {
    "Holonomic Pose Controller": [213, 0, 0],  # red
    "Holonomic Velocity Controller": [0, 114, 178],  # blue
    "Baseline P-Controller": [0, 158, 115],  # green
}


def _controller_style(label: str, fallback_idx: int) -> tuple[type[PoseStamped], list[int]]:
    """(pose class, color) for a controller -- fixed by identity for the 3
    known controllers above; an unrecognized label (not one of those 3)
    falls back to cycling the shared palette by its position in the list of
    dirs passed in, same as before this was made identity-based."""
    if label in _CONTROLLER_COLORS:
        idx = list(_CONTROLLER_COLORS).index(label)
        color = _CONTROLLER_COLORS[label]
    else:
        idx = fallback_idx
        color = _PALETTE[fallback_idx % len(_PALETTE)]
    return _ACTUAL_POSE_CLASSES[idx % len(_ACTUAL_POSE_CLASSES)], color


def _controller_for_prefix(prefix: str) -> tuple[str, list[int]] | None:
    """Recovers (label, color) for a run prefix built as
    _safe_ident(f"{label}_{path}_v{speed:.2f}") -- lets render_selected()
    give each plotted scalar series its controller's fixed color and a
    short legend name (see _CONTROLLER_COLORS) instead of rerun's default
    fallback (the full entity path), which is what was swamping the legend."""
    for label, color in _CONTROLLER_COLORS.items():
        if prefix.startswith(_safe_ident(label) + "_"):
            return label, color
    return None


# The scalar suffixes _write_run() writes, in display order -- used to lay
# out one TimeSeriesView per metric in _build_blueprint().
_METRIC_SUFFIXES = [
    "err_x_m",
    "err_y_m",
    "err_yaw_deg",
    "actual_x_m",
    "cmd_x_m",
    "actual_y_m",
    "cmd_y_m",
    "actual_yaw_deg",
    "cmd_yaw_deg",
]


def _build_blueprint(run_prefixes: list[str]) -> Blueprint:
    """One Spatial3D view (every selected run's pose/path, any controller)
    plus one TimeSeriesView per metric, each explicitly listing that metric's
    entity under every given run prefix -- e.g. "Error X (m)" shows every
    selected controller's curve overlaid in one panel. Sent explicitly rather
    than relying on rerun's default auto-layout, which groups entities by
    shared parent and so does NOT combine the same metric living under two
    different runs' prefixes into one panel on its own.

    Content queries can't do this with a wildcard: rerun's query-expression
    language only supports "/**" as a *suffix* (matching a whole subtree);
    a middle wildcard like "/**/err_x_m" (any prefix, exact suffix) isn't
    supported and silently matches nothing. Since the caller already knows
    the small, exact set of run_prefixes being rendered, listing each one
    explicitly sidesteps the limitation entirely."""
    import rerun.blueprint as rrb

    metric_views = [
        rrb.TimeSeriesView(
            name=_DISPLAY_NAMES[suffix],
            contents=[f"{prefix}/{suffix}" for prefix in run_prefixes],
            # pin to a small fixed corner -- without this, and without a
            # short per-series name (see render_selected()'s use of
            # rr.SeriesLines), the legend defaults to sizing itself off the
            # full entity path text and can swallow most of the panel.
            plot_legend=rrb.PlotLegend(corner=rrb.Corner2D.RightTop),
        )
        for suffix in _METRIC_SUFFIXES
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(name="3D Replay", contents=["/**"]),
            rrb.Grid(*metric_views, grid_columns=3),
            row_shares=[2, 3],
        ),
    )


class CommandedPose(PoseStamped):
    """PoseStamped whose to_rerun() is a visible arrow, not a bare transform."""

    def to_rerun(self) -> Archetype:
        import rerun as rr

        forward = self.orientation.rotate_vector(Vector3(_ARROW_LEN, 0, 0))
        return rr.Arrows3D(
            origins=[[self.x, self.y, self.z]],
            vectors=[[forward.x, forward.y, forward.z]],
            colors=[_COMMANDED_COLOR],
        )


# Each is a real, top-level class (not built by a factory function): memory2
# resolves a stream's payload type from cls.__module__ + cls.__qualname__, and
# a class defined inside a function gets a qualname like
# "_make_class.<locals>._ColoredPose", which isn't importable as a module
# path. Four separate static classes side-steps that entirely.
class ActualPose0(PoseStamped):
    def to_rerun(self) -> Archetype:
        import rerun as rr

        forward = self.orientation.rotate_vector(Vector3(_ARROW_LEN, 0, 0))
        return rr.Arrows3D(
            origins=[[self.x, self.y, self.z]],
            vectors=[[forward.x, forward.y, forward.z]],
            colors=[_PALETTE[0]],
        )


class ActualPose1(PoseStamped):
    def to_rerun(self) -> Archetype:
        import rerun as rr

        forward = self.orientation.rotate_vector(Vector3(_ARROW_LEN, 0, 0))
        return rr.Arrows3D(
            origins=[[self.x, self.y, self.z]],
            vectors=[[forward.x, forward.y, forward.z]],
            colors=[_PALETTE[1]],
        )


class ActualPose2(PoseStamped):
    def to_rerun(self) -> Archetype:
        import rerun as rr

        forward = self.orientation.rotate_vector(Vector3(_ARROW_LEN, 0, 0))
        return rr.Arrows3D(
            origins=[[self.x, self.y, self.z]],
            vectors=[[forward.x, forward.y, forward.z]],
            colors=[_PALETTE[2]],
        )


class ActualPose3(PoseStamped):
    def to_rerun(self) -> Archetype:
        import rerun as rr

        forward = self.orientation.rotate_vector(Vector3(_ARROW_LEN, 0, 0))
        return rr.Arrows3D(
            origins=[[self.x, self.y, self.z]],
            vectors=[[forward.x, forward.y, forward.z]],
            colors=[_PALETTE[3]],
        )


_ACTUAL_POSE_CLASSES = [ActualPose0, ActualPose1, ActualPose2, ActualPose3]

# Kept for the single-recording path (main/build_store) so its output is
# unchanged from before this batch/multi-controller mode was added.
ActualPose = ActualPose1


class Scalar:
    """Float wrapper with to_rerun(), so render_store()'s generic walk (which
    only logs payloads that implement to_rerun()) picks up error values as a
    plotted time series -- a bare float would otherwise be silently skipped."""

    def __init__(self, value: float) -> None:
        self.value = value

    def to_rerun(self) -> Archetype:
        import rerun as rr

        return rr.Scalars(self.value)


class PathLine:
    """A full 2D line strip (points already accumulated), so the 3D view shows
    where the path goes, not just a single moving arrow with no trail."""

    def __init__(self, points_xy: list[tuple[float, float]], color: list[int]) -> None:
        self.points_xy = points_xy
        self.color = color

    def to_rerun(self) -> Archetype:
        import rerun as rr

        strip = [[x, y, 0.0] for x, y in self.points_xy]
        return rr.LineStrips3D(strips=[strip], colors=[self.color], radii=0.01)


def _safe_ident(label: str) -> str:
    """memory2 stream names must be valid SQL identifiers (^[A-Za-z_][A-Za-z0-9_]*$)
    -- no spaces, no slashes -- but our controller labels are meant to be
    human-readable ("Holonomic Pose Controller"), so sanitize for the stream
    name while keeping the original label for display/printing."""
    ident = re.sub(r"\W+", "_", label).strip("_")
    if not ident or not re.match(r"^[A-Za-z_]", ident):
        ident = f"_{ident}"
    return ident


def _canonical_transform(reference: list[list[float]]) -> tuple[float, float, float]:
    """(ox, oy, th): reference start position + initial heading direction --
    identical convention to score.py's _canonicalize(), the transform the
    rest of the benchmark suite already trusts for overlaying runs. Each run
    is anchored wherever the robot physically stood that day
    (shift_path_to_start_at_pose() in benchmark.py), so two controllers'
    recordings of the "same" path sit at different absolute odom coordinates
    -- this is what actually lines them up for a direct overlay."""
    ox, oy = reference[0][0], reference[0][1]
    th = 0.0
    for px, py, _ in reference[1:]:
        if math.hypot(px - ox, py - oy) > 1e-6:
            th = math.atan2(py - oy, px - ox)
            break
    return ox, oy, th


def _transform_xy(xy: np.ndarray, ox: float, oy: float, th: float) -> np.ndarray:
    c, s = math.cos(-th), math.sin(-th)
    dx, dy = xy[:, 0] - ox, xy[:, 1] - oy
    return np.stack([dx * c - dy * s, dx * s + dy * c], axis=1)


def _derive_commanded(
    rec: RunRecording, *, canonicalize: bool = False
) -> list[tuple[float, float, float, float, float, float, float]]:
    """One (ts, x, y, yaw, cmd_x, cmd_y, cmd_yaw) tuple per tick -- the shared
    per-tick math build_store(), build_combined_store() and build_master_store()
    all need. With canonicalize=True, both the reference and the executed
    trajectory are rigid-transformed into the shared canonical frame (see
    _canonical_transform()) before the nearest-point search, so multiple
    controllers' runs of the same nominal path can be overlaid directly."""
    ref_xy = np.array([[p[0], p[1]] for p in rec.reference], dtype=np.float64)
    ref_yaw = np.unwrap(np.array([p[2] for p in rec.reference], dtype=np.float64))
    actual_xy = np.array([[tick[1], tick[2]] for tick in rec.ticks], dtype=np.float64)
    actual_yaw = np.unwrap(np.array([tick[3] for tick in rec.ticks], dtype=np.float64))

    th = 0.0
    if canonicalize:
        ox, oy, th = _canonical_transform(rec.reference)
        ref_xy = _transform_xy(ref_xy, ox, oy, th)
        actual_xy = _transform_xy(actual_xy, ox, oy, th)
        ref_yaw = ref_yaw - th
        actual_yaw = actual_yaw - th

    t0 = rec.ticks[0][0]
    out: list[tuple[float, float, float, float, float, float, float]] = []
    last_idx: int | None = None
    for i, tick in enumerate(rec.ticks):
        t = tick[0]
        x, y = actual_xy[i]
        yaw = actual_yaw[i]
        ts = t - t0

        seg_idx, _dist, t_along = _windowed_nearest_segment(actual_xy[i], ref_xy, last_idx)
        last_idx = seg_idx
        foot = ref_xy[seg_idx] + t_along * (ref_xy[seg_idx + 1] - ref_xy[seg_idx])
        cmd_yaw = _reference_yaw(ref_yaw, seg_idx, t_along)

        out.append((ts, float(x), float(y), float(yaw), float(foot[0]), float(foot[1]), float(cmd_yaw)))
    return out


def _canonical_reference_xy(rec: RunRecording, *, canonicalize: bool) -> list[tuple[float, float]]:
    """The reference path's (x, y) points, transformed the same way
    _derive_commanded() transforms this run's executed trajectory -- so the
    logged commanded_path line matches the actual/error streams' frame."""
    if not canonicalize:
        return [(p[0], p[1]) for p in rec.reference]
    ox, oy, th = _canonical_transform(rec.reference)
    ref_xy = np.array([[p[0], p[1]] for p in rec.reference], dtype=np.float64)
    return [(x, y) for x, y in _transform_xy(ref_xy, ox, oy, th)]


def _stream_name(prefix: str, suffix: str) -> str:
    """"<prefix>_<suffix>", or a bare "<suffix>" when prefix is empty -- lets
    build_store() reuse _write_run() below while keeping its original,
    unprefixed stream names ("actual_pose", not "_actual_pose")."""
    return suffix if not prefix else f"{prefix}_{suffix}"


def _write_run(
    store: SqliteStore,
    prefix: str,
    rec: RunRecording,
    pose_cls: type[PoseStamped],
    color: list[int],
    *,
    write_commanded: bool = True,
    canonicalize: bool = False,
) -> None:
    """Writes one run's full stream set (actual pose, growing path trail,
    error and actual-vs-commanded scalars) under "<prefix>_...". The
    single-recording and master-store modes each want their own commanded
    arrow + reference path (write_commanded=True, the default); the
    combined-overlay mode shares one reference path across controllers
    instead, so it draws that itself and passes write_commanded=False.

    canonicalize=True rigid-transforms this run into the shared canonical
    frame (see _canonical_transform()) before writing anything -- needed
    whenever multiple controllers' runs of the same nominal path will be
    overlaid, since each is otherwise anchored wherever the robot physically
    stood that day and would sit at different absolute coordinates."""
    actual_stream = store.stream(_stream_name(prefix, "actual_pose"), payload_type=pose_cls)
    actual_path_stream = store.stream(_stream_name(prefix, "actual_path"), payload_type=PathLine)
    err_x_stream = store.stream(_stream_name(prefix, "err_x_m"), payload_type=Scalar)
    err_y_stream = store.stream(_stream_name(prefix, "err_y_m"), payload_type=Scalar)
    err_yaw_stream = store.stream(_stream_name(prefix, "err_yaw_deg"), payload_type=Scalar)
    actual_x_stream = store.stream(_stream_name(prefix, "actual_x_m"), payload_type=Scalar)
    cmd_x_stream = store.stream(_stream_name(prefix, "cmd_x_m"), payload_type=Scalar)
    actual_y_stream = store.stream(_stream_name(prefix, "actual_y_m"), payload_type=Scalar)
    cmd_y_stream = store.stream(_stream_name(prefix, "cmd_y_m"), payload_type=Scalar)
    actual_yaw_stream = store.stream(_stream_name(prefix, "actual_yaw_deg"), payload_type=Scalar)
    cmd_yaw_stream = store.stream(_stream_name(prefix, "cmd_yaw_deg"), payload_type=Scalar)

    if write_commanded:
        cmd_stream = store.stream(_stream_name(prefix, "commanded_pose"), payload_type=CommandedPose)
        store.stream(_stream_name(prefix, "commanded_path"), payload_type=PathLine).append(
            PathLine(_canonical_reference_xy(rec, canonicalize=canonicalize), _COMMANDED_COLOR), ts=0.0
        )

    actual_trail: list[tuple[float, float]] = []
    for ts, x, y, yaw, cx, cy, cyaw in _derive_commanded(rec, canonicalize=canonicalize):
        actual_trail.append((x, y))
        actual_stream.append(
            pose_cls(ts=ts, position=Vector3(x, y, 0.0), orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw))),
            ts=ts,
        )
        if write_commanded:
            cmd_stream.append(
                CommandedPose(ts=ts, position=Vector3(cx, cy, 0.0), orientation=Quaternion.from_euler(Vector3(0.0, 0.0, cyaw))),
                ts=ts,
            )
        actual_path_stream.append(PathLine(list(actual_trail), color), ts=ts)

        err_x_stream.append(Scalar(x - cx), ts=ts)
        err_y_stream.append(Scalar(y - cy), ts=ts)
        err_yaw_stream.append(Scalar(math.degrees(angle_diff(yaw, cyaw))), ts=ts)
        actual_x_stream.append(Scalar(x), ts=ts)
        cmd_x_stream.append(Scalar(cx), ts=ts)
        actual_y_stream.append(Scalar(y), ts=ts)
        cmd_y_stream.append(Scalar(cy), ts=ts)
        actual_yaw_stream.append(Scalar(math.degrees(yaw)), ts=ts)
        cmd_yaw_stream.append(Scalar(math.degrees(cyaw)), ts=ts)


def build_store(rec: RunRecording, out_path: str | FsPath) -> None:
    """Single recording, single controller (orange actual vs. black commanded).

    Deliberately NOT canonicalized: a lone recording should sit in the real
    odom frame it was recorded in. Canonicalizing only matters once there's
    a second run to line it up against (see build_combined_store(),
    build_master_store())."""
    store = SqliteStore(path=str(out_path))
    _write_run(store, "", rec, ActualPose, _PALETTE[1])
    store.stop()  # checkpoints and closes the WAL files before this returns


def build_combined_store(labeled_recs: list[tuple[str, RunRecording]], out_path: str | FsPath) -> None:
    """Multiple controllers, same (path, speed), overlaid in one store: one
    shared static reference path drawn once (not per controller -- overlapping
    identical black arrows would add nothing), plus each controller's colored
    actual pose/path/scalar streams under a "<label>_" prefix.

    Canonicalized (see _canonical_transform()): each run is anchored wherever
    the robot physically stood that day, so without this the "same" nominal
    path from two controllers would sit at different absolute coordinates --
    two squares in different places instead of one overlaid comparison."""
    store = SqliteStore(path=str(out_path))

    store.stream("commanded_path", payload_type=PathLine).append(
        PathLine(_canonical_reference_xy(labeled_recs[0][1], canonicalize=True), _COMMANDED_COLOR), ts=0.0
    )

    for i, (label, rec) in enumerate(labeled_recs):
        pose_cls, color = _controller_style(label, i)
        # memory2 stream names must be valid SQL identifiers -- no spaces or
        # slashes -- so a human label like "Holonomic Pose Controller" is
        # sanitized here; the entity tree in rerun is flat, not nested, as a
        # result, but everything still renders correctly.
        _write_run(store, _safe_ident(label), rec, pose_cls, color, write_commanded=False, canonicalize=True)

    store.stop()  # checkpoints and closes the WAL files before this returns


def build_master_store(labeled_dirs: list[tuple[str, str]], out_path: str | FsPath) -> None:
    """Every (controller, path, speed) run's streams in one store, each under
    its own "<label>_<path>_v<speed>_" prefix -- one file to check in and
    share, queried afterwards with render_selected() for whichever run(s) you
    actually want to look at, without reconverting anything. A controller
    keeps the same color everywhere it appears (fixed by identity, see
    _CONTROLLER_COLORS).

    Canonicalized for the same reason as build_combined_store(): this store
    exists specifically to overlay different controllers' runs of the same
    path, and each run's raw odom-frame position is otherwise wherever the
    robot happened to be anchored that day."""
    from dimos.control.benchmarking.score import load_recordings

    store = SqliteStore(path=str(out_path))
    for label_idx, (d, label) in enumerate(labeled_dirs):
        pose_cls, color = _controller_style(label, label_idx)
        seen: set[tuple[str, float]] = set()
        for rec in load_recordings(d):
            # a benchmark dir can hold repeat runs of the same (path, speed)
            # (e.g. go2_square_v0.90_001.json and go2_square_v0.90_018.json)
            # -- since the prefix doesn't encode the file's run id, writing
            # more than one would silently append a second copy of every
            # observation into the same stream (e.g. two overlapping
            # reference-path squares). Keep only the first (sorted-filename
            # order, from load_recordings()), skip the rest.
            key = (rec.path, round(rec.speed, 2))
            if key in seen:
                print(f"  skip repeat recording: {label} {rec.path} @ {rec.speed:g} m/s already has a run")
                continue
            seen.add(key)
            prefix = _safe_ident(f"{label}_{rec.path}_v{rec.speed:.2f}")
            _write_run(store, prefix, rec, pose_cls, color, canonicalize=True)
    store.stop()  # checkpoints and closes the WAL files before this returns


def render_selected(
    store: SqliteStore, run_prefixes: list[str], out_path: str | FsPath, *, no_gui: bool = False
) -> str:
    """Render only the streams belonging to the given run prefixes into a
    fresh .rrd -- everything else in a (possibly much bigger) master store is
    left out. Same per-observation walk as memory2's own render_store(), just
    filtered to a chosen subset; that filter is the one thing the existing,
    unmodified core tool doesn't support."""
    import shutil
    import subprocess

    import rerun as rr

    from dimos.memory2.utils.progress import progress
    from dimos.visualization.rerun.init import rerun_init

    wanted = [name for name in store.list_streams() if any(name.startswith(p) for p in run_prefixes)]
    if not wanted:
        raise SystemExit(f"no streams matched any of {run_prefixes!r}")
    for p in run_prefixes:
        if not any(name.startswith(p) for name in wanted):
            print(f"  warning: no streams found for {p!r} -- that run isn't in this store, it won't show up")

    # (entity path, stream, series style) -- style is (label, color) for a
    # Scalar stream belonging to one of the known controllers, else None.
    # Without an explicit style, rerun assigns each series an arbitrary
    # color (can collide between controllers) and falls back to the full
    # entity path as its legend name, which is what was swamping the legend.
    renderable: list[tuple[str, Any, tuple[str, list[int]] | None]] = []
    t0: float | None = None
    for name in wanted:
        stream = store.streams[name]
        try:
            first = stream.first()
        except LookupError:
            continue
        if not hasattr(first.data, "to_rerun"):
            print(f"  skip {name}: {type(first.data).__name__} has no to_rerun()")
            continue
        # nest "<prefix>_<suffix>" as "<prefix>/<suffix>" so a run's streams
        # group under one collapsible row in the entity tree; the blueprint
        # (_build_blueprint()) is what actually gives panels their short,
        # human-readable titles and groups same-metric entities from
        # different runs into one overlaid view.
        prefix = max((p for p in run_prefixes if name.startswith(p)), key=len)
        path = f"{prefix}/{name[len(prefix):].lstrip('_')}"
        style = _controller_for_prefix(prefix) if isinstance(first.data, Scalar) else None
        renderable.append((path, stream, style))
        t0 = first.ts if t0 is None else min(t0, first.ts)

    if t0 is None:
        raise SystemExit("nothing renderable in the selected runs")

    rerun_init("dimos benchmark replay")
    rr.save(str(out_path))
    rr.send_blueprint(_build_blueprint(run_prefixes), make_active=True, make_default=True)

    for path, _stream, style in renderable:
        if style is not None:
            label, color = style
            rr.log(path, rr.SeriesLines(colors=[color], names=[label]), static=True)

    for path, stream, _style in renderable:
        with progress(stream.count(), label=path) as report:
            for obs in stream:
                if obs.data is None:
                    report(obs)
                    continue
                rr.set_time("time", duration=obs.ts - t0)
                data = obs.data.to_rerun()
                if isinstance(data, list):
                    for sub, arch in data:
                        rr.log(f"{path}/{sub}", arch)
                else:
                    rr.log(path, data)
                report(obs)

    rr.rerun_shutdown()  # flush + close the .rrd before opening it
    print(f"wrote {out_path}")
    if not no_gui:
        exe = shutil.which("rerun")
        if exe:
            # --new: always start a fresh viewer instead of streaming into
            # whatever's already running on the default port. Without this,
            # an old viewer session can keep an earlier, now-stale blueprint
            # active and never pick up the one just sent above.
            subprocess.Popen([exe, "--new", str(out_path)])
            print(f"  opening {out_path} in a new rerun window")
        else:
            print(f"  rerun viewer not found on PATH; open manually:\n    rerun --new {out_path}")
    return str(out_path)


def _parse_labeled_dirs(entries: list[str]) -> list[tuple[str, str]]:
    """Parses repeated "<dir>:<label>" CLI args, e.g. "data/benchmark/go2:Holonomic Pose Controller"."""
    labeled_dirs = []
    for entry in entries:
        if ":" not in entry:
            raise SystemExit(f"expected <dir>:<label>, got {entry!r}")
        d, label = entry.rsplit(":", 1)
        labeled_dirs.append((d, label))
    return labeled_dirs


def main_master() -> None:
    ap = argparse.ArgumentParser(
        description="Build one master memory2 store holding every (controller, path, speed) run"
    )
    ap.add_argument("dirs", nargs="+", help="one or more <recordings_dir>:<label> pairs")
    ap.add_argument("out", help="output .db path")
    args = ap.parse_args()

    labeled_dirs = _parse_labeled_dirs(args.dirs)

    # SqliteStore opens (not truncates) an existing file, so re-running this
    # against the same path would append a second copy of every observation
    # on top of whatever's already there. A rebuild always means "start
    # fresh," so clear out any previous store (+ its WAL sidecars) first.
    out_path = FsPath(args.out)
    for suffix in ("", "-shm", "-wal"):
        out_path.with_name(out_path.name + suffix).unlink(missing_ok=True)

    build_master_store(labeled_dirs, args.out)
    print(f"wrote {args.out}")

    store = SqliteStore(path=args.out, must_exist=True)
    run_prefixes = sorted(
        name[: -len("_actual_pose")] for name in store.list_streams() if name.endswith("_actual_pose")
    )
    print(f"\n{len(run_prefixes)} run(s) available -- pass any of these to main_view():")
    for p in run_prefixes:
        print(f"  {p}")


def main_view() -> None:
    ap = argparse.ArgumentParser(description="Render selected runs out of a master store into a fresh .rrd")
    ap.add_argument("store", help="path to the master .db")
    ap.add_argument("out", help="output .rrd path")
    ap.add_argument("--path", help="benchmark path name, e.g. circle_offset_45")
    ap.add_argument("--speed", type=float, help="speed in m/s, e.g. 0.9")
    for flag, label in _CONTROLLER_LABELS.items():
        ap.add_argument(f"--{flag}", action="store_true", help=f"overlay {label}")
    ap.add_argument(
        "--run-prefix",
        nargs="+",
        default=None,
        help="advanced: exact run-name prefix(es) (e.g. from main_master()'s listing), "
        "bypasses --path/--speed/--hpc/--hvc/--pc",
    )
    ap.add_argument("--no-gui", action="store_true", help="write the .rrd without opening the rerun viewer")
    args = ap.parse_args()

    if args.run_prefix:
        prefixes = args.run_prefix
    else:
        if not args.path or args.speed is None:
            raise SystemExit("need --path and --speed (or --run-prefix for exact matches)")
        chosen = [label for flag, label in _CONTROLLER_LABELS.items() if getattr(args, flag)]
        labels = chosen or list(_CONTROLLER_LABELS.values())  # none picked -> overlay all
        prefixes = [_safe_ident(f"{label}_{args.path}_v{args.speed:.2f}") for label in labels]

    store = SqliteStore(path=args.store, must_exist=True)
    render_selected(store, prefixes, args.out, no_gui=args.no_gui)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a benchmark RunRecording JSON into a memory2 store")
    ap.add_argument("recording", help="path to a RunRecording *.json")
    ap.add_argument("out", help="output .db path")
    args = ap.parse_args()

    data = json.loads(FsPath(args.recording).read_text())
    rec = RunRecording(**data)
    build_store(rec, args.out)
    print(f"wrote {args.out} -- run: dimos mem rerun {args.out}")


def main_batch() -> None:
    from dimos.control.benchmarking.score import load_recordings

    ap = argparse.ArgumentParser(
        description="Batch-convert every (path, speed) across labeled controller dirs into one combined "
        "rerun replay each -- all controllers overlaid, ready to view, no conversion needed by the viewer"
    )
    ap.add_argument("dirs", nargs="+", help="one or more <recordings_dir>:<label> pairs")
    ap.add_argument("--path", default=None, help="restrict to one path name (default: all)")
    ap.add_argument("--speed", type=float, default=None, help="restrict to one speed (default: all)")
    ap.add_argument("--out-dir", default="data/benchmark/rerun_replays")
    args = ap.parse_args()

    labeled_dirs = _parse_labeled_dirs(args.dirs)

    out_dir = FsPath(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recs_by_label = {label: load_recordings(d) for d, label in labeled_dirs}
    combos = sorted({(r.path, r.speed) for recs in recs_by_label.values() for r in recs})
    if args.path is not None:
        combos = [c for c in combos if c[0] == args.path]
    if args.speed is not None:
        combos = [c for c in combos if math.isclose(c[1], args.speed, abs_tol=1e-6)]
    if not combos:
        raise SystemExit("no matching (path, speed) combinations found")

    written = 0
    for path_name, speed in combos:
        labeled_recs = []
        for label, recs in recs_by_label.items():
            matches = [r for r in recs if r.path == path_name and math.isclose(r.speed, speed, abs_tol=1e-6)]
            if not matches:
                print(f"skip {label} for {path_name} @ {speed:g} m/s: no matching recording")
                continue
            labeled_recs.append((label, matches[0]))

        if not labeled_recs:
            print(f"skip {path_name} @ {speed:g} m/s: no controllers have data")
            continue

        db_path = out_dir / f"{path_name}_v{speed:.2f}.db"
        rrd_path = out_dir / f"{path_name}_v{speed:.2f}.rrd"
        build_combined_store(labeled_recs, db_path)
        render_store(SqliteStore(path=str(db_path), must_exist=True), out=str(rrd_path), no_gui=True)
        # the .rrd is the shareable artifact; the .db (+ its WAL sidecar files,
        # since SqliteStore opens in WAL mode) was just scratch.
        for suffix in ("", "-shm", "-wal"):
            sidecar = db_path.with_name(db_path.name + suffix)
            sidecar.unlink(missing_ok=True)
        written += 1

    print(f"done -- {written} combined replay(s) written to {out_dir}")


if __name__ == "__main__":
    main()

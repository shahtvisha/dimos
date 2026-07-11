"""Offline lidar-vs-stereo comparison from a db recorded by
scripts/lidar_stereo_record.py — ideally while the robot rotates in place, not
stationary (see the module-level notes on ``realsense_raw`` and ``stereo`` below
for why that matters here).

Reuses the exact same metric/alignment functions as the live
dimos/mapping/utils/cli/lidar_stereo_bench.py module — only the data source
changes (recorded db instead of live LCM subscriptions), so alignment/metric
code can be iterated on repeatedly without re-running the hardware pipeline.

Usage:
    python scripts/lidar_stereo_compare_offline.py lidar_stereo.db
"""

from __future__ import annotations

import sys

import numpy as np

from dimos.mapping.utils.cli.lidar_stereo_bench import (
    ROUGH_CAM_OFFSET_IN_LIDAR_FRAME,
    R_OPT_TO_LINK,
    apply_matrix,
    best_yaw_icp,
    compute_score,
    crop_forward_cone,
    drop_near_field,
    format_score,
    format_verdict_report,
    range_binned_density,
    voxel_downsample_xyz,
)
from dimos.memory2.store.sqlite import SqliteStore
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

MIN_LIDAR_RANGE_M = 0.5
ICP_MAX_CORR_DIST_M = 0.25
YAW_STEPS = 12
FSCORE_TIGHT_M = 0.05
FSCORE_LOOSE_M = 0.20
VOXEL_SIZE_M = 0.05
RANGE_BINS_M = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 100.0]
# Generous vs. the D435i's ~87 degree horizontal FOV (43.5 degree half-angle) since
# the two sensors' exact relative heading isn't known yet.
FORWARD_CONE_HALF_ANGLE_DEG = 70.0
FRAME_OFFSET_S = 6.0  # lidar: skip the first few seconds, same reasoning as before
REALSENSE_SAMPLE_FRAMES = 8  # independent headings to try (see _best_of_independent_frames)


def _frame_near(store: SqliteStore, name: str, offset_s: float) -> np.ndarray:
    """Pick the observation whose time is closest to (stream start + offset_s)."""
    first_ts: float | None = None
    best_obs = None
    best_dt = float("inf")
    for obs in store.stream(name, PointCloud2):
        if first_ts is None:
            first_ts = obs.ts
        dt = abs((obs.ts - first_ts) - offset_s)
        if dt < best_dt:
            best_dt = dt
            best_obs = obs
    if best_obs is None:
        raise SystemExit(f"No frames recorded for stream {name!r} — check `store.summary()` above")
    return best_obs.data.points_f32()


def _accumulate_xyz(store: SqliteStore, name: str, voxel_size: float) -> np.ndarray:
    """Concatenate every recorded frame's points, voxel-deduped.

    Only valid for streams already self-consistently registered in one frame over
    time — StereoPointCloud fuses its own VIO internally, so its frame_cloud
    messages are already comparable across time. If the robot rotated during the
    recording, this gives one map covering everywhere it looked, not just one
    narrow instant.
    """
    chunks = [obs.data.points_f32() for obs in store.stream(name, PointCloud2)]
    chunks = [c for c in chunks if len(c) > 0]
    if not chunks:
        raise SystemExit(f"No frames recorded for stream {name!r} — check `store.summary()` above")
    return voxel_downsample_xyz(np.concatenate(chunks, axis=0), voxel_size)


def _best_of_independent_frames(
    store: SqliteStore, name: str, lidar_xyz: np.ndarray, base_rotation: np.ndarray, n_samples: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Independently ICP-align several time-spread frames against lidar_xyz; keep
    whichever scores best.

    Unlike ``stereo``, ``realsense_raw`` has no persistent registration (raw SDK
    output) — concatenating frames from a rotating robot would smear geometry
    from different headings together with nothing to correct for the rotation
    between them. Instead, treat each frame as an independent alignment attempt;
    if the robot rotated, different frames faced different true headings, so
    trying several gets the benefit of that diversity without invalid merging.
    """
    all_obs = list(store.stream(name, PointCloud2))
    if not all_obs:
        raise SystemExit(f"No frames recorded for stream {name!r} — check `store.summary()` above")
    idxs = np.linspace(0, len(all_obs) - 1, num=min(n_samples, len(all_obs)), dtype=int)
    best: tuple[np.ndarray, dict[str, float]] | None = None
    for i in idxs:
        raw_xyz = all_obs[i].data.points_f32()
        if len(raw_xyz) < 50:
            continue
        icp_T = best_yaw_icp(
            raw_xyz, lidar_xyz, ICP_MAX_CORR_DIST_M,
            base_rotation=base_rotation, yaw_steps=YAW_STEPS,
            base_translation=ROUGH_CAM_OFFSET_IN_LIDAR_FRAME,
        )
        xyz = apply_matrix(raw_xyz, icp_T)
        score = compute_score(xyz, lidar_xyz, FSCORE_TIGHT_M, FSCORE_LOOSE_M, VOXEL_SIZE_M)
        if best is None or score["f_loose"] > best[1]["f_loose"]:
            best = (xyz, score)
    assert best is not None
    return best


def main(db_path: str) -> None:
    with SqliteStore(path=db_path) as store:
        print(store.summary())
        lidar_xyz = crop_forward_cone(
            drop_near_field(_frame_near(store, "lidar", FRAME_OFFSET_S), MIN_LIDAR_RANGE_M),
            FORWARD_CONE_HALF_ANGLE_DEG,
        )
        print(f"lidar (ref): n={len(lidar_xyz)}")
        scores: dict[str, dict[str, float]] = {}

        rs_xyz, scores["realsense (raw)"] = _best_of_independent_frames(
            store, "realsense_raw", lidar_xyz, R_OPT_TO_LINK.astype(np.float64), REALSENSE_SAMPLE_FRAMES,
        )
        print(format_score("realsense (raw, best-of-frames)", scores["realsense (raw)"], FSCORE_TIGHT_M, FSCORE_LOOSE_M, VOXEL_SIZE_M))

        stereo_raw_xyz = _accumulate_xyz(store, "stereo", VOXEL_SIZE_M)
        stereo_icp_T = best_yaw_icp(
            stereo_raw_xyz, lidar_xyz, ICP_MAX_CORR_DIST_M,
            base_rotation=np.eye(3), yaw_steps=YAW_STEPS,
            base_translation=ROUGH_CAM_OFFSET_IN_LIDAR_FRAME,
        )
        stereo_xyz = apply_matrix(stereo_raw_xyz, stereo_icp_T)
        scores["stereo (ours)"] = compute_score(stereo_xyz, lidar_xyz, FSCORE_TIGHT_M, FSCORE_LOOSE_M, VOXEL_SIZE_M)
        print(format_score("stereo (ours, accumulated)", scores["stereo (ours)"], FSCORE_TIGHT_M, FSCORE_LOOSE_M, VOXEL_SIZE_M))

    origin = np.zeros(3, dtype=np.float32)
    print(
        f"range-binned density {RANGE_BINS_M}: "
        f"lidar={range_binned_density(lidar_xyz, origin, RANGE_BINS_M)} "
        f"realsense={range_binned_density(rs_xyz, origin, RANGE_BINS_M)} "
        f"stereo={range_binned_density(stereo_xyz, origin, RANGE_BINS_M)}"
    )

    print()
    print(format_verdict_report(len(lidar_xyz), scores))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/lidar_stereo_compare_offline.py <db_path>")
    main(sys.argv[1])

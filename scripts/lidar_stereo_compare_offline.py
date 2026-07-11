"""Offline lidar-vs-stereo comparison from a db recorded by
scripts/lidar_stereo_record.py.

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
    accuracy_completeness_fscore,
    apply_matrix,
    best_yaw_icp,
    chamfer_distance,
    drop_near_field,
    range_binned_density,
    voxel_occupancy_iou,
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


def _last_xyz(store: SqliteStore, name: str) -> np.ndarray:
    last = None
    for obs in store.stream(name, PointCloud2):
        last = obs
    if last is None:
        raise SystemExit(f"No frames recorded for stream {name!r} — check `store.summary()` above")
    return last.data.points_f32()


def _fmt_score(pred: np.ndarray, gt: np.ndarray) -> str:
    cd = chamfer_distance(pred, gt)
    acc_t, comp_t, f_t = accuracy_completeness_fscore(pred, gt, FSCORE_TIGHT_M)
    _, _, f_l = accuracy_completeness_fscore(pred, gt, FSCORE_LOOSE_M)
    iou = voxel_occupancy_iou(pred, gt, VOXEL_SIZE_M)
    return (
        f"n={len(pred)}  chamfer={cd:.4f}  "
        f"acc@{FSCORE_TIGHT_M * 100:.0f}cm={acc_t:.2f}  comp@{FSCORE_TIGHT_M * 100:.0f}cm={comp_t:.2f}  "
        f"F@{FSCORE_TIGHT_M * 100:.0f}cm={f_t:.2f}  F@{FSCORE_LOOSE_M * 100:.0f}cm={f_l:.2f}  "
        f"IoU@{VOXEL_SIZE_M * 100:.0f}cm={iou:.2f}"
    )


def main(db_path: str) -> None:
    with SqliteStore(path=db_path) as store:
        print(store.summary())
        lidar_xyz = drop_near_field(_last_xyz(store, "lidar"), MIN_LIDAR_RANGE_M)
        rs_raw_xyz = _last_xyz(store, "realsense_raw")
        stereo_raw_xyz = _last_xyz(store, "stereo")

    print(f"lidar (ref): n={len(lidar_xyz)}")

    rs_icp_T = best_yaw_icp(
        rs_raw_xyz, lidar_xyz, ICP_MAX_CORR_DIST_M,
        base_rotation=R_OPT_TO_LINK.astype(np.float64), yaw_steps=YAW_STEPS,
        base_translation=ROUGH_CAM_OFFSET_IN_LIDAR_FRAME,
    )
    rs_xyz = apply_matrix(rs_raw_xyz, rs_icp_T)
    print(f"realsense (raw, ICP-aligned): {_fmt_score(rs_xyz, lidar_xyz)}")

    stereo_icp_T = best_yaw_icp(
        stereo_raw_xyz, lidar_xyz, ICP_MAX_CORR_DIST_M,
        base_rotation=np.eye(3), yaw_steps=YAW_STEPS,
        base_translation=ROUGH_CAM_OFFSET_IN_LIDAR_FRAME,
    )
    stereo_xyz = apply_matrix(stereo_raw_xyz, stereo_icp_T)
    print(f"stereo (ours, ICP-aligned): {_fmt_score(stereo_xyz, lidar_xyz)}")

    origin = np.zeros(3, dtype=np.float32)
    print(
        f"range-binned density {RANGE_BINS_M}: "
        f"lidar={range_binned_density(lidar_xyz, origin, RANGE_BINS_M)} "
        f"realsense={range_binned_density(rs_xyz, origin, RANGE_BINS_M)} "
        f"stereo={range_binned_density(stereo_xyz, origin, RANGE_BINS_M)}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/lidar_stereo_compare_offline.py <db_path>")
    main(sys.argv[1])

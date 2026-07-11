# Copyright 2026 Dimensional Inc.
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

"""Per-frame benchmark: Livox Mid-360 lidar vs RealSense (raw SDK cloud) vs
StereoPointCloud (our filtered pipeline).

Lidar is the reference ("ground truth") — absent survey equipment, it's the more
geometrically trustworthy sensor at range. Metrics: Chamfer distance, ETH3D-style
accuracy/completeness/F-score, voxel occupancy IoU, and a Newer-College-style
range-binned density falloff.

There is no calibrated D435i<->Mid-360 extrinsic for FlowBase anywhere in this repo
(checked: dimos/robot/assembly/mid360_realsense_30.py is a different, unrelated rig —
its mount is tilted ~36 degrees, contradicting the FlowBase README's "level
orientation"). So both candidate clouds are ICP-aligned to the lidar reference
instead of trusting any fixed offset.

Point-to-point ICP only converges from within roughly 30-45 degrees of the true
rotation, so a single identity-start guess isn't enough here: RealSense's raw cloud
is in *optical* convention (Z=depth, Y=down), a fixed ~90 degree rotation from the
lidar's body frame (Z=up) — that fixed rotation is known and applied. StereoPointCloud
is already reoriented to body-frame convention internally, but its Madgwick-derived
yaw has no absolute heading reference, so the remaining yaw is unknown — handled with
a multi-start sweep over heading hypotheses, keeping whichever gives the best ICP fit.

Once you have a measured extrinsic, thread it in as `base_rotation`/reduce
`yaw_steps` in `_evaluate` for faster, more robust convergence.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
from pydantic import Field
from reactivex.disposable import Disposable
from scipy.spatial import cKDTree

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.stereo_point_cloud.utils import R_OPT_TO_LINK
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Rough verbal estimate (not measured): D435i sits higher than the Mid-360 and at
# most 3cm further back, in the lidar's own body frame (X=forward, Y=left, Z=up).
# Height magnitude is still a guess — refine with a tape measure if ICP still struggles.
ROUGH_CAM_OFFSET_IN_LIDAR_FRAME = np.array([-0.03, 0.0, 0.10])


# ── Metrics ─────────────────────────────────────────────────────────────────


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric mean squared nearest-neighbor distance."""
    if len(pred) == 0 or len(gt) == 0:
        return float("nan")
    d_pred_to_gt, _ = cKDTree(gt).query(pred, k=1)
    d_gt_to_pred, _ = cKDTree(pred).query(gt, k=1)
    return float(np.mean(d_pred_to_gt**2) + np.mean(d_gt_to_pred**2))


def accuracy_completeness_fscore(
    pred: np.ndarray, gt: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    """ETH3D / Tanks-and-Temples convention: accuracy = pred covered by gt, completeness = gt covered by pred."""
    if len(pred) == 0 or len(gt) == 0:
        return 0.0, 0.0, 0.0
    d_pred_to_gt, _ = cKDTree(gt).query(pred, k=1)
    d_gt_to_pred, _ = cKDTree(pred).query(gt, k=1)
    accuracy = float(np.mean(d_pred_to_gt <= threshold))
    completeness = float(np.mean(d_gt_to_pred <= threshold))
    fscore = 0.0 if (accuracy + completeness) == 0 else 2 * accuracy * completeness / (accuracy + completeness)
    return accuracy, completeness, fscore


def voxel_occupancy_iou(pred: np.ndarray, gt: np.ndarray, voxel_size: float) -> float:
    if len(pred) == 0 or len(gt) == 0:
        return 0.0
    keys_pred = set(map(tuple, np.floor(pred / voxel_size).astype(np.int64)))
    keys_gt = set(map(tuple, np.floor(gt / voxel_size).astype(np.int64)))
    union = len(keys_pred | keys_gt)
    return len(keys_pred & keys_gt) / union if union else 0.0


def compute_score(
    pred: np.ndarray, gt: np.ndarray, fscore_tight: float, fscore_loose: float, voxel_size: float
) -> dict[str, float]:
    """All metrics for one candidate cloud against the reference — shared by the
    live module and the offline compare script so the two can't drift apart."""
    acc_t, comp_t, f_t = accuracy_completeness_fscore(pred, gt, fscore_tight)
    _, _, f_l = accuracy_completeness_fscore(pred, gt, fscore_loose)
    return {
        "n": len(pred),
        "chamfer": chamfer_distance(pred, gt),
        "acc_tight": acc_t, "comp_tight": comp_t, "f_tight": f_t,
        "f_loose": f_l,
        "iou": voxel_occupancy_iou(pred, gt, voxel_size),
    }


def format_score(name: str, s: dict[str, float], fscore_tight: float, fscore_loose: float, voxel_size: float) -> str:
    return (
        f"{name}: n={s['n']}  chamfer={s['chamfer']:.4f}  "
        f"acc@{fscore_tight * 100:.0f}cm={s['acc_tight']:.2f}  comp@{fscore_tight * 100:.0f}cm={s['comp_tight']:.2f}  "
        f"F@{fscore_tight * 100:.0f}cm={s['f_tight']:.2f}  F@{fscore_loose * 100:.0f}cm={s['f_loose']:.2f}  "
        f"IoU@{voxel_size * 100:.0f}cm={s['iou']:.2f}"
    )


def range_binned_density(points: np.ndarray, origin: np.ndarray, bins: list[float]) -> list[int]:
    """Point count per distance-from-origin bin — density/accuracy falloff vs range (Newer College style)."""
    if len(points) == 0:
        return [0] * (len(bins) - 1)
    d = np.linalg.norm(points - origin, axis=1)
    counts, _ = np.histogram(d, bins=bins)
    return counts.tolist()  # type: ignore[no-any-return]


def rms_from_chamfer(chamfer: float) -> float:
    """Approximate a human-readable RMS error (meters) from the Chamfer distance.

    ``chamfer_distance`` sums two mean-*squared*-distance terms (pred->gt and
    gt->pred); dividing by 2 and taking the square root gives a single
    representative RMS figure. It's an approximation for readability, not a
    formal statistic.
    """
    return float(np.sqrt(max(chamfer, 0.0) / 2.0))


def verdict_bucket(f_loose: float) -> str:
    """Rough, unvalidated pass/fail bucket from the loose-threshold F-score — a
    starting point for "is this even in the right ballpark", not a validated
    downstream requirement."""
    if f_loose >= 0.70:
        return "GOOD"
    if f_loose >= 0.40:
        return "OK"
    return "POOR"


def format_verdict_report(lidar_n: int, scores: dict[str, dict[str, float]]) -> str:
    """Plain-English summary: verdict bucket + headline numbers per candidate,
    plus how much worse each is than the best candidate present."""
    lines = [f"=== vs lidar (reference, forward-cone-cropped, n={lidar_n}) ==="]
    best_f = max((s["f_loose"] for s in scores.values()), default=0.0)
    for name, s in scores.items():
        verdict = verdict_bucket(s["f_loose"])
        rms = rms_from_chamfer(s["chamfer"])
        rel = ""
        if best_f > 0 and s["f_loose"] < best_f:
            worse_by = (best_f - s["f_loose"]) / best_f * 100
            rel = f" — {worse_by:.0f}% worse F-score than the best candidate here"
        lines.append(
            f"  {name:<28} {verdict:<4}  F@20cm={s['f_loose']:.2f}  "
            f"(~{rms:.1f}m RMS error, IoU@5cm={s['iou']:.2f}){rel}"
        )
    lines.append("  (GOOD >= 0.70, OK >= 0.40, POOR below — rough rule of thumb, not a validated spec)")
    return "\n".join(lines)


def icp_refine(
    source: np.ndarray, target: np.ndarray, init: np.ndarray, max_corr_dist: float
) -> o3d.pipelines.registration.RegistrationResult:
    """Snap ``source`` onto ``target`` starting from ``init`` (4x4)."""
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target)
    return o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr_dist, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def best_yaw_icp(
    source: np.ndarray,
    target: np.ndarray,
    max_corr_dist: float,
    base_rotation: np.ndarray,
    yaw_steps: int,
    base_translation: np.ndarray = np.zeros(3),
) -> np.ndarray:
    """Multi-start ICP over ``yaw_steps`` heading hypotheses around Z; keeps the best-fitness result.

    ``base_rotation`` is applied before the yaw sweep — use it for a *known* fixed
    rotation (e.g. optical -> body frame); leave as identity when only heading is
    unknown. ``base_translation`` is the source origin's position in the target
    frame (a rough prior is far better than none — blind full-6DOF search between a
    360-degree lidar and a narrow-FOV camera is underdetermined on translation).
    """
    best: o3d.pipelines.registration.RegistrationResult | None = None
    for i in range(yaw_steps):
        init = np.eye(4)
        init[:3, :3] = _yaw_matrix(2 * np.pi * i / yaw_steps) @ base_rotation
        init[:3, 3] = base_translation
        result = icp_refine(source, target, init, max_corr_dist)
        if best is None or result.fitness > best.fitness:
            best = result
    assert best is not None
    return best.transformation  # type: ignore[no-any-return]


def apply_matrix(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1))
    homogeneous = np.hstack([points, ones])
    return (matrix @ homogeneous.T).T[:, :3].astype(np.float32)  # type: ignore[no-any-return]


def drop_near_field(points: np.ndarray, min_range: float) -> np.ndarray:
    """Drop points within ``min_range`` of the sensor origin — self-return/housing noise.

    The raw Mid360 driver applies zero filtering (checked dimos/hardware/sensors/lidar/livox/cpp/main.cpp);
    FAST-LIO2 already gates this by default (`blind: float = 0.5` in fastlio2/module.py) — same idea here.
    """
    if len(points) == 0:
        return points
    return points[np.linalg.norm(points, axis=1) >= min_range]  # type: ignore[no-any-return]


def crop_forward_cone(points: np.ndarray, half_angle_deg: float) -> np.ndarray:
    """Keep only points within ``half_angle_deg`` of the +X (forward) axis.

    The Mid-360 sees the full 360-degree room; the D435i only sees a narrow
    (~87 degree) forward slice of it. Scoring/aligning the camera against the
    *entire* lidar sphere is structurally wrong on two counts: completeness/IoU
    are computed against points the camera could never see even with perfect
    data, and ICP's correspondence search is degraded by the huge fraction of
    lidar points with no valid camera match at all — both make results look
    worse than the actual sensor data. Cropped generously (wider than the
    D435i's real FOV) since the two sensors' exact relative heading isn't known.
    """
    if len(points) == 0:
        return points
    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    return points[np.abs(azimuth) <= half_angle_deg]  # type: ignore[no-any-return]


# ── Module ──────────────────────────────────────────────────────────────────


class BenchConfig(ModuleConfig):
    period_s: float = 2.0
    min_points: int = 50
    fscore_threshold_m: float = 0.05
    fscore_threshold_loose_m: float = 0.20
    voxel_size_m: float = 0.05
    icp_max_corr_dist_m: float = 0.25  # tighter now that a rough translation prior seeds the search
    min_lidar_range_m: float = 0.5  # drop near-field self-return noise (matches FAST-LIO2's `blind`)
    forward_cone_half_angle_deg: float = 70.0  # crop lidar's 360-degree sphere to ~camera FOV
    yaw_steps: int = 12  # heading hypotheses swept per ICP call (30 degree steps)
    range_bins_m: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 100.0])


class LidarStereoBenchmark(Module):
    """Buffers the latest lidar / realsense_raw / stereo frame and scores them every ``period_s``."""

    config: BenchConfig

    lidar: In[PointCloud2]
    realsense_raw: In[PointCloud2]
    stereo: In[PointCloud2]

    # Published (not logged directly) so the shared RerunBridgeModule/vis_module
    # picks these up the same way every other dimOS module does — no ad hoc rerun
    # SDK calls here, matching the existing mid360/mid360-fastlio-voxels pattern.
    vis_lidar: Out[PointCloud2]
    vis_realsense_aligned: Out[PointCloud2]
    vis_stereo_aligned: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._latest: dict[str, PointCloud2] = {}
        self._running = False

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.lidar.subscribe(lambda m: self._store("lidar", m))))
        self.register_disposable(
            Disposable(self.realsense_raw.subscribe(lambda m: self._store("realsense_raw", m)))
        )
        self.register_disposable(Disposable(self.stereo.subscribe(lambda m: self._store("stereo", m))))
        self._running = True
        self.spawn(self._benchmark_loop())

    @rpc
    def stop(self) -> None:
        self._running = False
        super().stop()

    def _store(self, name: str, msg: PointCloud2) -> None:
        with self._lock:
            self._latest[name] = msg

    async def _benchmark_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.config.period_s)
            try:
                self._evaluate()
            except Exception:
                logger.exception("LidarStereoBenchmark: evaluation failed")

    def _score(self, gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> dict[str, float]:
        cfg = self.config
        return compute_score(pred_xyz, gt_xyz, cfg.fscore_threshold_m, cfg.fscore_threshold_loose_m, cfg.voxel_size_m)

    def _evaluate(self) -> None:
        cfg = self.config
        with self._lock:
            lidar_msg = self._latest.get("lidar")
            rs_msg = self._latest.get("realsense_raw")
            stereo_msg = self._latest.get("stereo")

        if lidar_msg is None or rs_msg is None:
            logger.info("LidarStereoBenchmark: waiting for lidar + realsense_raw frames...")
            return

        # Lidar is the reference frame — nothing to transform. Everything else is
        # ICP-aligned onto it (see module docstring: no calibrated extrinsic exists).
        # Cropped to a forward cone: the D435i only ever sees a narrow slice of the
        # lidar's full 360-degree sphere, so scoring/aligning against the whole thing
        # both misrepresents completeness/IoU and degrades ICP's correspondence search.
        lidar_xyz = crop_forward_cone(
            drop_near_field(lidar_msg.points_f32(), cfg.min_lidar_range_m),
            cfg.forward_cone_half_angle_deg,
        )
        rs_raw_xyz = rs_msg.points_f32()
        if len(lidar_xyz) < cfg.min_points or len(rs_raw_xyz) < cfg.min_points:
            logger.info("LidarStereoBenchmark: not enough points yet (lidar=%d realsense=%d)",
                        len(lidar_xyz), len(rs_raw_xyz))
            return

        lines = [f"lidar (ref): n={len(lidar_xyz)}"]
        self._publish_vis(self.vis_lidar, lidar_xyz)
        scores: dict[str, dict[str, float]] = {}

        # Each candidate is isolated in its own try/except — an ICP failure on one
        # (e.g. realsense) must not silently swallow the other (e.g. stereo) and
        # block its publish/log. Previously they shared one un-guarded code path,
        # so a raised exception partway through realsense's alignment meant stereo
        # (and its vis_stereo_aligned publish) never ran at all, every cycle.
        rs_xyz = np.empty((0, 3), dtype=np.float32)
        try:
            # RealSense raw cloud is in optical convention (Z=depth, Y=down) — a known,
            # fixed rotation away from the lidar's body frame (Z=up) — plus an unknown heading.
            rs_icp_T = best_yaw_icp(
                rs_raw_xyz, lidar_xyz, cfg.icp_max_corr_dist_m,
                base_rotation=R_OPT_TO_LINK.astype(np.float64), yaw_steps=cfg.yaw_steps,
                base_translation=ROUGH_CAM_OFFSET_IN_LIDAR_FRAME,
            )
            rs_xyz = apply_matrix(rs_raw_xyz, rs_icp_T)
            rs_score = self._score(lidar_xyz, rs_xyz)
            scores["realsense (raw)"] = rs_score
            lines.append(format_score(
                "realsense (raw, ICP-aligned)", rs_score,
                cfg.fscore_threshold_m, cfg.fscore_threshold_loose_m, cfg.voxel_size_m,
            ))
            self._publish_vis(self.vis_realsense_aligned, rs_xyz)
        except Exception:
            logger.exception("LidarStereoBenchmark: realsense alignment/scoring failed")

        if stereo_msg is not None:
            stereo_raw_xyz = stereo_msg.points_f32()
            if len(stereo_raw_xyz) >= cfg.min_points:
                try:
                    # Already body-frame-oriented (Z=up) internally — only heading is unknown.
                    stereo_icp_T = best_yaw_icp(
                        stereo_raw_xyz, lidar_xyz, cfg.icp_max_corr_dist_m,
                        base_rotation=np.eye(3), yaw_steps=cfg.yaw_steps,
                        base_translation=ROUGH_CAM_OFFSET_IN_LIDAR_FRAME,
                    )
                    stereo_xyz = apply_matrix(stereo_raw_xyz, stereo_icp_T)
                    stereo_score = self._score(lidar_xyz, stereo_xyz)
                    scores["stereo (ours)"] = stereo_score
                    lines.append(format_score(
                        "stereo (ours, ICP-aligned)", stereo_score,
                        cfg.fscore_threshold_m, cfg.fscore_threshold_loose_m, cfg.voxel_size_m,
                    ))
                    self._publish_vis(self.vis_stereo_aligned, stereo_xyz)
                except Exception:
                    logger.exception("LidarStereoBenchmark: stereo alignment/scoring failed")

        origin = np.zeros(3, dtype=np.float32)
        lines.append(f"range-binned density {cfg.range_bins_m}: lidar={range_binned_density(lidar_xyz, origin, cfg.range_bins_m)} "
                     f"realsense={range_binned_density(rs_xyz, origin, cfg.range_bins_m)}")

        logger.info("LidarStereoBenchmark:\n  " + "\n  ".join(lines))
        if scores:
            logger.info(format_verdict_report(len(lidar_xyz), scores))

    def _publish_vis(self, stream: Out[PointCloud2], xyz: np.ndarray, cap: int = 40_000) -> None:
        if len(xyz) == 0:
            return
        if len(xyz) > cap:
            xyz = xyz[np.random.choice(len(xyz), cap, replace=False)]
        stream.publish(PointCloud2.from_numpy(xyz, frame_id="bench", timestamp=time.time()))

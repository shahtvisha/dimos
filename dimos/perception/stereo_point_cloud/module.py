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

"""D435i depth → per-frame voxel cloud + persistent log-odds global map.

Frames & conventions (matches the rest of dimos — rerun grid/floor, occupancy
algos, cmu_nav all assume this):

- ``world_frame``: gravity-aligned, **z = 0 at the floor**. The camera starts
  at (0, 0, cam_height).
- Pose: roll/pitch from the D435i IMU via Madgwick (yaw-stripped — gyro yaw
  drifts); translation and yaw are fixed at zero for this initial test pass
  (stationary camera assumption). Add wheel odometry input later.
- Floor points are KEPT in ``global_map``: CostMapper's ``height_cost`` needs
  ground returns to mark free/traversable space (deleted floor = unknown
  cells, not free cells).
- Occupancy is a log-odds voxel map with PROJECTIVE clearing: every stored
  voxel in the camera frustum is tested against the measured depth at its
  pixel every frame. 2 hits to appear (~130 ms at 15 fps), 3 misses to clear
  (~200 ms) — moved objects free their space almost immediately, single noisy
  frames cannot erase real obstacles (e.g. a thin floor mat), occlusion is
  handled correctly, and pixels with no depth return contribute no evidence.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.stereo_point_cloud.utils import (
    _R_OPT_TO_LINK,
    LogOddsVoxelMap,
    _gradient_mask,
    _pack,
    _projective_miss_keys,
    _unpack_centers,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_DEPTH_MM_THRESHOLD = 100


class Config(ModuleConfig):
    min_depth: float          = 0.1
    max_depth: float          = 8.0
    # points beyond this range are visualized (frame_cloud) but NOT written to
    # the global map — D435i depth sigma at 5 m is ~9 cm, useless at 2 cm voxels
    mapping_range: float      = 5.0
    gradient_threshold: float = 0.30
    # subsample depth image by this pixel stride before backprojection
    # (2 → 4x less compute; at 2 cm voxels nothing is lost)
    pixel_stride: int         = 2
    vox_size: float           = 0.020
    global_vox_size: float    = 0.020
    publish_every: int        = 2
    world_frame: str          = "world"
    cam_height_prior: float   = 1.0
    # projective clearing: a stored voxel is a 'miss' when the measured depth
    # at its pixel is behind it by max(clear_min_margin, 3 * sigma_z(range))
    clear_min_margin: float   = 0.06
    # rolling local map: voxels farther than this (xy) from the robot are
    # dropped — bounds memory AND odometry-drift error
    map_radius: float         = 12.0
    prune_every: int          = 150
    max_global_pts: int       = 800_000


class StereoPointCloud(Module):
    """D435i depth → frame_cloud + global_map (log-odds, floor at z = 0)."""

    config: Config

    depth_image:       In[Image]
    depth_camera_info: In[CameraInfo]

    frame_cloud: Out[PointCloud2]
    global_map:  Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock                       = threading.Lock()
        self._latest_info: CameraInfo | None = None
        self._voxmap                     = LogOddsVoxelMap()
        self._map_lock                   = threading.Lock()
        self._uv_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._frame                      = 0
        self._warned_no_intrinsics       = False

    # ------------------------------------------------------------------ setup

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.depth_camera_info.subscribe(self._on_info)))
        self.register_disposable(Disposable(self.depth_image.subscribe(self._on_depth)))

    # -------------------------------------------------------------- callbacks

    def _on_info(self, info: CameraInfo) -> None:
        with self._lock:
            self._latest_info = info

    def _uv_grid(self, H: int, W: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
        """Cached full-resolution pixel coordinate grids sampled with stride."""
        key = (H, W, stride)
        grids = self._uv_cache.get(key)
        if grids is None:
            uu, vv = np.meshgrid(
                np.arange(0, W, stride, dtype=np.float32),
                np.arange(0, H, stride, dtype=np.float32),
            )
            grids = (uu, vv)
            self._uv_cache[key] = grids
        return grids

    # ------------------------------------------------------------ depth frame

    def _on_depth(self, img: Image) -> None:  # noqa: C901
        with self._lock:
            info = self._latest_info

        depth = img.data
        if hasattr(depth, "get"):
            depth = depth.get()
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        H_full, W_full = depth.shape
        stride = max(1, int(self.config.pixel_stride))
        if stride > 1:
            depth = depth[::stride, ::stride]

        depth = depth.astype(np.float32)
        valid_d = depth[depth > 0]
        if len(valid_d) == 0:
            return
        is_mm = np.median(valid_d) > _DEPTH_MM_THRESHOLD
        if is_mm:
            depth /= 1000.0
        invalid = (depth <= 0) | (depth < self.config.min_depth) | (depth > self.config.max_depth)
        depth[invalid] = np.nan

        # Intrinsics — scaled if the image resolution differs from the
        # CameraInfo resolution (e.g. an upstream decimation filter).
        if info is not None:
            K = info.get_K_matrix()
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            info_w = int(getattr(info, "width", 0) or 0)
            info_h = int(getattr(info, "height", 0) or 0)
            if info_w and info_w != W_full:
                s = W_full / info_w
                fx, cx = fx * s, cx * s
            if info_h and info_h != H_full:
                s = H_full / info_h
                fy, cy = fy * s, cy * s
        else:
            if not self._warned_no_intrinsics:
                logger.warning(
                    "StereoPointCloud: no camera intrinsics — falling back to rough "
                    "guess, check depth_camera_info is connected"
                )
                self._warned_no_intrinsics = True
            fx = fy = float(max(H_full, W_full)) / 2.0
            cx, cy  = W_full / 2.0, H_full / 2.0

        stable = _gradient_mask(depth, self.config.gradient_threshold)
        valid  = np.isfinite(depth) & stable
        if not valid.any():
            return

        uu, vv  = self._uv_grid(H_full, W_full, stride)
        dd      = depth[valid]
        xyz_opt = np.column_stack([
            (uu[valid] - cx) * dd / fx,
            (vv[valid] - cy) * dd / fy,
            dd,
        ]).astype(np.float32)

        # Level mount assumed — no tilt correction.
        xyz_cam = (xyz_opt @ _R_OPT_TO_LINK.T).astype(np.float32)

        h = float(self.config.cam_height_prior)

        cam_pos = np.array([0.0, 0.0, h + 1.0], dtype=np.float32)

        # World frame: z = 0 at the floor (dimos-wide convention).
        xyz_world = (xyz_cam + cam_pos).astype(np.float32)

        # ---- per-frame voxel cloud (floor KEPT — height_cost needs it) ----
        vk       = np.floor(xyz_world / self.config.vox_size).astype(np.int32)
        _, first = np.unique(_pack(vk), return_index=True)
        xyz_vox  = xyz_world[first]
        self.frame_cloud.publish(
            PointCloud2.from_numpy(xyz_vox, frame_id=self.config.world_frame, timestamp=img.ts)
        )

        # ---- global map: log-odds hits + raycast misses, world frame ----
        rel  = xyz_vox - cam_pos
        near = np.einsum("ij,ij->i", rel, rel) <= self.config.mapping_range ** 2
        map_pts = xyz_vox[near]

        self._frame += 1
        gvox = self.config.global_vox_size
        hit_keys = (
            np.unique(_pack(np.floor(map_pts / gvox).astype(np.int32)))
            if len(map_pts)
            else np.empty(0, dtype=np.int64)
        )
        with self._map_lock:
            # Projective clearing: test every stored voxel in the frustum
            # against the measured depth at its pixel. ``depth`` here is the
            # strided image in meters with NaN for invalid pixels — dropout
            # pixels therefore contribute no clearing evidence.
            free_keys = _projective_miss_keys(
                self._voxmap.keys,
                gvox,
                depth,
                fx,
                fy,
                cx,
                cy,
                stride,
                np.eye(3, dtype=np.float32),
                cam_pos,
                self.config.mapping_range,
                min_margin=self.config.clear_min_margin,
            )
            self._voxmap.update(hit_keys, free_keys)

        with self._map_lock:
            if self._frame % self.config.prune_every == 0:
                self._voxmap.prune(
                    cam_pos,
                    self.config.map_radius,
                    self.config.global_vox_size,
                    self.config.max_global_pts,
                )
            occ = (
                self._voxmap.occupied_keys()
                if self._frame % self.config.publish_every == 0
                else None
            )

        if occ is not None and len(occ):
            centers = _unpack_centers(occ, self.config.global_vox_size)
            self.global_map.publish(
                PointCloud2.from_numpy(
                    centers, frame_id=self.config.world_frame, timestamp=img.ts
                )
            )

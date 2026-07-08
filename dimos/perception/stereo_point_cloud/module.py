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
    _FloorCalibrator,
    _gradient_mask,
    _pack,
    _projective_miss_keys,
    _unpack_centers,
)
from dimos.perception.stereo_point_cloud.vio import MadgwickFilter
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
    madgwick_beta: float      = 0.033
    # camera mount height prior (m). Used until floor calibration converges,
    # and as a sanity check afterwards. Set to the actual mount height.
    cam_height_prior: float   = 1.0
    floor_sanity_diff: float  = 0.15
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
        self._floor_calib                = _FloorCalibrator()
        self._madgwick: MadgwickFilter | None = None
        self._imu_lock                   = threading.Lock()
        self._last_accel                 = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        self._R_imu_to_link: np.ndarray  = np.eye(3, dtype=np.float32)
        self._motion_sensor              = None
        self._madgwick_seeded            = False
        self._voxmap                     = LogOddsVoxelMap()
        self._map_lock                   = threading.Lock()
        self._uv_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._frame                      = 0
        self._warned_no_intrinsics       = False
        self._warned_floor_sanity        = False
        self._logged_calibrated          = False

    # ------------------------------------------------------------------ setup

    @rpc
    def start(self) -> None:
        super().start()
        self._madgwick = MadgwickFilter(beta=self.config.madgwick_beta)
        if not self._init_imu():
            logger.warning(
                "StereoPointCloud: no IMU — roll/pitch assumed level (fine for a level mount)"
            )
        self.register_disposable(Disposable(self.depth_camera_info.subscribe(self._on_info)))
        self.register_disposable(Disposable(self.depth_image.subscribe(self._on_depth)))

    @rpc
    def stop(self) -> None:
        if self._motion_sensor is not None:
            try:
                self._motion_sensor.stop()
                self._motion_sensor.close()
            except Exception:
                pass
            self._motion_sensor = None
        super().stop()

    def _init_imu(self) -> bool:
        try:
            import pyrealsense2 as rs
        except ImportError:
            return False
        ctx     = rs.context()
        devices = ctx.query_devices()
        if not devices:
            return False
        device = devices[0]
        depth_profile = None
        for sensor in device.query_sensors():
            if sensor.is_motion_sensor():
                continue
            for p in sensor.get_stream_profiles():
                if p.stream_type() == rs.stream.depth:
                    depth_profile = p
                    break
            if depth_profile is not None:
                break
        for sensor in device.query_sensors():
            if not sensor.is_motion_sensor():
                continue
            all_profiles   = sensor.get_stream_profiles()
            gyro_profiles  = [p for p in all_profiles if p.stream_type() == rs.stream.gyro]
            accel_profiles = [p for p in all_profiles if p.stream_type() == rs.stream.accel]
            if not gyro_profiles or not accel_profiles:
                break
            gyro_p  = max(gyro_profiles,  key=lambda p: p.fps())
            accel_p = max(accel_profiles, key=lambda p: p.fps())
            if depth_profile is not None:
                ext                 = accel_p.get_extrinsics_to(depth_profile)
                R_imu_to_depth      = np.array(ext.rotation, dtype=np.float32).reshape(3, 3)
                self._R_imu_to_link = _R_OPT_TO_LINK @ R_imu_to_depth
            sensor.open([gyro_p, accel_p])
            sensor.start(self._on_motion)
            self._motion_sensor = sensor
            logger.info(f"StereoPointCloud: IMU — gyro@{gyro_p.fps()}Hz accel@{accel_p.fps()}Hz")
            return True
        return False

    # -------------------------------------------------------------- callbacks

    def _on_motion(self, frame: Any) -> None:
        try:
            import pyrealsense2 as rs
            st = frame.get_profile().stream_type()
            if st == rs.stream.gyro:
                g        = frame.as_motion_frame().get_motion_data()
                gyro_lnk = self._R_imu_to_link @ np.array([g.x, g.y, g.z], dtype=np.float32)
                ts_s     = frame.get_timestamp() / 1000.0
                with self._imu_lock:
                    if self._madgwick is not None and self._madgwick_seeded:
                        self._madgwick.update(gyro_lnk, self._last_accel, ts_s)
            elif st == rs.stream.accel:
                a = frame.as_motion_frame().get_motion_data()
                with self._imu_lock:
                    self._last_accel = self._R_imu_to_link @ np.array(
                        [a.x, a.y, a.z], dtype=np.float32
                    )
                    if not self._madgwick_seeded and self._madgwick is not None:
                        # seed roll/pitch from gravity so the filter is correct
                        # immediately — beta=0.033 converges too slowly from
                        # identity for floor calibration at frames 30-60
                        self._madgwick.init_from_accel(self._last_accel)
                        self._madgwick_seeded = True
        except Exception:
            pass

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

    def _cam_height(self) -> float:
        """Calibrated camera height above floor, falling back to the prior."""
        if self._floor_calib.ready:
            h = float(self._floor_calib.cam_height)  # type: ignore[arg-type]
            if not self._logged_calibrated:
                self._logged_calibrated = True
                logger.info(f"StereoPointCloud: using calibrated camera height {h:.3f} m")
            if (
                abs(h - self.config.cam_height_prior) > self.config.floor_sanity_diff
                and not self._warned_floor_sanity
            ):
                self._warned_floor_sanity = True
                logger.warning(
                    f"StereoPointCloud: calibrated height {h:.3f} m differs from "
                    f"prior {self.config.cam_height_prior:.3f} m by more than "
                    f"{self.config.floor_sanity_diff:.2f} m — check the mount / "
                    f"cam_height_prior config"
                )
            return h
        return float(self.config.cam_height_prior)

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

        # Orientation: roll/pitch from Madgwick (gravity); yaw fixed at 0.
        with self._imu_lock:
            R_tilt = (
                self._madgwick.R_rollpitch_only()
                if (self._madgwick is not None and self._madgwick_seeded)
                else np.eye(3, dtype=np.float32)
            )

        xyz_cam  = xyz_opt @ _R_OPT_TO_LINK.T
        xyz_grav = (xyz_cam @ R_tilt.T).astype(np.float32)  # gravity-aligned, camera-centered

        # Floor calibration runs in the gravity-aligned frame — mount pitch
        # does not affect it.
        self._floor_calib.update(xyz_grav)
        h = self._cam_height()

        cam_pos = np.array([0.0, 0.0, h + 1.0], dtype=np.float32)

        # World frame: z = 0 at the floor (dimos-wide convention).
        xyz_world = (xyz_grav + cam_pos).astype(np.float32)

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
                R_tilt.astype(np.float32),
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

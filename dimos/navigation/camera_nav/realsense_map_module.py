"""RealSense D435i depth mapping as a dimos Module.

Wraps the gradient-filtered voxel map + ray-cast ghost-clearing global map
into the dimos Module/stream system so it can be wired into any Blueprint.

Two Out[PointCloud2] streams published via LCM transport (PointCloud2 has
lcm_encode() so the transport layer selects LCM automatically):

  frame_cloud   per-frame voxel map — refreshed every frame
  global_map    persistent accumulated map with ray-cast ghost clearing

Usage
  from dimos.navigation.camera_nav.realsense_map_module import RealSenseMapModule
  from dimos.core.coordination.module_coordinator import ModuleCoordinator

  ModuleCoordinator.build(RealSenseMapModule.blueprint()).loop()
"""
from __future__ import annotations

import threading
import time

import numpy as np
from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

from dimos.navigation.camera_nav.realsense_depth_map import (
    DepthBackprojector,
    GradientStabilityFilter,
    RealSenseDepthSource,
    _FloorCalibrator,
    _VOX_SIZE,
    _pack,
    _raycast_free_keys,
)


class RealSenseMapConfig(ModuleConfig):
    width:  int   = 848
    height: int   = 480
    fps:    int   = 15
    frame_id: str = "world"


class RealSenseMapModule(Module):
    """RealSense D435i → frame_cloud + global_map as dimos PointCloud2 streams."""

    config: RealSenseMapConfig

    frame_cloud: Out[PointCloud2]
    global_map:  Out[PointCloud2]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._running    = False
        self._thread:    threading.Thread | None = None
        self._src:       RealSenseDepthSource | None = None
        self._acc_pts    = np.empty((0, 3), dtype=np.float32)
        self._map_ready  = False
        self._floor_z    = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        self._src     = RealSenseDepthSource(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
        )
        self._bp      = DepthBackprojector()
        self._grad    = GradientStabilityFilter()
        self._floor   = _FloorCalibrator()
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            ts  = time.monotonic()
            pkt = self._src.read(ts)

            stable = self._grad.compute(pkt.depth)
            xyz, _ = self._bp.project(pkt, stable)
            if not len(xyz):
                continue

            cam_z   = float(pkt.pose_t[2])
            xyz_cam = xyz @ pkt.pose_R
            self._floor.update(xyz_cam)

            keep = (
                xyz_cam[:, 2] > (self._floor.floor_z + 0.03)
                if self._floor.ready
                else (xyz_cam[:, 2] >= -1.4) & (xyz_cam[:, 2] <= 1.5)
            )
            xyz_kept = xyz[keep]
            if not len(xyz_kept):
                continue

            # Per-frame voxel map
            vk      = np.floor(xyz_kept / _VOX_SIZE).astype(np.int32)
            _, ui   = np.unique(_pack(vk), return_index=True)
            xyz_vox = xyz_kept[ui]
            self.frame_cloud.publish(
                PointCloud2.from_numpy(xyz_vox, frame_id=self.config.frame_id, timestamp=ts)
            )

            # Accumulated global map — world frame for CostMapper compatibility.
            # XY positions must be world-stable so the occupancy grid is coherent.
            # Voxel size is 4 cm (2× per-frame) to absorb ±2 cm ICP translation
            # noise — keeps the same surface in the same grid cell across frames.
            # Ghost clearing still runs in camera-relative space for ray-cast math.
            _GVOX = _VOX_SIZE * 2   # 4 cm accumulation voxel
            if self._floor.ready and self._src.pose_locked:
                if not self._map_ready:
                    self._floor_z   = cam_z + self._floor.floor_z
                    self._map_ready = True

                # Floor filter in world frame; xyz_vox already world-frame
                xyz_map = xyz_vox[xyz_vox[:, 2] > self._floor_z + 0.04]

                if len(self._acc_pts) and len(xyz_map):
                    # Ray-cast in camera-relative space
                    xyz_map_rel = xyz_map - pkt.pose_t
                    free = _raycast_free_keys(xyz_map_rel, _GVOX)
                    if len(free):
                        acc_rel = self._acc_pts - pkt.pose_t
                        keys    = _pack(np.floor(acc_rel / _GVOX).astype(np.int32))
                        self._acc_pts = self._acc_pts[~np.isin(keys, free)]

                if len(xyz_map):
                    self._acc_pts = (
                        np.vstack([self._acc_pts, xyz_map])
                        if len(self._acc_pts) else xyz_map.copy()
                    )
                    _, ui         = np.unique(
                        _pack(np.floor(self._acc_pts / _GVOX).astype(np.int32)),
                        return_index=True,
                    )
                    self._acc_pts = self._acc_pts[ui]
                    self.global_map.publish(
                        PointCloud2.from_numpy(self._acc_pts, frame_id=self.config.frame_id, timestamp=ts)
                    )

            if len(xyz_cam[keep]) >= 50:
                self._src.odom.update(xyz_cam[keep], pkt.pose_R)

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._src:
            self._src.stop()
        super().stop()

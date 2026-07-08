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

"""Map-building utilities: voxel packing, gradient mask, floor calibration,
raycast clearing, and a log-odds voxel occupancy map."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_VOFF  = np.int64(100_000)
_VMASK = np.int64(0x3FFFF)

# Optical frame (X=right, Y=down, Z=depth) → camera_link (X=fwd, Y=left, Z=up)
_R_OPT_TO_LINK = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)

# D435i axial noise model: sigma_z ≈ 0.0035 * z^2  (848x480, tuned unit)
_DEPTH_SIGMA_COEFF = 0.0035


def _pack(vkeys: np.ndarray) -> np.ndarray:
    v = (vkeys.astype(np.int64) + _VOFF) & _VMASK
    return (v[:, 0] << np.int64(36)) | (v[:, 1] << np.int64(18)) | v[:, 2]


def _unpack(keys: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_pack` — integer voxel indices (N, 3)."""
    k = keys.astype(np.int64)
    v0 = (k >> np.int64(36)) & _VMASK
    v1 = (k >> np.int64(18)) & _VMASK
    v2 = k & _VMASK
    return np.stack([v0, v1, v2], axis=1) - _VOFF


def _unpack_centers(keys: np.ndarray, vox_size: float) -> np.ndarray:
    """Voxel center coordinates (N, 3) float32 for packed keys."""
    return ((_unpack(keys).astype(np.float32) + 0.5) * vox_size).astype(np.float32)


def _gradient_mask(depth: np.ndarray, threshold: float) -> np.ndarray:
    valid   = np.isfinite(depth)
    depth_f = np.where(valid, depth, 0.0).astype(np.float64)
    grad    = np.hypot(sobel(depth_f, axis=1), sobel(depth_f, axis=0))
    return valid & (grad < threshold)


def _projective_miss_keys(
    keys: np.ndarray,
    vox_size: float,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int,
    R_link_to_world: np.ndarray,
    cam_pos: np.ndarray,
    max_range: float,
    min_margin: float = 0.06,
) -> np.ndarray:
    """Stored voxel keys contradicted by the current depth image (misses).

    Projective (depth-image-based) clearing: every stored voxel inside the
    camera frustum is projected into the current depth image; if the measured
    depth at its pixel is clearly BEHIND the voxel (by a range-dependent
    margin ``max(min_margin, 3 * sigma_z)``), the voxel is free evidence.

    Compared to sampled raycasting this tests *every* in-view voxel *every*
    frame (moved objects clear in ~3 frames), handles occlusion correctly
    (measured depth closer than the voxel → no evidence), never erodes
    observed surfaces (measured ≈ voxel depth → no miss, even at grazing
    incidence), and is conservative on dropout pixels (NaN → no evidence).

    Args:
        keys: packed int64 voxel keys currently stored in the map
        depth: strided depth image in meters with NaN for invalid pixels
        fx/fy/cx/cy: FULL-RESOLUTION intrinsics; ``stride`` maps to the
            strided image grid
        R_link_to_world: camera_link → world rotation used for this frame
        cam_pos: camera position in the world/map frame
    """
    if len(keys) == 0:
        return np.empty(0, dtype=np.int64)
    centers = _unpack_centers(keys, vox_size)
    rel = centers - np.asarray(cam_pos, dtype=np.float32)
    inr = np.einsum("ij,ij->i", rel, rel) <= np.float32(max_range * max_range)
    if not inr.any():
        return np.empty(0, dtype=np.int64)
    keys_r = keys[inr]
    cam = rel[inr] @ R_link_to_world.astype(np.float32)   # world → camera_link
    opt = cam @ _R_OPT_TO_LINK                            # camera_link → optical
    z = opt[:, 2]
    ok = z > np.float32(0.10)
    if not ok.any():
        return np.empty(0, dtype=np.int64)
    keys_r, opt, z = keys_r[ok], opt[ok], z[ok]
    H, W = depth.shape
    u = (fx * opt[:, 0] / z + cx) / float(stride)
    v = (fy * opt[:, 1] / z + cy) / float(stride)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    inb = (u0 >= 0) & (u0 < W - 1) & (v0 >= 0) & (v0 < H - 1)
    if not inb.any():
        return np.empty(0, dtype=np.int64)
    keys_r, z, u0, v0 = keys_r[inb], z[inb], u0[inb], v0[inb]
    # Closest measurement in the 2x2 pixel neighborhood: a voxel is only a
    # miss if even the NEAREST nearby surface is clearly behind it. This
    # prevents single boundary pixels (e.g. at a thin mat's edge, where
    # rounding can land on the floor behind) from falsely clearing thin
    # obstacles. fmin ignores NaN; all-NaN neighborhoods yield NaN → no miss.
    d_meas = np.fmin(
        np.fmin(depth[v0, u0], depth[v0, u0 + 1]),
        np.fmin(depth[v0 + 1, u0], depth[v0 + 1, u0 + 1]),
    )
    sigma = _DEPTH_SIGMA_COEFF * z * z
    margin = np.maximum(np.float32(min_margin), 3.0 * sigma)
    miss = ~np.isnan(d_meas) & (d_meas > z + margin)
    return keys_r[miss]


class _FloorCalibrator:
    """Floor height estimator in a GRAVITY-ALIGNED camera-centered frame (Z=up).

    Feed :meth:`update` with ``xyz_grav = xyz_cam @ R_rollpitch.T`` — never raw
    camera_link points. Gravity alignment makes the floor a horizontal plane,
    so its 2 cm z-histogram is sharp and the mode is meaningful regardless of
    mount pitch.

    Bin selection: the dominant bin within the lowest significant cluster —
    robust both to phantom sub-floor points (hole-fill/noise artifacts below
    the dominant bin never reach FLOOR_MIN_PTS) and to furniture surfaces
    (well above the lowest cluster).
    """

    SKIP_FRAMES      = 30
    CALIB_FRAMES     = 30
    BIN_M            = 0.02
    MIN_CAM_HEIGHT   = 0.20   # floor must be ≥ 0.2 m below camera
    MAX_CAM_HEIGHT   = 3.0
    FLOOR_MAX_DIST_M = 4.0
    FLOOR_MIN_PTS    = 200
    CLUSTER_BINS     = 3      # mode search window above lowest significant bin (6 cm)

    def __init__(self) -> None:
        self._frame   = 0
        self._samples: list[float] = []
        self.floor_z:    float | None = None
        self.cam_height: float | None = None

    @property
    def ready(self) -> bool:
        return self.floor_z is not None

    def update(self, xyz_grav: np.ndarray) -> None:
        self._frame += 1
        if self.ready or self._frame <= self.SKIP_FRAMES:
            return
        if len(xyz_grav) == 0:
            return
        z    = xyz_grav[:, 2]
        dist = np.linalg.norm(xyz_grav[:, :2], axis=1)
        mask = (
            (z < -self.MIN_CAM_HEIGHT)
            & (z > -self.MAX_CAM_HEIGHT)
            & (dist < self.FLOOR_MAX_DIST_M)
        )
        if mask.sum() < self.FLOOR_MIN_PTS:
            return
        zf = z[mask]
        lo, hi = float(zf.min()), float(zf.max())
        if hi - lo < self.BIN_M:
            return
        bins          = np.arange(lo, hi + self.BIN_M, self.BIN_M)
        counts, edges = np.histogram(zf, bins=bins)
        significant   = np.where(counts >= self.FLOOR_MIN_PTS)[0]
        if not len(significant):
            return
        window = significant[significant <= significant[0] + self.CLUSTER_BINS]
        best   = int(window[np.argmax(counts[window])])
        self._samples.append(float(edges[best] + self.BIN_M / 2))
        if len(self._samples) >= self.CALIB_FRAMES:
            self.floor_z    = float(np.median(self._samples))
            self.cam_height = -self.floor_z
            spread = float(np.std(self._samples))
            logger.info(
                f"StereoPointCloud: floor calibrated — camera "
                f"{self.cam_height:.3f} m above floor (sample std {spread:.3f} m)"
            )


class LogOddsVoxelMap:
    """Sorted-array log-odds voxel occupancy map (packed int64 keys).

    Replaces the previous hard insert/delete accumulation:
    - a voxel needs 2 confirming hits before it is published (kills
      single-frame ghosts from depth noise),
    - a voxel needs ~3 misses (rays passing through it) before it is
      un-published (moved objects clear in ~0.5-1.5 s without map flicker).

    All operations are vectorized numpy on sorted arrays; typical cost is
    < 10 ms/frame for a few hundred thousand voxels.
    """

    L_HIT  = 2
    L_MISS = -1
    L_MIN  = -4
    L_MAX  = 6
    L_OCC  = 4   # published when log-odds ≥ this (2 hits with no misses)

    def __init__(self) -> None:
        self._keys = np.empty(0, dtype=np.int64)
        self._lo   = np.empty(0, dtype=np.int16)

    def __len__(self) -> int:
        return int(len(self._keys))

    @property
    def keys(self) -> np.ndarray:
        """All stored voxel keys (sorted). Used for projective miss testing."""
        return self._keys

    def _find(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rows of existing keys matching ``queries`` (sorted unique int64)."""
        n = len(self._keys)
        if n == 0 or len(queries) == 0:
            return np.empty(0, dtype=np.int64), np.zeros(len(queries), dtype=bool)
        idx = np.searchsorted(self._keys, queries)
        inb = idx < n
        found = np.zeros(len(queries), dtype=bool)
        found[inb] = self._keys[idx[inb]] == queries[inb]
        return idx[found], found

    def update(self, hit_keys: np.ndarray, free_keys: np.ndarray) -> None:
        """Integrate one frame of evidence. Inputs: unique sorted int64 keys."""
        # A voxel observed occupied this frame must not also be decremented.
        if len(free_keys) and len(hit_keys):
            free_keys = free_keys[~np.isin(free_keys, hit_keys, assume_unique=True)]

        # Misses: decrement EXISTING voxels only (free space is not stored).
        if len(free_keys):
            rows, _ = self._find(free_keys)
            if len(rows):
                self._lo[rows] = np.maximum(
                    self.L_MIN, self._lo[rows] + self.L_MISS
                ).astype(np.int16)

        # Hits: increment existing, insert new at L_HIT.
        if len(hit_keys):
            rows, found = self._find(hit_keys)
            if len(rows):
                self._lo[rows] = np.minimum(
                    self.L_MAX, self._lo[rows] + self.L_HIT
                ).astype(np.int16)
            new = hit_keys[~found]
            if len(new):
                keys = np.concatenate([self._keys, new])
                lo   = np.concatenate(
                    [self._lo, np.full(len(new), self.L_HIT, dtype=np.int16)]
                )
                order = np.argsort(keys, kind="stable")
                self._keys = keys[order]
                self._lo   = lo[order]

    def occupied_keys(self) -> np.ndarray:
        return self._keys[self._lo >= self.L_OCC]

    def prune(
        self,
        center: np.ndarray,
        radius: float,
        vox_size: float,
        max_voxels: int,
    ) -> None:
        """Drop fully-decayed voxels and everything outside the rolling window.

        The rolling window (``radius`` around the robot) bounds both memory and
        odometry-drift error — the published map is a local map by design.
        """
        if len(self._keys) == 0:
            return
        keep = self._lo > np.int16(self.L_MIN)
        centers = _unpack_centers(self._keys, vox_size)
        d2 = np.sum(
            (centers[:, :2] - np.asarray(center[:2], dtype=np.float32)) ** 2, axis=1
        )
        keep &= d2 <= np.float32(radius * radius)
        if keep.sum() > max_voxels:
            # keep the strongest evidence first
            order = np.argsort(self._lo, kind="stable")[::-1]
            allowed = np.zeros(len(self._keys), dtype=bool)
            allowed[order[:max_voxels]] = True
            keep &= allowed
        self._keys = self._keys[keep]
        self._lo   = self._lo[keep]

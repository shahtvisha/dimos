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

"""Orientation estimation: Madgwick AHRS (roll/pitch) for depth registration.

Yaw and translation come from FlowBase wheel odometry (see
``flowbase_odometry.py``) — gyro-integrated yaw drifts 1-3 deg/min with no
magnetometer, while wheel yaw is drift-free at standstill. The Madgwick filter
here is used ONLY for gravity alignment (roll/pitch) via
:meth:`MadgwickFilter.R_rollpitch_only`.

``PointCloudOdometry`` (frame-to-frame ICP) is kept for optional scan-to-map
refinement experiments but is no longer the source of translation: mean-shift
point-to-point ICP cannot observe translation along self-similar geometry
(corridors), which made it unusable on a moving base.
"""

from __future__ import annotations

import threading

import numpy as np
from scipy.spatial import cKDTree

_MAX_DT_S        = 0.1  # don't integrate more than 100ms at once — big gaps would blow up the filter
_ACCEL_MIN_NORM  = 0.5  # skip accel correction if reading is basically zero (noise or free-fall)
_MIN_ICP_INLIERS = 30   # need at least 30 matched point pairs to trust the ICP translation update


class MadgwickFilter:
    """Gyro + accel → orientation R (camera_link → world, Z-up). beta=0.033."""

    def __init__(self, beta: float = 0.033) -> None:
        self._q      = np.array([1., 0., 0., 0.], dtype=np.float64)
        self._beta   = beta
        self._t_prev: float | None = None
        self._initialized = False

    def init_from_accel(self, accel: np.ndarray) -> None:
        """One-shot roll/pitch seed from a gravity sample.

        beta=0.033 takes tens of seconds to converge from identity when the
        mount is tilted; seeding makes R gravity-correct immediately, so floor
        calibration (frames 30-60) runs on a valid orientation.

        Convention check: for identity q the filter's accel objective expects
        a_link_normalized == [0, 0, +1], so we seed with the quaternion that
        rotates the measured accel direction onto +Z.
        """
        a_n = float(np.linalg.norm(accel))
        if a_n < _ACCEL_MIN_NORM:
            return
        a = np.asarray(accel, dtype=np.float64) / a_n
        v = np.array([0.0, 0.0, 1.0])
        c = float(np.dot(a, v))
        axis = np.cross(a, v)
        s = float(np.linalg.norm(axis))
        if s < 1e-8:
            # already aligned (or anti-aligned: flip around X)
            self._q = (
                np.array([1.0, 0.0, 0.0, 0.0])
                if c > 0
                else np.array([0.0, 1.0, 0.0, 0.0])
            )
        else:
            axis /= s
            half = np.arctan2(s, c) / 2.0
            self._q = np.array(
                [np.cos(half), *(np.sin(half) * axis)], dtype=np.float64
            )
        self._q /= np.linalg.norm(self._q)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, gyro: np.ndarray, accel: np.ndarray, t: float) -> None:
        if self._t_prev is None:
            self._t_prev = t
            return
        dt = min(t - self._t_prev, _MAX_DT_S)
        self._t_prev = t
        if dt <= 0:
            return
        q0, q1, q2, q3 = self._q
        gx, gy, gz = gyro.astype(np.float64)
        a_n = np.linalg.norm(accel)
        if a_n > _ACCEL_MIN_NORM:
            ax, ay, az = accel.astype(np.float64) / a_n
            f1 = 2*(q1*q3 - q0*q2) - ax
            f2 = 2*(q0*q1 + q2*q3) - ay
            f3 = 2*(0.5 - q1**2 - q2**2) - az
            J  = np.array([
                [-2*q2,  2*q3, -2*q0, 2*q1],
                [ 2*q1,  2*q0,  2*q3, 2*q2],
                [    0, -4*q1, -4*q2,    0],
            ])
            grad = J.T @ np.array([f1, f2, f3])
            gn   = np.linalg.norm(grad)
            if gn > 1e-9:
                grad /= gn
        else:
            grad = np.zeros(4)
        q_dot = 0.5 * np.array([
            -q1*gx - q2*gy - q3*gz,
             q0*gx + q2*gz - q3*gy,
             q0*gy - q1*gz + q3*gx,
             q0*gz + q1*gy - q2*gx,
        ]) - self._beta * grad
        self._q = self._q + q_dot * dt
        self._q /= np.linalg.norm(self._q) + 1e-12

    @property
    def R(self) -> np.ndarray:
        q0, q1, q2, q3 = self._q
        return np.array([
            [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)  ],
            [2*(q1*q2+q0*q3),   1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)  ],
            [2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1),   1-2*(q1**2+q2**2)],
        ], dtype=np.float32)

    def R_rollpitch_only(self) -> np.ndarray:
        """Gravity tilt with the (drifting) gyro yaw removed.

        Yaw is supplied by wheel odometry instead; composing
        ``R_z(yaw_wheel) @ R_rollpitch_only()`` gives the full camera_link →
        world rotation.
        """
        R = self.R
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        c, s = np.cos(-yaw), np.sin(-yaw)
        Rz = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        return (Rz @ R).astype(np.float32)


class PointCloudOdometry:
    """Rotation-decoupled ICP for translation t.

    LEGACY / OPTIONAL: no longer used as the translation source (wheel odometry
    is — see module.py). Retained for scan-to-map refinement experiments.
    """

    ITERS    = 4
    MAX_DIST = 0.40
    MIN_PTS  = 50     # minimum points needed to run ICP at all
    N_SRC    = 1_500
    N_DST    = 8_000
    N_STORE  = 10_000

    def __init__(self) -> None:
        self._t          = np.zeros(3, dtype=np.float32)
        self._lock       = threading.Lock()
        self._prev_world: np.ndarray | None = None

    @property
    def t(self) -> np.ndarray:
        with self._lock:
            return self._t.copy()

    def update(self, xyz_cam: np.ndarray, R: np.ndarray) -> np.ndarray:
        pts_ga = (xyz_cam @ R.T).astype(np.float32)
        with self._lock:
            t_est = self._t.copy()
        if self._prev_world is None or len(pts_ga) < self.MIN_PTS:
            t_new = t_est
        else:
            n_src = min(self.N_SRC, len(pts_ga))
            n_dst = min(self.N_DST, len(self._prev_world))
            src   = pts_ga[np.random.choice(len(pts_ga), n_src, replace=False)]
            dst   = self._prev_world[np.random.choice(len(self._prev_world), n_dst, replace=False)]
            tree  = cKDTree(dst)
            for _ in range(self.ITERS):
                dists, idx = tree.query(src + t_est, k=1, workers=1)
                mask = dists < self.MAX_DIST
                if mask.sum() < _MIN_ICP_INLIERS:
                    break
                delta = dst[idx[mask]].mean(axis=0) - (src[mask] + t_est).mean(axis=0)
                t_est = (t_est + 0.7 * delta).astype(np.float32)
            t_new = t_est
        n_st  = min(self.N_STORE, len(pts_ga))
        idx_s = np.random.choice(len(pts_ga), n_st, replace=False)
        self._prev_world = (pts_ga[idx_s] + t_new).astype(np.float32)
        with self._lock:
            self._t = t_new
        return t_new

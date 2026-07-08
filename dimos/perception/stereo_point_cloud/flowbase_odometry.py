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

"""Publishes FlowBase wheel odometry as an ``Odometry`` stream.

The FlowBase Portal RPC endpoint already exposes ``get_odometry`` (used by
``dimos.hardware.drive_trains.flowbase.adapter.FlowBaseAdapter.read_odometry``)
but nothing publishes it. StereoPointCloud needs it: wheel odometry is the
correct translation + yaw source for depth registration (drift-free at
standstill, ~1-3% of distance in motion), complementary to the D435i IMU's
roll/pitch.

Frame convention note: the FlowBase write path negates vy/wz for the
platform's inverted-Y frame (see the adapter docstring), but the raw
``get_odometry`` result is in the platform's own frame. ``invert_y=True``
(default) applies the matching negation to y and theta on read.
VERIFY ON HARDWARE: drive the base +y (standard-frame left); published ``y``
must increase. If it decreases, set ``invert_y=False``.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Matches FlowBaseAdapter.DEFAULT_ADDRESS
DEFAULT_ADDRESS = "172.6.2.20:11323"

_ERROR_LOG_PERIOD_S = 5.0


class Config(ModuleConfig):
    address: str = DEFAULT_ADDRESS
    rate_hz: float = 30.0
    invert_y: bool = True
    frame_id: str = "odom"
    child_frame_id: str = "base_link"


class FlowBaseOdometry(Module):
    """Polls FlowBase ``get_odometry`` over Portal RPC and publishes it."""

    config: Config

    wheel_odom: Out[Odometry]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Any = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_error_log = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        import portal

        self._client = portal.Client(self.config.address)
        logger.info(f"FlowBaseOdometry: connected to {self.config.address}")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        super().stop()

    def _loop(self) -> None:
        period = 1.0 / max(1.0, float(self.config.rate_hz))
        while self._running:
            t0 = time.monotonic()
            try:
                self._poll_once()
            except Exception as e:
                now = time.monotonic()
                if now - self._last_error_log > _ERROR_LOG_PERIOD_S:
                    self._last_error_log = now
                    logger.error(f"FlowBaseOdometry: read failed: {e}")
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)

    def _poll_once(self) -> None:
        if self._client is None:
            return
        odom = self._client.get_odometry({}).result()
        if odom is None:
            return
        ts = time.time()  # stamp at receipt — same clock as camera frames
        x  = float(odom["translation"][0])
        y  = float(odom["translation"][1])
        th = float(odom["rotation"])
        if self.config.invert_y:
            y, th = -y, -th
        pose = Pose(
            position=[x, y, 0.0],
            orientation=[0.0, 0.0, math.sin(th / 2.0), math.cos(th / 2.0)],
        )
        self.wheel_odom.publish(
            Odometry(
                ts=ts,
                frame_id=self.config.frame_id,
                child_frame_id=self.config.child_frame_id,
                pose=pose,
            )
        )

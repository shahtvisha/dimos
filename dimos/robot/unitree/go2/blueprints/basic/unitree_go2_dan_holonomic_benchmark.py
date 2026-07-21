#!/usr/bin/env python3
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

"""Unitree Go2 Dan holonomic benchmark -- controller + benchmark in ONE launchable.

Same shape as ``unitree-go2-holonomic-benchmark``, composed with
``unitree-go2-dan-holonomic-controller`` instead. DanHolonomicTC has no live
speed input (fixed at launch via DAN_SPEED_M_S), so the Benchmarker's own
speed sweep is pinned to that single value here -- otherwise it would run the
same physical speed five times under five different labels. Sweep speeds by
re-launching with a different DAN_SPEED_M_S each time.

    DAN_SPEED_M_S=0.5 dimos run unitree-go2-dan-holonomic-benchmark
    # afterwards, score offline:
    python -m dimos.control.benchmarking.score data/benchmark/go2
"""

from __future__ import annotations

import os
from typing import Any

from dimos.control.benchmarking.benchmark import Benchmarker
from dimos.core.coordination.blueprints import TransportSpec, autoconnect
from dimos.core.stream import Transport
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_dan_holonomic_controller import (
    unitree_go2_dan_holonomic_controller,
)

_speed_m_s = os.environ.get("DAN_SPEED_M_S", "0.5")

_BENCHMARK_TRANSPORTS: dict[tuple[str, type], TransportSpec | Transport[Any]] = {
    ("odom", PoseStamped): LCMTransport("/go2/odom", PoseStamped),
}

unitree_go2_dan_holonomic_benchmark = (
    autoconnect(
        unitree_go2_dan_holonomic_controller,
        Benchmarker.blueprint(
            robot="go2",
            battery="all",
            speeds=_speed_m_s,
            gate_source="stream",
            out_dir=os.environ.get("DAN_OUT_DIR"),
        ),
    )
    .remappings([(Benchmarker, "cmd_vel", "go2_cmd_vel")])
    .transports(_BENCHMARK_TRANSPORTS)
    .global_config(obstacle_avoidance=False, n_workers=6)
)

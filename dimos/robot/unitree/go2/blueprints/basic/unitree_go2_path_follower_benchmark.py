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

"""Unitree Go2 P-controller benchmark -- controller + benchmark in ONE launchable.

Same shape as ``unitree-go2-holonomic-benchmark``: the Benchmarker imports
nothing from the controller and talks only over LCM, composed here with
``unitree-go2-path-follower-controller`` instead of the holonomic one -- same
battery, same speeds, same tolerances, only the controller differs.

    dimos run unitree-go2-path-follower-benchmark
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
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_path_follower_controller import (
    unitree_go2_path_follower_controller,
)

_BENCHMARK_TRANSPORTS: dict[tuple[str, type], TransportSpec | Transport[Any]] = {
    ("odom", PoseStamped): LCMTransport("/go2/odom", PoseStamped),
}

unitree_go2_path_follower_benchmark = (
    autoconnect(
        unitree_go2_path_follower_controller,
        Benchmarker.blueprint(
            robot="go2",
            battery="all",
            gate_source="stream",
            out_dir=os.environ.get("PF_OUT_DIR"),
        ),
    )
    .remappings([(Benchmarker, "cmd_vel", "go2_cmd_vel")])
    .transports(_BENCHMARK_TRANSPORTS)
    .global_config(obstacle_avoidance=False, n_workers=6)
)

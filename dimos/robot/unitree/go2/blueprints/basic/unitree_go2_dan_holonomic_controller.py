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

"""Unitree Go2 Dan holonomic controller -- a standalone, path-following control process.

Same benchmark contract as ``unitree-go2-holonomic-controller`` and
``unitree-go2-path-follower-controller``: no planner and no map here, this
blueprint is a controller you feed paths to from any source over the
transport. Unlike the other two, ``DanHolonomicTC`` is a plain Module (not a
ControlCoordinator task), so it's wired via autoconnect + MovementManager
(mux against manual teleop) instead of a ControlCoordinator task list --
mirroring how it's wired in ``unitree-go2-mls-htc``, minus the planner pieces.

``DanHolonomicTC`` has no live speed input (only a blueprint-time
``speed_m_s``), so a multi-speed sweep means restarting this process once per
speed rather than one continuous session -- set via DAN_SPEED_M_S.

Interface (pure LCM pub/sub, identical contract to the other two controllers):

    IN   path   (nav_msgs/Path)              -> dan_tc.path
    OUT  odom   (geometry_msgs/PoseStamped, /go2/odom)  -- the Go2 leg odom
    OUT  cmd_vel(geometry_msgs/Twist,        /cmd_vel)  -- aggregated command echo

Run (one of two processes; the benchmark is the other)::

    DAN_SPEED_M_S=0.5 dimos run unitree-go2-dan-holonomic-controller
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.std_msgs.Int8 import Int8
from dimos.navigation.dannav.holonomic_tc.module import DanHolonomicTC
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.keyboard_teleop import KeyboardTeleop
from dimos.core.transport import LCMTransport

_speed_m_s = float(os.environ.get("DAN_SPEED_M_S", 0.5))

unitree_go2_dan_holonomic_controller = (
    autoconnect(
        GO2Connection.blueprint(velocity_api=True).remappings(
            [
                (GO2Connection, "cmd_vel", "go2_cmd_vel"),
                (GO2Connection, "odom", "go2_odom"),
            ]
        ),
        DanHolonomicTC.blueprint(
            speed_m_s=_speed_m_s,
            goal_tolerance=0.20,
            orientation_tolerance=0.25,
        ).remappings([(DanHolonomicTC, "odom", "go2_odom")]),
        KeyboardTeleop.blueprint(publish_only_when_active=True).remappings(
            [(KeyboardTeleop, "cmd_vel", "tele_cmd_vel")]
        ),
        MovementManager.blueprint().remappings([(MovementManager, "cmd_vel", "go2_cmd_vel")]),
    )
    .transports(
        {
            ("go2_cmd_vel", Twist): LCMTransport("/go2/cmd_vel", Twist),
            ("go2_odom", PoseStamped): LCMTransport("/go2/odom", PoseStamped),
            ("path", Path): LCMTransport("/path", Path),
            ("gate", Int8): LCMTransport("/benchmark/gate", Int8),
        }
    )
    .global_config(obstacle_avoidance=False)
)

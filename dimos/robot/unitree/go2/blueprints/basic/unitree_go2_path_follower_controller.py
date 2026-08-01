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

"""Unitree Go2 P-controller — a standalone, path-following control process.

Same shape as ``unitree-go2-rpp-controller``, swapping the RPP follower for
``PathFollowerTask`` (fixed-lookahead ``PController`` + rotate-then-drive).
NO planner and NO map here: this blueprint is a controller you feed paths to
from any source over the transport, so it can run against the same benchmark
battery as the other controllers.

Interface (pure LCM pub/sub -- identical to the RPP controller):

    IN   path   (nav_msgs/Path)        -> coordinator.path  -> path_follower.set_path
    IN   speed  (std_msgs/Float32, m/s)-> coordinator.speed -> path_follower.set_speed
    OUT  odom   (geometry_msgs/PoseStamped, /go2/odom)  -- the Go2 leg odom
    OUT  cmd_vel(geometry_msgs/Twist,        /cmd_vel)  -- aggregated command echo

Run (one of two processes; the benchmark is the other)::

    dimos run unitree-go2-path-follower-controller
"""

from __future__ import annotations

from dimos.control.components import HardwareComponent, HardwareType, make_twist_base_joints
from dimos.control.coordinator import TaskConfig
from dimos.control.path_following_coordinator import PathFollowingCoordinator
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Float32 import Float32
from dimos.msgs.std_msgs.Int8 import Int8
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.keyboard_teleop import KeyboardTeleop

_go2_joints = make_twist_base_joints("go2")


unitree_go2_path_follower_controller = (
    autoconnect(
        # velocity_api=True: matches unitree-go2-holonomic-controller's GO2Connection config.
        GO2Connection.blueprint(velocity_api=True),
        PathFollowingCoordinator.blueprint(
            publish_joint_state=True,
            hardware=[
                HardwareComponent(
                    hardware_id="go2",
                    hardware_type=HardwareType.BASE,
                    joints=_go2_joints,
                    adapter_type="transport_lcm",
                ),
            ],
            tasks=[
                TaskConfig(
                    name="vel_go2",
                    type="velocity",
                    joint_names=_go2_joints,
                    priority=20,
                    params={"zero_on_timeout": False},
                ),
                TaskConfig(
                    name="path_follower",
                    type="path_follower",
                    joint_names=_go2_joints,
                    priority=10,
                    params={
                        "speed": 0.7,
                        "goal_tolerance": 0.20,
                        "orientation_tolerance": 0.25,
                        "forward_only": True,
                    },
                ),
            ],
        ),
        KeyboardTeleop.blueprint(publish_only_when_active=True),
    )
    .remappings(
        [
            (GO2Connection, "cmd_vel", "go2_cmd_vel"),
            (GO2Connection, "odom", "go2_odom"),
        ]
    )
    .transports(
        {
            ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
            ("twist_command", Twist): LCMTransport("/cmd_vel", Twist),
            ("go2_cmd_vel", Twist): LCMTransport("/go2/cmd_vel", Twist),
            ("go2_odom", PoseStamped): LCMTransport("/go2/odom", PoseStamped),
            ("path", Path): LCMTransport("/path", Path),
            ("speed", Float32): LCMTransport("/speed", Float32),
            ("gate", Int8): LCMTransport("/benchmark/gate", Int8),
            ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
            ("coordinator_joint_state", JointState): LCMTransport(
                "/coordinator/joint_state", JointState
            ),
        }
    )
    .global_config(obstacle_avoidance=False)
)

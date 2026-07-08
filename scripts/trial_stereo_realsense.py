"""RealSense stereo depth → StereoPointCloud → Rerun.

No robot required. Camera is treated as fixed at base_link origin.

Usage:
    python scripts/trial_stereo_realsense.py
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.perception.stereo_point_cloud import StereoPointCloud
from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos.visualization.rerun.bridge import RerunBridgeModule

trial_stereo = autoconnect(
    RealSenseCamera.blueprint(align_depth_to_color=False),
    StereoPointCloud.blueprint(world_frame="base_link"),
    RerunBridgeModule.blueprint(
        pubsubs=[LCM()],
        rerun_open="web",
    ),
)

if __name__ == "__main__":
    ModuleCoordinator.build(trial_stereo).loop()

"""Runs on the machine wired to both the Mid-360 and the D435i (same rig as
dimos/robot/assembly/mid360_realsense_30.py): per-frame lidar-vs-stereo benchmark.

Compares three point clouds every `period_s` (see BenchConfig):
  - lidar           Mid-360 raw frame (reference / "ground truth")
  - realsense_raw   RealSense SDK-deprojected cloud (no filtering, no odometry)
  - stereo          our StereoPointCloud output (filtered, ICP-aligned to lidar for scoring)

Metrics: Chamfer distance, accuracy/completeness/F-score (ETH3D convention),
voxel occupancy IoU, range-binned point density. Logged to console.

Rerun viewer: same pattern as mid360/mid360-fastlio-voxels (vis_module bridge, not
ad hoc rerun SDK calls), bound to 0.0.0.0 so it's reachable over the network without
an SSH tunnel — open http://<this-host>:9878 from another machine on the same LAN.

Usage:
    DIMOS_MID360_LIDAR_IP=192.168.1.155 python scripts/lidar_stereo_benchmark.py
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.hardware.sensors.lidar.livox.module import Mid360
from dimos.mapping.utils.cli.lidar_stereo_bench import LidarStereoBenchmark
from dimos.perception.stereo_point_cloud.filtered_realsense import FilteredRealSenseCamera
from dimos.perception.stereo_point_cloud.module import StereoPointCloud
from dimos.visualization.vis_module import vis_module

lidar_stereo_benchmark = (
    autoconnect(
        Mid360.blueprint(),
        FilteredRealSenseCamera.blueprint(
            enable_depth=True, enable_pointcloud=True, publish_color=False
        ).remappings([
            (FilteredRealSenseCamera, "pointcloud", "realsense_raw"),
        ]),
        StereoPointCloud.blueprint(world_frame="stereo_odom").remappings([
            (StereoPointCloud, "frame_cloud", "stereo"),
        ]),
        LidarStereoBenchmark.blueprint(),
        # rerun_open is this module's own field (set here, not via global_config —
        # vis_module() only defaults it from the process-wide global_config.rerun_open
        # at blueprint-construction time, which a plain `python script.py` never sets).
        # rerun_host, by contrast, IS read from the blueprint's shared global config
        # (RerunBridgeModule.host reads self.config.g.rerun_host) — set via global_config below.
        vis_module("rerun", rerun_config={"rerun_open": "web"}),
    )
    .global_config(n_workers=5, rerun_host="0.0.0.0")
)

if __name__ == "__main__":
    ModuleCoordinator.build(lidar_stereo_benchmark).loop()

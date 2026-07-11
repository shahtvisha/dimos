"""Runs on the machine wired to both the Mid-360 and the D435i: records lidar,
realsense_raw, and stereo point clouds to a memory2 db for OFFLINE comparison.

Why: dimos/mapping/utils/cli/lidar_stereo_bench.py compares all three live, which
means re-running the whole hardware pipeline (native lidar driver + camera + VIO)
for every small alignment/metric tweak. Record once here, then iterate on
scripts/lidar_stereo_compare_offline.py against the same captured data — no
hardware needed for that part.

Robot should be stationary. A few seconds of recording is enough — stop with
Ctrl-C once you see all three streams logged in the startup output.

Usage:
    DIMOS_MID360_LIDAR_IP=192.168.1.155 python scripts/lidar_stereo_record.py
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.stream import In
from dimos.hardware.sensors.lidar.livox.module import Mid360
from dimos.memory2.module import Recorder, RecorderConfig
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.stereo_point_cloud.filtered_realsense import FilteredRealSenseCamera
from dimos.perception.stereo_point_cloud.module import StereoPointCloud


class LidarStereoRecorderConfig(RecorderConfig):
    db_path: str = "lidar_stereo.db"


class LidarStereoRecorder(Recorder):
    config: LidarStereoRecorderConfig

    lidar: In[PointCloud2]
    realsense_raw: In[PointCloud2]
    stereo: In[PointCloud2]


lidar_stereo_record = (
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
        LidarStereoRecorder.blueprint(),
    )
    .global_config(n_workers=4)
)

if __name__ == "__main__":
    ModuleCoordinator.build(lidar_stereo_record).loop()

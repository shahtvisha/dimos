from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.perception.stereo_point_cloud import StereoPointCloud
from dimos.mapping.costmapper import CostMapper
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator

if __name__ == '__main__':
    pipeline = (
        autoconnect(
            RealSenseCamera.blueprint(enable_depth=True, enable_pointcloud=False),
            StereoPointCloud.blueprint(),
            CostMapper.blueprint(),
        )
        .global_config(n_workers=4)
    )
    ModuleCoordinator.build(pipeline, {}).start()

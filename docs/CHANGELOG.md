# Changelog

## [Unreleased]

Accuracy and performance passes. Entries marked **(output change)** change the
default sonar / camera images: re-check gains, noise and water parameters that
were tuned against real data.

### Changed

- **(output change)** Imaging sonar geometry and radiometry: the render covers the
  full vertical FOV across the fan and points outside +-vfov/2 are dropped; points
  are weighted by pixel solid angle so beams integrate their true solid angle (edge
  beams were ~3-5x too bright, with 2/3-column stripes); min range is a slant range;
  back faces no longer subtract intensity; points come from this frame's render
  pose (`compact_depth_points`) instead of `get_pointcloud()`'s current pose
- **(output change)** Underwater camera model runs in linear light (sRGB decode ->
  `I = J e^(-beta_D z) + B_inf (1 - e^(-beta_B z))` -> encode); `B_inf` is the
  veiling colour as seen. On encoded values the effective beta was ~2x nominal
- **(output change)** Degraded depth keeps a pixel when its direct signal is at
  least `depth_visibility_threshold` (default 0.25) of the light reaching the
  camera, instead of absolute brightness > 0.25 (dark surfaces never returned depth)
- **(output change)** `UWCam_KittiWriter`'s default Jerlov tables are transmittances
  and are now converted to `beta = -ln N` (tables tagged
  `"coefficients": "transmittance"`; beta tables such as the SeaClear configs pass
  through). Its `UW_render_2` calls also had attenuation / backscatter swapped
- **(output change)** `rtx_acoustic` fold resamples A-scans as bin means of their
  linear interpolant in range and azimuth: at the runner defaults the point scatter
  left ~72% of range rows and ~88% of beam columns black

### Added

- Opt-in sonar model terms for `make_sonar_data` / `sonar_params.model_params`, all
  off by default: spreading / absorption / TVG range gain, Gamma speckle with a
  correlation cell, Gaussian beam-pattern blur
- ROS control mode `dynamic velocity control`: cmd_vel tracked by a PI loop with
  PhysX-damping feedforward through forces, so buoyancy / drag / collisions still
  act. `control_params` gains `ros2_mode`, topic names, `stamped_cmd_vel`
  (TwistStamped) and `command_timeout`
- Navigation outputs (opt-in): sonar LaserScan / PointCloud2 from a row-median
  detector, `odom -> base_link` TF from ground truth, static `map -> odom`; Nav2 /
  EasyNav / 3D notes in the bringup guide

### Fixed

- ROS force control applies the Wrench in N / N·m through a rigid-body view.
  `PhysxForceAPI` defaulted to acceleration mode, so newtons were read as m/s²
  (26x too strong on the 26 kg Revolution), and upstream found it stopped the IMU
- OmniGraph `/cmd_vel` subscriber zeroes the command after 1 s without a message
- DVL: lever-arm velocity (`v + omega x r`), altitude from the beams' vertical
  component, twist covariance from the beam noise; barometer variance reported
- OmniGraph ROS: IMU orientation order (xyzw), planar 32FC1 depth, image frame ids
- `ProjectedSonarImage` bearings / ranges at bin centres; FLS writer noise seeds
  decorrelated and its PNG snapshot taken before the async write
- Caustics world positions and UVs; `UWCam_KittiWriter` (H, W) swap; depth
  degradation honours `max_range`

### Performance

- Headless runner scans the sonar and renders the camera only on publish ticks
- ROS `uint8[]` payloads as `array('B')`; GUI sonar uses the GPU point filter
- KITTI writers: single-pass segmentation export (782 -> 165 ms per 1080p frame),
  debug-only annotators, reused buffers, constant JSON written once
- `rtx_acoustic` fold uses cached resampling matrices (~1.0 ms vs ~2.6 ms per frame)

## [0.4.0] - 2026-10-05

### Changed

- Merge upstream OceanSim 0.2 (umfieldrobotics/OceanSim main): SDG playground,
  trajectory recorder, water surface, KITTI writers, OmniGraph `*_ROS` sensor
  subclasses, depth degradation and the Seaclear standalone SDG scripts
- Target Isaac Sim 6.1.0: Docker base image `nvcr.io/nvidia/isaac-sim:6.1.0`, docs and
  links updated; the Dockerfile only pip-installs OpenCV when the bundled interpreter lacks cv2

### Fixed

- Import `read_camera_info` from `isaacsim.ros2.core` (the ROS 2 bridge extension no
  longer exports it), with the old path as a fallback
- Colorpicker caustics sliders get explicit bounds; Isaac Sim 6.1.0's slider clamp
  would otherwise force Max Depth / Time Speed / decal positions into 0..1
- FLS_KittiWriter rebuilds its ray-query points from per-pixel depth / normals /
  segmentation AOVs (new `compact_depth_points` kernel) instead of the `pointcloud`
  composite annotator, which crashed at `world.play()` on Isaac Sim 6.x
- OmniGraph ROS mode: scenario RESET detaches the replicator ROS writers and removes
  the TF / publisher graphs instead of stacking duplicate publishers
- Merge upstream `feature/remove_scenario_namespacing`: the OmniGraph ROS topics
  are now fixed names (`/RGBCamera/image`, `/ImagingSonar/image`, `/IMU`, `/DVL`, ...)

## [0.3.0] - 2025-09-06

### Changed

- Target Isaac Sim 6.0.1 (Ubuntu 24.04 / ROS 2 Jazzy); update documentation accordingly
- Merge the ROS2 bridge into the main line

### Added

- Dockerfile and run helper for Isaac Sim 6.0.1 + ROS 2 Jazzy with GPU and X11 display passthrough

### Fixed

- ROS2 image publisher now honors the configured publish frequency
- Coordinate the shared rclpy lifecycle across ROS2 components

## [0.2.0] - 2025-08-05

### Added

- Add ros2 control function
- Add ros2 publish uw image

## [0.1.0] - 2025-01-08

### Added

- Initial version of OceanSim Extension

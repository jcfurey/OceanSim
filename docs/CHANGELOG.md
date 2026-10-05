# Changelog

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

# Changelog

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

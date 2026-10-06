# ROS 2

OceanSim publishes sensor data and accepts velocity commands through ROS when enabled. Use **Enable ROS** in the included scenario to turn publishing on or off.

> **This fork:** the page below describes upstream OceanSim's OmniGraph publishers (the `*_ROS` sensor classes). Here they are an opt-in: tick **OmniGraph ROS** (default off) in the Sensor Example's *Sensors* panel before **LOAD**. Turning it on also switches off the underwater camera's own rclpy publisher, so topics are not published twice. With it off you get this fork's rclpy bridge instead (`/oceansim/robot/*` topics, *ROS control* mode, and the headless runner); see [ROS2 Bridge](../README.md#ros2-bridge) and [ROS2 Platform Bringup](ros2_platform_bringup.md). In this fork `/cmd_vel` is only subscribed with OmniGraph ROS on, and not in *Manual control* or *ROS control* mode.
Follow the example scenario to see how to utlilize ROS in your scenario.

**Important**: Ensure you have cloned [Oceansim_msgs](https://github.com/umfieldrobotics/oceansim_ros_msgs) into your ROS workspace. Build and source this package in the terminal from which you are starting Isaac Sim prior to launching. This is required as certain sensors depend on custom defined message types (this currently applies to the DVL only).

## Published topics

| Topic | Message | Sensor |
|-------|---------|--------|
| `/RGBCamera/image` | `sensor_msgs/Image` | Underwater camera |
| `/DepthImage` | `sensor_msgs/Image` (`32FC1`) | Underwater camera depth |
| `/RGBCamera/pointcloud` | `sensor_msgs/PointCloud2` | Underwater camera point cloud |
| `/RGBCamera/camera_info` | `sensor_msgs/CameraInfo` | Underwater camera parameters |
| `/ImagingSonar/image` | `sensor_msgs/Image` | Imaging sonar |
| `/ImagingSonar/camera_info` | `sensor_msgs/CameraInfo` | Imaging sonar camera parameters |
| `/ImagingSonar/pointcloud` | `sensor_msgs/PointCloud2` | Imaging sonar point cloud |
| `/IMU` | `sensor_msgs/Imu` | IMU |
| `/DVL` | `msgs/Dvl` | DVL |
| `/Barometer` | `sensor_msgs/FluidPressure` | Barometer |
| `/tf` | `tf2_msgs/TFMessage` | Camera / sonar frames |

## Subscribed topics

| Topic | Message | Notes |
|-------|---------|-------|
| `/cmd_vel` | `geometry_msgs/Twist` | Active when control mode is not Manual |

Use this topic to send control commands to your vehicle.

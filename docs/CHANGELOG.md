# Changelog

## [Unreleased]

Accuracy and performance passes. Entries marked **(output change)** change the
default sonar / camera images: re-check gains, noise and water parameters that
were tuned against real data.

### Changed

- **(output change)** Default imaging sonar and the Revolution demo model the
  Oculus M3000d's 1.2 MHz mode. Added the 3 MHz payload (`oculus_m3000d_hf`),
  with mode-specific apertures, range limits and beam responses from datasheet
  rev 10. The sensor window and OmniGraph image now show a metric Cartesian
  fan rather than a stretched polar rectangle; `ProjectedSonarImage` remains
  range-major. Each fan pixel shows the brightest bin it covers, so no bin is
  skipped where bins outnumber pixels; range rings are drawn in the window only.
  Sonar map coordinates now use the same bin centres as ROS. Constructor
  defaults equal the M3000d preset (10 m working range, 1024-bin budget).
- **(output change)** Every imaging-sonar preset normalises by the ping maximum
  (previously only the M3000d did) and shows the normalised echo with gamma 0.5,
  like an Oculus's gamma correction, so weak echoes stay visible beside a strong
  broadside return (`make_sonar_data(gamma=...)`, default 1 = linear; set
  `sonar_params.model_params.gamma` to override). Oculus presets apply their
  band's beam response (M370s 2.0°, 1.2 MHz 0.6°, 2.1 MHz 0.4°).
  `ProjectedSonarImage` `rx_beamwidths` report the beam width actually applied,
  or the beam spacing when unblurred, and `tx_beamwidths` the sensor's vertical
  aperture.
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

- **(output change)** Vehicle models: 6-DOF hydrodynamics (linear + quadratic
  drag, added mass, weight and buoyancy with righting moment and partial
  buoyancy at the surface) and thrusters (allocation, saturation, lag) for
  `bluerov2`, the new `bluerov2_heavy`, and `deeptrekker_revolution`, replacing
  the PhysX damping stand-in. BlueROV2 data from von Benzon et al. 2022 and the
  T200 datasheet; the Revolution is estimated from its specs. The BlueROV2's
  in-air mass is now 11.5 kg (was 5). Vehicles drift with their buoyancy when
  idle. `robot.hydrodynamics: false` restores the old behaviour. See
  `docs/subsections/vehicle_models.md`
- ROS `thruster control` mode: normalised per-thruster commands on
  `/oceansim/robot/thruster_cmd`
- Eight more vehicles: Deep Trekker DTG3 and PIVOT, VideoRay Pro 5 and
  Defender, Chasing M2 Pro Max, QYSEA FIFISH V6 Expert, Saab Seaeye Falcon,
  Teledyne SeaBotix vLBV300. Their models are estimated from published specs,
  and each reproduces its published thrust and speeds. Vehicles without a 3D
  asset are imported from a generated URDF
- Payload catalogue with datasheet values: Oculus M370s / M750d / M1200d /
  C550d, Tritech Gemini 720is / 720im, BlueView M900, Ping360, Ping2 altimeter,
  Water Linked A50 / A125, Nortek DVL500, Teledyne Pathfinder, grippers and
  lights. Payloads set the sonar / DVL / altimeter to the device, add mass,
  buoyancy and drag, and the vehicle is re-trimmed. `payloads` config,
  `--payload`, GUI pickers
- Altimeter sensor (`sensor_msgs/Range` on `/oceansim/robot/altimeter`)
- URDF export (primitive geometry, sensor frames, gripper jaws, a lossless
  `<oceansim>` block; optional Gazebo Sim plugins) and import of vehicles from
  URDFs carrying OceanSim, Gazebo Sim or UUV Simulator hydrodynamics;
  `scripts/oceansim_urdf.py` (list / show / export / inspect)
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

- **(output change)** The Revolution's sonar looks out level from the pivot
  head: `sonar_link` moves to (0.12, 0, 0.04) in the head frame with no pitch
  (was 0.0625, 0, 0.04, pitched 30° down). The old origin lay inside
  `pivot_head.stl`; once the M3000d's 0.1 m minimum range pulled the render
  camera's near clip in to 4 cm, the head's inside blocked 82% of the view and
  the sonar showed no returns. The fan now clears the vehicle for head angles
  from -105° to +120°; tilt the head to tilt the sonar.
- Infinite background depths no longer trigger invalid-arithmetic warnings
  during sonar point reconstruction.
- Docker launchers disable the optional Hub cache by default to avoid failed
  daemon-launch retries with local scenes; `OMNICLIENT_HUB_MODE=shared` enables
  it explicitly.
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

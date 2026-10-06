# ROS2 Platform Bringup (description, joints & sensors)

OceanSim's headless runner (`isaacsim/oceansim/standalone/oceansim_ros2.py`) can
present a vehicle to ROS2 as a fully articulated robot: its **description**
(URDF), its **joint states**, and its **sensors** — all driven by the selected
platform or a single URDF.

## 1. Pick a platform (or bring your own asset)

Vehicles live in a small registry (`isaacsim/oceansim/utils/platforms.py`). Three
ship with 3D assets:

| Platform key | Vehicle | Notes |
|---|---|---|
| `bluerov2` | Blue Robotics BlueROV2 | default |
| `bluerov2_heavy` | BlueROV2 Heavy (8 thrusters) | same 3D asset |
| `deeptrekker_revolution` | Deep Trekker REVOLUTION | 26 kg, 6 thrusters |

Eight more vehicles have no 3D asset; they are imported from a generated URDF:
Deep Trekker DTG3 and PIVOT, VideoRay Pro 5 and Defender, Chasing M2 Pro Max,
QYSEA FIFISH V6 Expert, Saab Seaeye Falcon and Teledyne SeaBotix vLBV300. Add
payloads with `--payload` (e.g. `--payload oculus_m750d --payload waterlinked_a50`).
[Vehicle Models](vehicle_models.md) lists them all, with their payload options.

A platform's spec supplies the USD/URDF asset path, mass/damping, collision,
spawn pose, sensor mount poses, and the default robot description. Select it:

```bash
./scripts/run_oceansim_ros2.sh --platform deeptrekker_revolution
```

For a first run, use the dedicated launcher. It selects the REVOLUTION,
publishes standalone sensor TF, and loads the scanned MHL environment when an
asset pack is mounted. Without that pack it uses the repository's textured
subsea inspection site and local CAD, with a generated URDF hull as a fallback:

```bash
./scripts/run_deeptrekker_revolution.sh
```

Set `OCEANSIM_ASSETS` to an asset-pack directory to have the same command load
the MHL scene and detailed REVOLUTION USD when available. Select explicitly
with `--environment auto|builtin|reef|mhl|/path/to/scene.usd`. See
[demo assets](demo_assets.md) for local CAD staging and Docker Compose. Extra runner flags are
forwarded, for example
`./scripts/run_deeptrekker_revolution.sh --no-sonar --no-camera`.

The asset itself lives under your registered asset root
(`<asset_root>/<usd_subpath>` / `<urdf_subpath>`), or override per run with
`--urdf` / config `robot.usd_path` / `robot.urdf_path`.

## 2. "I only have a URDF"

A URDF alone is enough — and gives *more* than a bare USD, because it defines the
articulation (joints) and the sensor frames:

```bash
./scripts/run_oceansim_ros2.sh --platform deeptrekker_revolution \
    --urdf /assets/DeepTrekker/revolution.urdf
```

From one URDF, OceanSim derives:

1. **the body** — imported with Isaac's URDF importer (creates the articulation);
2. **the joints** — published on `/joint_states`, driven from `/oceansim/robot/joint_command`;
3. **`/robot_description`** — the URDF latched for robot_state_publisher / RViz;
4. **the sensor mounts** — each sensor is placed at its URDF link (`sonar` /
   `camera` / `dvl` link, by its fixed-joint origin), falling back to the
   platform's spec mount if the URDF doesn't define that sensor;
5. **the frames** — the base frame is taken from the URDF root link and the
   sonar/camera frames from their URDF link names, so OceanSim's message stamps
   line up with robot_state_publisher's TF tree (and OceanSim doesn't publish a
   duplicate static TF for a frame the URDF already owns).

> The runtime URDF importer is experimental (it wraps Isaac's importer
> extension). The reliable alternative is to convert the URDF to USD once with
> Isaac's URDF importer and point `usd_subpath` at the result.

## 3. Topics & QoS

`utils/ros2_qos.py` is the single source of truth for QoS, and the pairings are
CI-verified (`tests/test_ros2_qos.py`) — a publisher/subscriber only connect if
their QoS is compatible.

| Topic | Type | Dir | QoS |
|---|---|---|---|
| `/clock` | rosgraph_msgs/Clock | pub | reliable |
| `/oceansim/robot/odom` | nav_msgs/Odometry | pub | sensor (best-effort) |
| `/oceansim/robot/imu` | sensor_msgs/Imu | pub | sensor (best-effort) |
| `/oceansim/robot/dvl/twist` | geometry_msgs/TwistWithCovarianceStamped | pub | sensor |
| `/oceansim/robot/pressure` | sensor_msgs/FluidPressure | pub | sensor |
| `/oceansim/robot/altimeter` | sensor_msgs/Range (altimeter payload) | pub | sensor |
| `/oceansim/robot/sonar` | marine_acoustic_msgs/ProjectedSonarImage | pub | sensor |
| `/oceansim/robot/sonar/scan` | sensor_msgs/LaserScan (opt-in, §7) | pub | sensor |
| `/oceansim/robot/sonar/points` | sensor_msgs/PointCloud2 (opt-in, §7) | pub | sensor |
| `/robot_description` | std_msgs/String | pub | **latched** (transient-local) |
| `/joint_states` | sensor_msgs/JointState | pub | **reliable** (robot_state_publisher) |
| `/oceansim/robot/joint_command` | sensor_msgs/JointState | sub | sensor |
| `/oceansim/robot/vel_cmd` | geometry_msgs/Twist (or TwistStamped, §6) | sub | reliable |
| `/oceansim/robot/force_cmd` | geometry_msgs/Wrench | sub | reliable |
| `/oceansim/robot/thruster_cmd` | std_msgs/Float64MultiArray (§6) | sub | reliable |

Sensor streams are **best-effort**: subscribe best-effort (RViz / `sonar_image_proc`
do by default), or you will silently receive nothing.

## 4. robot_state_publisher + RViz

With OceanSim running, bring up the standard consumers:

```bash
ros2 launch scripts/oceansim_bringup.launch.py \
    urdf:=/assets/DeepTrekker/revolution.urdf
```

This starts `robot_state_publisher` (same URDF in, `/joint_states` in, TF out)
and `rviz2`. In RViz add a **RobotModel** display and set its *Description Topic*
to `/robot_description` (the latched topic), plus a **TF** display. `use_sim_time`
is on, matching OceanSim's `/clock`.

## 5. Manipulating joints

Publish a `sensor_msgs/JointState` of position targets; the command may be a
subset / reordered — unspecified joints hold, unknown joint names are ignored:

```bash
ros2 topic pub --once /oceansim/robot/joint_command sensor_msgs/msg/JointState \
    '{name: ["arm_joint_1"], position: [0.5]}'
```

Joint manipulation is active only when the loaded asset is an articulation with
DOFs (a URDF, or a USD that defines joints); a plain rigid-body hull makes the
joint topics a graceful no-op.

## 6. Vehicle control modes (ROS control)

`control_mode: "ROS control"` subscribes to the velocity and force topics; the
config's `control_params` (or the GUI's *ROS2 Control Mode* dropdown) picks how
commands move the vehicle:

| `ros2_mode` | Input | What happens |
|---|---|---|
| `velocity control` (default) | Twist, body frame | Sets the body velocity every step (kinematic). Exact tracking, but buoyancy, drag and collision response are overwritten. |
| `dynamic velocity control` | Twist, body frame | A PI loop with damping feedforward turns the command into force / torque (`ros2_control_math.BodyVelocityPI`), so the physics still acts. Use this to tune controllers that must transfer to a real vehicle. |
| `force control` | Wrench, body frame, N / N·m | Applied at the centre of mass each step. |
| `thruster control` | Float64MultiArray on `/oceansim/robot/thruster_cmd`, one value in [−1, 1] per thruster | Through each thruster's thrust curve. Needs the vehicle model. |

With the platform's vehicle model on (the default; see [Vehicle Models](vehicle_models.md)),
the force, dynamic velocity and manual modes go through the thrusters, so thrust
limits and lag apply. The dynamic mode's feedforward is then the model's drag at
the commanded velocity rather than PhysX damping. Kinematic `velocity control`
bypasses the model.

Forces go through a rigid-body tensor view in newtons. The old `PhysxForceAPI`
path defaulted to *acceleration* mode, so a Wrench was read as m/s².

Other `control_params`:
- `vel_topic` / `force_topic` / `thruster_topic`: topic names, e.g. `"/cmd_vel"` for Nav2.
- `stamped_cmd_vel: true`: subscribe `TwistStamped`. Use it with Nav2's
  `enable_stamped_cmd_vel` (the default from Kilted on).
- `command_timeout`: dead-man timeout in seconds (default 2). A silent link
  zeroes the command; the dynamic mode then actively holds zero velocity.
- `velocity_pi`: gain overrides (`kp_lin`, `ki_lin`, `kp_ang`, `ki_ang`,
  `i_limit_*`, `damping_*`). By default the damping feedforward is the body's
  PhysX damping and `ki = (kp + damping)^2 / 8`.
- `max_linear_vel`, `max_angular_vel`, `max_force`, `max_torque`: magnitude clamps.

## 7. Navigation (Nav2, EasyNav, 3D)

OceanSim provides the simulator side; the navigation stack runs in your ROS
workspace. These options are off by default; turn them on under `publisher`:

```json
{
  "control_mode": "ROS control",
  "control_params": {"ros2_mode": "dynamic velocity control", "vel_topic": "/cmd_vel"},
  "publish_static_tf": true,
  "publisher": {
    "publish_odom_tf": true,
    "publish_map_odom_tf": true,
    "publish_sonar_scan": true,
    "publish_sonar_cloud": true
  }
}
```

- **`publish_odom_tf`**: broadcasts `odom → base_link` from the ground-truth
  pose with every odometry message. Leave it off if `robot_localization`
  publishes that transform.
- **`publish_map_odom_tf`**: a static identity `map → odom`, for stacks that
  expect a `map` frame when nothing localises.
- **`publish_sonar_scan`**: a LaserScan with the nearest sonar detection per
  beam. Beams with no detection are `+inf`, which lets costmaps clear along them.
- **`publish_sonar_cloud`**: a PointCloud2 (x, y, z, intensity) of every
  detection.

**How sonar detections work.** A cell counts as a detection when it is at least
`sonar_detect_factor` (default 5) times the median of its range row and at least
`sonar_detect_floor` (default 0.2). That works with either sonar normalisation,
and a flat seafloor band across the whole fan is not reported as an obstacle.

**Sonar points are placed at zero elevation**, on the sonar's horizontal plane.
An imaging sonar measures range and bearing, not elevation, so this is what a
real one gives you. The scan frame must be x-forward / z-up; the runner's sonar
static TF is. For true 3D points, use the camera depth: `/oceansim/robot/depth`
and `/oceansim/robot/camera_info` go through `depth_image_proc` to make a cloud.

**Nav2** plans in 2D (x, y, yaw). A vehicle that holds depth fits: both velocity
modes take the Twist's `linear.z` (Nav2 sends 0). Note that the dynamic mode
holds zero heave *velocity*, not depth, so depth can drift slowly. The settings
that matter for OceanSim (a sketch; not run in CI):

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      use_sim_time: true            # OceanSim publishes /clock
      global_frame: odom
      robot_base_frame: base_link
      rolling_window: true
      plugins: ["obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        observation_sources: sonar
        sonar:
          topic: /oceansim/robot/sonar/scan
          data_type: "LaserScan"
          marking: true
          clearing: true
          inf_is_valid: true
          # Heights are checked in the global frame. Underwater, z is negative,
          # and the defaults (0 to 2 m) drop every point.
          min_obstacle_height: -1000.0
          max_obstacle_height: 1000.0
controller_server:
  ros__parameters:
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      motion_model: "Omni"          # ROVs move sideways as well
```

**EasyNav** (URJC's representation-agnostic ROS 2 navigation system) consumes
the same inputs: TF, odometry, LaserScan / PointCloud2 and `cmd_vel`. Its 3D
representations (Octomap, NavMap meshes) model navigable *surfaces*, though,
not free water volume.

**Full 3D** (volumetric) navigation needs a 3D map and planner from elsewhere,
e.g. an OctoMap or voxel map built from the camera depth cloud, with an OMPL
planner. The simulator side is already there:
- the Twist's `linear.z` and angular rates are honoured in both velocity modes;
- the camera depth gives true 3D points;
- odometry, TF and the clock are published as above.

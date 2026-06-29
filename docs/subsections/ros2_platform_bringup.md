# ROS2 Platform Bringup (description, joints & sensors)

OceanSim's headless runner (`isaacsim/oceansim/standalone/oceansim_ros2.py`) can
present a vehicle to ROS2 as a fully articulated robot: its **description**
(URDF), its **joint states**, and its **sensors** — all driven by the selected
platform or a single URDF.

## 1. Pick a platform (or bring your own asset)

Vehicles live in a small registry (`isaacsim/oceansim/utils/platforms.py`). Two
ship today:

| Platform key | Vehicle | Notes |
|---|---|---|
| `bluerov2` | Blue Robotics BlueROV2 | default |
| `deeptrekker_revolution` | Deep Trekker REVOLUTION | 26 kg, 6 thrusters |

A platform's spec supplies the USD/URDF asset path, mass/damping, collision,
spawn pose, sensor mount poses, and the default robot description. Select it:

```bash
./scripts/run_oceansim_ros2.sh --platform deeptrekker_revolution
```

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
| `/oceansim/robot/sonar` | marine_acoustic_msgs/ProjectedSonarImage | pub | sensor |
| `/robot_description` | std_msgs/String | pub | **latched** (transient-local) |
| `/joint_states` | sensor_msgs/JointState | pub | **reliable** (robot_state_publisher) |
| `/oceansim/robot/joint_command` | sensor_msgs/JointState | sub | sensor |
| `/oceansim/robot/vel_cmd` | geometry_msgs/Twist | sub | reliable |
| `/oceansim/robot/force_cmd` | geometry_msgs/Wrench | sub | reliable |

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

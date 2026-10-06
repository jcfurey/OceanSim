# Vehicle Models (hydrodynamics and thrusters)

OceanSim's platforms have a 6-DOF underwater vehicle model. Before this model,
the scene ran without gravity and used PhysX damping (10–15 s⁻¹) as a stand-in
for water drag. Now drag, buoyancy, added mass and thrust limits come from
published or estimated vehicle data. The maths is in `utils/vehicle_dynamics.py`
(plain numpy, unit tested). `utils/vehicle_physics.py` applies it to the PhysX
body every physics step, and the parameters live with each platform in
`utils/platforms.py`.

## What is modelled

The model uses Fossen's equations in the body frame (x forward, y port, z up),
about the centre of gravity:

    (M_RB + M_A) ν̇ + C_RB(ν)ν + C_A(ν)ν + D(ν)ν + g(η) = τ_thrusters

| Term | Model |
|---|---|
| Rigid body `M_RB`, `C_RB` | PhysX, with the platform's mass and principal inertia; the CoG is at the prim origin |
| Added mass `M_A`, `C_A` | Diagonal. Applied by scaling the forces handed to PhysX per axis by m/(m+m_A), with Coriolis corrections. This reproduces the full equation exactly (tested against a reference integrator) |
| Damping `D(ν)` | Diagonal linear + quadratic drag |
| Restoring `g(η)` | Weight at the CoG and buoyancy at the CoB, so the vehicle rights itself. Buoyancy scales with the submerged fraction at the water surface, so a positively buoyant vehicle floats there |
| Thrusters | Fixed thrusters; a commanded wrench is allocated by least squares. If any thruster would saturate, the whole set is scaled down so the wrench keeps its direction. First-order thrust lag |

Not modelled:
- thruster–hull interaction;
- tether drag;
- currents and waves;
- off-diagonal hydrodynamic terms;
- added mass during contacts (PhysX resolves collisions with the rigid mass).

The scene's gravity stays zero; the vehicle's weight is applied by the model.

## Vehicles

The registry has 11 vehicles. Each row gives the steady speed each vehicle
reaches at full thrust, bare (no payloads). "—" means the thruster layout
can't produce that motion.

| Platform | Vehicle | Mass | Size L×W×H (m) | Thrusters | Model | Surge | Sway | Heave up | Yaw |
|---|---|---|---|---|---|---|---|---|---|
| `bluerov2` | Blue Robotics BlueROV2 | 11.5 kg | 0.457×0.338×0.254 | 6 | measured (von Benzon et al.) | 0.92 m/s | 0.69 m/s | 0.62 m/s | 4.0 rad/s |
| `bluerov2_heavy` | Blue Robotics BlueROV2 Heavy | 13.5 kg | 0.46×0.58×0.38 | 8 | measured (von Benzon et al.) | 0.92 m/s | 0.69 m/s | 0.91 m/s | 4.3 rad/s |
| `chasing_m2_pro_max` | Chasing M2 Pro Max | 8 kg | 0.608×0.294×0.196 | 8 | estimated | 1.50 m/s | 0.72 m/s | 0.61 m/s | 5.7 rad/s |
| `deeptrekker_dtg3` | Deep Trekker DTG3 | 8.5 kg | 0.279×0.325×0.258 | 3 | estimated | 1.29 m/s | — | 1.29 m/s | 11.8 rad/s |
| `deeptrekker_pivot` | Deep Trekker PIVOT | 23.6 kg | 0.576×0.36×0.31 | 6 | estimated | 1.03 m/s | 0.80 m/s | 0.51 m/s | 6.4 rad/s |
| `deeptrekker_revolution` | Deep Trekker REVOLUTION | 26 kg | 0.717×0.44×0.235 | 6 | estimated | 1.54 m/s | 1.03 m/s | 1.54 m/s | 6.4 rad/s |
| `qysea_fifish_v6_expert` | QYSEA FIFISH V6 Expert | 4.6 kg | 0.383×0.331×0.143 | 6 | estimated | 1.50 m/s | 1.39 m/s | 0.75 m/s | 17.1 rad/s |
| `saab_seaeye_falcon` | Saab Seaeye Falcon | 60 kg | 1×0.6×0.5 | 5 | estimated | 1.54 m/s | 0.94 m/s | 0.60 m/s | 4.7 rad/s |
| `seabotix_vlbv300` | Teledyne SeaBotix vLBV300 | 18 kg | 0.625×0.39×0.39 | 6 | estimated | 1.54 m/s | 1.06 m/s | 0.80 m/s | 7.5 rad/s |
| `videoray_defender` | VideoRay Mission Specialist Defender | 17.2 kg | 0.711×0.394×0.238 | 7 | estimated | 1.95 m/s | 0.46 m/s | 0.80 m/s | 3.4 rad/s |
| `videoray_pro5` | VideoRay Mission Specialist Pro 5 | 11.8 kg | 0.515×0.33×0.257 | 3 | estimated | 2.26 m/s | — | 0.80 m/s | 7.6 rad/s |

Every vehicle reproduces its published thrust and top-speed figures.
`tests/test_platforms.py` checks this against the sources. Only the BlueROV2
pair has measured hydrodynamics. For the others:
- **Hydrodynamics** are estimated from the published mass, size, thrust and
  speed (see *Estimated vehicles* below).
- **Thruster positions and unpublished vector angles** are estimates. Where both
  forward and lateral thrust are published, the angle comes from their ratio.
- **Sources and conflicts:** `utils/platforms.py` lists the sources for each
  vehicle, and notes where they conflict:
  - PIVOT weight: 16.8–23.6 kg;
  - Defender forward thrust: 23.6 or 26.7 kgf;
  - vLBV300 thrust ranges.

The BlueROV2, BlueROV2 Heavy and REVOLUTION use the OceanSim asset pack's 3D
models. The other vehicles have none, so the simulator imports a URDF generated
from their data: a hull box, thruster cylinders and payload boxes. The same
generated model is used when a platform's asset is missing, or when
`robot.prefer_source` is `"generated"`.

### Estimated vehicles

`platforms.HydroEstimate` derives a model from published specs:
- **Drag:** quadratic coefficients chosen so the thrusters' maximum along each
  axis reaches the listed top speed. Linear terms are 0.1 m/s × the quadratic
  ones. Axes without a listed speed use a drag coefficient of 1.
- **Rotational drag:** strip theory. On the BlueROV2 Heavy this lands within
  0.6–1.5× of the measured values.
- **Added mass:** Lamb's coefficients for the bounding ellipsoid.
- **Inertia:** 0.6 × a uniform box of the vehicle's mass.
- **Trim:** +0.5% buoyancy, with the CoB 2 cm above the CoG.
- **Unpublished thrust** (PIVOT, FIFISH, the Pro 5's vertical thruster): the
  thrusters are sized to reach the listed speed with drag coefficient 1.

Top yaw rates at full thrust come out high, because rotational drag scales with
L⁴. Their tip speed relative to sway speed matches the measured BlueROV2. Real
vehicles limit yaw rate in their controllers.

Other approximations, per vehicle:
- **DTG3:** climbs and dives by pitching its body. It is modelled with a central
  vertical thruster instead.
- **FIFISH V6:** free pitch and roll aren't reproduced, because its thruster
  layout isn't published.

## Payloads

Payloads come from `utils/payloads.py`, with datasheet values and sources:

| Payload | Kind | Description | In air / in water | Simulated |
|---|---|---|---|---|
| `blueview_m900` | sonar | Teledyne BlueView M900 Mk2 imaging sonar (900 kHz, 1000 m) | 2.5 / 1.2 kg | yes |
| `chasing_grabber_arm_2` | gripper | Chasing Grabber Arm 2 (two-jaw clamp) | 0.522 / 0 kg | body only |
| `lumen_light_pair` | light | Two Blue Robotics Lumen subsea lights (1500 lm each) | 0.236 / 0.1 kg | body only |
| `newton_gripper` | gripper | Blue Robotics Newton Subsea Gripper (62 mm jaws) | 0.524 / 0.267 kg | body only |
| `nortek_dvl500` | dvl | Nortek DVL500 300 m (500 kHz, 4-beam Janus) | 3.5 / 0.5 kg | yes |
| `oculus_c550d` | sonar | Oculus C550d imaging sonar, low-frequency mode (550 kHz) | 0.98 / 0.36 kg | yes |
| `oculus_c550d_hf` | sonar | Oculus C550d imaging sonar, high-frequency mode (820 kHz) | 0.98 / 0.36 kg | yes |
| `oculus_m1200d` | sonar | Oculus M1200d imaging sonar, low-frequency mode (1.2 MHz) | 0.98 / 0.36 kg | yes |
| `oculus_m1200d_hf` | sonar | Oculus M1200d imaging sonar, high-frequency mode (2.1 MHz) | 0.98 / 0.36 kg | yes |
| `oculus_m370s` | sonar | Oculus M370s imaging sonar (375 kHz) | 0.98 / 0.36 kg | yes |
| `oculus_m750d` | sonar | Oculus M750d imaging sonar, low-frequency mode (750 kHz) | 0.98 / 0.36 kg | yes |
| `oculus_m750d_hf` | sonar | Oculus M750d imaging sonar, high-frequency mode (1.2 MHz) | 0.98 / 0.36 kg | yes |
| `ping2` | altimeter | Blue Robotics Ping2 sonar altimeter / echosounder (115 kHz) | 0.187 / 0.1 kg | yes |
| `ping360` | sonar | Blue Robotics Ping360 mechanically scanning sonar (750 kHz) | 0.51 / 0.175 kg | no |
| `qysea_2finger_arm` | gripper | QYSEA FIFISH 2-Finger Robotic Arm (V6 Expert / V-EVO) | 0.8 / 0 kg | body only |
| `teledyne_pathfinder` | dvl | Teledyne RDI Pathfinder 600 kHz (ROV version) | 2 / 0.73 kg | yes |
| `tritech_gemini_720im` | sonar | Tritech Gemini 720im compact imaging sonar (720 kHz) | 0.435 / 0.244 kg | yes |
| `tritech_gemini_720is` | sonar | Tritech Gemini 720is imaging sonar (720 kHz, aluminium) | 3.4 / 1.3 kg | yes |
| `videoray_pro5_manipulator` | gripper | VideoRay Pro 5 rotating manipulator (Blueprint Lab, 78 mm claw) | 0.36 / 0.24 kg | body only |
| `videoray_rotating_manipulator` | gripper | VideoRay rotating manipulator (Pro 4 / Defender, parallel jaws) | 1.17 / 0.73 kg | body only |
| `waterlinked_a125` | dvl | Water Linked DVL A125 (420 kHz, 4-beam Janus) | 0.75 / 0.5 kg | yes |
| `waterlinked_a50` | dvl | Water Linked DVL A50 (1 MHz, 4-beam Janus) | 0.17 / 0.105 kg | yes |

Fitting a payload has these effects:
- **Body:** it adds its in-air mass and displaced volume at its mount, which
  moves the CoG and CoB and adds parallel-axis inertia. Its housing adds drag.
- **Trim:** the vehicle is then re-trimmed to its bare net buoyancy, as an
  operator would do it: syntactic foam high on the frame, or lead low if the
  payloads made it lighter. `robot.trim: false` turns this off.
- **Sensor payloads** set the simulated sensor to the device:
  - **Sonar:** the FOV and beam spacing (FOV / beams). `max_range` is a working
    range (an operator setting) and can't exceed the device's maximum. The
    range resolution is the device's, or range / 1024, whichever is coarser.
    `sonar_params` still override all of this.
  - **DVL:** Janus beam angle, altitude limits, and single-ping noise where the
    datasheet gives it.
  - **Ping2:** a downward single-ray altimeter, published as `sensor_msgs/Range`
    on `/oceansim/robot/altimeter`.
- **Grippers:** in a generated model, a gripper gets two revolute jaws. They are
  driven like any joint, through `/oceansim/robot/joint_command`. On the
  asset-pack 3D models a gripper is body only.
- **Not simulated:** the Ping360 scanning sonar. It contributes its body only.

At most one payload of each sensor kind can be fitted. Any payload can go on any
vehicle; each platform's `payload_options` lists the combinations commonly sold.

## Vehicle details: BlueROV2 and REVOLUTION

### BlueROV2 and BlueROV2 Heavy

Measured data:
- **Hydrodynamics:** von Benzon et al. 2022, *An Open-Source Benchmark
  Simulator: Control of a BlueROV2 Underwater Robot*, J. Mar. Sci. Eng. 10,
  1898, Table A1 (BlueROV2 Heavy).
  - Damping was identified in pool experiments. The tether could not be fully
    kept out of those experiments, so the drag may be high.
  - Added mass comes from the Eidsvik / DNV method; inertia comes from CAD.
  - The CoB is 1 cm above the CoG.
- **Thrust:** the Blue Robotics T200 datasheet at the 14.8 V battery's nominal
  voltage: 47 N forward, 37 N reverse. The thrust-versus-command shape comes
  from the paper's T200 regression.
- **Standard frame (`bluerov2`):**
  - thruster poses from the [`bluerov2_gz`](https://github.com/clydemcqueen/bluerov2_gz)
    Gazebo model;
  - mass 11.5 kg (Blue Robotics: 11–12 kg with ballast and battery);
  - trimmed about 1 N positive, as Blue Robotics recommends.

  It shares the Heavy's hydrodynamic coefficients.
- **Heavy (`bluerov2_heavy`):**
  - the paper's vehicle: 13.5 kg, 0.0134 m³, thruster poses from Table A1;
  - it uses the same 3D asset, since the Heavy kit isn't modelled visually.

The modelled top surge speed is about 0.9 m/s. Blue Robotics quotes 1.5 m/s;
the paper measured 0.72 m/s at its lower thrust, with the tether attached.

### Deep Trekker REVOLUTION (estimated, details)

Deep Trekker publishes the mass (26 kg), the size (717 × 440 × 235 mm) and the
thruster count: two vertical and four vectored horizontal. It publishes no
hydrodynamic data, so the rest is estimated:

| Quantity | Estimate |
|---|---|
| Thrust | 12 kgf in each axis (third-party listings): four 45° horizontal thrusters of 41.6 N, two vertical of 58.9 N |
| Drag | Quadratic coefficients set so that this thrust reaches the listed 3 kn forward, 2 kn lateral and 3 kn vertical; linear terms are 0.1 m/s × the quadratic ones. The implied drag coefficients are 0.90 (surge) and 1.20 (sway), which are plausible, and 0.29 (heave), which is low: the listed vertical speed is probably optimistic |
| Rotational drag | Strip theory from those coefficients. Checked on the BlueROV2 Heavy, the same formulas land within 0.6–1.5× of its measured values |
| Added mass | Lamb's coefficients for the bounding ellipsoid, scaled to the displaced volume |
| Inertia | 0.6 × a uniform 26 kg box (the BlueROV2 model's ratio) |
| Buoyancy | +0.5% net, CoB 2 cm above the CoG (assumed) |
| Thruster positions | Estimated from the hull |

With these estimates the vehicle can yaw at about 6 rad/s under full thrust.
Real vehicles limit rates in software.

## Configuration

Headless runner (`oceansim_ros2.py` config):

| Key | Default | Effect |
|---|---|---|
| `robot.hydrodynamics` | `true` | Use the model for platforms that have one. `false` restores the old PhysX damping stand-in |
| `robot.thruster_voltage` | platform (14.8 V) | Supply voltage for T200 platforms (12–20 V) |
| `robot.drag_scale` | `1.0` | Multiplies all damping, e.g. to match a top speed you measured |
| `robot.mass` | platform | In-air mass. The displaced volume, and so the buoyancy, is kept |
| `payloads` / `--payload NAME` | platform standard | Payloads to fit (`[]` / `--no-payloads` = bare) |
| `robot.trim` | `true` | Re-trim to the bare vehicle's buoyancy after fitting payloads |
| `robot.prefer_source` | `"usd"` | `"generated"`: always import the generated URDF model (shows payloads and gripper jaws) |
| `robot.urdf_path` / `--urdf` | — | Import this URDF. If it carries hydrodynamics it defines the vehicle (see below) |
| `robot.hydro_from_urdf` | `true` | `false`: use the URDF for visuals only and keep the platform's dynamics |
| `water_density` | `1000` | kg/m³ for buoyancy (≈1025 in seawater, which makes the BlueROV2 Heavy positive) |
| `water_surface_z` | `1.43389` | Water surface height for partial buoyancy (also the barometer's) |

In the GUI the model is on for every platform that has one, in fresh water. The *Sensors* panel has pickers for the sonar, DVL, gripper and lights; *standard* means the platform's default.

## Control with the model

| Mode | With the model |
|---|---|
| Manual control | Keyboard accelerations × (mass + added mass), sent through the thrusters |
| ROS `force control` | The Wrench is allocated to the thrusters, so limits and lag apply |
| ROS `dynamic velocity control` | Thrust = (M_RB + M_A) × PI output + the model's drag at the commanded velocity |
| ROS `thruster control` | `std_msgs/Float64MultiArray` on `/oceansim/robot/thruster_cmd`: one value in [−1, 1] per thruster, in the platform's `thrusters` order, through the thrust curve |
| ROS `velocity control`, Waypoints | Kinematic: the model is not applied |

With no thrust command the vehicle drifts with its buoyancy. The BlueROV2 rises
at a few cm/s and floats at the surface. The dynamic velocity mode holds zero
velocity against it.

To calibrate against your own vehicle:
1. Measure its top speed at full thrust.
2. Set `robot.drag_scale` to (modelled speed / measured speed)², since drag is
   mostly quadratic.
3. Set `robot.thruster_voltage` to your supply voltage.

## URDF import and export

**Export.** Any vehicle with payloads can be written as a URDF. The file
contains:
- the total mass, CoG and inertia, with payloads and trim included;
- primitive geometry;
- sensor frames (`sonar_link`, `camera_link`, `dvl_link`, `altimeter_link`, ...);
- gripper jaws;
- an `<oceansim>` element holding the complete model.

`--gazebo` adds continuous thruster joints plus Gazebo Sim's Hydrodynamics and
Thruster system plugins. Gazebo's buoyancy comes from its own world-level system,
so the net buoyancy there differs unless you tune it.

**Import.** A URDF given with `robot.urdf_path` (or `--urdf`) defines the
vehicle when it carries hydrodynamic data. It accepts:
1. OceanSim's `<oceansim>` block: lossless. Export then import reproduces the
   model exactly (tested for all 11 vehicles).
2. Gazebo Sim plugins: `gz-sim-hydrodynamics-system` and
   `gz-sim-thruster-system`, as in `bluerov2_gz` or orca4 models.
3. UUV Simulator plugins: `uuv_underwater_object` and `uuv_thruster` (Basic
   conversion).

Mass, CoG and inertia come from the base link's `<inertial>`. Anything a
format lacks is assumed and listed in the log, for example the displaced
volume of a Gazebo-only URDF (assumed 0.5% positive). Expand xacro first.

### Command-line tool

`scripts/oceansim_urdf.py` needs only Python 3 and numpy, not Isaac Sim:

```bash
scripts/oceansim_urdf.py list                               # vehicles + payloads
scripts/oceansim_urdf.py show bluerov2 -p oculus_m750d -p waterlinked_a50
scripts/oceansim_urdf.py export videoray_defender -p nortek_dvl500 -o defender.urdf
scripts/oceansim_urdf.py export bluerov2_heavy --gazebo -o brov2h_gz.urdf
scripts/oceansim_urdf.py inspect my_rov.urdf                # what OceanSim reads + assumes
```

`show` prints mass, trim, net buoyancy, and thrust and top speed per axis for a
configuration. `inspect` does the same for any URDF.

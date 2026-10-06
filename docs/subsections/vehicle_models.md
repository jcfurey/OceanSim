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

## Platforms

| Platform | Thrusters | Net buoyancy (fresh water) | Surge fwd / rev | Sway | Heave up / down | Yaw |
|---|---|---|---|---|---|---|
| `bluerov2` | 6 (pitch uncontrollable) | +1.0 N | 133 N → 0.92 m/s / 104 N → 0.81 m/s | 104 N → 0.69 m/s | 94 N → 0.62 m/s / 73 N → 0.54 m/s | 24 N·m → 4.0 rad/s |
| `bluerov2_heavy` | 8 | −1.0 N | 133 N → 0.92 m/s / 104 N → 0.81 m/s | 104 N → 0.69 m/s | 188 N → 0.91 m/s / 147 N → 0.80 m/s | 28 N·m → 4.3 rad/s |
| `deeptrekker_revolution` | 6 (pitch uncontrollable) | +1.3 N | 118 N → 1.54 m/s (3 kn) | 118 N → 1.03 m/s (2 kn) | 118 N → 1.54 m/s | 48 N·m → 6.4 rad/s |

Each cell gives the maximum thrust in that direction and the steady speed it
reaches. `tests/test_platforms.py` checks these figures.

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

### Deep Trekker REVOLUTION (estimated)

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
| `water_density` | `1000` | kg/m³ for buoyancy (≈1025 in seawater, which makes the BlueROV2 Heavy positive) |
| `water_surface_z` | `1.43389` | Water surface height for partial buoyancy (also the barometer's) |

In the GUI the model is on for every platform that has one, in fresh water.

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

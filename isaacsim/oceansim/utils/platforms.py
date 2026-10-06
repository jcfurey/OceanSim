"""Vehicle platform registry for OceanSim.

Bringing a vehicle into the sim used to mean hardcoding one robot's USD path,
rigid-body parameters, and sensor mount poses inline -- duplicated in both the
GUI extension (``modules/SensorExample_python/ui_builder``) and the headless
runner (``standalone/oceansim_ros2``). Adding a second vehicle meant copying that
whole block.

This module turns that into data: each supported platform is a :class:`PlatformSpec`
(USD asset, dynamics, collision, spawn pose, and per-sensor mount poses). Both
entry points select a platform by name (``get_platform``) and apply its spec
uniformly, so adding a vehicle is a registry entry, not a code change.

The USD assets themselves live under the registered OceanSim asset root (see
``assets_utils.get_oceansim_assets_path``); a spec only stores the path *relative*
to that root. Ship the asset at ``<asset_root>/<usd_subpath>`` (or override the
path in config) and the platform imports by name.

Pure data + stdlib only (no Isaac/USD imports) so it is unit tested in CI.
"""

import math
import os
from collections import namedtuple
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SensorMount:
    """Mount pose of a sensor in the vehicle body frame.

    translation: (x, y, z) metres. rpy_deg: (roll, pitch, yaw) degrees, applied
    as an intrinsic XYZ Euler rotation (the convention Isaac's
    ``euler_angles_to_quat(..., degrees=True)`` expects). The consumer converts
    rpy -> quaternion; the registry stays free of any quaternion/Isaac math.
    """
    translation: tuple
    rpy_deg: tuple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ThrusterSpec:
    """One fixed thruster in the body frame (x forward, y port, z up).

    direction: unit vector of positive ("forward") thrust on the vehicle.
    max_forward / max_reverse (N): full-throttle thrust, unless the platform's
    thruster_model supplies them (e.g. "T200" from its battery voltage)."""
    position: tuple
    direction: tuple
    max_forward: float = None
    max_reverse: float = None


@dataclass(frozen=True)
class HydroSpec:
    """6-DOF hydrodynamic model (see utils.vehicle_dynamics), body frame about
    the centre of gravity (the vehicle prim's origin).

    6-vectors are (surge, sway, heave, roll, pitch, yaw), positive magnitudes:
    added_mass (kg, kg m^2), linear_damping (N s/m, N m s), quadratic_damping
    (N s^2/m^2, N m s^2). inertia: principal moments (kg m^2). cob: centre of
    buoyancy relative to the CoG (m). height: vertical extent (m), for partial
    buoyancy at the surface. sources: where the numbers come from."""
    inertia: tuple
    displaced_volume: float
    cob: tuple
    height: float
    added_mass: tuple
    linear_damping: tuple
    quadratic_damping: tuple
    thrusters: tuple
    thruster_model: str = None       # "T200" -> limits from battery_voltage
    battery_voltage: float = None
    thruster_time_constant: float = 0.06
    sources: str = ""
    # Centre of gravity relative to the prim origin (m). (0, 0, 0) for the
    # bare vehicles; payloads move it (vehicle_dynamics.resolve_hydro).
    cog: tuple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class HydroEstimate:
    """Inputs for ESTIMATING a vehicle's hydrodynamics from published specs
    when no measured model exists (vehicle_dynamics.resolve_hydro turns it into
    a HydroSpec, using the platform's mass and dimensions):

    * drag: quadratic coefficients chosen so the thrusters' maximum along each
      axis reaches the listed top speed (surge, sway, heave; m/s), with linear
      terms of ``linear_fraction`` (m/s) x the quadratic ones. Axes without a
      listed speed use ``drag_coefficients`` on the box's projected area.
      Rotational drag by strip theory from the resulting drag coefficients.
    * added mass: Lamb's ellipsoid coefficients for the bounding ellipsoid.
    * inertia: ``inertia_factor`` x a uniform box of the platform's mass.
    * displaced volume: weight x (1 + net_buoyancy) in fresh water; CoB
      ``cob_height`` above the CoG.
    * unknown thrust (thrusters with max_forward None): drag from
      ``drag_coefficients`` on all axes, then horizontal thrusters sized so
      surge reaches the listed surge speed and vertical ones so heave reaches
      the listed heave speed (or, without one, as strong as the horizontal)."""
    thrusters: tuple
    top_speeds: tuple = (None, None, None)
    drag_coefficients: tuple = (1.0, 1.0, 1.0)
    linear_fraction: float = 0.1
    net_buoyancy: float = 0.005
    cob_height: float = 0.02
    inertia_factor: float = 0.6
    thruster_model: str = None
    battery_voltage: float = None
    thruster_time_constant: float = 0.08
    sources: str = ""


def vectored_thrusters(x, y, z=0.0, max_forward=None, max_reverse=None, angle_deg=45.0):
    """Four horizontal thrusters at (+-x, +-y, z) in the usual vectored-X
    layout (BlueROV2-style), each angled ``angle_deg`` off the x axis: each
    one's forward thrust has a +x component, so surge uses all four in their
    forward direction. max_forward None = unknown (estimated from speed)."""
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    rev = max_forward if max_reverse is None else max_reverse
    return (ThrusterSpec((x, -y, z), (c, s, 0.0), max_forward, rev),
            ThrusterSpec((x, y, z), (c, -s, 0.0), max_forward, rev),
            ThrusterSpec((-x, -y, z), (c, -s, 0.0), max_forward, rev),
            ThrusterSpec((-x, y, z), (c, s, 0.0), max_forward, rev))


def forward_thrusters(x, y, z=0.0, max_forward=None, max_reverse=None):
    """Two horizontal thrusters at (x, +-y, z) pointing straight ahead
    (surge + yaw only; DTG3 / VideoRay Pro style)."""
    rev = max_forward if max_reverse is None else max_reverse
    return (ThrusterSpec((x, -y, z), (1.0, 0.0, 0.0), max_forward, rev),
            ThrusterSpec((x, y, z), (1.0, 0.0, 0.0), max_forward, rev))


def vector_angle_from_thrust(forward, lateral, reverse=None):
    """Vector angle (deg) of a 4-thruster vectored-X layout from its published
    forward and lateral thrust. Pure sway runs two thrusters forward and two in
    reverse at equal magnitude, so it is bounded by the weaker direction:
    lateral = 4 f_min sin(a) and forward = 4 f_fwd cos(a), i.e.
    tan(a) = lateral / min(forward, reverse) in thrust units."""
    rev = forward if reverse is None else reverse
    return math.degrees(math.atan2(lateral, min(forward, rev)))


def vertical_thrusters(points, max_forward=None, max_reverse=None):
    """Vertical thrusters (forward thrust pushes up) at the given (x, y, z)."""
    rev = max_forward if max_reverse is None else max_reverse
    return tuple(ThrusterSpec(tuple(p), (0.0, 0.0, 1.0), max_forward, rev) for p in points)


@dataclass(frozen=True)
class PlatformSpec:
    """Everything needed to import a vehicle and place its sensors."""
    name: str
    usd_subpath: str                 # relative to the OceanSim asset root (None: no
                                     # asset -- the sim imports a generated URDF)
    mass: float                      # kg (in-air mass)
    linear_damping: float            # PhysX linear damping: water-drag proxy, used
    angular_damping: float           # only when the hydrodynamic model is off
    collision_approximation: str     # 'boundingCube' | 'convexHull' | 'convexDecomposition' | ...
    spawn_translation: tuple         # (x, y, z) initial world position
    sonar_mount: SensorMount
    camera_mount: SensorMount
    dvl_mount: SensorMount
    description: str = ""
    # ROS robot description (URDF) for this platform, relative to the asset root.
    # Published on /robot_description so robot_state_publisher / RViz can consume
    # the model. Optional -- None if the platform ships no URDF.
    urdf_subpath: str = None
    # Hydrodynamics + thrusters (utils.vehicle_dynamics): a measured HydroSpec
    # or a HydroEstimate. None -> the PhysX damping proxy above.
    hydro: object = None
    # Overall length x width x height (m): the generated URDF's hull box and the
    # basis of estimated hydrodynamics.
    dimensions: tuple = None
    manufacturer: str = ""
    # Mount poses for payload kinds other than sonar / camera / dvl (e.g.
    # "altimeter", "gripper", "usbl", "light"); missing kinds get a default
    # from the hull dimensions (see mount()).
    extra_mounts: dict = field(default_factory=dict)
    # Payload catalogue names (utils.payloads) commonly fitted to this vehicle,
    # and the ones fitted by default (its standard configuration).
    payload_options: tuple = ()
    default_payloads: tuple = ()
    # Built-in camera's horizontal field of view underwater (deg); None keeps
    # the simulator's default lens.
    camera_hfov_deg: float = None

    def usd_path(self, asset_root):
        """Absolute USD path for this platform under ``asset_root`` (None if the
        platform ships no USD asset)."""
        if not self.usd_subpath:
            return None
        return os.path.join(asset_root, self.usd_subpath)

    def mount(self, kind):
        """SensorMount for payload ``kind`` in the body frame."""
        if kind == "sonar":
            return self.sonar_mount
        if kind == "camera":
            return self.camera_mount
        if kind == "dvl":
            return self.dvl_mount
        if kind in self.extra_mounts:
            return self.extra_mounts[kind]
        length, width, height = self.dimensions or (0.4, 0.3, 0.25)
        defaults = {
            # beam along the mount's -z, i.e. straight down (like the DVL beams)
            "altimeter": SensorMount((0.25 * length, 0.0, -0.5 * height)),
            "gripper": SensorMount((0.5 * length, 0.0, -0.4 * height)),
            "usbl": SensorMount((-0.35 * length, 0.0, 0.5 * height)),
            "light": SensorMount((0.5 * length, 0.0, 0.25 * height)),
            "laser": SensorMount((0.5 * length, 0.0, 0.0)),
            "skid": SensorMount((0.0, 0.0, -0.5 * height)),
        }
        return defaults.get(kind, SensorMount((0.0, 0.0, 0.0)))

    def urdf_path(self, asset_root):
        """Absolute URDF path under ``asset_root``, or None if this platform has
        no registered robot description."""
        if not self.urdf_subpath:
            return None
        return os.path.join(asset_root, self.urdf_subpath)


_S = 0.7071067811865476   # cos 45 deg

# --- BlueROV2 ----------------------------------------------------------------
# Hydrodynamics: von Benzon et al. 2022, "An Open-Source Benchmark Simulator:
# Control of a BlueROV2 Underwater Robot", J. Mar. Sci. Eng. 10, 1898, Table A1
# (BlueROV2 Heavy): damping identified from pool experiments (tether effects
# could not be fully removed, so drag errs high), added mass from the Eidsvik /
# DNV method, inertia from CAD, CoB 1 cm above the CoG. Thrust: Blue Robotics
# T200 datasheet at the 14.8 V battery's nominal voltage (47 N fwd / 37 N rev).
# With these, full surge is ~0.9 m/s (Blue Robotics quotes 1.5 m/s; von Benzon
# measured 0.72 m/s at their lower thrust). Body frame here is x fwd / y port /
# z up (the paper's NED values with y and z negated).
_BROV_DAMPING_LIN = (13.7, 0.0, 33.0, 0.0, 0.8, 0.0)
_BROV_DAMPING_QUAD = (141.0, 217.0, 190.0, 1.19, 0.47, 1.5)
_BROV_ADDED_MASS = (6.36, 7.12, 18.68, 0.189, 0.135, 0.222)

# Standard 6-thruster frame: thruster poses from the bluerov2_gz Gazebo model
# (github.com/clydemcqueen/bluerov2_gz); four vectored at 45 deg, two vertical.
# Pitch is not controllable. Mass 11.5 kg (Blue Robotics: 11-12 kg with ballast
# and battery); inertia = bluerov2_gz's (10 kg) scaled to 11.5 kg; displaced
# volume trims it ~1 N positive in fresh water, as Blue Robotics recommends.
_BLUEROV2_HYDRO = HydroSpec(
    inertia=(0.114, 0.205, 0.309),
    displaced_volume=0.0116,
    cob=(0.0, 0.0, 0.01),
    height=0.254,
    added_mass=_BROV_ADDED_MASS,
    linear_damping=_BROV_DAMPING_LIN,
    quadratic_damping=_BROV_DAMPING_QUAD,
    thrusters=(
        ThrusterSpec((0.14, -0.092, 0.0), (_S, _S, 0.0)),
        ThrusterSpec((0.14, 0.092, 0.0), (_S, -_S, 0.0)),
        ThrusterSpec((-0.14, -0.092, 0.0), (_S, -_S, 0.0)),
        ThrusterSpec((-0.14, 0.092, 0.0), (_S, _S, 0.0)),
        ThrusterSpec((0.0, -0.109, 0.077), (0.0, 0.0, 1.0)),
        ThrusterSpec((0.0, 0.109, 0.077), (0.0, 0.0, 1.0)),
    ),
    thruster_model="T200",
    battery_voltage=14.8,
    sources="von Benzon et al. 2022 (JMSE 10:1898) Table A1; bluerov2_gz thruster poses; "
            "Blue Robotics BlueROV2 and T200 datasheets",
)

# Heavy configuration (8 thrusters, pitch controllable): the paper's own vehicle
# -- mass 13.5 kg, displaced volume 0.0134 m^3 (about 1 N negative in fresh
# water, 2 N positive in sea water), inertia and thruster poses from Table A1.
_BLUEROV2_HEAVY_HYDRO = HydroSpec(
    inertia=(0.26, 0.23, 0.37),
    displaced_volume=0.0134,
    cob=(0.0, 0.0, 0.01),
    height=0.38,
    added_mass=_BROV_ADDED_MASS,
    linear_damping=_BROV_DAMPING_LIN,
    quadratic_damping=_BROV_DAMPING_QUAD,
    thrusters=(
        ThrusterSpec((0.156, -0.111, -0.085), (_S, _S, 0.0)),
        ThrusterSpec((0.156, 0.111, -0.085), (_S, -_S, 0.0)),
        ThrusterSpec((-0.156, -0.111, -0.085), (_S, -_S, 0.0)),
        ThrusterSpec((-0.156, 0.111, -0.085), (_S, _S, 0.0)),
        ThrusterSpec((0.12, -0.218, 0.0), (0.0, 0.0, 1.0)),
        ThrusterSpec((0.12, 0.218, 0.0), (0.0, 0.0, 1.0)),
        ThrusterSpec((-0.12, -0.218, 0.0), (0.0, 0.0, 1.0)),
        ThrusterSpec((-0.12, 0.218, 0.0), (0.0, 0.0, 1.0)),
    ),
    thruster_model="T200",
    battery_voltage=14.8,
    sources="von Benzon et al. 2022 (JMSE 10:1898) Table A1; Blue Robotics T200 datasheet",
)

# The spawn pose, collision and sensor mounts reproduce the values hardcoded in
# the GUI/runner before the registry existed. The in-air mass was 5.0 kg there;
# it is the real 11.5 kg now that the vehicle has a hydrodynamic model.
_BLUEROV2 = PlatformSpec(
    name="bluerov2",
    usd_subpath=os.path.join("Bluerov", "BROV_low.usd"),
    mass=11.5,
    linear_damping=10.0,
    angular_damping=10.0,
    collision_approximation="boundingCube",
    spawn_translation=(-2.0, 0.0, -0.8),
    sonar_mount=SensorMount((0.3, 0.0, 0.3), (0.0, 45.0, 0.0)),
    camera_mount=SensorMount((0.3, 0.0, 0.1)),
    dvl_mount=SensorMount((0.0, 0.0, -0.1)),
    description="Blue Robotics BlueROV2 (small observation-class ROV).",
    urdf_subpath=os.path.join("Bluerov", "bluerov2.urdf"),
    hydro=_BLUEROV2_HYDRO,
    dimensions=(0.457, 0.338, 0.254),
    manufacturer="Blue Robotics",
    camera_hfov_deg=110.0,       # BlueROV2 page: 110 deg horizontal underwater
    default_payloads=("lumen_light_pair",),
    payload_options=("ping360", "ping2", "oculus_m750d", "oculus_m1200d", "waterlinked_a50",
                     "waterlinked_a125", "newton_gripper", "lumen_light_pair"),
)

_BLUEROV2_HEAVY = PlatformSpec(
    name="bluerov2_heavy",
    usd_subpath=_BLUEROV2.usd_subpath,     # same asset: the heavy kit is not modelled visually
    mass=13.5,
    linear_damping=_BLUEROV2.linear_damping,
    angular_damping=_BLUEROV2.angular_damping,
    collision_approximation=_BLUEROV2.collision_approximation,
    spawn_translation=_BLUEROV2.spawn_translation,
    sonar_mount=_BLUEROV2.sonar_mount,
    camera_mount=_BLUEROV2.camera_mount,
    dvl_mount=_BLUEROV2.dvl_mount,
    description="Blue Robotics BlueROV2 Heavy (8 thrusters, controllable pitch).",
    urdf_subpath=_BLUEROV2.urdf_subpath,
    hydro=_BLUEROV2_HEAVY_HYDRO,
    dimensions=(0.46, 0.58, 0.38),
    manufacturer="Blue Robotics",
    camera_hfov_deg=110.0,
    default_payloads=("lumen_light_pair",),
    payload_options=_BLUEROV2.payload_options + ("tritech_gemini_720is",),
)

# DeepTrekker REVOLUTION: 26 kg in air, 717 x 440 x 235 mm, 6 thrusters (two
# vertical, four horizontal vectored), 305 m rated (Deep Trekker REVOLUTION spec
# sheet; the current product page lists 33 kg for the newer build).
#
# No hydrodynamic data is published, so the model is an ESTIMATE:
# * thrust: third-party listings (Geo-matching / NauticExpo) give 12 kgf in each
#   axis and top speeds of 3 kn forward, 2 kn lateral, 3 kn vertical. Four 45 deg
#   horizontal thrusters of 41.6 N and two vertical of 58.9 N give 12 kgf per axis.
# * drag: quadratic coefficients chosen so that thrust reaches those speeds, with
#   linear terms of 0.1 m/s x the quadratic ones. Implied drag coefficients:
#   0.90 surge, 1.20 sway (plausible), 0.29 heave (low for the 0.32 m^2 plan
#   area -- the listed 3 kn vertical is probably optimistic). Rotational drag by
#   strip theory from those coefficients (K = rho Cd_z L W^4 / 64, M = rho Cd_z
#   W L^4 / 64, N = rho Cd_y H L^4 / 64); on the BlueROV2 Heavy the same
#   formulas land within 0.6-1.5x of the measured values. With the estimated
#   thruster lever arms this gives a fast top yaw rate (~6 rad/s at full
#   thrust); real pilots / controllers limit rates well below that.
# * added mass: Lamb's ellipsoid coefficients for the bounding ellipsoid,
#   scaled to the displaced volume (vehicle_dynamics.ellipsoid_added_mass).
# * inertia: 0.6 x a uniform 26 kg box (the BlueROV2 model's ratio -- mass is
#   concentrated near the middle); CoB 2 cm above the CoG and +0.5% net
#   buoyancy, both assumed. Thruster positions estimated from the hull.
# Per run, robot.drag_scale / robot.mass override the drag and mass (see the runner).
#
# The sensor mounts are taken from the ROS-stack URDF (description_platform_
# deeptrekker_revolution + settings_erdc/urdf/sensors.urdf.xacro), expressed in
# base_link at pivot_head angle 0, so the sim places sensors where
# robot_state_publisher's TF tree expects them:
#   sonar:  base->pivot_head (0.215,0,0) + Oculus mount (0.0625,0,0.04, pitch 30)
#           = (0.2775,0,0.04) pitch 30. NOTE the real sonar rides the articulated
#           pivot_head (revolute joint); this static mount is exact only at angle 0.
#   dvl:    Waterlinked A50 at (-0.209,0,-0.06).
#   camera: child of the sonar connection link, ~(0.2675,0,0.0227).
# These act as FALLBACKS: when the robot is imported from revolution.urdf the
# runner reads each mount from the URDF directly (urdf_parse). The sonar/dvl
# frames are slashed (sonar0/..., dvl0/...); urdf_parse.find_link resolves them
# live via namespaced-leaf matching (e.g. the "optical_frame" sonar candidate ->
# sonar0/optical_frame), and the sonar additionally anchors under its real
# articulated parent (pivot_head). These values are used only when a live match
# is missing or ambiguous.
_DEEPTREKKER_REVOLUTION = PlatformSpec(
    name="deeptrekker_revolution",
    usd_subpath=os.path.join("DeepTrekker", "revolution.usd"),
    mass=26.0,
    linear_damping=15.0,
    angular_damping=15.0,
    collision_approximation="boundingCube",
    spawn_translation=(-2.0, 0.0, -0.8),
    sonar_mount=SensorMount((0.2775, 0.0, 0.04), (0.0, 30.0, 0.0)),
    camera_mount=SensorMount((0.2675, 0.0, 0.0227)),
    dvl_mount=SensorMount((-0.209, 0.0, -0.06)),
    description="Deep Trekker REVOLUTION (mid-size inspection ROV, 26 kg, 6 thrusters).",
    urdf_subpath=os.path.join("DeepTrekker", "revolution.urdf"),
    dimensions=(0.717, 0.44, 0.235),
    manufacturer="Deep Trekker",
    payload_options=("oculus_m750d", "oculus_m1200d", "oculus_c550d", "waterlinked_a50"),
    hydro=HydroEstimate(
        thrusters=vectored_thrusters(0.25, 0.16, 0.0, 41.62)
        + vertical_thrusters([(0.0, -0.17, 0.0), (0.0, 0.17, 0.0)], 58.86),
        top_speeds=(3 * 0.514444, 2 * 0.514444, 3 * 0.514444),
        net_buoyancy=0.005,
        cob_height=0.02,
        thruster_time_constant=0.1,
        sources="Deep Trekker REVOLUTION spec sheet (mass, size, thruster count); "
                "third-party listings (12 kgf / 3 kn); estimates as documented above",
    ),
)



# ---------------------------------------------------------------------------
# More vehicles. None of these makers publish hydrodynamic data, so every one
# is a HydroEstimate (see HydroEstimate for the method) from the published
# mass, size, thrust and speed; thruster positions are placed on the hull and
# are estimates. They have no 3D asset: the sim imports a URDF generated from
# this data (urdf_export), with primitive shapes.
# ---------------------------------------------------------------------------
_KN = 0.514444            # m/s per knot
_KGF = 9.80665            # N per kgf


def _std_mounts(length, height, sonar_pitch=15.0):
    """Sonar high at the bow pitched down, camera at the bow, DVL underneath."""
    return dict(sonar_mount=SensorMount((0.42 * length, 0.0, 0.3 * height), (0.0, sonar_pitch, 0.0)),
                camera_mount=SensorMount((0.48 * length, 0.0, 0.0)),
                dvl_mount=SensorMount((-0.2 * length, 0.0, -0.5 * height)))


def _estimated(name, mass, dims, hydro, description, manufacturer, payload_options=(),
               default_payloads=(), camera_hfov_deg=None, **mounts):
    return PlatformSpec(
        name=name, usd_subpath=None, mass=mass, linear_damping=10.0, angular_damping=10.0,
        collision_approximation="boundingCube", spawn_translation=(-2.0, 0.0, -0.8),
        description=description, hydro=hydro, dimensions=dims, manufacturer=manufacturer,
        payload_options=payload_options, default_payloads=default_payloads,
        camera_hfov_deg=camera_hfov_deg, **(mounts or _std_mounts(dims[0], dims[2])))


# Deep Trekker DTG3: 8.5 kg, 279 x 325 x 258 mm, 200 m (DT spec sheet); two main
# thrusters + an optional rear vertical "precision thruster"; 2.5 kgf forward /
# backward / vertical and 2.5 kn forward / up / down (geo-matching listing).
# The real vehicle climbs and dives by pitching its body (patented pitch
# system); here a vertical thruster at the centre provides the listed vertical
# thrust instead. No lateral thruster: sway is uncontrollable.
_DTG3 = _estimated(
    "deeptrekker_dtg3", 8.5, (0.279, 0.325, 0.258),
    HydroEstimate(
        thrusters=forward_thrusters(-0.05, 0.14, 0.0, 2.5 * _KGF / 2)
        + vertical_thrusters([(0.0, 0.0, 0.0)], 2.5 * _KGF),
        top_speeds=(2.5 * _KN, None, 2.5 * _KN),
        sources="Deep Trekker DTG3 spec sheet (mass, size); geo-matching DTG3 listing "
                "(2.5 kgf, 2.5 kn); estimates"),
    "Deep Trekker DTG3 (compact 3-thruster ROV, 8.5 kg; vertical by pitching in reality).",
    "Deep Trekker", payload_options=("oculus_c550d", "oculus_c550d_hf"))

# Deep Trekker PIVOT: 23.6 kg (DT spec sheet; other sources 21.7 and 16.8 kg),
# 576 x 360 x 310 mm, 305 m, six vectored thrusters (layout not published --
# REVOLUTION-like 4 vectored + 2 vertical assumed). Thrust not published; 2 kn
# forward, 1 kn vertical (geo-matching listing), so thrusters are sized to
# reach those speeds with drag coefficient 1.
_PIVOT = _estimated(
    "deeptrekker_pivot", 23.6, (0.576, 0.36, 0.31),
    HydroEstimate(
        thrusters=vectored_thrusters(0.2, 0.14, 0.0)
        + vertical_thrusters([(0.0, -0.15, 0.0), (0.0, 0.15, 0.0)]),
        top_speeds=(2.0 * _KN, None, 1.0 * _KN),
        sources="Deep Trekker PIVOT spec sheet (mass, size, 6 thrusters); geo-matching listing "
                "(2 kn / 1 kn); thrust and layout estimated"),
    "Deep Trekker PIVOT (mid-size ROV with pivoting tool platform, 6 thrusters).",
    "Deep Trekker", payload_options=("oculus_m1200d", "oculus_c550d", "waterlinked_a50"))

# VideoRay Mission Specialist Pro 5 (successor to the Pro 4): 11.8 kg (datasheet;
# product page 10 kg), 515 x 330 x 257 mm, 300 m; 2 horizontal + 1 vertical
# thrusters; forward 20.3 kgf, reverse 13.0 kgf; 4.4 kn forward, 0.8 m/s vertical
# (VideoRay Pro 5 datasheet 2025). The listed 1.4 kg "lift" is ambiguous, so the
# vertical thruster is sized to the 0.8 m/s.
_PRO5 = _estimated(
    "videoray_pro5", 11.8, (0.515, 0.33, 0.257),
    HydroEstimate(
        thrusters=forward_thrusters(-0.15, 0.13, 0.0, 20.3 * _KGF / 2, 13.0 * _KGF / 2)
        + vertical_thrusters([(0.0, 0.0, 0.0)]),
        top_speeds=(4.4 * _KN, None, 0.8),
        sources="VideoRay Mission Specialist Pro 5 datasheet (AV_Pro5_Datasheet_250825)"),
    "VideoRay Mission Specialist Pro 5 (portable 3-thruster inspection ROV).",
    "VideoRay", payload_options=("oculus_m750d", "oculus_m1200d", "tritech_gemini_720im",
                                 "tritech_gemini_720is", "videoray_pro5_manipulator"))

# VideoRay Mission Specialist Defender: 17.2 kg, 711 x 394 x 238 mm (2025
# datasheet); 4 vectored horizontal + 3 vertical (2 forward, 1 aft) thrusters.
# Thrust forward 23.6 kgf (2025; 2022 sheet 26.7), reverse 15.0, lateral 8.6,
# vertical up 23.1 / down 12.9 kgf; 3.8 kn forward, 0.9 kn sway, 0.8 m/s
# vertical. The vector angle (not published) follows from lateral vs reverse
# thrust: 30 deg. Aft vertical thruster placed so the three balance in pitch.
_DEF_ANGLE = vector_angle_from_thrust(23.6, 8.6, 15.0)
_DEF_COS = math.cos(math.radians(_DEF_ANGLE))
_DEFENDER = _estimated(
    "videoray_defender", 17.2, (0.711, 0.394, 0.238),
    HydroEstimate(
        thrusters=vectored_thrusters(0.25, 0.15, 0.0, 23.6 * _KGF / (4 * _DEF_COS),
                                     15.0 * _KGF / (4 * _DEF_COS), _DEF_ANGLE)
        + vertical_thrusters([(0.15, -0.14, 0.0), (0.15, 0.14, 0.0), (-0.3, 0.0, 0.0)],
                             23.1 * _KGF / 3, 12.9 * _KGF / 3),
        top_speeds=(3.8 * _KN, 0.9 * _KN, 0.8),
        sources="VideoRay Defender datasheets (2025 commercial, 2022 MSS) and manual"),
    "VideoRay Mission Specialist Defender (modular 7-thruster ROV).",
    "VideoRay", payload_options=("oculus_m750d", "oculus_m1200d", "blueview_m900",
                                 "tritech_gemini_720is", "nortek_dvl500",
                                 "videoray_rotating_manipulator"))

# Chasing M2 Pro Max: ~8 kg, 608 x 294 x 196 mm, 200 m; 8 vectored thrusters
# (layout not published -- 4 vectored horizontal + 4 vertical assumed, giving
# the advertised free pitch / roll). "Load" forward / upward / sideways 5.7 /
# 4.0 / 3.6 kgf (Chasing spec page), taken as thrust; vector angle from forward
# vs sideways: 32 deg. 1.5 m/s (3 kn) forward (dealer; original M2 spec).
# (For every estimated vehicle the top yaw rate at full thrust is fast --
# rotational drag scales with L^4 -- with the same tip-speed / sway-speed ratio
# as the measured BlueROV2; real vehicles limit yaw rate in their controllers.)
_M2_ANGLE = vector_angle_from_thrust(5.7, 3.6)
_M2_PRO_MAX = _estimated(
    "chasing_m2_pro_max", 8.0, (0.608, 0.294, 0.196),
    HydroEstimate(
        thrusters=vectored_thrusters(0.22, 0.11, 0.0,
                                     5.7 * _KGF / (4 * math.cos(math.radians(_M2_ANGLE))),
                                     angle_deg=_M2_ANGLE)
        + vertical_thrusters([(0.18, -0.12, 0.0), (0.18, 0.12, 0.0),
                              (-0.18, -0.12, 0.0), (-0.18, 0.12, 0.0)], 4.0 * _KGF / 4),
        top_speeds=(1.5, None, None),
        sources="Chasing M2 Pro Max spec page; dealer (3 kn); layout estimated"),
    "Chasing M2 Pro Max (8-thruster prosumer / light-industrial ROV).",
    "Chasing", payload_options=("oculus_m750d", "tritech_gemini_720im", "waterlinked_a50",
                                "ping360", "chasing_grabber_arm_2"))

# QYSEA FIFISH V6 Expert: 4.6 kg, 383 x 331 x 143 mm, 100 m, 1.5 m/s (QYSEA
# store); "6 vectored thrusters" (dealer) -- layout and thrust not published:
# 4 vectored + 2 vertical assumed, sized to 1.5 m/s. The real vehicle's free
# pitch / roll is not reproduced by this layout. Camera 96 deg underwater.
_FIFISH_V6 = _estimated(
    "qysea_fifish_v6_expert", 4.6, (0.383, 0.331, 0.143),
    HydroEstimate(
        thrusters=vectored_thrusters(0.13, 0.12, 0.0)
        + vertical_thrusters([(0.0, -0.13, 0.0), (0.0, 0.13, 0.0)]),
        top_speeds=(1.5, None, None),
        sources="QYSEA FIFISH V6 Expert store page; layout and thrust estimated"),
    "QYSEA FIFISH V6 Expert (compact 6-thruster omnidirectional ROV).",
    "QYSEA", payload_options=("oculus_m750d", "qysea_2finger_arm"), camera_hfov_deg=96.0)

# Saab Seaeye Falcon (300 m): 60 kg, 1000 x 600 x 500 mm, 14 kg payload; 4
# vectored horizontal + 1 vertical thrusters; forward 42, lateral 25, vertical
# 13 kgf; > 3 kn (Saab Falcon page and brochure rev 19.3, 2024). Vector angle
# from forward vs lateral: 31 deg.
_FALCON_ANGLE = vector_angle_from_thrust(42.0, 25.0)
_FALCON = _estimated(
    "saab_seaeye_falcon", 60.0, (1.0, 0.6, 0.5),
    HydroEstimate(
        thrusters=vectored_thrusters(0.35, 0.22, 0.0,
                                     42.0 * _KGF / (4 * math.cos(math.radians(_FALCON_ANGLE))),
                                     angle_deg=_FALCON_ANGLE)
        + vertical_thrusters([(0.0, 0.0, 0.0)], 13.0 * _KGF),
        top_speeds=(3.0 * _KN, None, None),
        thruster_time_constant=0.15,
        sources="Saab Seaeye Falcon product page and brochure rev 19.3 (2024)"),
    "Saab Seaeye Falcon (electric light-work-class ROV, 60 kg).",
    "Saab Seaeye", payload_options=("tritech_gemini_720is", "blueview_m900", "nortek_dvl500",
                                    "teledyne_pathfinder"))

# Teledyne SeaBotix vLBV300 (discontinued March 2025; widely fielded): 18 kg,
# 625 x 390 x 390 mm, 300 m; 4 vectored (45 / 35 / 20 deg settings) + 2 vertical;
# forward 18.1-22.5, lateral 7.3-15.2, vertical 9 kgf; 3 kn (Teledyne brochure
# rev 3, 2016). Modelled at the 45 deg setting (18.1 forward / 15.2 lateral --
# lateral bounded by reverse thrust, so reverse = 15.2 kgf).
_VLBV_COS = math.cos(math.radians(45.0))
_VLBV300 = _estimated(
    "seabotix_vlbv300", 18.0, (0.625, 0.39, 0.39),
    HydroEstimate(
        thrusters=vectored_thrusters(0.22, 0.14, 0.0, 18.1 * _KGF / (4 * _VLBV_COS),
                                     15.2 * _KGF / (4 * _VLBV_COS))
        + vertical_thrusters([(0.0, -0.15, 0.0), (0.0, 0.15, 0.0)], 9.0 * _KGF / 2),
        top_speeds=(3.0 * _KN, None, None),
        sources="Teledyne SeaBotix brochure rev 3 (2016); EOL notice 2025"),
    "Teledyne SeaBotix vLBV300 (6-thruster inspection ROV; discontinued 2025).",
    "Teledyne SeaBotix", payload_options=("tritech_gemini_720is", "blueview_m900",
                                          "teledyne_pathfinder"))

PLATFORMS = {p.name: p for p in (
    _BLUEROV2, _BLUEROV2_HEAVY, _DEEPTREKKER_REVOLUTION, _DTG3, _PIVOT, _PRO5, _DEFENDER,
    _M2_PRO_MAX, _FIFISH_V6, _FALCON, _VLBV300)}

# Convenience aliases so common spellings resolve to a canonical platform.
_ALIASES = {
    "bluerov": "bluerov2",
    "brov": "bluerov2",
    "bluerov2heavy": "bluerov2_heavy",
    "bluerov_heavy": "bluerov2_heavy",
    "revolution": "deeptrekker_revolution",
    "deeptrekker": "deeptrekker_revolution",
    "deep_trekker_revolution": "deeptrekker_revolution",
    "dtg3": "deeptrekker_dtg3",
    "pivot": "deeptrekker_pivot",
    "pro5": "videoray_pro5",
    "videoray": "videoray_defender",
    "defender": "videoray_defender",
    "m2_pro_max": "chasing_m2_pro_max",
    "chasing": "chasing_m2_pro_max",
    "fifish": "qysea_fifish_v6_expert",
    "fifish_v6": "qysea_fifish_v6_expert",
    "falcon": "saab_seaeye_falcon",
    "seaeye_falcon": "saab_seaeye_falcon",
    "vlbv300": "seabotix_vlbv300",
}

DEFAULT_PLATFORM = "bluerov2"


def available_platforms():
    """Sorted list of canonical platform names."""
    return sorted(PLATFORMS)


def _normalize(name):
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(key, key)


def get_platform(name):
    """Look up a :class:`PlatformSpec` by name (case-insensitive, alias-aware).

    Raises ``KeyError`` naming the available platforms if ``name`` is unknown,
    rather than failing deep in the stage-loading code.
    """
    key = _normalize(name)
    if key not in PLATFORMS:
        raise KeyError(
            f"Unknown OceanSim platform {name!r}. Available platforms: "
            f"{available_platforms()} (aliases: {sorted(_ALIASES)}).")
    return PLATFORMS[key]


# How to bring a vehicle onto the stage: from a prebuilt USD (add_reference) or
# by importing a URDF (Isaac's URDF importer creates the articulation). ``kind``
# is "usd" or "urdf"; ``path`` is the absolute file.
RobotSource = namedtuple("RobotSource", ["kind", "path"])


def resolve_robot_source(asset_root=None, platform=None,
                         usd_path=None, urdf_path=None, prefer="usd"):
    """Decide what to load a vehicle from -- a prebuilt USD or a URDF to import.

    Precedence: an explicit ``usd_path`` then ``urdf_path`` (when the file
    exists) always win; otherwise the platform's own ``usd_subpath`` /
    ``urdf_subpath`` under ``asset_root`` are tried, in the order set by
    ``prefer`` (``"usd"`` default, or ``"urdf"`` if you only have / prefer a
    URDF). Only paths whose file actually exists are considered, so "I only have
    a URDF" resolves to the URDF automatically even with the default preference.

    Returns ``(RobotSource, "ok")`` or ``(None, reason)`` where reason is
    ``"none"`` (nothing configured) or ``"missing:<a>,<b>"`` (configured assets
    don't exist) -- so the caller can raise a clear error instead of feeding a
    bad path to the stage loader. Pure: only stats files, never loads them.
    """
    tried = []

    def _ok(path):
        if not path:
            return False
        tried.append(path)
        return os.path.isfile(path)

    if _ok(usd_path):
        return RobotSource("usd", usd_path), "ok"
    if _ok(urdf_path):
        return RobotSource("urdf", urdf_path), "ok"

    spec = None
    if platform is not None:
        spec = platform if isinstance(platform, PlatformSpec) else get_platform(platform)
    if spec is not None and asset_root:
        spec_usd = spec.usd_path(asset_root)
        spec_urdf = spec.urdf_path(asset_root)
        order = [("urdf", spec_urdf), ("usd", spec_usd)] if prefer == "urdf" \
            else [("usd", spec_usd), ("urdf", spec_urdf)]
        for kind, path in order:
            if _ok(path):
                return RobotSource(kind, path), "ok"

    if not tried:
        return None, "none"
    return None, "missing:" + ",".join(tried)


def resolve_robot_description(asset_root=None, platform=None, inline=None, path=None):
    """Resolve the URDF text to publish on ``/robot_description``.

    Precedence (so an explicit override always wins over the platform default):

    1. ``inline`` -- a URDF XML string supplied directly in config.
    2. ``path``   -- an explicit URDF file path.
    3. ``platform`` -- the selected platform's registered ``urdf_subpath`` under
       ``asset_root``.

    Returns ``(urdf_text, source)``. ``source`` is ``"inline"`` or the file path
    that was read. If nothing is available or the chosen file is missing, returns
    ``(None, reason)`` where reason is ``"none"`` or ``"missing:<path>"`` -- the
    caller can log it and carry on without a description rather than crashing.

    Pure apart from reading the single file that precedence selects, so it is
    unit tested without Isaac/ROS.
    """
    if inline:
        return inline, "inline"

    candidate = path
    if not candidate and platform is not None:
        spec = platform if isinstance(platform, PlatformSpec) else get_platform(platform)
        candidate = spec.urdf_path(asset_root) if asset_root else None

    if not candidate:
        return None, "none"
    if not os.path.isfile(candidate):
        return None, f"missing:{candidate}"
    with open(candidate, "r") as f:
        return f.read(), candidate

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


@dataclass(frozen=True)
class PlatformSpec:
    """Everything needed to import a vehicle and place its sensors."""
    name: str
    usd_subpath: str                 # relative to the OceanSim asset root
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
    # Hydrodynamics + thrusters (utils.vehicle_dynamics). None -> the PhysX
    # damping proxy above.
    hydro: HydroSpec = None

    def usd_path(self, asset_root):
        """Absolute USD path for this platform under ``asset_root``."""
        return os.path.join(asset_root, self.usd_subpath)

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
    hydro=HydroSpec(
        inertia=(0.3235, 0.7401, 0.92),
        displaced_volume=0.02613,
        cob=(0.0, 0.0, 0.02),
        height=0.235,
        added_mass=(4.575, 10.354, 34.246, 0.1045, 0.5211, 0.0804),
        linear_damping=(4.642, 10.135, 4.642, 0.01236, 0.05347, 0.11675),
        quadratic_damping=(46.42, 101.35, 46.42, 0.1236, 0.5347, 1.1675),
        thrusters=(
            ThrusterSpec((0.25, -0.16, 0.0), (_S, _S, 0.0), 41.62, 41.62),
            ThrusterSpec((0.25, 0.16, 0.0), (_S, -_S, 0.0), 41.62, 41.62),
            ThrusterSpec((-0.25, -0.16, 0.0), (_S, -_S, 0.0), 41.62, 41.62),
            ThrusterSpec((-0.25, 0.16, 0.0), (_S, _S, 0.0), 41.62, 41.62),
            ThrusterSpec((0.0, -0.17, 0.0), (0.0, 0.0, 1.0), 58.86, 58.86),
            ThrusterSpec((0.0, 0.17, 0.0), (0.0, 0.0, 1.0), 58.86, 58.86),
        ),
        thruster_time_constant=0.1,
        sources="Deep Trekker REVOLUTION spec sheet (mass, size, thruster count); "
                "third-party listings (12 kgf / 3 kn); estimates as documented above",
    ),
)

PLATFORMS = {
    _BLUEROV2.name: _BLUEROV2,
    _BLUEROV2_HEAVY.name: _BLUEROV2_HEAVY,
    _DEEPTREKKER_REVOLUTION.name: _DEEPTREKKER_REVOLUTION,
}

# Convenience aliases so common spellings resolve to a canonical platform.
_ALIASES = {
    "bluerov": "bluerov2",
    "brov": "bluerov2",
    "bluerov2heavy": "bluerov2_heavy",
    "bluerov_heavy": "bluerov2_heavy",
    "revolution": "deeptrekker_revolution",
    "deeptrekker": "deeptrekker_revolution",
    "deep_trekker_revolution": "deeptrekker_revolution",
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

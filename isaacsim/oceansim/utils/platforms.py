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
class PlatformSpec:
    """Everything needed to import a vehicle and place its sensors."""
    name: str
    usd_subpath: str                 # relative to the OceanSim asset root
    mass: float                      # kg (in-air mass, for PhysX inertia)
    linear_damping: float            # PhysX linear damping (water-drag proxy, tunable)
    angular_damping: float           # PhysX angular damping (water-drag proxy, tunable)
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

    def usd_path(self, asset_root):
        """Absolute USD path for this platform under ``asset_root``."""
        return os.path.join(asset_root, self.usd_subpath)

    def urdf_path(self, asset_root):
        """Absolute URDF path under ``asset_root``, or None if this platform has
        no registered robot description."""
        if not self.urdf_subpath:
            return None
        return os.path.join(asset_root, self.urdf_subpath)


# NOTE: the BlueROV2 entry reproduces the exact values that were hardcoded in the
# GUI/runner before the registry existed (mass 5.0, damping 10/10, boundingCube,
# spawn (-2, 0, -0.8), and the sonar/camera/DVL mounts) -- so selecting it is a
# behaviour-preserving refactor. test_platforms.py locks those values down.
_BLUEROV2 = PlatformSpec(
    name="bluerov2",
    usd_subpath=os.path.join("Bluerov", "BROV_low.usd"),
    mass=5.0,
    linear_damping=10.0,
    angular_damping=10.0,
    collision_approximation="boundingCube",
    spawn_translation=(-2.0, 0.0, -0.8),
    sonar_mount=SensorMount((0.3, 0.0, 0.3), (0.0, 45.0, 0.0)),
    camera_mount=SensorMount((0.3, 0.0, 0.1)),
    dvl_mount=SensorMount((0.0, 0.0, -0.1)),
    description="Blue Robotics BlueROV2 (small observation-class ROV).",
    urdf_subpath=os.path.join("Bluerov", "bluerov2.urdf"),
)

# DeepTrekker REVOLUTION: 26 kg in air, 717 x 440 x 235 mm, 6 thrusters, 305 m
# rated (Deep Trekker REVOLUTION spec sheet). Mass is the real in-air figure;
# damping is a tunable water-drag proxy (PhysX has no hydrodynamics model), set a
# bit higher than the BlueROV2 to reflect the larger, heavier frame. The mount
# poses are sensible defaults for the larger hull and should be trimmed to the
# actual USD geometry. Ship the asset at <asset_root>/DeepTrekker/revolution.usd
# (or override robot.usd_path in config).
_DEEPTREKKER_REVOLUTION = PlatformSpec(
    name="deeptrekker_revolution",
    usd_subpath=os.path.join("DeepTrekker", "revolution.usd"),
    mass=26.0,
    linear_damping=15.0,
    angular_damping=15.0,
    collision_approximation="boundingCube",
    spawn_translation=(-2.0, 0.0, -0.8),
    sonar_mount=SensorMount((0.35, 0.0, 0.1), (0.0, 45.0, 0.0)),
    camera_mount=SensorMount((0.35, 0.0, 0.0)),
    dvl_mount=SensorMount((0.0, 0.0, -0.12)),
    description="Deep Trekker REVOLUTION (mid-size inspection ROV, 26 kg, 6 thrusters).",
    urdf_subpath=os.path.join("DeepTrekker", "revolution.urdf"),
)

PLATFORMS = {
    _BLUEROV2.name: _BLUEROV2,
    _DEEPTREKKER_REVOLUTION.name: _DEEPTREKKER_REVOLUTION,
}

# Convenience aliases so common spellings resolve to a canonical platform.
_ALIASES = {
    "bluerov": "bluerov2",
    "brov": "bluerov2",
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

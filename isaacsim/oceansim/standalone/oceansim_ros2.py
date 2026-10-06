#!/usr/bin/env python3
"""Headless OceanSim + ROS2 runner.

The stock OceanSim ``SensorExample`` is an interactive Isaac Sim UI extension
(``isaacsim/oceansim/modules/SensorExample_python``).  This script reproduces
the same scene assembly and physics loop as a *standalone, headless* Isaac Sim
application so it can be launched from a ROS2 launch file the same way the
HoloOcean ``holoocean_node`` is.  It:

1. boots a (headless) ``SimulationApp`` and activates ``isaacsim.ros2.bridge``
   so ``rclpy`` is importable inside Isaac Sim;
2. builds the underwater scene + BlueROV exactly as
   ``SensorExample_python/ui_builder.py:_setup_scene`` does (or references a
   user-supplied USD scene);
3. runs OceanSim's ``MHL_Sensor_Example_Scenario`` in ``"ROS control"`` mode so
   the vehicle is driven by ``/oceansim/robot/vel_cmd`` (Twist) /
   ``/oceansim/robot/force_cmd`` (Wrench);
4. publishes ``/clock`` + vehicle state + sensors through
   :class:`isaacsim.oceansim.utils.ros2_sensors.OceanSimSensorPublisher`.

Because Isaac Sim must be running for any of this to import, the script is
launched via the Isaac Sim python environment, e.g.::

    $ISAAC_SIM_ROOT/python.sh oceansim_ros2.py --config scenario.json

See ``OceanSim/scripts/run_oceansim_ros2.sh`` for the wrapper used by the ROS2
workspace bringup.

Configuration is taken from a JSON file (``--config``) and/or CLI flags; CLI
flags win.  Example config::

    {
      "headless": true,
      "renderer": "RayTracedLighting",
      "physics_dt": 0.0166667,
      "rendering_dt": 0.0166667,
      "control_mode": "ROS control",
      "scene_usd": "",
      "asset_path": "/path/to/oceansim/assets",
      "platform": "bluerov2",
      "robot": {"mass": 5.0, "urdf_path": "/path/to/robot.urdf"},
      "sensors": {"sonar": true, "camera": true, "dvl": true, "baro": true},
      "publisher": {"sonar_frame_id": "sonar0/optical_frame"}
    }

``platform`` selects a vehicle from ``utils.platforms`` (``bluerov2`` or
``deeptrekker_revolution``); its spec supplies the USD, dynamics, sensor mounts
and default URDF, each overridable via ``robot``. The platform URDF is latched on
``/robot_description`` and the articulation's joints are published on
``/joint_states`` (and driven from ``/oceansim/robot/joint_command``) so
robot_state_publisher / RViz get a fully articulated model.
"""

import argparse
import json
import os
import signal
import sys


def parse_args(argv):
    p = argparse.ArgumentParser(description="Headless OceanSim ROS2 runner")
    p.add_argument("--config", default=os.environ.get("OCEANSIM_CONFIG", ""),
                   help="Path to a JSON config file.")
    p.add_argument("--scene-usd", default=None,
                   help="USD scene to load instead of the default MHL scene.")
    p.add_argument("--asset-path", default=None,
                   help="OceanSim assets directory (registers asset_path.json).")
    p.add_argument("--platform", default=None,
                   help="Vehicle platform to import (e.g. 'bluerov2', "
                        "'deeptrekker_revolution'). See utils.platforms.")
    p.add_argument("--payload", dest="payloads", action="append", default=None,
                   help="Payload to fit (repeat), e.g. --payload oculus_m750d --payload "
                        "waterlinked_a50. Default: the platform's standard set. "
                        "scripts/oceansim_urdf.py list shows the catalogue.")
    p.add_argument("--no-payloads", dest="no_payloads", action="store_true",
                   help="Bare vehicle (no payloads).")
    p.add_argument("--urdf", dest="urdf", default=None,
                   help="Import the robot from this URDF (creates the articulation; "
                        "used when you have a URDF but no prebuilt USD). It is also "
                        "published on /robot_description unless --robot-description is set.")
    p.add_argument("--robot-description", dest="robot_description", default=None,
                   help="Path to a URDF to latch on /robot_description only, "
                        "without changing the imported robot.")
    p.add_argument("--control-mode", default=None,
                   choices=["No control", "Straight line", "Waypoints",
                            "Manual control", "ROS control"],
                   help="Scenario control mode (default: ROS control).")
    p.add_argument("--headless", dest="headless", action="store_true", default=None)
    p.add_argument("--no-headless", dest="headless", action="store_false")
    p.add_argument("--no-sonar", dest="sonar", action="store_false", default=None)
    p.add_argument("--sonar-backend", default=None,
                   choices=["oceansim", "rtx_acoustic"],
                   help="Sonar implementation: 'oceansim' (custom imaging sonar, "
                        "default) or 'rtx_acoustic' (Isaac native RTX acoustic, "
                        "experimental).")
    p.add_argument("--publish-static-tf", dest="publish_static_tf",
                   action="store_true", default=None,
                   help="Broadcast base_link->{sonar,camera} static TF from the "
                        "sensor mounts (for standalone runs without a stack URDF).")
    p.add_argument("--sensor-compute-rate", dest="sensor_compute_rate",
                   type=float, default=None,
                   help="Hz cap on heavy sensor compute (sonar/camera); 0 = every "
                        "physics step. Default 15.")
    p.add_argument("--no-camera", dest="camera", action="store_false", default=None)
    p.add_argument("--no-dvl", dest="dvl", action="store_false", default=None)
    p.add_argument("--no-baro", dest="baro", action="store_false", default=None)
    return p.parse_args(argv)


def load_config(args):
    cfg = {
        "headless": True,
        "renderer": "RayTracedLighting",
        "physics_dt": 1.0 / 60.0,
        "rendering_dt": 1.0 / 60.0,
        "control_mode": "ROS control",
        # ROS control. ros2_mode: "velocity control" (kinematic: sets the body
        # velocity every step), "force control" (geometry_msgs/Wrench in N /
        # N*m, body frame), "dynamic velocity control" (cmd_vel tracked by a
        # PI loop through forces, so buoyancy / drag / collisions still act --
        # see ros2_control_math.BodyVelocityPI; velocity_pi overrides its
        # gains) or "thruster control" (std_msgs/Float64MultiArray of
        # normalised per-thruster commands on thruster_topic; needs the
        # hydrodynamic model). With the model, force and dynamic velocity
        # control go through the thrusters, so their limits apply. stamped_cmd_vel subscribes TwistStamped (Nav2
        # enable_stamped_cmd_vel). command_timeout is the dead-man timeout (s).
        # max_* clamp command / wrench magnitudes (direction-preserving); None =
        # unbounded -- no repo-documented physical limits exist yet.
        "control_params": {"max_linear_vel": None, "max_angular_vel": None,
                           "max_force": None, "max_torque": None,
                           "ros2_mode": "velocity control",
                           "vel_topic": "/oceansim/robot/vel_cmd",
                           "force_topic": "/oceansim/robot/force_cmd",
                           "thruster_topic": "/oceansim/robot/thruster_cmd",
                           "stamped_cmd_vel": False, "command_timeout": 2.0,
                           "velocity_pi": {}},
        "scene_usd": "",
        "asset_path": "",
        # Vehicle platform (utils.platforms). Its spec provides the USD, mass,
        # damping, collision, spawn pose, sensor mounts, and default URDF. The
        # optional "robot" dict overrides individual fields (e.g. mass,
        # translation, usd_path, urdf_path, robot_description).
        "platform": "bluerov2",
        # robot: per-field overrides of the platform spec (mass, translation,
        # usd_path, urdf_path, ...). Hydrodynamics (utils.vehicle_dynamics):
        # "hydrodynamics" (default true for platforms that have a model) applies
        # drag, buoyancy, added mass and thrusters instead of the PhysX damping
        # proxy; "thruster_voltage" (T200 platforms) and "drag_scale" tune it.
        "robot": {},
        # Water density (kg/m^3) for buoyancy: 1000 fresh (the MHL tank), ~1025 sea.
        "water_density": 1000.0,
        # Payloads fitted to the vehicle (utils.payloads names, e.g.
        # ["oculus_m750d", "waterlinked_a50", "ping2", "newton_gripper"]). None =
        # the platform's standard set, [] = bare vehicle. Sensor payloads set the
        # simulated sonar / DVL / altimeter to that device's datasheet values
        # (sonar_params still override); every payload adds its mass, buoyancy and
        # drag, and the vehicle is re-trimmed (robot.trim, default true).
        # scripts/oceansim_urdf.py list shows what exists.
        "payloads": None,
        "sensors": {"sonar": True, "camera": True, "dvl": True, "baro": True},
        # Per-sensor GUI windows (sonar + UW camera viewports). False also skips
        # their per-frame set_bytes_data_from_gpu readback; AOVs still render for ROS
        # and the main Isaac viewport stays. None = auto (on with a GUI, off when
        # headless -- headless paid the readback for windows nobody can see).
        "sensor_viewports": None,
        # Waypoint file for control_mode "Waypoints" (one x y z qw qx qy qz per
        # line); required in that mode, unused otherwise.
        "waypoints_file": "",
        # Sonar backend: "oceansim" = custom imaging sonar (Camera + pointcloud
        # annotator); "rtx_acoustic" = Isaac native RTX acoustic sensor
        # (experimental, avoids the 6.0.1 pointcloud-annotator crash).
        "sonar_backend": "oceansim",
        # Imaging-sonar tuning (oceansim backend). range_res (m) + angular_res
        # (deg) set the OUTPUT image resolution (range bins x beams). hori_res is
        # the raytrace SUPERSAMPLING (vert auto = hori_res/AR); it drives compute
        # (~hori_res^2 points) but NOT output resolution -- lower it to speed up
        # the scan (and unblock odom, which shares the sim loop) without losing
        # resolution. gpu_point_filter compacts points on-device (skips a host
        # round-trip; self-heals to the numpy path if outputs aren't on-device).
        # async_compute runs the post-scan kernels + readback on a worker thread so
        # the sonar doesn't block the sim loop (frees odom/imu); scan() stays on the
        # sim thread. render_rate caps how often the sonar CAMERA renders (the
        # dominant per-step cost): its render product is disabled on the other steps
        # so physics/odom/GUI run unblocked. <=0 = no cap (async worker throughput
        # is the real limit). With async_compute, render is also gated on the worker
        # being idle so no rendered frame is wasted.
        "sonar_params": {"hori_res": 2500, "gpu_point_filter": True,
                         "async_compute": False, "render_rate": 0.0,
                         "range_res": 0.005, "angular_res": 0.25},
        # Physics simulation device: None -> Isaac default (GPU/cuda:0). Set to
        # "cpu" to run PhysX on the CPU -- needed when the host NVIDIA driver is too
        # new for Isaac 6.0.1's bundled CUDA (e.g. driver 595.80 / CUDA 13.2 leaves
        # the GPU physics-tensor SimulationView invalid, so get_velocities fails and
        # odom/IMU/control all break). Rendering (camera/sonar/RTX) stays on the GPU.
        "physics_device": None,
        # Hz cap on the heavy per-step sensor compute (sonar scan + camera
        # UW_render). They run in update_scenario every physics step (~60 Hz) but
        # are published ~5 Hz, so the rest is wasted. 15 Hz keeps published frames
        # fresh while cutting that compute ~4x; raise it if you raise the publish
        # rates, or set 0 to compute every physics step.
        "sensor_compute_rate": 15.0,
        "publisher": {},
        # Publish base_link->{sonar,camera} static TF from the sensor mounts.
        # OFF by default: in a robot-stack deployment the URDF / robot_state_publisher
        # owns those frames. Turn ON for standalone runs so RViz / sonar_image_proc
        # have the sensor frames in the TF tree.
        "publish_static_tf": False,
        "water_surface_z": 1.43389,
    }
    if args.config:
        with open(args.config, "r") as f:
            user = json.load(f)
        _deep_update(cfg, user)

    # CLI overrides
    if args.headless is not None:
        cfg["headless"] = args.headless
    if args.scene_usd is not None:
        cfg["scene_usd"] = args.scene_usd
    if args.asset_path is not None:
        cfg["asset_path"] = args.asset_path
    if args.platform is not None:
        cfg["platform"] = args.platform
    if args.no_payloads:
        cfg["payloads"] = []
    elif args.payloads is not None:
        cfg["payloads"] = list(args.payloads)
    if args.urdf is not None:
        cfg.setdefault("robot", {})["urdf_path"] = args.urdf
    if args.robot_description is not None:
        cfg.setdefault("robot", {})["robot_description_path"] = args.robot_description
    if args.control_mode is not None:
        cfg["control_mode"] = args.control_mode
    if args.sonar_backend is not None:
        cfg["sonar_backend"] = args.sonar_backend
    if args.publish_static_tf is not None:
        cfg["publish_static_tf"] = args.publish_static_tf
    if args.sensor_compute_rate is not None:
        cfg["sensor_compute_rate"] = args.sensor_compute_rate
    for key, val in (("sonar", args.sonar), ("camera", args.camera),
                     ("dvl", args.dvl), ("baro", args.baro)):
        if val is not None:
            cfg["sensors"][key] = val
    return cfg


def _deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def maybe_register_assets(asset_path):
    """Write OceanSim's asset_path.json when a path is supplied. An explicit
    --asset-path / config asset_path is honored even if a (possibly stale)
    asset_path.json already exists -- the old code returned early on any existing
    file, so an explicit override was silently ignored after the first run."""
    if not asset_path:
        return
    import isaacsim.oceansim.utils as _utils_pkg
    json_path = os.path.join(os.path.dirname(_utils_pkg.__file__), "asset_path.json")
    abspath = os.path.abspath(asset_path)
    if os.path.isfile(json_path):
        try:
            with open(json_path) as f:
                existing = json.load(f).get("asset_path")
        except Exception:  # noqa: BLE001 - malformed json -> rewrite it
            existing = None
        if existing == abspath:
            return
        print(f"[oceansim_ros2] overriding asset path {existing} -> {abspath}")
    with open(json_path, "w") as f:
        json.dump({"asset_path": abspath}, f, indent=2)
    print(f"[oceansim_ros2] registered asset path -> {abspath}")


def main(argv):
    args = parse_args(argv)
    cfg = load_config(args)

    # 1) Boot Isaac Sim FIRST -- nothing from omni/isaacsim is importable before this.
    # The RTX acoustic sensor needs Motion BVH (/renderer/raytracingMotion) so the
    # raytracer handles the moving robot/sonar geometry. It MUST be set as a BOOT
    # setting, not at runtime: the old RtxAcousticSensor.sonar_initialize flipped
    # /renderer/raytracingMotion/enabled mid-run, which forces a hydra-engine
    # reconfiguration. With the Kit viewport + sonar render products both live, that
    # reconfiguration fails to spawn new engine threads (deviceMask 0) -> the viewport
    # AND the sonar silently stop rendering (thousands of "failed to create Hydra
    # Engine thread for viewport" warnings + empty GMO frames). Passing it as a kit
    # arg here boots the renderer with Motion BVH already on, so nothing reconfigures.
    import sys as _sys
    if cfg.get("sonar_backend") == "rtx_acoustic":
        _sys.argv += ["--/renderer/raytracingMotion/enabled=True"]
    from isaacsim import SimulationApp
    sim_app = SimulationApp({
        "headless": bool(cfg["headless"]),
        "renderer": cfg["renderer"],
    })

    # Ensure the kit process is closed on ANY exit. The run loop's finally
    # covers the loop, but an exception in the ~200 setup lines between here and
    # the loop (asset resolution, robot import, sensor init, publisher init)
    # used to leave the SimulationApp -- a full kit process with a GPU context --
    # hanging. The idempotent guard means the normal finally-path close and this
    # atexit hook can't double-close.
    import atexit

    _app_closed = {"done": False}

    def _close_sim_app():
        if not _app_closed["done"]:
            _app_closed["done"] = True
            try:
                sim_app.close()
            except Exception as _e:  # noqa: BLE001
                print(f"[oceansim_ros2] sim_app.close warning: {_e}")

    atexit.register(_close_sim_app)

    # 2) ROS2 bridge must be enabled before OceanSim imports rclpy.
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.ros2.bridge")
    sim_app.update()

    # 3) Asset registration must happen before importing assets_utils (it
    #    validates the path at import time).
    maybe_register_assets(cfg["asset_path"])

    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.semantics import add_labels  # Isaac 6.0.1 renamed add_update_semantics -> add_labels
    from isaacsim.core.prims import SingleRigidPrim, SingleGeometryPrim
    from pxr import PhysxSchema

    # Isaac 6.0.1 turns `isaacsim` into a regular package with a fixed __path__
    # (['.../python_packages/isaacsim']), so the OceanSim dir on PYTHONPATH is NOT
    # merged into the namespace and `import isaacsim.oceansim` fails. Splice the
    # OceanSim package dir ($OCEANSIM_ROOT/isaacsim) into isaacsim.__path__ so the
    # submodule resolves. (Inside Isaac, OceanSim is normally a kit extension; the
    # standalone runner has to wire it up by hand.)
    import importlib
    import isaacsim as _isaacsim_pkg
    _oceansim_ns = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))  # -> $OCEANSIM_ROOT/isaacsim
    if _oceansim_ns not in _isaacsim_pkg.__path__:
        _isaacsim_pkg.__path__.append(_oceansim_ns)
    importlib.invalidate_caches()

    # The headless runner only needs the GUI-free `scenario` module, but importing
    # it would run `SensorExample_python/__init__.py` -> `from .extension import *`,
    # which drags in the GUI `ui_builder` -- omni.ui widgets plus LoadButton/
    # ResetButton from `isaacsim.examples.extension.core_connectors` (deprecated
    # in Isaac 6.0, but still importable). omni.ui is not available in a headless
    # kit app, so executing that import chain fails regardless of LoadButton.
    # Pre-seed a stub for that package so `scenario` loads as a submodule WITHOUT
    # executing the package __init__. (modules/ has no __init__; oceansim/'s
    # __init__ only does guarded convenience re-exports, so it is safe to run.)
    import sys
    import types
    _sep_name = "isaacsim.oceansim.modules.SensorExample_python"
    if _sep_name not in sys.modules:
        _sep = types.ModuleType(_sep_name)
        _sep.__path__ = [os.path.join(_oceansim_ns, "oceansim", "modules",
                                      "SensorExample_python")]
        _sep.__package__ = _sep_name
        sys.modules[_sep_name] = _sep

    from isaacsim.oceansim.modules.SensorExample_python.scenario import (
        MHL_Sensor_Example_Scenario)
    from isaacsim.oceansim.utils.ros2_sensors import OceanSimSensorPublisher

    create_new_stage()
    _world_kwargs = dict(physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"],
                         stage_units_in_meters=1.0)
    if cfg.get("physics_device"):
        _world_kwargs["device"] = cfg["physics_device"]
        print(f"[oceansim_ros2] physics device override: {cfg['physics_device']}")
    world = World(**_world_kwargs)

    # Underwater vehicles are operated near neutral buoyancy, so the net vertical
    # force is ~0 -- we model that as NO gravity rather than gravity+buoyancy. Per-
    # body DisableGravity does NOT propagate to URDF articulation links (they sink to
    # the floor and tip), so zero the SCENE gravity globally: the robust no-gravity
    # model for the whole sim. (Per-link DisableGravity below is then belt-and-suspenders.)
    try:
        _pc = world.get_physics_context()
        try:
            _pc.set_gravity(0.0)            # scalar magnitude API
        except TypeError:
            _pc.set_gravity([0.0, 0.0, 0.0])  # vector API fallback
        print("[oceansim_ros2] scene gravity = 0 (neutral-buoyancy underwater model)")
    except Exception as exc:  # noqa: BLE001
        print(f"[oceansim_ros2] could not zero scene gravity: {exc}")

    # ---- scene (mirrors ui_builder._setup_scene) --------------------------
    if cfg["scene_usd"]:
        add_reference_to_stage(usd_path=cfg["scene_usd"], prim_path="/World/scene")
        print(f"[oceansim_ros2] loaded user scene: {cfg['scene_usd']}")
    else:
        from isaacsim.oceansim.utils.assets_utils import get_oceansim_assets_path
        assets = get_oceansim_assets_path()
        mhl_path = "/World/mhl"
        add_reference_to_stage(usd_path=assets + "/collected_MHL/mhl_scaled.usd",
                               prim_path=mhl_path)
        SingleGeometryPrim(prim_path=mhl_path, collision=True)
        add_labels(get_prim_at_path(mhl_path + "/Mesh/mesh"),
                   labels=["1.0"], instance_name="reflectivity")
        rock_path = "/World/rock"
        add_reference_to_stage(usd_path=assets + "/collected_rock/rock.usd",
                               prim_path=rock_path)
        add_labels(get_prim_at_path(rock_path + "/Mesh/mesh"),
                   labels=["2.0"], instance_name="reflectivity")
        rock = SingleGeometryPrim(prim_path=rock_path, collision=True)
        rock.set_collision_approximation("convexDecomposition")
        SingleRigidPrim(prim_path=rock_path, translation=np.array([1.0, 0.1, -1.5]),
                        orientation=euler_angles_to_quat(np.array([0.0, 0.0, 90]),
                                                         degrees=True))

    # ---- robot (selected platform from utils.platforms) -------------------
    from isaacsim.oceansim.utils.assets_utils import get_oceansim_assets_path
    from isaacsim.oceansim.utils import platforms
    assets = get_oceansim_assets_path()
    from isaacsim.oceansim.utils import payloads as payload_catalogue
    spec = platforms.get_platform(cfg.get("platform", platforms.DEFAULT_PLATFORM))
    rob_cfg = cfg.get("robot", {})
    # A URDF carrying hydrodynamics (OceanSim export, Gazebo Sim or UUV
    # Simulator plugins) defines the vehicle itself: its dynamics replace the
    # selected platform's (robot.hydro_from_urdf: false to keep the platform's).
    _urdf_override = rob_cfg.get("urdf_path")
    if _urdf_override and os.path.isfile(_urdf_override) and rob_cfg.get("hydro_from_urdf", True):
        from isaacsim.oceansim.utils import urdf_platform
        with open(_urdf_override) as _f:
            _utext = _f.read()
        if urdf_platform.hydro_source(_utext):
            spec, _notes = urdf_platform.platform_from_urdf(_utext, _urdf_override, fallback=spec)
            for _n in _notes:
                print(f"[oceansim_ros2] URDF vehicle: {_n}")
    print(f"[oceansim_ros2] platform: {spec.name} -- {spec.description}")
    fitted = payload_catalogue.select_payloads(spec, cfg.get("payloads"))
    print(f"[oceansim_ros2] payloads: {[p.name for p in fitted] or 'none'}")

    robot_path = "/World/rob"
    # Each field falls back to the platform spec unless explicitly overridden in
    # cfg["robot"]. (For bluerov2 the spec values equal the old hardcoded ones,
    # so this is behaviour-preserving.)
    use_hydro = bool(rob_cfg.get("hydrodynamics", True)) and spec.hydro is not None
    # With the hydrodynamic model the drag comes from it, not PhysX damping.
    lin_d = 0.0 if use_hydro else float(rob_cfg.get("linear_damping", spec.linear_damping))
    ang_d = 0.0 if use_hydro else float(rob_cfg.get("angular_damping", spec.angular_damping))
    spawn = np.array(rob_cfg.get("translation", spec.spawn_translation), dtype=float)

    # USD or URDF? An explicit robot.usd_path / robot.urdf_path wins, else the
    # platform's own assets are used (prefer "usd", or set robot.prefer_source
    # = "urdf"). Resolves to a URDF automatically if that's all that exists.
    _prefer = rob_cfg.get("prefer_source", "usd")
    src, why = (None, "generated") if _prefer == "generated" else platforms.resolve_robot_source(
        asset_root=assets, platform=spec,
        usd_path=rob_cfg.get("usd_path"), urdf_path=rob_cfg.get("urdf_path"), prefer=_prefer)
    generated_urdf = None
    if src is None and spec.dimensions and spec.hydro is not None:
        if why.startswith("missing:"):
            print(f"[oceansim_ros2] WARNING: {spec.name} asset not found ({why[8:]}); "
                  f"using the generated primitive-shape model instead")
        # No 3D asset (or robot.prefer_source "generated"): import a URDF built
        # from the platform data, with primitive shapes and the payloads.
        from isaacsim.oceansim.utils import urdf_export
        _gpath, generated_urdf = urdf_export.write_generated_urdf(
            spec, fitted, out_dir=rob_cfg.get("generated_urdf_dir"),
            rho=float(cfg.get("water_density", 1000.0)))
        src = platforms.RobotSource("urdf", _gpath)
        print(f"[oceansim_ros2] generated URDF for {spec.name} -> {_gpath}")
    if src is None:
        raise FileNotFoundError(
            f"[oceansim_ros2] no robot asset for platform '{spec.name}' ({why}). "
            f"Provide a USD or URDF under the asset root, or set robot.usd_path / "
            f"robot.urdf_path.")

    if src.kind == "urdf":
        # Import the URDF (creates the articulation). Our staged platform URDFs
        # carry joints + (visual-derived) collisions but NO <inertial>, so set the
        # base body's mass to the platform figure -- without it PhysX assigns an
        # invalid/negative mass to the root. urdf_import's link_density gives the
        # other links a valid geometry-based mass. Otherwise just match the
        # underwater setup (no gravity, damping, spawn).
        from isaacsim.oceansim.utils import urdf_import
        from pxr import UsdPhysics
        mass = float(rob_cfg.get("mass", spec.mass))
        print(f"[oceansim_ros2] importing URDF -> {src.path}")
        robot_path = urdf_import.import_urdf_to_stage(
            src.path, fix_base=False,
            merge_fixed_joints=rob_cfg.get("merge_fixed_joints", True))
        # Disable gravity + set damping on EVERY rigid-body link, not just the root:
        # for a URDF articulation, DisableGravity on the root does NOT propagate to
        # the child links, so they fall under gravity -- the robot sinks (looks "in
        # the ground" with no collision, or falls onto the floor and TIPS once
        # collision is enabled). Iterating all links keeps the whole vehicle neutrally
        # floating at its spawn pose.
        from pxr import Usd
        _root_prim = get_prim_at_path(robot_path)
        _n_links = 0
        for _p in Usd.PrimRange(_root_prim):
            if _p.HasAPI(UsdPhysics.RigidBodyAPI) or _p.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                _rb = PhysxSchema.PhysxRigidBodyAPI.Apply(_p)
                _rb.CreateDisableGravityAttr(True)
                try:
                    _rb.GetLinearDampingAttr().Set(lin_d)
                    _rb.GetAngularDampingAttr().Set(ang_d)
                except Exception:  # noqa: BLE001
                    pass
                _n_links += 1
        rob_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(_root_prim)
        rob_rb.CreateDisableGravityAttr(True)
        print(f"[oceansim_ros2] disabled gravity on {_n_links} robot link(s)")
        UsdPhysics.MassAPI.Apply(get_prim_at_path(robot_path)).GetMassAttr().Set(mass)
        # NOTE: robot collision is intentionally NOT enabled on the URDF articulation.
        # With gravity disabled (above) the vehicle floats at its spawn pose and never
        # sinks into the seafloor, so a collider is unnecessary -- and enabling one
        # (convex hull / boundingCube on the articulation root) caused a persistent
        # spawn-contact TIP (~28deg) without any benefit, since velocity control
        # (set_linear_velocity) kinematically overrides contact response anyway. Revisit
        # only with force/thrust-based control, where contacts can actually resist motion.
        SingleRigidPrim(prim_path=robot_path, translation=spawn)
    else:
        print(f"[oceansim_ros2] referencing USD -> {src.path}")
        mass = float(rob_cfg.get("mass", spec.mass))
        collision = rob_cfg.get("collision_approximation", spec.collision_approximation)
        add_reference_to_stage(usd_path=src.path, prim_path=robot_path)
        rob_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(get_prim_at_path(robot_path))
        rob_rb.CreateDisableGravityAttr(True)
        rob_rb.GetLinearDampingAttr().Set(lin_d)
        rob_rb.GetAngularDampingAttr().Set(ang_d)
        rob_collider = SingleGeometryPrim(prim_path=robot_path, collision=True)
        rob_collider.set_collision_approximation(collision)
        SingleRigidPrim(prim_path=robot_path, mass=mass, translation=spawn)
    robot_prim = get_prim_at_path(robot_path)

    vehicle_model = None
    if use_hydro:
        from isaacsim.oceansim.utils import vehicle_dynamics, vehicle_physics
        mass = float(rob_cfg.get("mass", spec.mass))
        _rho = float(cfg.get("water_density", 1000.0))
        _trim = bool(rob_cfg.get("trim", True))
        _resolved = vehicle_dynamics.resolve_hydro(spec, fitted, rho=_rho, mass=mass, trim=_trim)
        vehicle_physics.configure_prim(robot_prim, _resolved)
        vehicle_model = vehicle_dynamics.from_platform(
            spec, rho=_rho, surface_z=cfg.get("water_surface_z"),
            voltage=rob_cfg.get("thruster_voltage"),
            drag_scale=float(rob_cfg.get("drag_scale", 1.0)), mass=mass,
            payloads=fitted, trim=_trim)
        _net = (_rho * _resolved["displaced_volume"] - _resolved["mass"]) * vehicle_model.g
        print(f"[oceansim_ros2] hydrodynamics: {spec.name}, {vehicle_model.thrusters.count} "
              f"thrusters (max {vehicle_model.thrusters.max_forward.max():.1f} N fwd), "
              f"mass {_resolved['mass']:.2f} kg, net buoyancy {_net:+.1f} N, "
              f"water density {_rho:g}")
        if _resolved.get("trim"):
            _t = _resolved["trim"]
            print(f"[oceansim_ros2] trimmed for payloads: {_t['kind']} {_t['mass']:.3f} kg / "
                  f"{_t['volume'] * 1e3:.2f} L")
    else:
        print(f"[oceansim_ros2] hydrodynamics off: PhysX damping {lin_d:g}/{ang_d:g} "
              f"stands in for drag")

    # If the robot came from a URDF, let the URDF's sensor links define the mount
    # poses (a "sonar" / "camera" / "dvl" link's fixed-joint origin relative to
    # base), falling back to the platform spec mount for any sensor the URDF
    # doesn't define. (urdf_parse is pure + unit tested.)
    from isaacsim.oceansim.utils import urdf_parse
    urdf_text = None
    if src.kind == "urdf":
        try:
            with open(src.path, "r") as _f:
                urdf_text = _f.read()
        except Exception as e:  # noqa: BLE001
            print(f"[oceansim_ros2] could not read URDF for sensor mounts: {e}")

    def _resolve_link_prim(link_name):
        """USD prim path of URDF link ``link_name`` within the imported robot,
        or None if no matching prim exists (e.g. merge_fixed_joints folded a
        purely-structural link into its rigid-body ancestor on import -- the
        default, see the gravity-disable loop above where only the REAL rigid
        bodies, e.g. base_link + pivot_head, survive as distinct prims)."""
        direct = robot_path + "/" + link_name
        if get_prim_at_path(direct).IsValid():
            return direct
        from pxr import Usd as _Usd
        target = link_name.rsplit("/", 1)[-1].lower()
        for _p in _Usd.PrimRange(robot_prim):
            if _p.GetName().lower() == target:
                return _p.GetPath().pathString
        return None

    def _mount(kind, fallback_mount):
        """Resolve sensor ``kind``'s USD parent prim + LOCAL mount pose.

        Prefers the URDF's own sensor frame, anchored at the nearest ancestor
        link reachable via only FIXED joints (urdf_parse.mount_anchor) -- e.g.
        the DeepTrekker's sonar/camera hang off the revolute pivot_head via a
        chain of fixed sub-frames. Parenting the sensor prim there (instead of
        always at the rigid-body root) lets USD's ordinary parent/child
        transform composition carry the sensor as PhysX animates that joint,
        rather than baking a stale zero-angle snapshot that never updates
        (previously EVERY sensor was parented at the root regardless of which
        link its URDF mount was read from -- correct only when the whole
        chain to the root is fixed, e.g. the DVL). Falls back to the old
        static, root-relative behaviour when the URDF has no matching sensor
        link, or the resolved anchor link has no prim in the imported stage.
        """
        anchored = urdf_parse.sensor_mount_anchor(urdf_text, kind) if urdf_text is not None else None
        if anchored is not None:
            anchor_link, tr, rpy = anchored
            parent_path = (robot_path if anchor_link == urdf_parse.root_link(urdf_text)
                           else _resolve_link_prim(anchor_link))
            if parent_path is not None:
                print(f"[oceansim_ros2] {kind} mount from URDF link -> {tr} "
                      f"(anchored to {parent_path})")
                return parent_path, np.array(tr, dtype=float), np.array(rpy, dtype=float)
            print(f"[oceansim_ros2] {kind} mount anchor '{anchor_link}' has no prim "
                  f"in the imported stage -- falling back to a static root-relative mount")
        # Parse the URDF once (sensor_mount_or also parses, so the old extra
        # sensor_mount() call just to gate the log re-parsed the whole URDF).
        m = urdf_parse.sensor_mount(urdf_text, kind) if urdf_text is not None else None
        if m is not None:
            tr, rpy = m
            print(f"[oceansim_ros2] {kind} mount from URDF link -> {tr} (static, root-relative)")
        else:
            tr, rpy = fallback_mount.translation, fallback_mount.rpy_deg
        return robot_path, np.array(tr, dtype=float), np.array(rpy, dtype=float)

    # ---- sensors (mounts from the URDF if present, else the platform spec) -
    sensors = cfg["sensors"]
    sonar = cam = dvl = baro = None
    if sensors.get("sonar"):
        sonar_backend = cfg.get("sonar_backend", "oceansim")
        _sonar_parent, _sonar_tr, _sonar_rpy = _mount("sonar", spec.sonar_mount)
        _sonar_xform = dict(
            prim_path=_sonar_parent + "/sonar",
            translation=_sonar_tr,
            orientation=euler_angles_to_quat(_sonar_rpy, degrees=True))
        if sonar_backend == "rtx_acoustic":
            # Isaac native RTX acoustic sensor (experimental). Avoids the 6.0.1
            # pointcloud-annotator SIGSEGV; output mapping is a scaffold (see class).
            from isaacsim.oceansim.sensors.RtxAcousticSensor import RtxAcousticSensor
            _sp = cfg.get("sonar_params", {})
            # rtx_acoustic emits ONE signal way (= one azimuth return) per receiver
            # mount, so the number of returns == n_elements. Bump it for a denser
            # fan (the default 8 gives only ~8 returns). Tunable via sonar_params.
            _n_el = int(_sp.get("n_elements", 64))
            _cf = float(_sp.get("center_frequency", 1.2e6))  # M300d LF default
            print(f"[oceansim_ros2] sonar backend: rtx_acoustic (native, experimental), "
                  f"n_elements={_n_el} receivers, center_frequency={_cf:.3g} Hz")
            sonar = RtxAcousticSensor(
                # The sensor's own physics caps real range at ~6.05 m (see
                # RtxAcousticSensor's meters_per_sample/range_offset comment) --
                # RtxAcousticSensor's class default max_range=10.0 m is sized for
                # the geometric `oceansim` backend and, applied here, pads ~40%
                # of the (n_range, n_beams) grid with always-zero rows. That
                # bloats drawn_sonar to ~21 MB/frame (1980x3590 rgb8), which is
                # what was overrunning foxglove_bridge's send buffer. Default to
                # the physical ceiling here; still overridable via sonar_params.
                min_range=_sp.get("min_range", 0.1),
                max_range=_sp.get("max_range", 6.2),
                range_res=_sp.get("range_res", 0.005),
                angular_res=_sp.get("angular_res", 0.25),
                hori_fov=_sp.get("hori_fov_deg", 130.0),
                vert_fov=_sp.get("vert_fov_deg", 20.0),
                center_frequency=_cf,
                n_elements=_n_el, **_sonar_xform)
        else:
            from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor
            from isaacsim.oceansim.utils import sensor_presets
            sp = cfg.get("sonar_params", {})
            # A fitted sonar payload (e.g. oculus_m750d) supplies the device's
            # FOV, beam spacing and range defaults; sonar_params still win.
            _sonar_pl = payload_catalogue.sensor_payload(fitted, "sonar")
            _sk = sensor_presets.sonar_kwargs(_sonar_pl, sp) if _sonar_pl is not None else dict(
                min_range=sp.get("min_range", 0.2), max_range=sp.get("max_range", 3.0),
                hori_fov=sp.get("hori_fov_deg", 130.0), vert_fov=sp.get("vert_fov_deg", 20.0),
                range_res=sp.get("range_res", 0.005), angular_res=sp.get("angular_res", 0.25))
            if _sonar_pl is not None:
                print(f"[oceansim_ros2] sonar: {_sonar_pl.description} -> {_sk}")
            elif any(p.kind == "sonar" for p in fitted):
                print("[oceansim_ros2] sonar payload is not simulated (scanning sonar); "
                      "the imaging sonar uses the default parameters")
            _hori_res = int(sp.get("hori_res", 2500))
            _gpu_filter = bool(sp.get("gpu_point_filter", True))
            _async = bool(sp.get("async_compute", False))
            print("[oceansim_ros2] sonar backend: oceansim (custom imaging sonar) "
                  f"hori_res={_hori_res} gpu_point_filter={_gpu_filter} async_compute={_async}")
            sonar = ImagingSonarSensor(
                **_sk,
                hori_res=_hori_res,
                gpu_point_filter=_gpu_filter,
                async_compute=_async,
                **_sonar_xform)
            # Optional make_sonar_data model terms (spreading_exponent,
            # absorption, tvg_exponent, speckle_looks, speckle_cell,
            # beam_fwhm_deg, noise params, normalizing_method, ...): all off
            # unless set under sonar_params.model_params in the config.
            if _sonar_pl is not None:
                sonar.acoustic_frequency = sensor_presets.sonar_frequency(_sonar_pl)
            sonar.make_sonar_data_params = dict(sp.get("model_params") or {})
            if sonar.make_sonar_data_params:
                print(f"[oceansim_ros2] sonar model params: {sonar.make_sonar_data_params}")
    if sensors.get("camera"):
        from isaacsim.oceansim.sensors.UW_Camera import UW_Camera
        _cam_parent, _cam_translation, _cam_rpy = _mount("camera", spec.camera_mount)
        # Apply the URDF/spec mount rotation too (a tilted, e.g. down-looking,
        # camera): dropping the rpy rendered the camera body-aligned, so its
        # image orientation no longer matched the robot_state_publisher TF frame
        # consumers reproject against. UW_Camera takes a quaternion orientation.
        cam = UW_Camera(prim_path=_cam_parent + "/UW_camera",
                        resolution=[1920, 1080], translation=_cam_translation,
                        orientation=euler_angles_to_quat(_cam_rpy, degrees=True))
        if spec.camera_hfov_deg:
            from isaacsim.oceansim.utils import sensor_presets
            cam.set_focal_length(sensor_presets.focal_length_for_hfov(
                spec.camera_hfov_deg, cam.get_horizontal_aperture()))
            print(f"[oceansim_ros2] camera: {spec.camera_hfov_deg:g} deg horizontal FOV")
        else:
            cam.set_focal_length(0.1 * 21)
        cam.set_clipping_range(0.1, 100)
    if sensors.get("dvl"):
        from isaacsim.oceansim.sensors.DVLsensor import DVLsensor
        _dvl_parent, _dvl_translation, _ = _mount("dvl", spec.dvl_mount)
        from isaacsim.oceansim.utils import sensor_presets
        _dvl_pl = payload_catalogue.sensor_payload(fitted, "dvl")
        if _dvl_pl is not None:
            _dk = sensor_presets.dvl_kwargs(_dvl_pl)
            print(f"[oceansim_ros2] DVL: {_dvl_pl.description} -> {_dk}")
            dvl = DVLsensor(**_dk)
        else:
            dvl = DVLsensor(max_range=10)
        dvl.attachDVL(rigid_body_path=_dvl_parent, translation=_dvl_translation)
    altimeter = None
    _alt_pl = payload_catalogue.sensor_payload(fitted, "altimeter")
    if _alt_pl is not None and sensors.get("altimeter", True):
        from isaacsim.oceansim.sensors.AltimeterSensor import AltimeterSensor
        from isaacsim.oceansim.utils import sensor_presets
        _alt_parent, _alt_tr, _alt_rpy = _mount("altimeter", spec.mount("altimeter"))
        altimeter = AltimeterSensor(**sensor_presets.altimeter_kwargs(_alt_pl))
        altimeter.attach(_alt_parent, translation=_alt_tr,
                         orientation=euler_angles_to_quat(_alt_rpy, degrees=True))
        altimeter.rate_hz = float(_alt_pl.params.get("rate_hz", 10.0))
        print(f"[oceansim_ros2] altimeter: {_alt_pl.description}")
    if sensors.get("baro"):
        from isaacsim.oceansim.sensors.BarometerSensor import BarometerSensor
        baro = BarometerSensor(prim_path=robot_path + "/Baro",
                               water_surface_z=float(cfg["water_surface_z"]))

    # ---- scenario + sensor publisher --------------------------------------
    world.reset()
    scenario = MHL_Sensor_Example_Scenario()
    # sensor_viewports: None (default) = auto -- per-sensor GUI windows only make
    # sense with a GUI, and in headless mode they still paid the per-frame
    # set_bytes_data_from_gpu readback for windows nobody can see.
    _sv = cfg.get("sensor_viewports")
    _sv = (not cfg["headless"]) if _sv is None else bool(_sv)
    scenario.setup_scenario(robot_prim, sonar, cam, dvl, baro, cfg["control_mode"],
                            sensor_viewports=_sv,
                            control_params=cfg.get("control_params"),
                            vehicle_model=vehicle_model)
    # Throttle the heavy sensor compute to sensor_compute_rate (0 = every step).
    _scr = float(cfg.get("sensor_compute_rate", 0.0) or 0.0)
    scenario._sensor_update_period = (1.0 / _scr) if _scr > 0 else 0.0
    # The headless pipeline consumes DVL/baro via the ROS publisher; the
    # scenario's own per-tick reads only feed the GUI extension's plots.
    scenario._poll_gui_readings = False

    # Waypoints mode needs a waypoint file wired in: the GUI loads one via its
    # file picker, but the runner never called setup_waypoints, so the first
    # update_scenario step crashed on the missing self.waypoints. Load it from
    # cfg["waypoints_file"], or fail at startup with an actionable message
    # instead of a mid-run AttributeError.
    if cfg["control_mode"] == "Waypoints":
        _wp = cfg.get("waypoints_file", "")
        if not _wp or not os.path.isfile(_wp):
            raise SystemExit(
                "[oceansim_ros2] control_mode 'Waypoints' requires config key "
                f"'waypoints_file' pointing at an existing waypoint file (got {_wp!r}).")
        scenario.setup_waypoints(_wp, _wp)

    pub_cfg = dict(cfg.get("publisher", {}))

    # Robot description for ROS: latch the platform URDF on /robot_description so
    # robot_state_publisher / RViz can articulate the model from the published
    # /joint_states. Precedence: inline string > explicit path (--robot-description
    # / robot.urdf_path) > the platform's registered URDF under the asset root.
    if "robot_description" not in pub_cfg and generated_urdf is not None:
        pub_cfg["robot_description"] = generated_urdf
        print(f"[oceansim_ros2] robot_description from the generated URDF "
              f"({len(generated_urdf)} chars) -> /robot_description")
    if "robot_description" not in pub_cfg:
        desc_text, desc_src = platforms.resolve_robot_description(
            asset_root=assets, platform=spec,
            inline=rob_cfg.get("robot_description"),
            path=rob_cfg.get("robot_description_path") or rob_cfg.get("urdf_path"))
        if desc_text:
            pub_cfg["robot_description"] = desc_text
            print(f"[oceansim_ros2] robot_description from {desc_src} "
                  f"({len(desc_text)} chars) -> /robot_description")
        else:
            print(f"[oceansim_ros2] no robot_description ({desc_src}); "
                  f"robot_state_publisher/RViz will need one from elsewhere.")

    # When the robot was imported from a URDF, align the published frame_ids with
    # the URDF tree so robot_state_publisher's TF and OceanSim's message stamps
    # agree: base frame = the URDF root link; sonar/camera frames = their URDF
    # sensor-link names (each only if the user hasn't overridden it).
    _urdf_frames = {}   # sensor kind -> URDF link (frames robot_state_publisher owns)
    if urdf_text is not None:
        _base = urdf_parse.root_link(urdf_text)
        if _base:
            for key in ("base_frame_id", "imu_frame_id", "dvl_frame_id", "baro_frame_id"):
                pub_cfg.setdefault(key, _base)
            print(f"[oceansim_ros2] base frame from URDF root link -> {_base}")
        for kind, fid in (("sonar", "sonar_frame_id"), ("camera", "camera_frame_id")):
            link = urdf_parse.sensor_link(urdf_text, kind)
            if link:
                _urdf_frames[kind] = link
                pub_cfg.setdefault(fid, link)

    if cfg.get("publish_static_tf"):
        # The sensors are children of the robot prim, so their local mount pose
        # IS base->sensor. DVL/baro/IMU report in the base frame, so no TF needed.
        # Skip any sensor whose frame the URDF already defines -- robot_state_publisher
        # publishes base->that link from the URDF, so OceanSim must not duplicate it.
        static_tfs = []
        if sonar is not None and "sonar" not in _urdf_frames:
            static_tfs.append({
                "child_frame_id": pub_cfg.get("sonar_frame_id", "sonar0/optical_frame"),
                "translation": [float(x) for x in _sonar_xform["translation"]],
                "rotation_wxyz": [float(x) for x in _sonar_xform["orientation"]],
            })
        if cam is not None and "camera" not in _urdf_frames:
            # REP-103/104: Image/depth/CameraInfo are stamped in an OPTICAL frame
            # (z forward, x right, y down). The camera prim is body-aligned at the
            # mount (created with translation only), so broadcast
            # base->camera_optical with the standard optical rotation
            # (rpy -90,0,-90) rather than identity -- otherwise depth_image_proc /
            # RViz reproject the planar z-depth along the wrong axis. (The exact
            # sign is the textbook optical transform; confirm against a real frame.)
            _cam_opt_quat = euler_angles_to_quat(np.array([-90.0, 0.0, -90.0]), degrees=True)
            static_tfs.append({
                "child_frame_id": pub_cfg.get("camera_frame_id", "camera_optical_frame"),
                "translation": [float(x) for x in _cam_translation],
                "rotation_wxyz": [float(x) for x in _cam_opt_quat],
            })
        pub_cfg["publish_static_tf"] = True
        pub_cfg["static_transforms"] = static_tfs
    if altimeter is not None:
        pub_cfg.setdefault("altimeter_rate", altimeter.rate_hz)
    publisher = OceanSimSensorPublisher(
        robot_prim=robot_prim, sonar=sonar, dvl=dvl, baro=baro, config=pub_cfg,
        altimeter=altimeter)
    publisher.initialize()

    # ---- run loop ---------------------------------------------------------
    running = {"flag": True}

    def _stop(*_):
        running["flag"] = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    world.play()
    fixed_dt = cfg["physics_dt"]
    prev_time = world.current_time
    # The camera and imaging sonar both read Isaac render products, so rendering
    # must run when either is active even in headless mode.
    need_render = (not cfg["headless"]) or (cam is not None) or (sonar is not None)

    # Sonar render-cadence gating (opt-in via render_rate>0; OFF by default).
    # KNOWN-LIMITED: the only per-step lever wired here is
    # render_product.hydra_texture.set_updates_enabled(), which stops the sonar
    # AOV readback (so scan() gets no data) WITHOUT removing the underlying RTX
    # raytrace from the GPU pipeline -- so it breaks the sonar and does NOT free
    # odom (the loop is GPU-render-bound; see nvidia-smi 100%). Left dormant until a
    # lever that actually drops the sonar render product from the per-step GPU
    # pipeline (with a re-enable warmup for render latency) is wired. async alone
    # does NOT force it on.
    _sonar_render_rate = float(cfg.get("sonar_params", {}).get("render_rate", 0.0) or 0.0)
    _sonar_render_period = (1.0 / _sonar_render_rate) if _sonar_render_rate > 0 else 0.0
    # render_rate/gating only applies to the "oceansim" backend's sonar CAMERA
    # (ready_for_scan()/set_render_enabled() are ImagingSonarSensor-only methods
    # that RtxAcousticSensor does not implement -- calling them on that backend
    # raises AttributeError). Warn instead of crashing if the config combo is set.
    _sonar_backend = cfg.get("sonar_backend", "oceansim")
    if _sonar_render_period > 0.0 and _sonar_backend != "oceansim":
        print(f"[oceansim_ros2] sonar_params.render_rate is ignored for "
              f"sonar_backend={_sonar_backend!r} (only the 'oceansim' backend "
              f"supports render gating)")
    _sonar_gating = (sonar is not None and _sonar_render_period > 0.0
                      and _sonar_backend == "oceansim")
    _last_sonar_scan = -1e9
    # Scan-on-publish: headless with no sonar viewport, the synchronous oceansim
    # sonar's scan only feeds the ROS publisher, so scan exactly on the ticks
    # the publisher will send a sonar image (sonar_rate) instead of at
    # sensor_compute_rate -- 2/3 fewer scans at the 15 Hz / 5 Hz defaults, and
    # the published frame is captured on its publish tick. Not with render
    # gating (it drives sonar_tick itself) or async compute (whose worker needs
    # the lead time).
    # Needs a positive sonar_rate: with 0 (publish every tick) it would scan
    # every physics step, above the sensor_compute_rate throttle.
    _scan_on_publish = (sonar is not None and not _sonar_gating and not _sv
                        and _sonar_backend == "oceansim"
                        and not getattr(sonar, "async_compute", False)
                        and float(publisher._cfg.get("sonar_rate", 0.0) or 0.0) > 0.0)
    print("[oceansim_ros2] simulation running; publishing ROS2 sensor data"
          + (f" (sonar render gated, cap={_sonar_render_rate or 'worker'} Hz)"
             if _sonar_gating else "")
          + (" (sonar scans on publish ticks)" if _scan_on_publish else ""))
    try:
        while sim_app.is_running() and running["flag"]:
            # Decide -- before the step that would render it -- whether to render +
            # scan the sonar this iteration. Gated on is_playing so a PAUSED sim
            # doesn't keep force-enabling the sonar render (paying the raytrace)
            # for scans that never run.
            sonar_tick = None
            _decision_time = prev_time
            if _sonar_gating:
                sonar_tick = (world.is_playing()
                              and (_decision_time - _last_sonar_scan) >= _sonar_render_period
                              and sonar.ready_for_scan())
                sonar.set_render_enabled(sonar_tick)
            world.step(render=need_render)
            if world.is_playing():
                now = world.current_time
                # Pass the ACTUAL sim-time advance (= rendering_dt when it differs
                # from physics_dt), so the scenario's sensor-compute throttle is
                # rate-correct. Equals physics_dt at the default config.
                step = now - prev_time
                if step <= 0.0:
                    step = fixed_dt    # first tick / after a reset
                prev_time = now
                if _scan_on_publish:
                    sonar_tick = publisher.sonar_due(now)
                if sonar_tick and _sonar_gating:
                    # Record on the SAME clock the decision compares against
                    # (pre-step time): recording the post-step `now` stretched
                    # the effective period to period + one step per scan.
                    _last_sonar_scan = _decision_time
                scenario.update_scenario(step, now, sonar_tick=sonar_tick)
                publisher.publish(now)
    finally:
        print("[oceansim_ros2] shutting down")
        # Best-effort teardown: a failure in publisher.close() or
        # teardown_scenario() must NOT skip sim_app.close() (the old nested
        # finally dropped sim_app.close when teardown raised, orphaning the kit
        # process + the rclpy context).
        for _what, _fn in (("publisher.close", publisher.close),
                           ("scenario.teardown", scenario.teardown_scenario)):
            try:
                _fn()
            except Exception as _e:  # noqa: BLE001
                print(f"[oceansim_ros2] {_what} warning: {_e}")
        _close_sim_app()   # idempotent; also registered atexit for setup-phase failures


if __name__ == "__main__":
    main(sys.argv[1:])

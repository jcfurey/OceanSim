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
        "scene_usd": "",
        "asset_path": "",
        # Vehicle platform (utils.platforms). Its spec provides the USD, mass,
        # damping, collision, spawn pose, sensor mounts, and default URDF. The
        # optional "robot" dict overrides individual fields (e.g. mass,
        # translation, usd_path, urdf_path, robot_description).
        "platform": "bluerov2",
        "robot": {},
        "sensors": {"sonar": True, "camera": True, "dvl": True, "baro": True},
        # Sonar backend: "oceansim" = custom imaging sonar (Camera + pointcloud
        # annotator); "rtx_acoustic" = Isaac native RTX acoustic sensor
        # (experimental, avoids the 6.0.1 pointcloud-annotator crash).
        "sonar_backend": "oceansim",
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
    from isaacsim import SimulationApp
    sim_app = SimulationApp({
        "headless": bool(cfg["headless"]),
        "renderer": cfg["renderer"],
    })

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
    # executing the package __init__. (oceansim/ and modules/ have no __init__.)
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
    spec = platforms.get_platform(cfg.get("platform", platforms.DEFAULT_PLATFORM))
    rob_cfg = cfg.get("robot", {})
    print(f"[oceansim_ros2] platform: {spec.name} -- {spec.description}")

    robot_path = "/World/rob"
    # Each field falls back to the platform spec unless explicitly overridden in
    # cfg["robot"]. (For bluerov2 the spec values equal the old hardcoded ones,
    # so this is behaviour-preserving.)
    lin_d = float(rob_cfg.get("linear_damping", spec.linear_damping))
    ang_d = float(rob_cfg.get("angular_damping", spec.angular_damping))
    spawn = np.array(rob_cfg.get("translation", spec.spawn_translation), dtype=float)

    # USD or URDF? An explicit robot.usd_path / robot.urdf_path wins, else the
    # platform's own assets are used (prefer "usd", or set robot.prefer_source
    # = "urdf"). Resolves to a URDF automatically if that's all that exists.
    src, why = platforms.resolve_robot_source(
        asset_root=assets, platform=spec,
        usd_path=rob_cfg.get("usd_path"), urdf_path=rob_cfg.get("urdf_path"),
        prefer=rob_cfg.get("prefer_source", "usd"))
    if src is None:
        raise FileNotFoundError(
            f"[oceansim_ros2] no robot asset for platform '{spec.name}' ({why}). "
            f"Provide a USD or URDF under the asset root, or set robot.usd_path / "
            f"robot.urdf_path.")

    if src.kind == "urdf":
        # Import the URDF (creates the articulation). The URDF already defines
        # inertials / joints / collisions, so do NOT override mass or collision
        # here -- only match the underwater setup (no gravity, damping, spawn).
        from isaacsim.oceansim.utils import urdf_import
        print(f"[oceansim_ros2] importing URDF -> {src.path}")
        robot_path = urdf_import.import_urdf_to_stage(
            src.path, fix_base=False,
            merge_fixed_joints=rob_cfg.get("merge_fixed_joints", True))
        rob_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(get_prim_at_path(robot_path))
        rob_rb.CreateDisableGravityAttr(True)
        try:
            rob_rb.GetLinearDampingAttr().Set(lin_d)
            rob_rb.GetAngularDampingAttr().Set(ang_d)
        except Exception:  # noqa: BLE001 - articulation root may differ
            pass
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

    def _mount(kind, fallback_mount):
        # Parse the URDF once (sensor_mount_or also parses, so the old extra
        # sensor_mount() call just to gate the log re-parsed the whole URDF).
        m = urdf_parse.sensor_mount(urdf_text, kind) if urdf_text is not None else None
        if m is not None:
            tr, rpy = m
            print(f"[oceansim_ros2] {kind} mount from URDF link -> {tr}")
        else:
            tr, rpy = fallback_mount.translation, fallback_mount.rpy_deg
        return np.array(tr, dtype=float), np.array(rpy, dtype=float)

    # ---- sensors (mounts from the URDF if present, else the platform spec) -
    sensors = cfg["sensors"]
    sonar = cam = dvl = baro = None
    if sensors.get("sonar"):
        sonar_backend = cfg.get("sonar_backend", "oceansim")
        _sonar_tr, _sonar_rpy = _mount("sonar", spec.sonar_mount)
        _sonar_xform = dict(
            prim_path=robot_path + "/sonar",
            translation=_sonar_tr,
            orientation=euler_angles_to_quat(_sonar_rpy, degrees=True))
        if sonar_backend == "rtx_acoustic":
            # Isaac native RTX acoustic sensor (experimental). Avoids the 6.0.1
            # pointcloud-annotator SIGSEGV; output mapping is a scaffold (see class).
            from isaacsim.oceansim.sensors.RtxAcousticSensor import RtxAcousticSensor
            print("[oceansim_ros2] sonar backend: rtx_acoustic (native, experimental)")
            sonar = RtxAcousticSensor(
                range_res=0.005, angular_res=0.25, **_sonar_xform)
        else:
            from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor
            print("[oceansim_ros2] sonar backend: oceansim (custom imaging sonar)")
            sonar = ImagingSonarSensor(
                range_res=0.005, angular_res=0.25, hori_res=4000, **_sonar_xform)
    if sensors.get("camera"):
        from isaacsim.oceansim.sensors.UW_Camera import UW_Camera
        _cam_translation, _ = _mount("camera", spec.camera_mount)
        cam = UW_Camera(prim_path=robot_path + "/UW_camera",
                        resolution=[1920, 1080], translation=_cam_translation)
        cam.set_focal_length(0.1 * 21)
        cam.set_clipping_range(0.1, 100)
    if sensors.get("dvl"):
        from isaacsim.oceansim.sensors.DVLsensor import DVLsensor
        _dvl_translation, _ = _mount("dvl", spec.dvl_mount)
        dvl = DVLsensor(max_range=10)
        dvl.attachDVL(rigid_body_path=robot_path, translation=_dvl_translation)
    if sensors.get("baro"):
        from isaacsim.oceansim.sensors.BarometerSensor import BarometerSensor
        baro = BarometerSensor(prim_path=robot_path + "/Baro",
                               water_surface_z=float(cfg["water_surface_z"]))

    # ---- scenario + sensor publisher --------------------------------------
    world.reset()
    scenario = MHL_Sensor_Example_Scenario()
    scenario.setup_scenario(robot_prim, sonar, cam, dvl, baro, cfg["control_mode"])
    # Throttle the heavy sensor compute to sensor_compute_rate (0 = every step).
    _scr = float(cfg.get("sensor_compute_rate", 0.0) or 0.0)
    scenario._sensor_update_period = (1.0 / _scr) if _scr > 0 else 0.0

    pub_cfg = dict(cfg.get("publisher", {}))

    # Robot description for ROS: latch the platform URDF on /robot_description so
    # robot_state_publisher / RViz can articulate the model from the published
    # /joint_states. Precedence: inline string > explicit path (--robot-description
    # / robot.urdf_path) > the platform's registered URDF under the asset root.
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
    publisher = OceanSimSensorPublisher(
        robot_prim=robot_prim, sonar=sonar, dvl=dvl, baro=baro, config=pub_cfg)
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
    print("[oceansim_ros2] simulation running; publishing ROS2 sensor data")
    try:
        while sim_app.is_running() and running["flag"]:
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
                scenario.update_scenario(step, now)
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
        sim_app.close()


if __name__ == "__main__":
    main(sys.argv[1:])

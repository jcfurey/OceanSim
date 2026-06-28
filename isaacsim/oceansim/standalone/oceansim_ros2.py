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
    p.add_argument("--robot-description", dest="robot_description", default=None,
                   help="Path to a URDF to latch on /robot_description, "
                        "overriding the platform's default robot description.")
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
    if args.robot_description is not None:
        cfg.setdefault("robot", {})["urdf_path"] = args.robot_description
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
    """Write OceanSim's asset_path.json if a path was supplied and not yet set."""
    if not asset_path:
        return
    import isaacsim.oceansim.utils as _utils_pkg
    json_path = os.path.join(os.path.dirname(_utils_pkg.__file__), "asset_path.json")
    if os.path.isfile(json_path):
        return
    with open(json_path, "w") as f:
        json.dump({"asset_path": os.path.abspath(asset_path)}, f, indent=2)
    print(f"[oceansim_ros2] registered asset path -> {asset_path}")


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
    world = World(physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"],
                  stage_units_in_meters=1.0)

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
    usd_path = rob_cfg.get("usd_path") or spec.usd_path(assets)
    mass = float(rob_cfg.get("mass", spec.mass))
    lin_d = float(rob_cfg.get("linear_damping", spec.linear_damping))
    ang_d = float(rob_cfg.get("angular_damping", spec.angular_damping))
    collision = rob_cfg.get("collision_approximation", spec.collision_approximation)
    spawn = np.array(rob_cfg.get("translation", spec.spawn_translation), dtype=float)

    add_reference_to_stage(usd_path=usd_path, prim_path=robot_path)
    rob_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(get_prim_at_path(robot_path))
    rob_rb.CreateDisableGravityAttr(True)
    rob_rb.GetLinearDampingAttr().Set(lin_d)
    rob_rb.GetAngularDampingAttr().Set(ang_d)
    rob_collider = SingleGeometryPrim(prim_path=robot_path, collision=True)
    rob_collider.set_collision_approximation(collision)
    SingleRigidPrim(prim_path=robot_path, mass=mass, translation=spawn)
    robot_prim = get_prim_at_path(robot_path)

    def _mount_quat(mount):
        return euler_angles_to_quat(np.array(mount.rpy_deg, dtype=float), degrees=True)

    # ---- sensors (mount poses come from the platform spec) ----------------
    sensors = cfg["sensors"]
    sonar = cam = dvl = baro = None
    if sensors.get("sonar"):
        sonar_backend = cfg.get("sonar_backend", "oceansim")
        _sonar_xform = dict(
            prim_path=robot_path + "/sonar",
            translation=np.array(spec.sonar_mount.translation, dtype=float),
            orientation=_mount_quat(spec.sonar_mount))
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
        _cam_translation = np.array(spec.camera_mount.translation, dtype=float)
        cam = UW_Camera(prim_path=robot_path + "/UW_camera",
                        resolution=[1920, 1080], translation=_cam_translation)
        cam.set_focal_length(0.1 * 21)
        cam.set_clipping_range(0.1, 100)
    if sensors.get("dvl"):
        from isaacsim.oceansim.sensors.DVLsensor import DVLsensor
        dvl = DVLsensor(max_range=10)
        dvl.attachDVL(rigid_body_path=robot_path,
                      translation=np.array(spec.dvl_mount.translation, dtype=float))
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
        urdf_text, src = platforms.resolve_robot_description(
            asset_root=assets, platform=spec,
            inline=rob_cfg.get("robot_description"),
            path=rob_cfg.get("urdf_path"))
        if urdf_text:
            pub_cfg["robot_description"] = urdf_text
            print(f"[oceansim_ros2] robot_description from {src} "
                  f"({len(urdf_text)} chars) -> /robot_description")
        else:
            print(f"[oceansim_ros2] no robot_description ({src}); "
                  f"robot_state_publisher/RViz will need one from elsewhere.")

    if cfg.get("publish_static_tf"):
        # The sensors are children of the robot prim, so their local mount pose
        # IS base_link->sensor. DVL/baro/IMU report in base_link, so no TF needed.
        static_tfs = []
        if sonar is not None:
            static_tfs.append({
                "child_frame_id": pub_cfg.get("sonar_frame_id", "sonar0/optical_frame"),
                "translation": [float(x) for x in _sonar_xform["translation"]],
                "rotation_wxyz": [float(x) for x in _sonar_xform["orientation"]],
            })
        if cam is not None:
            static_tfs.append({
                "child_frame_id": pub_cfg.get("camera_frame_id", "camera"),
                "translation": [float(x) for x in _cam_translation],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
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
    dt = cfg["physics_dt"]
    # The camera and imaging sonar both read Isaac render products, so rendering
    # must run when either is active even in headless mode.
    need_render = (not cfg["headless"]) or (cam is not None) or (sonar is not None)
    print("[oceansim_ros2] simulation running; publishing ROS2 sensor data")
    try:
        while sim_app.is_running() and running["flag"]:
            world.step(render=need_render)
            if world.is_playing():
                scenario.update_scenario(dt)
                publisher.publish(world.current_time)
    finally:
        print("[oceansim_ros2] shutting down")
        try:
            publisher.close()
        finally:
            scenario.teardown_scenario()
            sim_app.close()


if __name__ == "__main__":
    main(sys.argv[1:])

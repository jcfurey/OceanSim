"""ROS2 sensor publishing layer for OceanSim.

The stock OceanSim ``SensorExample`` only publishes the underwater camera image
(see :class:`isaacsim.oceansim.sensors.UW_Camera.UW_Camera`) and accepts
velocity/force commands (see
:class:`isaacsim.oceansim.utils.ros2_control.ROS2ControlReceiver`).  Nothing
publishes the *vehicle state* (odometry / IMU) or the acoustic / pressure
sensors over ROS2, which a downstream robot autonomy stack needs.

This module fills that gap.  Given the robot rigid-body prim and whichever
OceanSim sensor objects are active, :class:`OceanSimSensorPublisher` publishes:

================================  ==========================================  =================================
Topic (default)                   Message type                                Source
================================  ==========================================  =================================
``/clock``                        ``rosgraph_msgs/Clock``                     simulation time
``/oceansim/robot/odom``          ``nav_msgs/Odometry``                       robot rigid-body world pose+twist
``/oceansim/robot/imu``           ``sensor_msgs/Imu``                         robot orientation/angular-vel/accel
``/oceansim/robot/dvl/twist``     ``geometry_msgs/TwistWithCovarianceStamped``  ``DVLsensor.get_linear_vel()``
``/oceansim/robot/pressure``      ``sensor_msgs/FluidPressure``               ``BarometerSensor.get_pressure()``
``/oceansim/robot/sonar``         ``marine_acoustic_msgs/ProjectedSonarImage``  ``ImagingSonarSensor``
================================  ==========================================  =================================

The message types for odom / imu / dvl / pressure deliberately match what
``robot_localization`` and the ``sonar_image_proc`` / ``sonar_proc`` pipelines
in the companion ROS2 workspace expect, so the workspace bringup only has to
*relay* topics rather than reshape them.

Design notes
------------
* Like the rest of OceanSim's ROS2 code, ``rclpy`` is provided by the
  ``isaacsim.ros2.bridge`` extension which must be activated *before* OceanSim.
  We share the process-wide rclpy context through
  :mod:`isaacsim.oceansim.utils.ros2_context` so this publisher coexists with
  the camera publisher and control receiver.
* Every sensor publish is independently guarded and rate-limited.  A failure
  building one message (e.g. ``marine_acoustic_msgs`` not installed) disables
  only that publisher; the simulation keeps running.
* OceanSim has no IMU sensor, so IMU orientation/angular-velocity come straight
  from the rigid body and linear acceleration is a finite difference of the
  body-frame linear velocity.
"""

import math

import numpy as np

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from isaacsim.oceansim.utils import ros2_context
from isaacsim.oceansim.utils import ros2_math
from isaacsim.oceansim.utils import joint_control
from isaacsim.oceansim.utils import ros2_qos
# Pure (numpy-only) math lives in ros2_math so it can be unit tested without ROS.
_quat_wxyz_to_xyzw = ros2_math.quat_wxyz_to_xyzw
_RateGate = ros2_math.RateGate

try:
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
except Exception:  # pragma: no cover - only importable inside Isaac Sim
    quat_to_rot_matrix = None


def _to_profile(q) -> QoSProfile:
    """Build an rclpy QoSProfile from a pure ros2_qos.QoS spec, so the profiles
    that are verified for compatibility in tests are exactly the ones used on the
    wire."""
    return QoSProfile(
        reliability=(ReliabilityPolicy.RELIABLE if q.reliability == ros2_qos.RELIABLE
                     else ReliabilityPolicy.BEST_EFFORT),
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if q.durability == ros2_qos.TRANSIENT_LOCAL
                    else DurabilityPolicy.VOLATILE),
        history=HistoryPolicy.KEEP_LAST,
        depth=q.depth,
    )


def _sensor_qos(depth: int = 10) -> QoSProfile:
    """Best-effort, keep-last sensor QoS (matches qos_profile_sensor_data)."""
    return _to_profile(ros2_qos.SENSOR_DATA._replace(depth=depth))


def _clock_qos() -> QoSProfile:
    return _to_profile(ros2_qos.CLOCK)


def _latched_qos() -> QoSProfile:
    """Latched (TRANSIENT_LOCAL) QoS for the robot description: published once,
    yet still delivered to subscribers that join later (RViz, robot_state_publisher)."""
    return _to_profile(ros2_qos.LATCHED)


def _reliable_state_qos() -> QoSProfile:
    """RELIABLE QoS for /joint_states. robot_state_publisher subscribes with the
    rclcpp default (RELIABLE), so a BEST_EFFORT publisher would be silently
    dropped -- see ros2_qos.TOPIC_CONTRACTS / tests/test_ros2_qos.py."""
    return _to_profile(ros2_qos.RELIABLE_STATE)


class OceanSimSensorPublisher:
    """Publishes OceanSim vehicle state + sensors over ROS2.

    Args:
        robot_prim: the robot ``Usd.Prim`` (as created in the scenario).  Used to
            build a ``SingleRigidPrim`` for pose/velocity queries.
        sonar: optional ``ImagingSonarSensor`` instance.
        dvl: optional ``DVLsensor`` instance.
        baro: optional ``BarometerSensor`` instance.
        config: optional dict overriding defaults (topics / frames / rates).
            Recognised keys are documented in ``DEFAULT_CONFIG`` below.
    """

    DEFAULT_CONFIG = {
        # frames
        "odom_frame_id": "odom",
        "base_frame_id": "base_link",
        "imu_frame_id": "base_link",
        "dvl_frame_id": "base_link",
        "baro_frame_id": "base_link",
        "sonar_frame_id": "sonar0/optical_frame",
        # topics
        "odom_topic": "/oceansim/robot/odom",
        "imu_topic": "/oceansim/robot/imu",
        "dvl_topic": "/oceansim/robot/dvl/twist",
        "baro_topic": "/oceansim/robot/pressure",
        "sonar_topic": "/oceansim/robot/sonar",
        "clock_topic": "/clock",
        "robot_description_topic": "/robot_description",
        "joint_states_topic": "/joint_states",
        "joint_command_topic": "/oceansim/robot/joint_command",
        # rates (Hz, <=0 means "every sim step")
        "odom_rate": 60.0,
        "imu_rate": 100.0,
        "dvl_rate": 10.0,
        "baro_rate": 10.0,
        "sonar_rate": 5.0,
        "joint_state_rate": 30.0,
        # Robot description (URDF XML string) to latch on robot_description_topic
        # so robot_state_publisher / RViz can load the platform model. None ->
        # the description publisher is simply not created. The runner fills this
        # from the selected platform's URDF (see utils.platforms).
        "robot_description": None,
        # Joint manipulation: publish the articulation's joint states on
        # joint_states_topic (for robot_state_publisher TF), and -- when
        # enable_joint_command is True -- subscribe to joint_command_topic
        # (sensor_msgs/JointState of position targets) to drive the joints.
        # Both are no-ops if the robot prim is not an Isaac articulation.
        "enable_joint_command": True,
        # toggles
        "publish_clock": True,
        # Static TF base_link->{sensor} from the mount poses. OFF by default so it
        # never conflicts with a robot-stack URDF / robot_state_publisher that
        # already owns those transforms; enable for standalone runs (RViz /
        # sonar_image_proc need the sensor frames in the TF tree). The transforms
        # themselves are supplied by the runner via "static_transforms".
        "publish_static_tf": False,
        # sonar acoustic params (used to fill ProjectedSonarImage.ping_info)
        "sound_speed": 1500.0,
        # Explicit override (Hz) for the sonar acoustic carrier reported in
        # ping_info.frequency. None -> use the sensor's own modelled value
        # (ImagingSonarSensor.acoustic_frequency / RtxAcousticSensor.center_frequency).
        "sonar_acoustic_freq": None,
        # gravity magnitude (m/s^2) used to build the IMU specific force
        "gravity": 9.81,
    }

    def __init__(self, robot_prim, sonar=None, dvl=None, baro=None, config=None):
        self._robot_prim = robot_prim
        self._sonar = sonar
        self._dvl = dvl
        self._baro = baro

        self._cfg = dict(self.DEFAULT_CONFIG)
        if config:
            self._cfg.update({k: v for k, v in config.items() if v is not None})

        self._node = None
        self._ros2_acquired = False
        self._rigid_prim = None
        self._static_tf_broadcaster = None  # kept alive so /tf_static stays latched

        # publishers (created lazily in initialize, may be None if msg pkg missing)
        self._odom_pub = None
        self._imu_pub = None
        self._dvl_pub = None
        self._baro_pub = None
        self._sonar_pub = None
        self._clock_pub = None
        self._robot_desc_pub = None    # latched /robot_description (URDF)
        self._joint_state_pub = None   # /joint_states from the articulation
        self._joint_cmd_sub = None     # joint position-command subscription
        self._articulation = None      # lazily-resolved Isaac articulation (if any)
        self._joint_names = None       # articulation dof order (cached)

        # rate gates
        self._odom_gate = _RateGate(self._cfg["odom_rate"])
        self._imu_gate = _RateGate(self._cfg["imu_rate"])
        self._dvl_gate = _RateGate(self._cfg["dvl_rate"])
        self._baro_gate = _RateGate(self._cfg["baro_rate"])
        self._sonar_gate = _RateGate(self._cfg["sonar_rate"])
        self._joint_state_gate = _RateGate(self._cfg["joint_state_rate"])

        # IMU finite-difference state (world-frame velocity, for specific force)
        self._last_world_lin_vel = None
        self._last_imu_time = None

        # Cached ProjectedSonarImage geometry (beam directions / ranges /
        # beamwidths). These depend only on the fixed sonar config (FOV, beam &
        # range counts), so they are built once and reused every publish instead
        # of being reconstructed -- the per-publish work then is just the image.
        self._sonar_geom = None

    # ------------------------------------------------------------------ setup
    def initialize(self):
        """Create the rclpy node and publishers for the active sensors."""
        ros2_context.acquire()
        self._ros2_acquired = True
        self._node = rclpy.create_node("oceansim_sensor_publisher")

        # NOTE: the robot rigid-prim wrapper is acquired lazily (see
        # _ensure_rigid_prim), NOT here: initialize() runs before world.play(),
        # and under Isaac Sim 6.0.1 SingleRigidPrim.__init__ eagerly reads
        # rigid-body velocities, which raises before the physics backend is live.

        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, FluidPressure

        self._odom_pub = self._node.create_publisher(
            Odometry, self._cfg["odom_topic"], _sensor_qos())
        self._imu_pub = self._node.create_publisher(
            Imu, self._cfg["imu_topic"], _sensor_qos())

        if self._cfg["publish_clock"]:
            try:
                from rosgraph_msgs.msg import Clock
                self._clock_pub = self._node.create_publisher(
                    Clock, self._cfg["clock_topic"], _clock_qos())
            except Exception as e:  # pragma: no cover
                self._node.get_logger().warn(f"clock publisher disabled: {e}")

        if self._dvl is not None:
            try:
                from geometry_msgs.msg import TwistWithCovarianceStamped  # noqa: F401
                self._dvl_pub = self._node.create_publisher(
                    TwistWithCovarianceStamped, self._cfg["dvl_topic"], _sensor_qos())
            except Exception as e:  # pragma: no cover
                self._node.get_logger().warn(f"DVL publisher disabled: {e}")

        if self._baro is not None:
            self._baro_pub = self._node.create_publisher(
                FluidPressure, self._cfg["baro_topic"], _sensor_qos())

        if self._sonar is not None:
            try:
                from marine_acoustic_msgs.msg import ProjectedSonarImage  # noqa: F401
                self._sonar_pub = self._node.create_publisher(
                    ProjectedSonarImage, self._cfg["sonar_topic"], _sensor_qos(depth=2))
            except Exception as e:  # pragma: no cover
                self._node.get_logger().warn(
                    f"sonar publisher disabled (marine_acoustic_msgs missing?): {e}")

        self._setup_robot_description()
        self._setup_joints()
        self._setup_static_tf()
        self._node.get_logger().info("OceanSimSensorPublisher initialized")

    def _setup_robot_description(self):
        """Latch the platform URDF on robot_description_topic (if provided), so
        robot_state_publisher / RViz can load the model. Published once with
        TRANSIENT_LOCAL durability, so subscribers that join later still get it."""
        urdf = self._cfg.get("robot_description")
        if not urdf:
            return
        try:
            from std_msgs.msg import String
            self._robot_desc_pub = self._node.create_publisher(
                String, self._cfg["robot_description_topic"], _latched_qos())
            self._robot_desc_pub.publish(String(data=urdf))
            self._node.get_logger().info(
                f"latched robot_description ({len(urdf)} chars) on "
                f"{self._cfg['robot_description_topic']}")
        except Exception as e:  # pragma: no cover
            self._node.get_logger().warn(f"robot_description publisher disabled: {e}")

    def _setup_joints(self):
        """Resolve the robot's articulation (if any) and set up joint state
        publishing + the optional joint-command subscription. All a no-op when
        the robot prim is a plain rigid body with no DOFs."""
        # GUARD: only touch an articulation when the robot prim actually IS one.
        # Constructing a SingleArticulation on a plain rigid hull fails *and* deletes
        # that prim's physics tensor view ("prim '/World/rob' was deleted ... the
        # physics.tensors simulationView was invalidated"), which then breaks the
        # odom/IMU velocity reads here AND OceanSim's own ROS2ControlReceiver. Detect
        # via USD (no tensor view), so a rigid body is a true no-op.
        try:
            from pxr import Usd, UsdPhysics
            if not any(p.HasAPI(UsdPhysics.ArticulationRootAPI)
                       for p in Usd.PrimRange(self._robot_prim)):
                return  # rigid body -> no joints; leave its rigid tensor view intact
        except Exception as e:  # noqa: BLE001 - if the USD probe fails, skip joint I/O
            self._node.get_logger().warn(
                f"articulation probe failed ({e}); skipping joint I/O")
            return

        try:
            from isaacsim.core.prims import SingleArticulation
            from isaacsim.core.utils.prims import get_prim_path
            art = SingleArticulation(prim_path=get_prim_path(self._robot_prim))
            try:
                art.initialize()
            except Exception:  # pragma: no cover - may already be initialized
                pass
            dof_names = list(getattr(art, "dof_names", None) or [])
            if not dof_names:
                return  # not an articulation / no joints -> nothing to manipulate
            self._articulation = art
            self._joint_names = dof_names
        except Exception as e:  # pragma: no cover
            self._node.get_logger().warn(f"articulation unavailable (no joint I/O): {e}")
            return

        from sensor_msgs.msg import JointState
        self._joint_state_pub = self._node.create_publisher(
            JointState, self._cfg["joint_states_topic"], _reliable_state_qos())
        if self._cfg.get("enable_joint_command"):
            self._joint_cmd_sub = self._node.create_subscription(
                JointState, self._cfg["joint_command_topic"], self._on_joint_command,
                _sensor_qos())
        self._node.get_logger().info(
            f"joint manipulation ready: {len(self._joint_names)} DOFs "
            f"{self._joint_names} (states -> {self._cfg['joint_states_topic']}"
            f"{', commands <- ' + self._cfg['joint_command_topic'] if self._joint_cmd_sub else ''})")

    def _setup_static_tf(self):
        """Broadcast base_link->{sensor} static transforms (latched /tf_static).

        Opt-in (config publish_static_tf): each entry of static_transforms is
        {child_frame_id, translation:[x,y,z], rotation_wxyz:[w,x,y,z]} expressed
        in the robot base frame (the sensors are children of the robot prim, so
        their local mount pose is exactly base_link->sensor)."""
        transforms = self._cfg.get("static_transforms", [])
        if not self._cfg.get("publish_static_tf") or not transforms:
            return
        try:
            from tf2_ros import StaticTransformBroadcaster
            from geometry_msgs.msg import TransformStamped
        except Exception as e:  # pragma: no cover - tf2_ros not installed
            self._node.get_logger().warn(f"static TF disabled (tf2_ros unavailable): {e}")
            return

        self._static_tf_broadcaster = StaticTransformBroadcaster(self._node)
        base = self._cfg["base_frame_id"]
        stamp = self._node.get_clock().now().to_msg()
        msgs = []
        for t in transforms:
            child = t.get("child_frame_id")
            if not child or child == base:
                continue  # skip identity / unnamed (e.g. sensors reported in base_link)
            ts = TransformStamped()
            ts.header.stamp = stamp
            ts.header.frame_id = base
            ts.child_frame_id = child
            tr = t.get("translation", [0.0, 0.0, 0.0])
            ts.transform.translation.x = float(tr[0])
            ts.transform.translation.y = float(tr[1])
            ts.transform.translation.z = float(tr[2])
            qx, qy, qz, qw = _quat_wxyz_to_xyzw(t.get("rotation_wxyz", [1.0, 0.0, 0.0, 0.0]))
            ts.transform.rotation.x = qx
            ts.transform.rotation.y = qy
            ts.transform.rotation.z = qz
            ts.transform.rotation.w = qw
            msgs.append(ts)
        if msgs:
            self._static_tf_broadcaster.sendTransform(msgs)
            self._node.get_logger().info(
                f"published {len(msgs)} static transform(s) from {base}: "
                f"{[m.child_frame_id for m in msgs]}")

    # --------------------------------------------------------------- per-step
    def publish(self, sim_time: float):
        """Publish all due sensor messages for the current sim time (seconds)."""
        if self._node is None:
            return
        stamp = self._stamp(sim_time)

        if self._clock_pub is not None:
            self._safe(self._publish_clock, sim_time)
        if self._odom_gate.ready(sim_time):
            self._safe(self._publish_odom, stamp)
        if self._imu_gate.ready(sim_time):
            self._safe(self._publish_imu, stamp, sim_time)
        if self._dvl_pub is not None and self._dvl_gate.ready(sim_time):
            self._safe(self._publish_dvl, stamp)
        if self._baro_pub is not None and self._baro_gate.ready(sim_time):
            self._safe(self._publish_baro, stamp)
        if self._sonar_pub is not None and self._sonar_gate.ready(sim_time):
            self._safe(self._publish_sonar, stamp)
        if self._joint_state_pub is not None and self._joint_state_gate.ready(sim_time):
            self._safe(self._publish_joint_state, stamp)

        # Service any timers/callbacks on our node without blocking the sim
        # (this is also what dispatches incoming joint commands).
        rclpy.spin_once(self._node, timeout_sec=0.0)

    def _safe(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:  # pragma: no cover - keep sim alive on publish error
            self._node.get_logger().warn(f"{fn.__name__} failed: {e}", throttle_duration_sec=5.0)

    # --------------------------------------------------------------- builders
    def _stamp(self, sim_time: float):
        from builtin_interfaces.msg import Time
        sec, nanosec = ros2_math.sim_time_to_sec_nanosec(sim_time)
        return Time(sec=sec, nanosec=nanosec)

    def _publish_joint_state(self, stamp):
        """Publish the articulation's joint positions/velocities as
        sensor_msgs/JointState (the input robot_state_publisher needs, together
        with the latched /robot_description, to broadcast the moving TF tree)."""
        from sensor_msgs.msg import JointState
        pos = np.asarray(self._articulation.get_joint_positions(), dtype=float).reshape(-1)
        try:
            vel = np.asarray(self._articulation.get_joint_velocities(), dtype=float).reshape(-1)
        except Exception:  # pragma: no cover - velocity optional
            vel = np.zeros_like(pos)
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(self._joint_names)
        msg.position = [float(x) for x in pos]
        msg.velocity = [float(x) for x in vel]
        self._joint_state_pub.publish(msg)

    def _on_joint_command(self, msg):
        """Apply an incoming sensor_msgs/JointState position command to the
        articulation. The command may be a subset / reordered / contain unknown
        joints -- joint_control.map_named_command reconciles it against the DOF
        order, holding the current target for any joint the command omits."""
        if self._articulation is None or not msg.name:
            return
        try:
            from isaacsim.core.utils.types import ArticulationAction
            current = np.asarray(self._articulation.get_joint_positions(), dtype=float).reshape(-1)
            targets, ignored = joint_control.map_named_command(
                list(msg.name), list(msg.position), self._joint_names, current=current)
            if ignored:
                self._node.get_logger().warn(
                    f"joint command names not on this robot, ignored: {ignored}",
                    throttle_duration_sec=5.0)
            self._articulation.apply_action(ArticulationAction(joint_positions=targets))
        except Exception as e:  # keep the sim alive on a malformed command / API change
            self._node.get_logger().warn(
                f"joint command failed: {e}", throttle_duration_sec=5.0)

    def _publish_clock(self, sim_time: float):
        from rosgraph_msgs.msg import Clock
        msg = Clock()
        msg.clock = self._stamp(sim_time)
        self._clock_pub.publish(msg)

    def _ensure_rigid_prim(self):
        """Wrap the robot prim once the physics backend is live (lazy).

        Built on first use inside the run loop -- after world.play() and the first
        world.step() -- not in initialize(): under Isaac Sim 6.0.1
        SingleRigidPrim.__init__ -> RigidPrim._on_physics_ready() eagerly calls
        get_velocities(), which raises "Failed to get rigid body velocities from
        backend" when the physics tensor view isn't initialized yet, and
        initialize() runs before play()."""
        if self._rigid_prim is None:
            from isaacsim.core.prims import SingleRigidPrim
            from isaacsim.core.utils.prims import get_prim_path
            self._rigid_prim = SingleRigidPrim(prim_path=get_prim_path(self._robot_prim))
        return self._rigid_prim

    def _robot_state(self):
        """Return (pos, quat_wxyz, lin_vel_world, ang_vel_world) as numpy arrays."""
        rp = self._ensure_rigid_prim()
        pos, quat = rp.get_world_pose()
        lin_vel = np.asarray(rp.get_linear_velocity(), dtype=float)
        ang_vel = np.asarray(rp.get_angular_velocity(), dtype=float)
        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float), lin_vel, ang_vel

    def _publish_odom(self, stamp):
        from nav_msgs.msg import Odometry
        pos, quat, lin_vel_w, ang_vel_w = self._robot_state()

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self._cfg["odom_frame_id"]
        msg.child_frame_id = self._cfg["base_frame_id"]

        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        qx, qy, qz, qw = _quat_wxyz_to_xyzw(quat)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Odometry twist is expressed in the child (body) frame.
        lin_b, ang_b = self._to_body(quat, lin_vel_w, ang_vel_w)
        msg.twist.twist.linear.x = float(lin_b[0])
        msg.twist.twist.linear.y = float(lin_b[1])
        msg.twist.twist.linear.z = float(lin_b[2])
        msg.twist.twist.angular.x = float(ang_b[0])
        msg.twist.twist.angular.y = float(ang_b[1])
        msg.twist.twist.angular.z = float(ang_b[2])

        # Modest diagonal covariance so robot_localization will fuse it.
        msg.pose.covariance = self._diag6(1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3)
        msg.twist.covariance = self._diag6(1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2)
        self._odom_pub.publish(msg)

    def _publish_imu(self, stamp, sim_time):
        from sensor_msgs.msg import Imu
        _, quat, lin_vel_w, ang_vel_w = self._robot_state()

        # world -> body rotation (R^T) shared by angular velocity and accel.
        if quat_to_rot_matrix is not None:
            rot = quat_to_rot_matrix(np.asarray(quat, dtype=float))
        else:  # pragma: no cover - fallback if Isaac util unavailable
            rot = self._rot_from_quat(quat)
        rt = rot.T
        ang_b = rt @ ang_vel_w

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self._cfg["imu_frame_id"]
        qx, qy, qz, qw = _quat_wxyz_to_xyzw(quat)
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        msg.angular_velocity.x = float(ang_b[0])
        msg.angular_velocity.y = float(ang_b[1])
        msg.angular_velocity.z = float(ang_b[2])

        # Linear acceleration = specific force, i.e. what a real accelerometer
        # measures: f = R^T * (a_world - g_world). Gravity points down (-z), so at
        # rest the reading is +g on the axis opposing gravity (REP-145). a_world is
        # the finite difference of *world*-frame linear velocity; before a second
        # sample exists (or right after a reset) we assume a_world = 0 so the
        # static gravity term is still reported instead of a spurious zero.
        g = float(self._cfg["gravity"])
        a_world = np.zeros(3)
        if self._last_world_lin_vel is not None and self._last_imu_time is not None:
            dt = sim_time - self._last_imu_time
            if dt > 1e-6:
                a_world = (lin_vel_w - self._last_world_lin_vel) / dt
        f_body = ros2_math.specific_force_body(rot, a_world, g)
        msg.linear_acceleration.x = float(f_body[0])
        msg.linear_acceleration.y = float(f_body[1])
        msg.linear_acceleration.z = float(f_body[2])
        self._last_world_lin_vel = lin_vel_w
        self._last_imu_time = sim_time

        msg.orientation_covariance = self._diag3(1e-3, 1e-3, 1e-3)
        msg.angular_velocity_covariance = self._diag3(1e-3, 1e-3, 1e-3)
        msg.linear_acceleration_covariance = self._diag3(1e-2, 1e-2, 1e-2)
        self._imu_pub.publish(msg)

    def _publish_dvl(self, stamp):
        from geometry_msgs.msg import TwistWithCovarianceStamped
        vel = np.asarray(self._dvl.get_linear_vel(), dtype=float)  # body frame
        msg = TwistWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._cfg["dvl_frame_id"]
        msg.twist.twist.linear.x = float(vel[0])
        msg.twist.twist.linear.y = float(vel[1])
        msg.twist.twist.linear.z = float(vel[2])
        msg.twist.covariance = self._diag6(1e-2, 1e-2, 1e-2, 1e9, 1e9, 1e9)
        self._dvl_pub.publish(msg)

    def _publish_baro(self, stamp):
        from sensor_msgs.msg import FluidPressure
        msg = FluidPressure()
        msg.header.stamp = stamp
        msg.header.frame_id = self._cfg["baro_frame_id"]
        msg.fluid_pressure = float(self._baro.get_pressure())  # Pascals
        msg.variance = 0.0
        self._baro_pub.publish(msg)

    def _publish_sonar(self, stamp):
        """Build a ProjectedSonarImage from the OceanSim imaging sonar.

        OceanSim bins each ping onto a polar grid stored in
        ``ImagingSonarSensor.sonar_map`` -- a Warp ``vec3`` array of shape
        ``(n_range, n_azimuth)`` whose third channel is the processed,
        display-normalised intensity in ``[0, 1]`` (see
        ``utils/ImagingSonar_kernels.make_sonar_map_*``).  We expose it as the
        standard ``marine_acoustic_msgs/ProjectedSonarImage`` used by
        ``sonar_image_proc`` / ``sonar_proc``, emitting one ``uint8`` intensity
        per (beam, range) cell, row-major over beams (``SonarImageData``
        convention).

        Note: ``make_sonar_image()`` is deliberately *not* used here -- it
        returns the RGBA viewport texture (shape ``(n_range, n_azimuth, 4)``),
        not the per-beam intensities the sonar pipeline expects.
        """
        from marine_acoustic_msgs.msg import (
            ProjectedSonarImage, PingInfo, SonarImageData)

        sonar_map = getattr(self._sonar, "sonar_map", None)
        if sonar_map is None:
            return
        grid = sonar_map.numpy() if hasattr(sonar_map, "numpy") else np.asarray(sonar_map)
        # (n_range, n_azimuth, 3) vec3 grid; channel 2 is intensity in [0, 1].
        if grid.ndim != 3 or grid.shape[2] < 3 or grid.size == 0:
            return
        # marine_acoustic_msgs SonarImageData is RANGE-major: element (range r,
        # beam b) lives at flat index r*n_beams + b. This matches sonar_image_proc
        # (sonar_image_msg_interface.h), sonar_proc, and the real Oculus driver
        # (oculus_sonar_driver/ping_to_sonar_image.h, which pushes r outer / b
        # inner). The grid is already (n_range, n_azimuth), so use it directly --
        # transposing to beam-major scrambles the image for every consumer.
        # Range-major uint8 image (see ros2_math.sonar_intensity_uint8).
        img8, n_range, n_beams = ros2_math.sonar_intensity_uint8(grid)

        min_range, max_range = self._sonar.get_range()
        hori_fov, vert_fov = self._sonar.get_fov()  # degrees
        geom = self._sonar_geometry(n_range, n_beams, min_range, max_range,
                                    hori_fov, vert_fov)

        msg = ProjectedSonarImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self._cfg["sonar_frame_id"]

        ping = PingInfo()
        # Acoustic carrier frequency for ping_info. Prefer an explicit config
        # override, then the sensor's own MODELLED acoustic frequency
        # (ImagingSonarSensor.acoustic_frequency = 375 kHz for the Oculus M370s;
        # RtxAcousticSensor.center_frequency for the RTX backend). Do NOT use the
        # Camera `frequency` attribute -- on the imaging sonar that is the render
        # frame rate (e.g. 5-60 Hz), not the acoustic carrier, which would publish
        # a wildly wrong ping frequency to sonar_image_proc / sonar_proc.
        _sonar_freq = (self._cfg.get("sonar_acoustic_freq")
                       or getattr(self._sonar, "acoustic_frequency", None)
                       or getattr(self._sonar, "center_frequency", None))
        ping.frequency = float(_sonar_freq) if _sonar_freq else 375e3
        ping.sound_speed = float(self._cfg["sound_speed"])
        ping.tx_beamwidths = geom["tx_beamwidths"]
        ping.rx_beamwidths = geom["rx_beamwidths"]
        msg.ping_info = ping

        msg.beam_directions = geom["beam_directions"]
        msg.ranges = geom["ranges"]

        data = SonarImageData()
        data.is_bigendian = False
        data.dtype = SonarImageData.DTYPE_UINT8
        data.beam_count = int(n_beams)
        data.data = img8.reshape(-1).tobytes()
        msg.image = data
        self._sonar_pub.publish(msg)

    def _sonar_geometry(self, n_range, n_beams, min_range, max_range, hori_fov, vert_fov):
        """Cached ProjectedSonarImage geometry, rebuilt only if the sonar grid
        shape / FOV changes (it does not, frame to frame).

        * beam_directions: bearings spread symmetrically across the horizontal
          FOV, encoded with the SAME convention as the real Oculus driver
          (oculus_sonar_driver/ping_to_sonar_image.h) and as sonar_image_proc /
          sonar_proc expect: azimuth in -y, range/forward in +z, elevation x=0,
          so az = atan2(-y, z). Putting the spread in x instead (y=0) collapses
          every azimuth to 0 -> draw_sonar builds an empty remap LUT (assertion
          crash) and sonar_proc emits a degenerate cloud.
        * tx/rx_beamwidths: BOTH per-beam (length n_beams), per the Oculus driver
          / marine_acoustic_msgs convention. rx is the azimuth (horizontal)
          beamwidth of each receive beam; tx is the elevation (vertical)
          beamwidth of the transmit swath. sonar_proc indexes both per beam, so a
          length-1 tx array reads out of bounds.
        """
        key = (n_range, n_beams, min_range, max_range, hori_fov, vert_fov)
        cache = self._sonar_geom
        if cache is not None and cache["key"] == key:
            return cache
        from geometry_msgs.msg import Vector3
        geom = {
            "key": key,
            "beam_directions": [
                Vector3(x=x, y=y, z=z)
                for (x, y, z) in ros2_math.sonar_beam_directions(hori_fov, n_beams)
            ],
            "ranges": ros2_math.sonar_ranges(min_range, max_range, n_range),
            "tx_beamwidths": [math.radians(vert_fov)] * n_beams,
            "rx_beamwidths": [math.radians(hori_fov / max(n_beams, 1))] * n_beams,
        }
        self._sonar_geom = geom
        return geom

    # ----------------------------------------------------------------- helpers
    def _to_body(self, quat_wxyz, lin_vel_w, ang_vel_w):
        """Rotate world-frame linear/angular velocity into the body frame."""
        if quat_to_rot_matrix is not None:
            rot = quat_to_rot_matrix(np.asarray(quat_wxyz, dtype=float))
        else:  # pragma: no cover - fallback if Isaac util unavailable
            rot = self._rot_from_quat(quat_wxyz)
        rt = rot.T
        return rt @ lin_vel_w, rt @ ang_vel_w

    @staticmethod
    def _rot_from_quat(q):
        return ros2_math.rot_from_quat_wxyz(q)

    @staticmethod
    def _diag3(a, b, c):
        return ros2_math.diag3(a, b, c)

    @staticmethod
    def _diag6(*vals):
        return ros2_math.diag6(*vals)

    # ------------------------------------------------------------------ close
    def close(self):
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._ros2_acquired:
            ros2_context.release()
            self._ros2_acquired = False

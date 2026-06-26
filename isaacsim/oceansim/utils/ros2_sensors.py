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

try:
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
except Exception:  # pragma: no cover - only importable inside Isaac Sim
    quat_to_rot_matrix = None


def _sensor_qos(depth: int = 10) -> QoSProfile:
    """Best-effort, keep-last QoS matching ``rclpy.qos.qos_profile_sensor_data``."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _clock_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
    )


def _quat_wxyz_to_xyzw(q):
    """Isaac uses scalar-first (w, x, y, z); ROS uses scalar-last (x, y, z, w)."""
    return (float(q[1]), float(q[2]), float(q[3]), float(q[0]))


class _RateGate:
    """Returns True at most every ``1/hz`` seconds of sim time (hz<=0 => always)."""

    def __init__(self, hz: float):
        self._period = (1.0 / hz) if hz and hz > 0 else 0.0
        self._last = None

    def ready(self, now: float) -> bool:
        if self._period <= 0.0:
            return True
        # Reset if sim time jumped backward (e.g. a world reset) so publishing
        # resumes immediately instead of stalling until `now` catches back up.
        if self._last is None or now < self._last or (now - self._last) >= self._period:
            self._last = now
            return True
        return False


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
        # rates (Hz, <=0 means "every sim step")
        "odom_rate": 60.0,
        "imu_rate": 100.0,
        "dvl_rate": 10.0,
        "baro_rate": 10.0,
        "sonar_rate": 5.0,
        # toggles
        "publish_clock": True,
        # sonar acoustic params (used to fill ProjectedSonarImage.ping_info)
        "sound_speed": 1500.0,
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

        # publishers (created lazily in initialize, may be None if msg pkg missing)
        self._odom_pub = None
        self._imu_pub = None
        self._dvl_pub = None
        self._baro_pub = None
        self._sonar_pub = None
        self._clock_pub = None

        # rate gates
        self._odom_gate = _RateGate(self._cfg["odom_rate"])
        self._imu_gate = _RateGate(self._cfg["imu_rate"])
        self._dvl_gate = _RateGate(self._cfg["dvl_rate"])
        self._baro_gate = _RateGate(self._cfg["baro_rate"])
        self._sonar_gate = _RateGate(self._cfg["sonar_rate"])

        # IMU finite-difference state (world-frame velocity, for specific force)
        self._last_world_lin_vel = None
        self._last_imu_time = None

    # ------------------------------------------------------------------ setup
    def initialize(self):
        """Create the rclpy node and publishers for the active sensors."""
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.prims import get_prim_path

        ros2_context.acquire()
        self._ros2_acquired = True
        self._node = rclpy.create_node("oceansim_sensor_publisher")

        self._rigid_prim = SingleRigidPrim(prim_path=get_prim_path(self._robot_prim))

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

        self._node.get_logger().info("OceanSimSensorPublisher initialized")

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

        # Service any timers/callbacks on our node without blocking the sim.
        rclpy.spin_once(self._node, timeout_sec=0.0)

    def _safe(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:  # pragma: no cover - keep sim alive on publish error
            self._node.get_logger().warn(f"{fn.__name__} failed: {e}", throttle_duration_sec=5.0)

    # --------------------------------------------------------------- builders
    def _stamp(self, sim_time: float):
        from builtin_interfaces.msg import Time
        sec = int(sim_time)
        nanosec = int(round((sim_time - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        return Time(sec=sec, nanosec=nanosec)

    def _publish_clock(self, sim_time: float):
        from rosgraph_msgs.msg import Clock
        msg = Clock()
        msg.clock = self._stamp(sim_time)
        self._clock_pub.publish(msg)

    def _robot_state(self):
        """Return (pos, quat_wxyz, lin_vel_world, ang_vel_world) as numpy arrays."""
        pos, quat = self._rigid_prim.get_world_pose()
        lin_vel = np.asarray(self._rigid_prim.get_linear_velocity(), dtype=float)
        ang_vel = np.asarray(self._rigid_prim.get_angular_velocity(), dtype=float)
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
        f_world = a_world - np.array([0.0, 0.0, -g])  # subtract gravity vector
        f_body = rt @ f_world
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
        from geometry_msgs.msg import Vector3

        sonar_map = getattr(self._sonar, "sonar_map", None)
        if sonar_map is None:
            return
        grid = sonar_map.numpy() if hasattr(sonar_map, "numpy") else np.asarray(sonar_map)
        # (n_range, n_azimuth, 3) vec3 grid; channel 2 is intensity in [0, 1].
        if grid.ndim != 3 or grid.shape[2] < 3 or grid.size == 0:
            return
        # Transpose polar (range, azimuth) -> (beam, range), row-major over beams.
        intensity = np.ascontiguousarray(grid[:, :, 2].T)
        n_beams, n_range = intensity.shape

        min_range, max_range = self._sonar.get_range()
        hori_fov, _vert_fov = self._sonar.get_fov()  # degrees

        msg = ProjectedSonarImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self._cfg["sonar_frame_id"]

        ping = PingInfo()
        ping.frequency = float(getattr(self._sonar, "frequency", 1.2e6))
        ping.sound_speed = float(self._cfg["sound_speed"])
        # An imaging sonar has a SINGLE transmit beam insonifying the whole
        # horizontal swath, and one receive beam per bearing. So tx_beamwidths is
        # a single element (the tx swath), while rx_beamwidths is per receive beam.
        ping.tx_beamwidths = [math.radians(hori_fov)]
        ping.rx_beamwidths = [math.radians(hori_fov / max(n_beams, 1))] * n_beams
        msg.ping_info = ping

        # Bearings spread symmetrically across the horizontal FOV.
        half = math.radians(hori_fov) / 2.0
        bearings = np.linspace(-half, half, n_beams)
        msg.beam_directions = [
            Vector3(x=float(math.sin(b)), y=0.0, z=float(math.cos(b))) for b in bearings
        ]

        msg.ranges = np.linspace(
            float(min_range), float(max_range), n_range).astype(np.float32).tolist()

        data = SonarImageData()
        data.is_bigendian = False
        data.dtype = SonarImageData.DTYPE_UINT8
        data.beam_count = int(n_beams)
        # Intensity is normalised to [0, 1]; scale to the full uint8 range.
        img8 = np.clip(intensity * 255.0, 0.0, 255.0).astype(np.uint8)
        data.data = img8.reshape(-1).tobytes()
        msg.image = data
        self._sonar_pub.publish(msg)

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
        w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    @staticmethod
    def _diag3(a, b, c):
        return [a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, c]

    @staticmethod
    def _diag6(*vals):
        m = [0.0] * 36
        for i, v in enumerate(vals):
            m[i * 6 + i] = v
        return m

    # ------------------------------------------------------------------ close
    def close(self):
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._ros2_acquired:
            ros2_context.release()
            self._ros2_acquired = False

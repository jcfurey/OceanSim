import time
import os
import numpy as np
from enum import Enum

from isaacsim.core.prims import RigidPrim, SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
from isaacsim.core.utils.rotations import quat_to_rot_matrix

'''
Attention:

Before OceanSim extension being activated, the extension isaacsim.ros2.bridge should be activated, otherwise rclpy will
fail to be loaded.

so, we suggest that make sure the extension isaacsim.ros2.bridge is being setup to "AUTOLOADED" in Window->Extension.
'''
import rclpy

from isaacsim.oceansim.utils import ros2_context
from isaacsim.oceansim.utils import ros2_control_math

try:
    from pxr import PhysxSchema
    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False
    PhysxSchema = None

ROS2_AVAILABLE = False

class ROS2_CONTROL_MODE(Enum):
    VEL = 1          # velocity control: kinematic, sets the body velocity each step
    FORCE = 2        # force control: body-frame Wrench applied as force / torque
    VEL_DYNAMIC = 3  # dynamic velocity control: cmd_vel -> PI -> force / torque
    THRUSTER = 4     # thruster control: normalised per-thruster commands (needs a vehicle model)


# Mode names as the GUI dropdown / runner config spell them.
CONTROL_MODE_NAMES = {
    "velocity control": ROS2_CONTROL_MODE.VEL,
    "force control": ROS2_CONTROL_MODE.FORCE,
    "dynamic velocity control": ROS2_CONTROL_MODE.VEL_DYNAMIC,
    "thruster control": ROS2_CONTROL_MODE.THRUSTER,
}

class ROS2ControlReceiver:
    """
    ROS2 Control Receiver
    
    for recieving velocity and force command
    """
    
    def __init__(self, robot_prim, name="ROS2ControlReceiver",
                 max_linear_vel=None, max_angular_vel=None,
                 max_force=None, max_torque=None, velocity_pi=None, vehicle=None):
        """
        initialize ROS2 Control Receiver

        Args:
            robot_prim: robot prim path
            name (str): receiver name
            max_linear_vel (float | None): clamp on the incoming linear-velocity
                vector's magnitude (m/s). None (default) = unbounded.
            max_angular_vel (float | None): clamp on the incoming angular-velocity
                vector's magnitude (rad/s). None (default) = unbounded.
            max_force (float | None): clamp on the force magnitude (N), for both
                force control and the dynamic velocity loop's output. None
                (default) = unbounded.
            max_torque (float | None): clamp on the torque magnitude (N*m), as
                max_force. None (default) = unbounded.
            velocity_pi (dict | None): ros2_control_math.BodyVelocityPI keyword
                overrides for dynamic velocity control (kp_lin, ki_lin, kp_ang,
                ki_ang, i_limit_lin, i_limit_ang, damping_lin, damping_ang). The
                damping feedforward defaults to the prim's PhysX linear /
                angular damping.
            vehicle (vehicle_physics.IsaacVehicle | None): the platform's
                hydrodynamics + thruster model. When set, force and dynamic
                velocity control command its thrusters (so thrust limits and
                lag apply) instead of applying the wrench directly, the velocity
                loop's feedforward is the model's drag, and thruster control is
                available.
        """
        self._name = name
        self._robot_prim = robot_prim
        self._rigid_prim = None  # lazily created SingleRigidPrim, cached across ticks

        # configuration
        self._enable_ros2 = False
        self._ros2_acquired = False

        self._ros2_control_mode = ROS2_CONTROL_MODE.VEL  # control mode
        self._ros2_vel_node = None
        self._ros2_force_node = None

        # Command clamping (magnitude-preserving-direction). No repo-documented
        # physical limits exist for this vehicle today, so these default to
        # None (unbounded) -- set real numbers here once they're known.
        self._max_linear_vel = max_linear_vel
        self._max_angular_vel = max_angular_vel
        self._max_force = max_force
        self._max_torque = max_torque

        # command cache
        self.force_cmd = [0.0, 0.0, 0.0]
        self.torque_cmd = [0.0, 0.0, 0.0]
        self.linear_vel = [0.0, 0.0, 0.0]
        self.angular_vel = [0.0, 0.0, 0.0]
        # Watchdog clock: time.monotonic(), NOT time.time() -- a wall-clock step
        # (NTP correction, DST, manual clock set) must not spuriously trip or
        # indefinitely defer the dead-man timeout.
        self.last_command_time = time.monotonic()
        self.command_timeout = 2.0
        # Set once we've warned about a stale command, so the watchdog logs a
        # single message per disconnect instead of spamming every tick; reset
        # as soon as a fresh command arrives.
        self._stale_warned = False
        # While True, incoming command callbacks are DISCARDED (used to flush
        # messages queued for the inactive mode's node at a mode switch, so an
        # arbitrarily old queued command is not replayed as "fresh").
        self._discard_commands = False
        # Incremented by every accepted command; lets update_control's drain
        # loop stop as soon as a spin delivers nothing instead of guessing.
        self._rx_count = 0
        # Per-message receive prints (2 per command) run on the sim thread at
        # teleop rate; keep the scaffolding available but off by default.
        self.verbose = False
        self._update_count = 0
        
        # Forces go through a RigidPrim tensor view (N, N*m, body frame), like
        # the scenario's manual control: PhysxForceAPI's default "acceleration"
        # mode read a Wrench's newtons as m/s^2 (26x too strong on a 26 kg
        # vehicle) and, per upstream, stopped the IMU updating. Built lazily
        # after world.play(); mass / inertia diagonal read from it.
        self._view = None
        self._view_failed_warned = False
        self._mass = 1.0
        self._inertia = np.ones(3)
        self._velocity_pi_params = dict(velocity_pi or {})
        self._vel_pi = None
        self._stamped_vel = False
        self._vehicle = vehicle
        self.thruster_cmd = None
        self._ros2_thruster_node = None
        
        print(f"[{self._name}] Initialized for robot prim")
        
    def initialize(self, enable_ros2=True, vel_topic="/oceansim/robot/vel_cmd",
                   force_topic="/oceansim/robot/force_cmd", stamped_vel=False,
                   command_timeout=None, thruster_topic="/oceansim/robot/thruster_cmd"):
        """
        initialize receiver function
        
        Args:
            enable_ros2 (bool): whether using ros2
            vel_topic (str): topic name of vel
            force_topic (str): topic name of force(include torque)
            stamped_vel (bool): subscribe to geometry_msgs/TwistStamped instead
                of Twist on vel_topic (Nav2's enable_stamped_cmd_vel; the
                default from Kilted on).
            command_timeout (float | None): dead-man timeout (s); None keeps 2.0.
            thruster_topic (str): std_msgs/Float64MultiArray of normalised
                per-thruster commands in [-1, 1] for thruster control (only
                subscribed with a vehicle model).
        """
        self._enable_ros2 = enable_ros2
        self._thruster_topic = thruster_topic
        self._vel_topic = vel_topic
        self._force_topic = force_topic
        self._stamped_vel = bool(stamped_vel)
        if command_timeout is not None:
            self.command_timeout = float(command_timeout)
        
        if not self._enable_ros2:
            print(f'[{self._name}] ROS2 disabled by configuration')
            return
        
        self._setup_subscriber()
        
        print(f'[{self._name}] Control Receiver Initialized:')
        print(f'[{self._name}] ROS2 Bridge: {self._enable_ros2}')
        if PXR_AVAILABLE and self._robot_prim:
            print(f'[{self._name}] Robot Prim: {self._robot_prim.GetPath()}')
        else:
            print(f'[{self._name}] Robot Prim: {self._robot_prim}')

    def set_scenario_force_api(self, scenario_force_api):
        """Deprecated no-op: forces are applied through a RigidPrim view now
        (see __init__), not PhysxForceAPI. Kept so older callers don't break."""

    def _ensure_view(self):
        """The robot's RigidPrim view, created on first use after world.play()
        (it needs the physics simulation view). Returns None until then."""
        if self._view is not None:
            return self._view
        try:
            view = RigidPrim(prim_paths_expr=get_prim_path(self._robot_prim))
            view.initialize()
        except Exception as e:  # noqa: BLE001 - physics not live yet: retry next step
            if not self._view_failed_warned:
                self._view_failed_warned = True
                print(f'[{self._name}] rigid-body view not ready yet ({e}); retrying')
            return None
        self._view = view

        def _np(a):
            return a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        try:
            mass = float(_np(view.get_masses()).reshape(-1)[0])
            inertia = _np(view.get_inertias()).reshape(-1, 9)[0]
            diag = np.array([inertia[0], inertia[4], inertia[8]], dtype=float)
            if mass > 0.0:
                self._mass = mass
            if np.all(np.isfinite(diag)) and np.all(diag > 0.0):
                self._inertia = diag
        except Exception as e:  # noqa: BLE001
            print(f'[{self._name}] could not read mass / inertia ({e}); '
                  f'dynamic velocity control uses unit mass / inertia')
        return view

    def _prim_damping(self):
        """(linear, angular) PhysX damping (1/s) of the robot body -- the
        platforms' water-drag proxy, used as the velocity loop's feedforward."""
        if not PXR_AVAILABLE or self._robot_prim is None:
            return 0.0, 0.0
        try:
            api = PhysxSchema.PhysxRigidBodyAPI(self._robot_prim)
            lin = api.GetLinearDampingAttr().Get()
            ang = api.GetAngularDampingAttr().Get()
            return float(lin or 0.0), float(ang or 0.0)
        except Exception:  # noqa: BLE001 - API not applied: no feedforward
            return 0.0, 0.0

    def _ensure_vel_pi(self):
        if self._vel_pi is None:
            params = dict(self._velocity_pi_params)
            lin_d, ang_d = self._prim_damping()
            params.setdefault("damping_lin", lin_d)
            params.setdefault("damping_ang", ang_d)
            self._vel_pi = ros2_control_math.BodyVelocityPI(**params)
            ff = ("drag from the vehicle model" if self._vehicle is not None
                  else f"PhysX damping {self._vel_pi.damping[[0, 3]].tolist()} 1/s")
            print(f'[{self._name}] dynamic velocity control: kp={self._vel_pi.kp[[0, 3]].tolist()}, '
                  f'ki={self._vel_pi.ki[[0, 3]].tolist()}, feedforward: {ff}')
        return self._vel_pi
    
    def _setup_subscriber(self):
        """
        setting the ROS2 subscriber
        """
        try:
            # import ROS2 module
            from geometry_msgs.msg import Twist, TwistStamped, Wrench
            
            # Initialize/share the rclpy context (ref-counted across components)
            ros2_context.acquire()
            self._ros2_acquired = True

            # Create velocity subscriber node
            node_name = f'oceansim_rob_velocity_control_{self._name.lower()}'.replace(' ', '_')
            self._ros2_vel_node = rclpy.create_node(node_name)
            self._ros2_vel_subscriber = self._ros2_vel_node.create_subscription(
                TwistStamped if self._stamped_vel else Twist,
                self._vel_topic,
                self._vel_callback,
                10
            )

            # Per-thruster commands, when a vehicle model with thrusters exists
            if self._vehicle is not None:
                from std_msgs.msg import Float64MultiArray
                node_name = f'oceansim_rob_thruster_control_{self._name.lower()}'.replace(' ', '_')
                self._ros2_thruster_node = rclpy.create_node(node_name)
                self._ros2_thruster_node.create_subscription(
                    Float64MultiArray, self._thruster_topic, self._thruster_callback, 10)

            # Create force subscriber node
            node_name = f'oceansim_rob_force_control_{self._name.lower()}'.replace(' ', '_')
            self._ros2_force_node = rclpy.create_node(node_name)
            self._force_subscriber = self._ros2_force_node.create_subscription(
                Wrench,
                self._force_topic,
                self._force_callback,
                10
            )
            
        except Exception as e:
            self._enable_ros2 = False
            # Destroy any node already created before the failure (e.g. the vel
            # node when force-node creation raises), or it leaks for the process
            # lifetime -- close()'s teardown was gated on _enable_ros2, which we
            # just set False.
            for _attr in ("_ros2_vel_node", "_ros2_force_node", "_ros2_thruster_node"):
                _node = getattr(self, _attr, None)
                if _node is not None:
                    try:
                        _node.destroy_node()
                    except Exception:  # noqa: BLE001
                        pass
                    setattr(self, _attr, None)
            print(f'[{self._name}] ROS2 subscriber setup failed: {e}')

    def _setup_ros2_control_mode(self, ctrl_mode):
        new_mode = CONTROL_MODE_NAMES.get(ctrl_mode)
        if new_mode is None:
            print(f'[{self._name}] unknown ROS2 control mode {ctrl_mode!r}; '
                  f'expected one of {list(CONTROL_MODE_NAMES)}')
            return
        if new_mode == self._ros2_control_mode:
            return
        if new_mode == ROS2_CONTROL_MODE.THRUSTER and self._vehicle is None:
            print(f'[{self._name}] thruster control needs a vehicle model (platform '
                  f'hydrodynamics); staying in {self._ros2_control_mode.name}')
            return
        # Mode switch safety:
        # 1. Flush both nodes' queues with callbacks discarding, so a command
        #    queued while the OTHER mode was active (up to depth 10, arbitrarily
        #    old) is not delivered right after the switch, stamped "now", and
        #    replayed as a fresh command -- defeating the dead-man watchdog.
        # 2. Zero the cached commands and mark them stale, so nothing actuates
        #    until a genuinely fresh post-switch command arrives.
        # (Forces go through the tensor view and last one step, so nothing
        # persists across the switch; the velocity loop's integral is reset.)
        self._discard_commands = True
        try:
            for node in (self._ros2_vel_node, self._ros2_force_node, self._ros2_thruster_node):
                if node is None:
                    continue
                for _ in range(16):
                    rclpy.spin_once(node, timeout_sec=0.0)
        finally:
            self._discard_commands = False
        self.linear_vel = [0.0, 0.0, 0.0]
        self.angular_vel = [0.0, 0.0, 0.0]
        self.force_cmd = [0.0, 0.0, 0.0]
        self.torque_cmd = [0.0, 0.0, 0.0]
        self.last_command_time = time.monotonic() - (self.command_timeout + 1.0)
        self.thruster_cmd = None
        if self._vel_pi is not None:
            self._vel_pi.reset()
        if self._vehicle is not None:
            self._vehicle.stop()
        self._ros2_control_mode = new_mode
    
    def _vel_callback(self, msg):
        """
        msg type: geometry_msgs/Twist, or TwistStamped with stamped_vel

        include linear and angular velocity
        """
        if self._discard_commands:
            return  # mode-switch queue flush in progress
        if self._stamped_vel:
            msg = msg.twist
        if self.verbose:
            print(f'[{self._name}] receive ROS2 msg, type: {type(msg).__name__}, linear: {msg.linear}, angular: {msg.angular}')

        if not self._enable_ros2:
            return

        try:
            self.linear_vel = self._clamp_magnitude(
                [msg.linear.x, msg.linear.y, msg.linear.z], self._max_linear_vel)
            self.angular_vel = self._clamp_magnitude(
                [msg.angular.x, msg.angular.y, msg.angular.z], self._max_angular_vel)
            self.last_command_time = time.monotonic()
            self._stale_warned = False  # fresh command -- watchdog can warn again next time it goes stale
            self._rx_count += 1

            if self.verbose:
                print(f'Received velocity - Linear: {self.linear_vel}, Angular: {self.angular_vel}')
            # self._update_receive_stats(current_time)

        except Exception as e:
            print(f'[{self._name}] Vel Receive Failed: {e}')

    def _force_callback(self, msg):
        """
        msg type: geometry_msgs/Wrench

        include force and torque
        """
        if self._discard_commands:
            return  # mode-switch queue flush in progress
        if self.verbose:
            print(f'[{self._name}] receive ROS2 msg, type: {type(msg).__name__}, force: {msg.force}, torque: {msg.torque}')

        if not self._enable_ros2:
            return

        try:
            self.force_cmd = self._clamp_magnitude(
                [msg.force.x, msg.force.y, msg.force.z], self._max_force)
            self.torque_cmd = self._clamp_magnitude(
                [msg.torque.x, msg.torque.y, msg.torque.z], self._max_torque)
            self.last_command_time = time.monotonic()
            self._stale_warned = False  # fresh command -- watchdog can warn again next time it goes stale
            self._rx_count += 1

            if self.verbose:
                print(f'Received force - Force: {self.force_cmd}, Torque: {self.torque_cmd}')

        except Exception as e:
            print(f'[{self._name}] force Receive Failed: {e}')

    def _thruster_callback(self, msg):
        """msg type: std_msgs/Float64MultiArray, one normalised command in
        [-1, 1] per thruster (the platform's thruster order)."""
        if self._discard_commands or not self._enable_ros2:
            return
        data = list(msg.data)
        if self._vehicle is not None and len(data) != self._vehicle.thruster_count:
            print(f'[{self._name}] thruster command has {len(data)} values, '
                  f'expected {self._vehicle.thruster_count}; ignored')
            return
        self.thruster_cmd = data
        self.last_command_time = time.monotonic()
        self._stale_warned = False
        self._rx_count += 1

    @staticmethod
    def _clamp_magnitude(vec, max_mag):
        """Clamp a 3-vector's magnitude to max_mag, preserving direction.

        max_mag=None (the default everywhere in this class) disables clamping.
        Delegates to ros2_control_math (pure, unit tested independently of
        rclpy/Isaac Sim -- see tests/test_ros2_control.py).
        """
        return ros2_control_math.clamp_magnitude(vec, max_mag)

    def update_control(self, dt=None):
        """
        update control
        
        this function will be called in each simulation step. ( in scenario.update_scenario() )
        dt (float | None): the physics step, for the dynamic velocity loop's
        integral (None: P + feedforward only this step).
        """
        if not self._enable_ros2 or not self._ros2_vel_node or not self._ros2_force_node:
            return
        
        try:
            if self._ros2_control_mode == ROS2_CONTROL_MODE.VEL: # velocity mode
                # DRAIN the cmd_vel queue every step: rclpy.spin_once dispatches at
                # most ONE callback per call, so with a KEEP_LAST depth-10 queue a
                # publisher faster than the physics tick built a backlog and the
                # vehicle actuated commands up to 10 messages old. The bounded loop
                # stops as soon as a spin delivers nothing (_rx_count unchanged), so
                # the idle cost stays one no-op spin per tick.
                self._drain_node(self._ros2_vel_node)
                # Dead-man's-switch, evaluated AFTER the drain: if the link has
                # dropped, zero the command instead of actuating the last one
                # forever. Checking before the spin (the old order) zeroed a
                # command that had already arrived in the queue this tick -- a
                # one-tick false dropout on every reconnect.
                stale = self._is_command_stale()

                if self._rigid_prim is None:
                    self._rigid_prim = SingleRigidPrim(prim_path=get_prim_path(self._robot_prim))
                lin_cmd = [0.0, 0.0, 0.0] if stale else self.linear_vel
                ang_cmd = [0.0, 0.0, 0.0] if stale else self.angular_vel
                # The incoming Twist is a body-frame command (ROS cmd_vel
                # convention), but Isaac's set_*_velocity take world-frame
                # vectors. Rotate body -> world by the robot's orientation.
                lin_w, ang_w = self._body_to_world(lin_cmd, ang_cmd)
                self._rigid_prim.set_linear_velocity(lin_w)
                self._rigid_prim.set_angular_velocity(ang_w)

            elif self._ros2_control_mode == ROS2_CONTROL_MODE.VEL_DYNAMIC:
                self._drain_node(self._ros2_vel_node)
                stale = self._is_command_stale()
                view = self._ensure_view()
                if view is None:
                    return
                # Dead-man: a stale link commands zero velocity -- the loop
                # actively brakes and holds, like the kinematic mode's stop.
                lin_cmd = [0.0, 0.0, 0.0] if stale else self.linear_vel
                ang_cmd = [0.0, 0.0, 0.0] if stale else self.angular_vel
                lin_b, ang_b = self._body_velocity(view)
                acc_lin, acc_ang = self._ensure_vel_pi().update(
                    lin_cmd, ang_cmd, lin_b, ang_b, dt)
                if self._vehicle is not None:
                    # Thrust = (M_RB + M_A) a + the model's drag at the command.
                    eff = self._vehicle.effective_mass()
                    ff = self._vehicle.feedforward(np.concatenate([lin_cmd, ang_cmd]))
                    wrench = eff * np.concatenate([acc_lin, acc_ang]) + ff
                    self._vehicle.command_wrench(
                        self._clamp_magnitude(wrench[:3], self._max_force),
                        self._clamp_magnitude(wrench[3:], self._max_torque))
                    return
                force, torque = ros2_control_math.body_wrench(
                    acc_lin, acc_ang, self._mass, self._inertia)
                self._apply_body_wrench(view, force, torque)

            elif self._ros2_control_mode == ROS2_CONTROL_MODE.FORCE: # force mode
                self._drain_node(self._ros2_force_node)
                if self._vehicle is not None:
                    if self._is_command_stale():
                        self._vehicle.stop()
                    else:
                        self._vehicle.command_wrench(
                            self._clamp_magnitude(self.force_cmd, self._max_force),
                            self._clamp_magnitude(self.torque_cmd, self._max_torque))
                    return
                if self._is_command_stale():
                    return  # view forces last one step: nothing applied = zero wrench
                view = self._ensure_view()
                if view is None:
                    return
                self._apply_body_wrench(view, self.force_cmd, self.torque_cmd)

            elif self._ros2_control_mode == ROS2_CONTROL_MODE.THRUSTER:
                self._drain_node(self._ros2_thruster_node)
                if self._is_command_stale() or self.thruster_cmd is None:
                    self._vehicle.stop()
                else:
                    self._vehicle.command_thrusters(self.thruster_cmd)

        except Exception as e:
            print(f'[{self._name}] Control Update Failed: {e}')

    def _body_velocity(self, view):
        """Measured (linear m/s, angular rad/s) body-frame velocity."""
        def _np(a):
            return a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        _, quats = view.get_world_poses()
        quat = _np(quats).reshape(-1, 4)[0]
        lin_w = _np(view.get_linear_velocities()).reshape(-1, 3)[0]
        ang_w = _np(view.get_angular_velocities()).reshape(-1, 3)[0]
        return (ros2_control_math.world_to_body(quat, lin_w),
                ros2_control_math.world_to_body(quat, ang_w))

    def _apply_body_wrench(self, view, force, torque):
        """Apply a body-frame force (N) / torque (N*m) at the centre of mass for
        this physics step, clamped to max_force / max_torque."""
        force = self._clamp_magnitude(force, self._max_force)
        torque = self._clamp_magnitude(torque, self._max_torque)
        view.apply_forces_and_torques_at_pos(
            forces=np.asarray(force, dtype=np.float32).reshape(1, 3),
            torques=np.asarray(torque, dtype=np.float32).reshape(1, 3),
            is_global=False)

    def _drain_node(self, node, max_msgs: int = 16):
        """Dispatch every queued callback on ``node`` (bounded), stopping as soon
        as a spin delivers nothing. One no-op spin per tick when idle."""
        if node is None:
            return
        for _ in range(max_msgs):
            before = self._rx_count
            rclpy.spin_once(node, timeout_sec=0.0)
            if self._rx_count == before:
                break

    def _is_command_stale(self):
        """True if no vel/force command has arrived within command_timeout.

        Staleness check delegates to ros2_control_math (pure, unit tested).
        Logs a single warning per disconnect (not every tick) via
        self._stale_warned, which the vel/force callbacks reset as soon as a
        fresh command arrives.
        """
        stale = ros2_control_math.is_command_stale(
            time.monotonic(), self.last_command_time, self.command_timeout)
        if stale and not self._stale_warned:
            print(f'[{self._name}] no command received for over '
                  f'{self.command_timeout}s -- zeroing command (watchdog)')
            self._stale_warned = True
        return stale
    
    def _body_to_world(self, lin_body, ang_body):
        """Rotate a body-frame velocity command into the world frame.

        Isaac's set_linear_velocity / set_angular_velocity expect world-frame
        vectors, while ROS Twist commands are conventionally body-frame. The
        robot's current orientation (wxyz) gives the body->world rotation R, so
        v_world = R @ v_body.
        """
        lin_b = np.asarray(lin_body, dtype=float)
        ang_b = np.asarray(ang_body, dtype=float)
        _, quat_wxyz = self._rigid_prim.get_world_pose()
        rot = quat_to_rot_matrix(np.asarray(quat_wxyz, dtype=float))
        return rot @ lin_b, rot @ ang_b

    def close(self):
        try:
            # Clean up ROS2 resources. Gate on the node existing, not on
            # _enable_ros2 (which a failed setup flips to False while a node may
            # already have been created).
            if self._ros2_vel_node:
                self._ros2_vel_node.destroy_node()
                self._ros2_vel_node = None
            if self._ros2_force_node:
                self._ros2_force_node.destroy_node()
                self._ros2_force_node = None
            if self._ros2_thruster_node:
                self._ros2_thruster_node.destroy_node()
                self._ros2_thruster_node = None

            self._update_count = 0
            self._view = None
            self._vel_pi = None
            self.force_cmd = [0.0, 0.0, 0.0]
            self.torque_cmd = [0.0, 0.0, 0.0]
            self.linear_vel = [0.0, 0.0, 0.0]
            self.angular_vel = [0.0, 0.0, 0.0]
            self._stale_warned = False
        finally:
            # Always release the shared rclpy context, even if node teardown
            # raised, so the ref count never leaks.
            if self._ros2_acquired:
                ros2_context.release()
                self._ros2_acquired = False
            print(f'[{self._name}] ROS2_Control_receiver closed.') 


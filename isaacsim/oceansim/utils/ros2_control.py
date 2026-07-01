import time
import os
import numpy as np
from enum import Enum

from isaacsim.core.prims import SingleRigidPrim
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
    from pxr import Gf, PhysxSchema
    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False
    Gf = None 
    PhysxSchema = None

ROS2_AVAILABLE = False

class ROS2_CONTROL_MODE(Enum):
    VEL = 1     # velocity control mode
    FORCE = 2   # force control mode

class ROS2ControlReceiver:
    """
    ROS2 Control Receiver
    
    for recieving velocity and force command
    """
    
    def __init__(self, robot_prim, name="ROS2ControlReceiver",
                 max_linear_vel=None, max_angular_vel=None,
                 max_force=None, max_torque=None):
        """
        initialize ROS2 Control Receiver

        Args:
            robot_prim: robot prim path
            name (str): receiver name
            max_linear_vel (float | None): clamp on the incoming linear-velocity
                vector's magnitude (m/s). None (default) = unbounded.
            max_angular_vel (float | None): clamp on the incoming angular-velocity
                vector's magnitude (rad/s). None (default) = unbounded.
            max_force (float | None): clamp on the incoming force vector's
                magnitude (N). None (default) = unbounded.
            max_torque (float | None): clamp on the incoming torque vector's
                magnitude (N*m). None (default) = unbounded.
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
        self.last_command_time = time.time()
        self.command_timeout = 2.0
        # Set once we've warned about a stale command, so the watchdog logs a
        # single message per disconnect instead of spamming every tick; reset
        # as soon as a fresh command arrives.
        self._stale_warned = False
        self._update_count = 0
        
        # Physics API - using scenario.py created instance
        self._force_api = None
        self._scenario_force_api = None 
        
        print(f"[{self._name}] Initialized for robot prim")
        
    def initialize(self, enable_ros2=True, vel_topic="/oceansim/robot/vel_cmd", force_topic="/oceansim/robot/force_cmd"):
        """
        initialize receiver function
        
        Args:
            enable_ros2 (bool): whether using ros2
            vel_topic (str): topic name of vel
            force_topic (str): topic name of force(include torque)
        """
        self._enable_ros2 = enable_ros2
        self._vel_topic = vel_topic
        self._force_topic = force_topic
        
        if not self._enable_ros2:
            print(f'[{self._name}] ROS2 disabled by configuration')
            return
        
        self._setup_subscriber()
        self._setup_physics()
        
        print(f'[{self._name}] Control Receiver Initialized:')
        print(f'[{self._name}] ROS2 Bridge: {self._enable_ros2}')
        if PXR_AVAILABLE and self._robot_prim:
            print(f'[{self._name}] Robot Prim: {self._robot_prim.GetPath()}')
        else:
            print(f'[{self._name}] Robot Prim: {self._robot_prim}')

    def set_scenario_force_api(self, scenario_force_api):
        """
        setting the force api
        """
        self._scenario_force_api = scenario_force_api
        
    def _setup_physics(self):
        """
        setting the physics control API(PXR)
        """
        if not PXR_AVAILABLE:
            print(f'[{self._name}] PXR not available, physics API disabled')
            return
            
        try:
            if self._scenario_force_api is not None:
                self._force_api = self._scenario_force_api
            else:
                if self._robot_prim.HasAPI(PhysxSchema.PhysxForceAPI):
                    self._force_api = PhysxSchema.PhysxForceAPI(self._robot_prim)
                else:
                    self._force_api = PhysxSchema.PhysxForceAPI.Apply(self._robot_prim)
                
        except Exception as e:
            print(f'[{self._name}] Physics API set failed: {e}')
    
    def _setup_subscriber(self):
        """
        setting the ROS2 subscriber
        """
        try:
            # import ROS2 module
            from sensor_msgs.msg import Image
            from geometry_msgs.msg import Twist, Wrench
            from std_msgs.msg import Header
            
            # Initialize/share the rclpy context (ref-counted across components)
            ros2_context.acquire()
            self._ros2_acquired = True

            # Create velocity subscriber node
            node_name = f'oceansim_rob_velocity_control_{self._name.lower()}'.replace(' ', '_')
            self._ros2_vel_node = rclpy.create_node(node_name)
            self._ros2_vel_subscriber = self._ros2_vel_node.create_subscription(
                Twist,
                self._vel_topic,
                self._vel_callback,
                10
            )

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
            for _attr in ("_ros2_vel_node", "_ros2_force_node"):
                _node = getattr(self, _attr, None)
                if _node is not None:
                    try:
                        _node.destroy_node()
                    except Exception:  # noqa: BLE001
                        pass
                    setattr(self, _attr, None)
            print(f'[{self._name}] ROS2 subscriber setup failed: {e}')

    def _setup_ros2_control_mode(self, ctrl_mode):
        if ctrl_mode == "velocity control":
            self._ros2_control_mode = ROS2_CONTROL_MODE.VEL
        elif ctrl_mode == "force control":
            self._ros2_control_mode = ROS2_CONTROL_MODE.FORCE
    
    def _vel_callback(self, msg):
        """
        msg type: geometry_msgs/Twist
        
        include linear and angular velocity
        """
        print(f'[{self._name}] receive ROS2 msg, type: {type(msg).__name__}, linear: {msg.linear}, angular: {msg.angular}')
        
        if not self._enable_ros2:
            print(f'[{self._name}] ROS2 is not enabled, ignore msg')
            return
        
        try:
            current_time = time.time()

            self.linear_vel = self._clamp_magnitude(
                [msg.linear.x, msg.linear.y, msg.linear.z], self._max_linear_vel)
            self.angular_vel = self._clamp_magnitude(
                [msg.angular.x, msg.angular.y, msg.angular.z], self._max_angular_vel)
            self.last_command_time = current_time
            self._stale_warned = False  # fresh command -- watchdog can warn again next time it goes stale

            print(f'Received velocity - Linear: {self.linear_vel}, Angular: {self.angular_vel}')
            # self._update_receive_stats(current_time)

        except Exception as e:
            print(f'[{self._name}] Vel Receive Failed: {e}')
        
    def _force_callback(self, msg):
        """
        msg type: geometry_msgs/Wrench
        
        include force and torque
        """
        print(f'[{self._name}] receive ROS2 msg, type: {type(msg).__name__}, force: {msg.force}, torque: {msg.torque}')

        if not self._enable_ros2:
            print(f'[{self._name}] ROS2 is not enabled, ignore msg')
            return

        try:
            current_time = time.time()

            self.force_cmd = self._clamp_magnitude(
                [msg.force.x, msg.force.y, msg.force.z], self._max_force)
            self.torque_cmd = self._clamp_magnitude(
                [msg.torque.x, msg.torque.y, msg.torque.z], self._max_torque)
            self.last_command_time = current_time
            self._stale_warned = False  # fresh command -- watchdog can warn again next time it goes stale

            print(f'Received force - Force: {self.force_cmd}, Torque: {self.torque_cmd}')

        except Exception as e:
            print(f'[{self._name}] force Receive Failed: {e}')

    @staticmethod
    def _clamp_magnitude(vec, max_mag):
        """Clamp a 3-vector's magnitude to max_mag, preserving direction.

        max_mag=None (the default everywhere in this class) disables clamping.
        Delegates to ros2_control_math (pure, unit tested independently of
        rclpy/Isaac Sim -- see tests/test_ros2_control.py).
        """
        return ros2_control_math.clamp_magnitude(vec, max_mag)

    def update_control(self):
        """
        update control
        
        this function will be called in each simulation step. ( in scenario.update_scenario() )
        """
        if not self._enable_ros2 or not self._ros2_vel_node or not self._ros2_force_node:
            return
        
        try:
            # Dead-man's-switch: if the ROS2 link that feeds this receiver has
            # dropped (teleop crash, network partition, lost zenoh session),
            # do not keep actuating the last command received forever -- zero
            # it out once command_timeout has elapsed with no fresh message.
            stale = self._is_command_stale()

            if self._ros2_control_mode == ROS2_CONTROL_MODE.VEL: # velocity mode
                # Drain the cmd_vel queue every step with a non-blocking spin --
                # the same pattern the sensor publisher (ros2_sensors) and camera
                # (UW_Camera) already use, so it cannot "block the scene". The old
                # `% 10` gate spun (and applied the command) only every 10th step,
                # adding up to ~10 physics-steps of teleop latency; and because it
                # re-applies the commanded velocity only intermittently, other
                # forces (drag/gravity) perturbed the body in between. Spinning and
                # holding the commanded velocity every step is the standard
                # velocity-control behaviour.
                rclpy.spin_once(self._ros2_vel_node, timeout_sec=0.0)

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

            elif self._ros2_control_mode == ROS2_CONTROL_MODE.FORCE: # force mode
                # using PXR API to control
                if PXR_AVAILABLE:
                    rclpy.spin_once(self._ros2_force_node, timeout_sec=0.0)

                    force_cmd = [0.0, 0.0, 0.0] if stale else self.force_cmd
                    torque_cmd = [0.0, 0.0, 0.0] if stale else self.torque_cmd
                    force_gf = Gf.Vec3f(float(force_cmd[0]), float(force_cmd[1]), float(force_cmd[2]))
                    torque_gf = Gf.Vec3f(float(torque_cmd[0]), float(torque_cmd[1]), float(torque_cmd[2]))

                    if self._force_api:
                        try:
                            self._force_api.CreateForceAttr().Set(force_gf)
                            self._force_api.CreateTorqueAttr().Set(torque_gf)
                        except Exception as e:
                            print(f'[{self._name}] Force API Update Failed: {e}')

        except Exception as e:
            print(f'[{self._name}] Control Update Failed: {e}')

    def _is_command_stale(self):
        """True if no vel/force command has arrived within command_timeout.

        Staleness check delegates to ros2_control_math (pure, unit tested).
        Logs a single warning per disconnect (not every tick) via
        self._stale_warned, which the vel/force callbacks reset as soon as a
        fresh command arrives.
        """
        stale = ros2_control_math.is_command_stale(
            time.time(), self.last_command_time, self.command_timeout)
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

            self._update_count = 0
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


# Omniverse import
import numpy as np
from pxr import Gf, PhysxSchema

# Isaac sim import
from isaacsim.core.prims import RigidPrim, SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path

# ROS Control import
try:
    from isaacsim.oceansim.utils.ros2_control import ROS2ControlReceiver
    ROS2_CONTROL_AVAILABLE = True
    print("[Scenario] Simple ROS2 Control receiver found")
except ImportError as e:
    ROS2_CONTROL_AVAILABLE = False
    print(f"[Scenario] Simple ROS2 Control not available: {e}")
    print("[Scenario] ROS2 Control functionality will be disabled")

class MHL_Sensor_Example_Scenario():
    def __init__(self):
        self._rob = None
        self._rob_rigid = None
        self._rob_view = None
        self._imu = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None

        self._ctrl_mode = None

        self._running_scenario = False
        self._time = 0.0

        # Sensor compute throttle. The sonar scan + camera UW_render run here every
        # physics step (e.g. 60 Hz), but their output is typically consumed far
        # slower (ROS publish ~5 Hz / viewport). _sensor_update_period > 0 limits
        # the heavy per-frame sensor compute to 1/period Hz; 0 = every step
        # (default, so the GUI extension is unaffected). Control still runs every
        # step. Set via the runner's sensor_compute_rate.
        self._sensor_update_period = 0.0
        self._sensor_accum = 0.0

        # ROS2 Control
        self._ros2_control_receiver = None
        self._enable_ros2_control = True
        self._ros2_control_mode = "velocity control"

        # Upstream OmniGraph ROS2 publishers (opt-in, see setup_scenario)
        self._use_omnigraph_ros = False
        self.omni_ros = None
        self._cmd_vel_controller = None

    def setup_scenario(self, rob, sonar, cam, DVL, baro, ctrl_mode, sensor_viewports=True,
                       control_params=None, imu=None, use_omnigraph_ros=False):
        """use_omnigraph_ros=True wires upstream OceanSim's OmniGraph ROS2
        publishers (ros2_helpers.OmniHandler + the *_ROS sensor classes, which
        the caller must have constructed) and a /cmd_vel subscriber. The
        default (False) keeps this fork's rclpy bridge: UW_Camera's own
        publisher, ROS control mode and the headless runner's ros2_sensors."""
        self._use_omnigraph_ros = use_omnigraph_ros
        self._imu = imu
        self._rob = rob
        # The rigid-body wrapper is created LAZILY on first use in update_scenario
        # (after world.play()), NOT here. A SingleRigidPrim (physics tensor view)
        # created in setup_scenario -- which runs between world.reset() and
        # world.play() -- is invalidated when play() rebuilds the physics scene
        # ("prim '/World/rob' was deleted ... simulationView was invalidated"),
        # which poisons the global SimulationView and breaks EVERY velocity read in
        # the sim (this scenario's control, the ROS2 control receiver, and the
        # odom/IMU publisher). It's still cached after the first lazy creation, so
        # it is not rebuilt per physics step.
        self._rob_rigid = None
        # Same lazy rule for the manual-control RigidPrim view (see update_scenario).
        self._rob_view = None
        self._sonar = sonar
        self._cam = cam
        self._DVL = DVL
        self._baro = baro
        self._ctrl_mode = ctrl_mode
        # sensor_viewports=False skips the per-sensor GUI windows (sonar + UW camera)
        # AND their per-frame set_bytes_data_from_gpu readback, while still rendering
        # the AOVs for ROS publishing and keeping the main Isaac viewport. (Manually
        # closing the windows in the UI does NOT stop the readback -- the _viewport
        # flag stays set -- so this is the way to actually drop that GPU work.)
        if self._use_omnigraph_ros:
            self._setup_omnigraph_ros(sensor_viewports)
        else:
            self.omni_ros = None
            if self._imu is not None and hasattr(self._imu, "initialize"):
                self._imu.initialize()
            if self._sonar is not None:
                self._sonar.sonar_initialize(include_unlabelled=True, viewport=sensor_viewports)
            if self._cam is not None:
                self._cam.initialize(viewport=sensor_viewports)
        if self._DVL is not None:
            self._DVL_reading = [0.0, 0.0, 0.0]
        if self._baro is not None:
            self._baro_reading = 101325.0 # atmospheric pressure (Pa)

        # Upstream's /cmd_vel subscriber rides with the OmniGraph publishers. Not
        # in Manual control (keyboard drives) or ROS control (the rclpy receiver
        # drives), so two command sources never fight over the same body.
        if self._use_omnigraph_ros and ctrl_mode not in ("Manual control", "ROS control"):
            from ...utils.cmd_vel_subscriber import CmdVelController
            self._cmd_vel_controller = CmdVelController(robot_prim_path=get_prim_path(self._rob))

        # Manual control applies forces through a RigidPrim tensor view (upstream
        # moved off PhysxForceAPI because it stopped the IMU updating). The view
        # is built lazily in update_scenario, after world.play().
        if ctrl_mode == "Manual control":
            from ...utils.keyboard_cmd import keyboard_cmd

            self._force_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # forward command
                                        "W": [10.0, 0.0, 0.0],
                                        # backward command
                                        "S": [-10.0, 0.0, 0.0],
                                        # leftward command
                                        "A": [0.0, 10.0, 0.0],
                                        # rightward command
                                        "D": [0.0, -10.0, 0.0],
                                         # rise command
                                        "UP": [0.0, 0.0, 10.0],
                                        # sink command
                                        "DOWN": [0.0, 0.0, -10.0],
                                      })
            self._torque_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # yaw command (left)
                                        "J": [0.0, 0.0, 10.0],
                                        # yaw command (right)
                                        "L": [0.0, 0.0, -10.0],
                                        # pitch command (up)
                                        "I": [0.0, -10.0, 0.0],
                                        # pitch command (down)
                                        "K": [0.0, 10.0, 0.0],
                                        # row command (left)
                                        "LEFT": [-10.0, 0.0, 0.0],
                                        # row command (negative)
                                        "RIGHT": [10.0, 0.0, 0.0],
                                      })
        elif ctrl_mode == "ROS control":
            self._rob_forceAPI = PhysxSchema.PhysxForceAPI.Apply(self._rob)

            # initialize ROS2ControlReceiver
            self._setup_ros2_control(control_params)

        self._running_scenario = True

    def _setup_omnigraph_ros(self, sensor_viewports=True):
        """Upstream OceanSim's OmniGraph ROS2 publishing: one OmniHandler graph
        plus the og_node each *_ROS sensor writes its frames into."""
        from isaacsim.oceansim.sensors import ros2_helpers

        self.omni_ros = ros2_helpers.OmniHandler(
            name="SensorExample",
            use_camera=self._cam is not None,
            use_sonar=self._sonar is not None,
            use_imu=self._imu is not None,
            use_dvl=self._DVL is not None,
            use_baro=self._baro is not None,
        )
        approx_freq = 30

        if self._imu is not None:
            self._imu.initialize(og_node=self.omni_ros._imu_node)

        if self._sonar is not None:
            self._sonar.sonar_initialize(
                include_unlabelled=True, viewport=sensor_viewports,
                og_node=self.omni_ros._sonar_node
            )
            ros2_helpers.publish_camera_info(self._sonar, approx_freq)
            ros2_helpers.publish_pointcloud_from_depth(self._sonar, approx_freq)
            ros2_helpers.publish_camera_tf(self._sonar)

        if self._cam is not None:
            self._cam.initialize(
                viewport=sensor_viewports,
                og_node=self.omni_ros._rgb_node,
                depth_og_node=self.omni_ros._depth_node,
                pointcloud_og_node=self.omni_ros._pointcloud_node,
            )
            ros2_helpers.publish_camera_info(
                self._cam,
                approx_freq,
                topic_name="RGBCamera/camera_info",
            )
            ros2_helpers.publish_camera_tf(self._cam)

        if self._DVL is not None:
            self._DVL.initialize(og_node=self.omni_ros._dvl_node)

        if self._baro is not None:
            self._baro.initialize(og_node=self.omni_ros._baro_node)

    def _setup_ros2_control(self, control_params=None):
        """setup ROS2 control receiver"""
        if not ROS2_CONTROL_AVAILABLE:
            return

        control_params = control_params or {}
        try:
            self._ros2_control_receiver = ROS2ControlReceiver(
                self._rob, "ROS2ControlReceiver",
                max_linear_vel=control_params.get("max_linear_vel"),
                max_angular_vel=control_params.get("max_angular_vel"),
                max_force=control_params.get("max_force"),
                max_torque=control_params.get("max_torque"))

            if hasattr(self, '_rob_forceAPI') and self._rob_forceAPI is not None:
                self._ros2_control_receiver.set_scenario_force_api(self._rob_forceAPI)

            self._ros2_control_receiver.initialize(
                enable_ros2=True
            )

            self._ros2_control_receiver._setup_ros2_control_mode(
                self._ros2_control_mode
            )
                
        except Exception as e:
            print(f"[Scenario] setup ros2 control receiver failed: {e}")
            self._ros2_control_receiver = None

    # This function will only be called if ctrl_mode==waypoints and waypoints files are changed
    def setup_waypoints(self, waypoint_path, default_waypoint_path):
        def read_data_from_file(file_path):
            # Initialize an empty list to store the floats
            data = []
            
            # Open the file in read mode
            with open(file_path, 'r') as file:
                # Read each line in the file
                for line in file:
                    # Strip any leading/trailing whitespace and split the line by spaces
                    float_strings = line.strip().split()
                    
                    # Convert the list of strings to a list of floats
                    floats = [float(x) for x in float_strings]
                    
                    # Append the list of floats to the data list
                    data.append(floats)
            
            return data
        try:
            self.waypoints = read_data_from_file(waypoint_path)
            print('Waypoints loaded successfully.')
            print(f'Waypoint[0]: {self.waypoints[0]}')
        except Exception as e:
            self.waypoints = read_data_from_file(default_waypoint_path)
            print(f'Fail to load waypoints from {waypoint_path} ({e}). Back to default waypoints.')

        
    def teardown_scenario(self):

        # Per-resource guards: one close() raising must not skip the remaining
        # teardown (e.g. a sonar annotator-detach failure used to leak the
        # camera's rclpy context AND the control receiver's nodes).
        def _safe_teardown(fn, name):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f'[Scenario] {name} teardown warning: {e}')

        # Because these two sensors create annotator cache in GPU,
        # close() will detach annotator from render product and clear the cache.
        if self._sonar is not None:
            _safe_teardown(self._sonar.close, "sonar")
        if self._cam is not None:
            _safe_teardown(self._cam.close, "camera")

        # Clear cmd_vel controller
        if self._cmd_vel_controller is not None:
            _safe_teardown(self._cmd_vel_controller.cleanup, "cmd_vel controller")
            self._cmd_vel_controller = None

        # clear the keyboard subscription
        if self._ctrl_mode=="Manual control":
            _safe_teardown(self._force_cmd.cleanup, "keyboard force cmd")
            _safe_teardown(self._torque_cmd.cleanup, "keyboard torque cmd")

        # clear the ROS2 control receiver
        if self._ros2_control_receiver is not None:
            _safe_teardown(self._ros2_control_receiver.close, "ROS2 control receiver")

        self._rob = None
        self._rob_rigid = None
        self._rob_view = None
        self._imu = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self.omni_ros = None
        self._running_scenario = False
        self._time = 0.0


    _safe_call_fail_counts = {}

    @staticmethod
    def _safe_call(fn, *args, name, **kwargs):
        """Run fn(*args, **kwargs), logging and swallowing any exception instead
        of letting it propagate. A single transient GPU/CUDA hiccup in sonar or
        camera compute must not tear down the whole simulation -- every
        publish call in ros2_sensors.py already gets this same treatment via
        its _safe() helper; this mirrors it for the compute side.

        Log throttled per call-site: a PERSISTENTLY failing sensor otherwise
        prints identical lines at the full sensor-tick rate (15-60 Hz)."""
        try:
            fn(*args, **kwargs)
            MHL_Sensor_Example_Scenario._safe_call_fail_counts.pop(name, None)
        except Exception as e:  # noqa: BLE001 -- keep sim alive on compute error
            counts = MHL_Sensor_Example_Scenario._safe_call_fail_counts
            n = counts.get(name, 0)
            counts[name] = n + 1
            if n == 0 or (n + 1) % 100 == 0:
                print(f'[Scenario] {name} failed ({n + 1}x): {e}')

    def update_scenario(self, step: float, sim_time: float = None, sonar_tick: bool = None):


        if not self._running_scenario:
            return

        self._time += step

        # IMU is cheap and high-rate: read (and, in OmniGraph mode, publish)
        # every step, outside the heavy-sensor throttle below.
        if self._imu is not None:
            if self._use_omnigraph_ros:
                self._safe_call(self._imu.read, name="IMU read")
            else:
                self._safe_call(self._imu.get_current_frame, name="IMU read")

        # Throttle the heavy sensor compute (sonar scan + camera UW_render) to
        # _sensor_update_period; 0 means every step. Control below is unaffected.
        self._sensor_accum += step
        do_sensors = (self._sensor_update_period <= 0.0
                      or self._sensor_accum >= self._sensor_update_period)
        # The runner can drive a separate sonar render cadence (sonar_tick): only
        # scan on the steps where the sonar render product was enabled, so the
        # expensive sonar render isn't wasted. Falls back to the shared do_sensors
        # throttle when sonar_tick is None.
        do_sonar = do_sensors if sonar_tick is None else bool(sonar_tick)
        if do_sonar and self._sonar is not None:
            self._safe_call(self._sonar.make_sonar_data, sim_time=sim_time,
                            name="sonar make_sonar_data")
        if do_sensors:
            if self._sensor_update_period > 0.0:
                # Subtract the period instead of resetting to zero: a reset
                # discards the overshoot every cycle and quantises the effective
                # rate down by up to one physics step per fire. Cap the residual
                # so a long stall doesn't queue a burst of catch-up fires.
                self._sensor_accum = min(self._sensor_accum - self._sensor_update_period,
                                         self._sensor_update_period)
            else:
                self._sensor_accum = 0.0
            if self._cam is not None:
                # Pass the authoritative sim time (headless runner) so the camera
                # rate-gates + stamps on the same clock as the other publishers;
                # None (GUI) keeps the camera's wall-clock gate.
                self._safe_call(self._cam.render, sim_time, name="camera render")
            # DVL/baro polled here ONLY for the GUI extension's readout plots --
            # the ROS publisher does its own reads. The runner sets
            # _poll_gui_readings False so headless runs skip this dead compute
            # (the DVL read is 4 beam queries + pose + noise per tick). Shielded
            # by _safe_call like the other sensor compute (a transient physics-
            # view error here used to escape update_scenario entirely).
            # In OmniGraph mode these reads are also what publishes the DVL /
            # barometer messages, so they run regardless of _poll_gui_readings.
            if self._use_omnigraph_ros or getattr(self, "_poll_gui_readings", True):
                if self._DVL is not None:
                    self._safe_call(self._read_dvl, name="DVL read")
                if self._baro is not None:
                    self._safe_call(self._read_baro, name="baro read")

        # Upstream /cmd_vel subscriber (OmniGraph mode only; see setup_scenario)
        if self._cmd_vel_controller is not None:
            self._safe_call(self._cmd_vel_controller.update, self._rob, name="cmd_vel update")

        if self._ctrl_mode=="Manual control":
            # Built here, after world.play(), for the same reason as _rob_rigid.
            if self._rob_view is None and self._rob is not None:
                self._init_manual_control_view()
            # Keyboard commands keep the meaning they had with PhysxForceAPI
            # (default mode "acceleration"): linear m/s^2 and angular deg/s^2.
            # apply_forces_and_torques_at_pos takes newtons / N*m, so scale by the
            # body's mass and inertia diagonal (else a 26 kg vehicle would get
            # 1/26 of the thrust it had before).
            force = (np.asarray(self._force_cmd._base_command, dtype=np.float32)
                     * self._manual_mass).reshape(1, 3)
            torque = (np.deg2rad(np.asarray(self._torque_cmd._base_command, dtype=np.float32))
                      * self._manual_inertia).reshape(1, 3).astype(np.float32)
            self._rob_view.apply_forces_and_torques_at_pos(
                forces=force, torques=torque, is_global=False
            )
        elif self._ctrl_mode=="Waypoints":
            if len(self.waypoints) > 0:
                waypoints = self.waypoints[0]
                self._rob.GetAttribute('xformOp:translate').Set(Gf.Vec3f(waypoints[0], waypoints[1], waypoints[2]))
                self._rob.GetAttribute('xformOp:orient').Set(Gf.Quatd(waypoints[3], waypoints[4], waypoints[5], waypoints[6]))
                self.waypoints.pop(0)
            else:
                print('Waypoints finished')                
        elif self._ctrl_mode=="Straight line":
            if self._rob_rigid is None and self._rob is not None:
                self._rob_rigid = SingleRigidPrim(prim_path=get_prim_path(self._rob))
            self._rob_rigid.set_linear_velocity(np.array([0.5,0,0]))
        elif self._ctrl_mode=="ROS control":
            if self._ros2_control_receiver is not None:
                self._ros2_control_receiver.update_control()
            elif not getattr(self, "_warned_no_receiver", False):
                # Once, not every physics step (60 Hz of identical lines).
                self._warned_no_receiver = True
                print("[Scenario] ROS2 Control receiver is not initialized, skipping update.")

    def _init_manual_control_view(self):
        """Create the manual-control RigidPrim view (post-play) and cache the
        mass / inertia diagonal used to turn keyboard accelerations into forces.
        Falls back to unit scaling (raw N / N*m) if they can't be read."""
        self._manual_mass = 1.0
        self._manual_inertia = np.ones(3, dtype=np.float32)
        view = RigidPrim(prim_paths_expr=get_prim_path(self._rob))
        view.initialize()
        # Only publish the view once it initialized; a failure retries next step.
        self._rob_view = view
        try:
            def _np(a):
                return a.numpy() if hasattr(a, "numpy") else np.asarray(a)
            mass = float(_np(self._rob_view.get_masses()).reshape(-1)[0])
            inertia = _np(self._rob_view.get_inertias()).reshape(-1, 9)[0]
            diag = np.array([inertia[0], inertia[4], inertia[8]], dtype=np.float32)
            if mass > 0.0:
                self._manual_mass = mass
            if np.all(np.isfinite(diag)) and np.all(diag > 0.0):
                self._manual_inertia = diag
        except Exception as e:  # noqa: BLE001
            print(f"[Scenario] manual control: could not read mass/inertia ({e}); "
                  f"applying keyboard commands as raw forces/torques")

    def _read_dvl(self):
        if self._use_omnigraph_ros:
            self._DVL_reading = self._DVL.read()
        else:
            self._DVL_reading = self._DVL.get_linear_vel()

    def _read_baro(self):
        if self._use_omnigraph_ros:
            self._baro_reading = float(self._baro.read())
        else:
            self._baro_reading = self._baro.get_pressure()




        

        



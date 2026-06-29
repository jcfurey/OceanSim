# Omniverse import
import numpy as np
from pxr import Gf, PhysxSchema

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
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

    def setup_scenario(self, rob, sonar, cam, DVL, baro, ctrl_mode):
        self._rob = rob
        # Cache the rigid-body wrapper once (the prim never changes); the
        # Straight-line control path built a fresh SingleRigidPrim -- prim-path
        # resolution + PhysX view setup -- every physics step.
        self._rob_rigid = SingleRigidPrim(prim_path=get_prim_path(rob)) if rob is not None else None
        self._sonar = sonar
        self._cam = cam
        self._DVL = DVL
        self._baro = baro
        self._ctrl_mode = ctrl_mode
        if self._sonar is not None:
            self._sonar.sonar_initialize(include_unlabelled=True)
        if self._cam is not None:
            self._cam.initialize()
        if self._DVL is not None:
            self._DVL_reading = [0.0, 0.0, 0.0]
        if self._baro is not None:
            self._baro_reading = 101325.0 # atmospheric pressure (Pa)
        
        
        # Apply the physx force schema if manual control
        if ctrl_mode == "Manual control":
            from ...utils.keyboard_cmd import keyboard_cmd

            self._rob_forceAPI = PhysxSchema.PhysxForceAPI.Apply(self._rob)
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
            self._setup_ros2_control()
            
        self._running_scenario = True

    def _setup_ros2_control(self):
        """setup ROS2 control receiver"""
        if not ROS2_CONTROL_AVAILABLE:
            return
        
        try:
            self._ros2_control_receiver = ROS2ControlReceiver(self._rob, "ROS2ControlReceiver")
            
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

        # Because these two sensors create annotator cache in GPU,
        # close() will detach annotator from render product and clear the cache.
        if self._sonar is not None:
            self._sonar.close()
        if self._cam is not None:
            self._cam.close()

        # clear the keyboard subscription
        if self._ctrl_mode=="Manual control":
            self._force_cmd.cleanup()
            self._torque_cmd.cleanup()

        # clear the ROS2 control receiver
        if self._ros2_control_receiver is not None:
            self._ros2_control_receiver.close()

        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self._running_scenario = False
        self._time = 0.0


    def update_scenario(self, step: float, sim_time: float = None):


        if not self._running_scenario:
            return

        self._time += step

        # Throttle the heavy sensor compute (sonar scan + camera UW_render) to
        # _sensor_update_period; 0 means every step. Control below is unaffected.
        self._sensor_accum += step
        do_sensors = (self._sensor_update_period <= 0.0
                      or self._sensor_accum >= self._sensor_update_period)
        if do_sensors:
            self._sensor_accum = 0.0
            if self._sonar is not None:
                self._sonar.make_sonar_data()
            if self._cam is not None:
                # Pass the authoritative sim time (headless runner) so the camera
                # rate-gates + stamps on the same clock as the other publishers;
                # None (GUI) keeps the camera's wall-clock gate.
                self._cam.render(sim_time)
            if self._DVL is not None:
                self._DVL_reading = self._DVL.get_linear_vel()
            if self._baro is not None:
                self._baro_reading = self._baro.get_pressure()

        if self._ctrl_mode=="Manual control":
            force_cmd = Gf.Vec3f(*self._force_cmd._base_command)
            torque_cmd = Gf.Vec3f(*self._torque_cmd._base_command)
            self._rob_forceAPI.CreateForceAttr().Set(force_cmd)
            self._rob_forceAPI.CreateTorqueAttr().Set(torque_cmd)
        elif self._ctrl_mode=="Waypoints":
            if len(self.waypoints) > 0:
                waypoints = self.waypoints[0]
                self._rob.GetAttribute('xformOp:translate').Set(Gf.Vec3f(waypoints[0], waypoints[1], waypoints[2]))
                self._rob.GetAttribute('xformOp:orient').Set(Gf.Quatd(waypoints[3], waypoints[4], waypoints[5], waypoints[6]))
                self.waypoints.pop(0)
            else:
                print('Waypoints finished')                
        elif self._ctrl_mode=="Straight line":
            self._rob_rigid.set_linear_velocity(np.array([0.5,0,0])) 
        elif self._ctrl_mode=="ROS control":
            if self._ros2_control_receiver is not None:
                self._ros2_control_receiver.update_control()
            else:
                print("[Scenario] ROS2 Control receiver is not initialized, skipping update.")




        

        



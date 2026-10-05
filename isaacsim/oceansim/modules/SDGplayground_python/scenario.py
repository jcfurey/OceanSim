# Omniverse import
import numpy as np
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf
class SDGplayground_Scenario():
    def __init__(self):

        self._running_scenario = False
        self._time = 0.0
        self._id = 0
        self._cam = None


    def setup_scenario(self, camera: "UW_Camera"):  # isaacsim.oceansim.sensors.UW_Camera

        self._running_scenario = True
        self._cam = camera
        if self._cam is not None:
            # The SDG playground only renders/saves frames; keep the fork's
            # UW_Camera rclpy publisher (on by default) off.
            self._cam.initialize(enable_ros2_pub=False)

    def teardown_scenario(self):
        self._running_scenario = False
        # Detach the camera's annotators + viewport before dropping it: every
        # LOAD/RESET re-runs initialize(), which would otherwise leak them.
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                print(f"[SDGplayground] camera teardown warning: {exc}")
        self._cam = None
        self._time = 0.0
        self._id = 0



    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        self._cam.render()
        
        self._time += step
        
        self._id += 1

       
   
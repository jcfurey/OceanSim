"""Single-beam sonar altimeter / echosounder (e.g. Blue Robotics Ping2).

One PhysX light-beam ray along the mount's -z axis (straight down for the
default mount) gives the range to the first surface hit. A real echosounder
reports the nearest return inside its beam cone (25 deg for the Ping2); the
single ray is the beam axis only, so slopes read slightly long. Range noise is
Gaussian with a standard deviation of ``range_noise_fraction`` x range (the
Ping2's published range resolution is 0.5 % of range).
"""

import numpy as np
import omni.kit.commands
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.sensors.physx import _range_sensor
from pxr import Gf


class AltimeterSensor:
    def __init__(self, name="Altimeter", min_range=0.3, max_range=100.0,
                 beamwidth_deg=25.0, range_noise_fraction=0.005, frequency_hz=115e3, seed=None):
        self._name = name
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.beamwidth_deg = float(beamwidth_deg)
        self.frequency_hz = float(frequency_hz)
        self._noise = float(range_noise_fraction)
        self._rng = np.random.default_rng(seed)
        self._beam_path = None
        self._interface = None

    def attach(self, rigid_body_path, translation=(0.0, 0.0, 0.0), orientation=None):
        """Create the beam under ``rigid_body_path`` at the mount pose."""
        self._beam_path = f"{rigid_body_path}/{self._name}"
        ok, _ = omni.kit.commands.execute(
            "IsaacSensorCreateLightBeamSensor", path=self._beam_path,
            min_range=self.min_range, max_range=self.max_range,
            forward_axis=Gf.Vec3d(0, 0, -1), num_rays=1)
        if not ok:
            raise RuntimeError(f"[{self._name}] could not create the light-beam sensor")
        SingleXFormPrim(prim_path=self._beam_path).set_local_pose(
            translation=np.asarray(translation, dtype=float),
            orientation=None if orientation is None else np.asarray(orientation, dtype=float))
        self._interface = _range_sensor.acquire_lightbeam_sensor_interface()

    def get_range(self):
        """Range (m) to the bottom with noise, or NaN without a return."""
        if self._interface is None:
            return float("nan")
        hit = self._interface.get_beam_hit_data(self._beam_path)[0]
        if not hit:
            return float("nan")
        r = float(self._interface.get_linear_depth_data(self._beam_path)[0])
        if self._noise > 0.0:
            r += self._rng.normal(0.0, self._noise * r)
        return float(np.clip(r, self.min_range, self.max_range))

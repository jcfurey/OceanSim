# SPDX-License-Identifier: Apache-2.0
"""Native-RTX-acoustic sonar backend for OceanSim (Isaac Sim 6.0.1+).

ALTERNATIVE to ``ImagingSonarSensor``.  Instead of the (6.0.1-incompatible)
Camera + ``pointcloud`` Replicator pipeline -- whose composite ``pointcloud``
annotator SIGSEGVs at ``world.play()`` on 6.0.1 -- this wraps Isaac Sim's built-in
RTX acoustic sensor (``isaacsim.sensors.experimental.rtx.AcousticSensor`` over an
``OmniAcoustic`` prim).  It is RTX-raytraced, NVIDIA-maintained (the runtime owns
its render product + hydra texture), and uses native non-visual material
reflectivity instead of OceanSim's semantic-label hack.

Data delivery (the hard-won part):

* Acoustic GMO is delivered via a Replicator **Writer**, NOT the
  ``get_data("generic-model-output")`` annotator path -- that path returns a
  buffer with an invalid GMO magic number for acoustic (verified on cpu + cuda).
  So we create the runtime with ``annotators=[]`` and attach a custom Writer that
  captures GenericModelOutput (Isaac's create_acoustic_basic.py pattern).
* Replicator writers only fire when the orchestrator evaluates the SDG pipeline.
  We enable ``rep.orchestrator.set_capture_on_play(True)`` in ``sonar_initialize``
  so the writer captures on the renderer advances the OceanSim runner already
  drives via ``world.step(render=True)``.  ``make_sonar_data()`` then just reads
  the writer's latest stashed frame -- it does NOT call
  ``rep.orchestrator.step()`` itself, because an explicit pump conflicts with
  ``world.step``'s render control ("renderer failed to advance").

Acoustic GMO field semantics (per create_acoustic_basic.py):

    x      -> transmitter sensor-mount ID
    y      -> receiver sensor-mount ID
    z      -> channel ID
    scalar -> amplitude sample
    timeOffsetNs -> sample time offset (range = sound_speed * t / 2)

We fold those into the ``(n_range, n_beams)`` intensity grid the rest of the
OceanSim pipeline expects (``sonar_map``: Warp ``vec3``, channel 2 = intensity in
[0, 1], consumed by ``OceanSimSensorPublisher``), exposing the same interface as
``ImagingSonarSensor`` so the two backends are interchangeable via ``sonar_backend``.

STATUS: EXPERIMENTAL.  The Tx/Rx fan authored below is a placeholder array and the
time->range / receiver->azimuth mapping is a first pass needing calibration
against real sonar geometry.  Range comes from real time-of-flight; azimuth from
the fanned receiver geometry.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from isaacsim.oceansim.utils import rtx_acoustic_math


_GMO_WRITER_NAME = "OceanSimAcousticGmoSink"
_gmo_writer_registered = False


def _gmo_field(field, n):
    """Copy an n-length GMO field to numpy, tolerating slice-able array-likes
    (``scalar``/``timeOffsetNs``) or raw ctypes pointers (``x``/``y``/``z``)."""
    try:
        return np.array(field[:n])
    except Exception:  # noqa: BLE001
        import numpy.ctypeslib as npct
        return np.array(npct.as_array(field, shape=(n,)))


def _ensure_gmo_writer():
    """Register the acoustic GMO sink writer once; return its registry name."""
    global _gmo_writer_registered
    import omni.replicator.core as rep
    from omni.replicator.core import Writer
    from isaacsim.sensors.experimental.rtx import parse_generic_model_output_data

    if _gmo_writer_registered:
        return _GMO_WRITER_NAME

    class OceanSimAcousticGmoSink(Writer):
        """Stash the latest acoustic GMO (numpy copies) each render frame."""

        def __init__(self):
            self.data_structure = "renderProduct"
            self.annotators = [rep.annotators.get("GenericModelOutput")]
            self.latest = None
            self.frame_count = 0

        def write(self, data):
            try:
                if "renderProducts" not in data:
                    return
                for _rp, rpd in data["renderProducts"].items():
                    raw = rpd.get("GenericModelOutput")
                    if isinstance(raw, dict):
                        raw = raw.get("data")
                    gmo = parse_generic_model_output_data(raw)
                    n = int(getattr(gmo, "numElements", 0) or 0)
                    if n <= 0:
                        continue
                    self.latest = {
                        "n": n,
                        "tx": _gmo_field(gmo.x, n).astype(np.int64),
                        "rx": _gmo_field(gmo.y, n).astype(np.int64),
                        "amp": _gmo_field(gmo.scalar, n).astype(np.float32),
                        "t_ns": _gmo_field(gmo.timeOffsetNs, n).astype(np.float64),
                    }
            except Exception as exc:  # noqa: BLE001 - never throw inside the SDG pipeline
                print(f"[{_GMO_WRITER_NAME}] write error: {exc}", flush=True)
            finally:
                self.frame_count += 1

    rep.WriterRegistry.register(OceanSimAcousticGmoSink)
    _gmo_writer_registered = True
    return _GMO_WRITER_NAME


class RtxAcousticSensor:
    def __init__(self,
                 prim_path: str,
                 translation=None,
                 orientation=None,
                 min_range: float = 0.1,
                 max_range: float = 10.0,
                 range_res: float = 0.01,
                 hori_fov: float = 130.0,
                 vert_fov: float = 20.0,
                 angular_res: float = 0.5,
                 n_elements: int = 8,
                 center_frequency: float = 51200.0,
                 sound_speed: float = 1500.0,
                 tick_rate: float = 30.0,
                 name: str = "RtxAcousticSonar"):
        self._name = name
        self._prim_path = prim_path
        self._translation = None if translation is None else np.asarray(translation, dtype=float)
        self._orientation = None if orientation is None else np.asarray(orientation, dtype=float)

        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.range_res = float(range_res)
        self.hori_fov = float(hori_fov)
        self.vert_fov = float(vert_fov)
        self.angular_res = float(angular_res)
        self.center_frequency = float(center_frequency)
        # Alias so OceanSimSensorPublisher's `getattr(sonar, "frequency", ...)`
        # reports the real acoustic frequency in ProjectedSonarImage.ping_info.
        self.frequency = self.center_frequency
        self.sound_speed = float(sound_speed)
        self.tick_rate = float(tick_rate)
        self._n_elements = int(n_elements)

        self.n_range = max(1, int(round((self.max_range - self.min_range) / self.range_res)))
        self.n_beams = max(1, int(round(self.hori_fov / self.angular_res)))

        self._acoustic = None
        self._sensor = None
        self._writer = None
        self._rep = None
        self._frame_i = 0
        self._logged_valid = False
        self._map_np = np.zeros((self.n_range, self.n_beams, 3), dtype=np.float32)
        self.sonar_map = self._map_np  # reuse the buffer (publisher accepts numpy); no per-frame GPU alloc

    # -- interface parity with ImagingSonarSensor -------------------------------
    def get_range(self):
        return [self.min_range, self.max_range]

    def get_fov(self):
        return [self.hori_fov, self.vert_fov]

    def make_sonar_viewport(self):
        return None

    def make_sonar_image(self):
        return None

    def _build_acoustic_attributes(self) -> dict:
        attrs = {"omni:sensor:WpmAcoustic:centerFrequency": self.center_frequency}
        half = np.deg2rad(self.hori_fov) / 2.0
        angles = np.linspace(-half, half, self._n_elements)
        for i, a in enumerate(angles):
            m = f"m{i + 1:03d}"
            attrs[f"omni:sensor:WpmAcoustic:sensorMount:{m}:position"] = (0.0, float(0.02 * np.sin(a)), 0.0)
            attrs[f"omni:sensor:WpmAcoustic:sensorMount:{m}:rotation"] = (0.0, 0.0, float(np.rad2deg(a)))
        for g in range(self._n_elements - 1):
            attrs[f"omni:sensor:WpmAcoustic:rxGroup:g{g + 1:03d}:receiverIndices"] = [g, g + 1]
        return attrs

    def sonar_initialize(self, output_dir=None, viewport=True,
                         include_unlabelled=False, if_array_copy=True):
        try:
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("isaacsim.sensors.experimental.rtx")
        except Exception as exc:  # noqa: BLE001
            print(f"[{self._name}] could not enable isaacsim.sensors.experimental.rtx: {exc}")

        # NOTE: Motion BVH (/renderer/raytracingMotion/enabled) MUST be enabled as a
        # BOOT setting -- the standalone runner passes it as a kit arg before
        # SimulationApp (see oceansim_ros2.py). Do NOT flip it here: changing that
        # render setting at runtime (this used to do `carb.settings.set(...)`) forces a
        # hydra-engine reconfiguration that fails to spawn engine threads while the Kit
        # viewport + sonar render products are live -- the viewport AND the sonar both
        # silently stop rendering (thousands of "failed to create Hydra Engine thread"
        # warnings + empty GMO frames). That single mid-run set was the entire reason
        # the rtx_acoustic backend produced no data.

        import omni.replicator.core as rep
        self._rep = rep
        try:
            rep.orchestrator.set_capture_on_play(True)
        except Exception:  # noqa: BLE001
            pass

        from isaacsim.sensors.experimental.rtx import Acoustic, AcousticSensor

        self._acoustic = Acoustic(
            self._prim_path,
            aux_output_level="BASIC",
            tick_rate=self.tick_rate,
            attributes=self._build_acoustic_attributes(),
        )
        if self._translation is not None:
            try:
                self._acoustic.set_world_poses(positions=self._translation.reshape(1, 3))
            except Exception:  # noqa: BLE001
                pass
        # annotators=[] -- the writer brings its own GenericModelOutput annotator.
        self._sensor = AcousticSensor(self._acoustic, annotators=[])
        writer_name = _ensure_gmo_writer()
        self._writer = self._sensor.attach_writer(writer_name)

        self._map_np[:] = 0.0
        self.sonar_map = self._map_np  # reuse the buffer (publisher accepts numpy); no per-frame GPU alloc
        print(f"[{self._name}] native RTX acoustic sensor ready @ {self._prim_path}: "
              f"{self._n_elements} receiver mounts, tick_rate={self.tick_rate} Hz, "
              f"grid {self.n_beams} beams x {self.n_range} range (writer={writer_name})",
              flush=True)

    def make_sonar_data(self, *args, **kwargs):
        """Fold the writer's latest captured acoustic GMO frame into ``sonar_map``.

        The frame is captured by the GMO writer via replicator capture-on-play as
        ``world.step(render=True)`` advances the renderer (see ``sonar_initialize``);
        this method only reads the stashed result, it does not step Replicator."""
        if self._writer is None:
            return
        self._frame_i += 1
        # The writer fires via replicator capture-on-play (set in sonar_initialize)
        # on the renderer advances driven by the runner's world.step(render=True).
        # (Explicitly pumping rep.orchestrator.step() here instead conflicts with
        # world.step's render control -> "renderer failed to advance".)
        latest = getattr(self._writer, "latest", None)
        if self._frame_i <= 3:
            print(f"[{self._name}] frame {self._frame_i}: writer frames="
                  f"{getattr(self._writer, 'frame_count', 0)}, "
                  f"latest={'yes' if latest else 'none'}", flush=True)
        if latest is None:
            return

        n = int(latest["n"])
        tx, rx = latest["tx"], latest["rx"]
        amp, t_ns = latest["amp"], latest["t_ns"]

        if not self._logged_valid:
            self._logged_valid = True
            print(f"[{self._name}] FIRST VALID acoustic frame {self._frame_i}: "
                  f"numElements={n}, tx={np.unique(tx).tolist()[:8]}, "
                  f"rx={np.unique(rx).tolist()[:8]}, "
                  f"t_ns=[{t_ns.min():.0f},{t_ns.max():.0f}], "
                  f"|amp|=[{np.abs(amp).min():.4g},{np.abs(amp).max():.4g}]", flush=True)

        # Fold the GMO samples into the (n_range, n_beams) intensity grid
        # (rtx_acoustic_math.fold_gmo_to_grid is pure numpy + unit tested).
        self._map_np[:] = 0.0
        self._map_np[:, :, 2] = rtx_acoustic_math.fold_gmo_to_grid(
            rx, amp, t_ns, self.sound_speed, self.min_range, self.range_res,
            self.n_range, self.n_beams, self._n_elements)
        self.sonar_map = self._map_np  # reuse the buffer (publisher accepts numpy); no per-frame GPU alloc

    def close(self):
        try:
            if self._sensor is not None and self._writer is not None:
                try:
                    self._sensor.detach_writer(_GMO_WRITER_NAME)
                except Exception:  # noqa: BLE001
                    pass
            self._writer = None
            self._sensor = None
            self._acoustic = None
            print(f"[{self._name}] closed.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{self._name}] close error: {exc}", flush=True)

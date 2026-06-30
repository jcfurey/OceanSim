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

Acoustic GMO layout (measured against Isaac 6.0.1, aux_output_level="BASIC"): the
GMO is organised as **signal ways**, NOT a per-sample point cloud.
``numElements = numSgws * numSamplesPerSgw``, laid out as numSgws *contiguous
row-major A-scan blocks*. Each block is one signal way's amplitude envelope
(``scalar``) vs SAMPLE INDEX -- the sample index is the range/TOF axis. The
per-element ``timeOffsetNs`` is ALWAYS 0 for acoustic (the old ``range =
sound_speed * timeOffsetNs / 2`` collapsed every sample to range 0); ``x``/``y``/
``z`` (tx/rx/channel ids) are likewise unreliable at BASIC, so azimuth is taken
from the signal-way INDEX. Sample ``k`` -> range ``range_offset + k *
meters_per_sample`` with ``meters_per_sample = c_sensor * sampleDuration / 2``
(the sensor models AIR, c~=343 m/s; ``sampleDuration`` ~= 1.024e-4 s is a readable
prim attr) -> a ~6 m range window over 320 samples.

We fold those A-scans into the ``(n_range, n_beams)`` intensity grid the rest of
the OceanSim pipeline expects (``sonar_map``: Warp ``vec3``, channel 2 = intensity
in [0, 1], consumed by ``OceanSimSensorPublisher``), exposing the same interface as
``ImagingSonarSensor`` so the two backends are interchangeable via ``sonar_backend``.

STATUS: EXPERIMENTAL.  Range is now calibrated (sample index -> range via the
sensor's sampleDuration). REMAINING: azimuth resolution == number of signal ways
(no per-sample azimuth; needs delay-and-sum beamforming -- see RTX_SONAR_BACKENDS.md
item #2), the ~6 m air-medium range cap, and the 90deg sensor FOV vs Oculus 130deg.
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
                    # numSamplesPerSgw is the A-scan length: numElements = numSgws *
                    # numSamplesPerSgw, the 640/2560/... samples laid out as numSgws
                    # contiguous A-scan blocks. ONLY populated at aux_output_level=BASIC.
                    self.latest = {
                        "n": n,
                        "nspg": int(getattr(gmo, "numSamplesPerSgw", 0) or 0),
                        "amp": _gmo_field(gmo.scalar, n).astype(np.float32),
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
                 sensor_sound_speed: float = 343.0,
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
        # The Isaac RTX acoustic sensor raytraces time-of-flight in AIR (c~=343 m/s;
        # there is NO sound-speed attribute on the WpmAcoustic schema). Range per
        # A-scan sample is c_sensor * sampleDuration / 2, so this -- NOT the
        # underwater self.sound_speed -- sets the sample->range mapping below.
        self.sensor_sound_speed = float(sensor_sound_speed)
        self.tick_rate = float(tick_rate)
        self._n_elements = int(n_elements)

        # sample->range mapping, finalised in sonar_initialize from the prim's
        # sampleDuration/pulseDuration attrs. Fallback assumes the measured
        # sampleDuration=1.024e-4 s, pulseDuration=2.5e-3 s (Isaac 6.0.1 defaults).
        self.meters_per_sample = self.sensor_sound_speed * 1.024e-4 / 2.0
        self.range_offset = self.sensor_sound_speed * 2.5e-3 / 2.0

        self.n_range = max(1, int(round((self.max_range - self.min_range) / self.range_res)))
        self.n_beams = max(1, int(round(self.hori_fov / self.angular_res)))

        self._acoustic = None
        self._sensor = None
        self._writer = None
        self._rep = None
        self._frame_i = 0
        self._logged_valid = False
        self._logged_fold = False
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

        # Read the acoustic timing from the prim so the sample->range mapping tracks
        # the real sensor config: range(k) = range_offset + k * meters_per_sample,
        # meters_per_sample = c_sensor * sampleDuration / 2 (sensor models air), and
        # range_offset ~= c_sensor * pulseDuration / 2 (the pulse-length echo delay).
        try:
            import omni.usd
            from pxr import Usd  # noqa: F401
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(str(self._acoustic.paths[0]))
            sd = prim.GetAttribute("omni:sensor:WpmAcoustic:sampleDuration")
            pd = prim.GetAttribute("omni:sensor:WpmAcoustic:pulseDuration")
            if sd and sd.IsValid() and sd.Get():
                self.meters_per_sample = self.sensor_sound_speed * float(sd.Get()) / 2.0
            if pd and pd.IsValid() and pd.Get():
                self.range_offset = self.sensor_sound_speed * float(pd.Get()) / 2.0
        except Exception as exc:  # noqa: BLE001
            print(f"[{self._name}] could not read acoustic timing attrs "
                  f"(using fallback mps={self.meters_per_sample:.5f}): {exc}", flush=True)
        print(f"[{self._name}] sample->range: meters_per_sample={self.meters_per_sample:.5f} m, "
              f"range_offset={self.range_offset:.3f} m, "
              f"sensor range window~={self.range_offset + self.meters_per_sample * 320:.2f} m "
              f"(c_sensor={self.sensor_sound_speed:.0f} m/s, air)", flush=True)
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
        amp = latest["amp"]
        nspg = int(latest.get("nspg", 0) or 0)
        n_sgw = (n // nspg) if nspg > 0 else 0

        if not self._logged_valid:
            self._logged_valid = True
            print(f"[{self._name}] FIRST VALID acoustic frame {self._frame_i}: "
                  f"numElements={n}, numSamplesPerSgw={nspg}, numSgws={n_sgw}, "
                  f"|amp|=[{np.abs(amp).min():.4g},{np.abs(amp).max():.4g}], "
                  f"meters_per_sample={self.meters_per_sample:.5f}, "
                  f"range_offset={self.range_offset:.3f}", flush=True)

        # Fold the signal-way A-scans into the (n_range, n_beams) intensity grid:
        # range(k) = range_offset + k * meters_per_sample (sample index = range axis;
        # the per-element timeOffsetNs is always 0 for acoustic). Pure numpy + unit tested.
        self._map_np[:] = 0.0
        self._map_np[:, :, 2] = rtx_acoustic_math.fold_gmo_to_grid(
            amp, nspg, self.meters_per_sample, self.range_offset,
            self.min_range, self.range_res, self.n_range, self.n_beams)
        self.sonar_map = self._map_np  # reuse the buffer (publisher accepts numpy); no per-frame GPU alloc

        # One-time post-fold sanity: prove the image is NOT collapsed (the old
        # timeOffsetNs=0 bug put everything in/below bin 0). Reports how many
        # range/beam cells are lit and the range extent actually hit.
        if not self._logged_fold:
            self._logged_fold = True
            inten = self._map_np[:, :, 2]
            rbins, beams = np.nonzero(inten)
            if rbins.size:
                print(f"[{self._name}] FOLDED grid: {rbins.size} lit cells, "
                      f"range bins [{rbins.min()},{rbins.max()}] "
                      f"= [{self.min_range + rbins.min()*self.range_res:.2f},"
                      f"{self.min_range + rbins.max()*self.range_res:.2f}] m, "
                      f"beams hit={np.unique(beams).size}/{self.n_beams}", flush=True)
            else:
                print(f"[{self._name}] FOLDED grid is EMPTY (collapsed!) -- check calibration", flush=True)

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

"""Payload catalogue for OceanSim vehicles: imaging sonars, DVLs, altimeters,
cameras, grippers, lights and positioning, as fitted to real ROVs.

A payload does two things in the sim:

* it is a body on the vehicle: its in-air mass and displaced volume at its
  mount move the vehicle's CoG, CoB and inertia, and its housing adds drag
  (vehicle_dynamics.resolve_hydro); it appears in the generated URDF;
* sensor payloads configure the simulated sensor (``params``, keys as below)
  so e.g. choosing an Oculus M750d sets the imaging sonar's field of view,
  beam count and range to that device's.

``params`` keys by kind:
  sonar      hori_fov_deg, vert_fov_deg, n_beams, frequency_hz, min_range,
             max_range (device limit), working_range (default sim range),
             range_res (device range resolution, m)
  dvl        beam_angle_deg (from vertical), min_range, max_range, rate_hz,
             velocity_noise (m/s, 1 sigma)
  altimeter  frequency_hz, beamwidth_deg, min_range, max_range, rate_hz
  camera     width, height, hfov_deg

Each mounted payload sits at the platform's mount for its kind
(PlatformSpec.mount), plus its own ``offset``. Payloads with
``simulated=False`` (e.g. a mechanically scanning sonar the imaging-sonar
model can't reproduce) only contribute mass, buoyancy, drag and URDF geometry.

Pure data + stdlib, like platforms.py, so it is unit tested without Isaac Sim.
"""

from dataclasses import dataclass, field

# Kinds a vehicle carries at most one of (they map to one simulated sensor).
SINGLE_KINDS = ("sonar", "dvl", "altimeter", "camera", "gripper", "usbl")


@dataclass(frozen=True)
class PayloadSpec:
    name: str
    kind: str
    description: str
    manufacturer: str
    mass: float                     # kg in air
    volume: float                   # displaced volume (m^3)
    size: tuple                     # housing box L x W x H (m): URDF + drag area
    params: dict = field(default_factory=dict)
    offset: tuple = (0.0, 0.0, 0.0)  # from the platform's mount for this kind
    simulated: bool = True
    sources: str = ""

    @property
    def weight_in_water(self):
        """kg in fresh water (positive sinks)."""
        return self.mass - 1000.0 * self.volume


def _vol(mass_air, mass_water):
    """Displaced volume (m^3, fresh water) from in-air and in-water mass (kg)."""
    return (mass_air - mass_water) / 1000.0


PAYLOADS = {}


def _add(p):
    if p.name in PAYLOADS:
        raise ValueError(f"duplicate payload {p.name!r}")
    PAYLOADS[p.name] = p
    return p


def available_payloads(kind=None):
    """Sorted payload names (optionally of one kind)."""
    return sorted(n for n, p in PAYLOADS.items() if kind is None or p.kind == kind)


def get_payload(name):
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in PAYLOADS:
        raise KeyError(f"Unknown payload {name!r}. Available: {available_payloads()}")
    return PAYLOADS[key]


def select_payloads(platform, names=None):
    """Payloads to fit to ``platform``: its ``default_payloads`` when ``names``
    is None, else exactly ``names`` ([] = bare vehicle). At most one payload of
    each kind in SINGLE_KINDS. Any catalogue payload may be fitted to any
    vehicle; platform.payload_options only lists the common ones."""
    chosen = [get_payload(n) for n in (platform.default_payloads if names is None else names)]
    seen = {}
    for p in chosen:
        if p.kind in SINGLE_KINDS:
            if p.kind in seen:
                raise ValueError(f"two {p.kind} payloads: {seen[p.kind]!r} and {p.name!r}")
            seen[p.kind] = p.name
    return chosen


def sensor_payload(payloads, kind):
    """The fitted, simulated payload of ``kind`` (or None)."""
    for p in payloads:
        if p.kind == kind and p.simulated:
            return p
    return None


# ---------------------------------------------------------------------------
# Catalogue. Numbers are datasheet values (sources per entry); "working_range"
# is the default simulated range (an imaging sonar's range is an operator
# setting, and the sim's cost grows with range / resolution).
# Housings are modelled as boxes in the mount frame (x = look direction).
# ---------------------------------------------------------------------------
_OCULUS_SRC = ("Blueprint Subsea Oculus M-series product page "
               "(blueprintsubsea.com/oculus/oculus-m-series) and datasheet DA-148-P01443-10")
_OCULUS_BODY = dict(mass=0.98, volume=_vol(0.98, 0.36), size=(0.062, 0.122, 0.124),
                    manufacturer="Blueprint Subsea")
# beamwidth_h_deg (azimuth resolving power) per carrier band, as in liboculus
# Constants.h (ros2_math.oculus_beamwidths): 375 kHz 2.0, 1.2 MHz 0.6,
# 2.1 MHz 0.4 deg. The M3000d entries take theirs from datasheet rev 10. No
# value is recorded for 750 kHz, so the M750d's low-frequency mode is unblurred.

_add(PayloadSpec("oculus_m370s", "sonar", "Oculus M370s imaging sonar (375 kHz)",
                 params=dict(hori_fov_deg=130.0, vert_fov_deg=20.0, n_beams=256, frequency_hz=375e3,
                             min_range=0.2, max_range=200.0, working_range=20.0, range_res=0.008,
                             beamwidth_h_deg=2.0),
                 sources=_OCULUS_SRC, **_OCULUS_BODY))
_add(PayloadSpec("oculus_m750d", "sonar", "Oculus M750d imaging sonar, low-frequency mode (750 kHz)",
                 params=dict(hori_fov_deg=130.0, vert_fov_deg=20.0, n_beams=512, frequency_hz=750e3,
                             min_range=0.1, max_range=120.0, working_range=20.0, range_res=0.004),
                 sources=_OCULUS_SRC, **_OCULUS_BODY))
_add(PayloadSpec("oculus_m750d_hf", "sonar",
                 "Oculus M750d imaging sonar, high-frequency mode (1.2 MHz)",
                 params=dict(hori_fov_deg=130.0, vert_fov_deg=20.0, n_beams=512, frequency_hz=1.2e6,
                             min_range=0.1, max_range=40.0, working_range=10.0, range_res=0.0025,
                             beamwidth_h_deg=0.6),
                 sources=_OCULUS_SRC, **_OCULUS_BODY))
_add(PayloadSpec("oculus_m1200d", "sonar", "Oculus M1200d imaging sonar, low-frequency mode (1.2 MHz)",
                 params=dict(hori_fov_deg=130.0, vert_fov_deg=20.0, n_beams=512, frequency_hz=1.2e6,
                             min_range=0.1, max_range=40.0, working_range=10.0, range_res=0.0025,
                             beamwidth_h_deg=0.6),
                 sources=_OCULUS_SRC, **_OCULUS_BODY))
_add(PayloadSpec("oculus_m1200d_hf", "sonar", "Oculus M1200d imaging sonar, high-frequency mode (2.1 MHz)",
                 params=dict(hori_fov_deg=60.0, vert_fov_deg=12.0, n_beams=512, frequency_hz=2.1e6,
                             min_range=0.1, max_range=10.0, working_range=5.0, range_res=0.0025,
                             beamwidth_h_deg=0.4),
                 sources=_OCULUS_SRC, **_OCULUS_BODY))
# DA-148-P01443-10 (Feb 2026): beamwidth is the resolving power, not
# FOV / output beam count. Both modes support up to 512 output beams.
_add(PayloadSpec("oculus_m3000d", "sonar", "Oculus M3000d imaging sonar, low-frequency mode (1.2 MHz)",
                 params=dict(hori_fov_deg=130.0, vert_fov_deg=20.0, n_beams=512, frequency_hz=1.2e6,
                             min_range=0.1, max_range=30.0, working_range=10.0, range_res=0.0025,
                             beamwidth_h_deg=0.6, rate_hz=40.0),
                 sources=_OCULUS_SRC, **_OCULUS_BODY))
_add(PayloadSpec("oculus_m3000d_hf", "sonar", "Oculus M3000d imaging sonar, high-frequency mode (3.0 MHz)",
                 params=dict(hori_fov_deg=40.0, vert_fov_deg=20.0, n_beams=512, frequency_hz=3.0e6,
                             min_range=0.1, max_range=5.0, working_range=5.0, range_res=0.002,
                             beamwidth_h_deg=0.25, rate_hz=40.0),
                 sources=_OCULUS_SRC, **_OCULUS_BODY))
_C550D_SRC = ("Blueye Robotics' Oculus C550d integration page (blueyerobotics.com) and Deep "
              "Trekker's C550d page; not listed on Blueprint's own site")
_add(PayloadSpec("oculus_c550d", "sonar", "Oculus C550d imaging sonar, low-frequency mode (550 kHz)",
                 manufacturer="Blueprint Subsea", mass=0.98, volume=_vol(0.98, 0.36),
                 size=(0.062, 0.122, 0.122),
                 params=dict(hori_fov_deg=120.0, vert_fov_deg=20.0, n_beams=256, frequency_hz=550e3,
                             min_range=0.2, max_range=100.0, working_range=20.0, range_res=0.008),
                 sources=_C550D_SRC))
_add(PayloadSpec("oculus_c550d_hf", "sonar", "Oculus C550d imaging sonar, high-frequency mode (820 kHz)",
                 manufacturer="Blueprint Subsea", mass=0.98, volume=_vol(0.98, 0.36),
                 size=(0.062, 0.122, 0.122),
                 params=dict(hori_fov_deg=90.0, vert_fov_deg=20.0, n_beams=256, frequency_hz=820e3,
                             min_range=0.2, max_range=30.0, working_range=10.0, range_res=0.005),
                 sources=_C550D_SRC))
_add(PayloadSpec("tritech_gemini_720im", "sonar", "Tritech Gemini 720im compact imaging sonar (720 kHz)",
                 manufacturer="Tritech", mass=0.435, volume=_vol(0.435, 0.244), size=(0.098, 0.063, 0.040),
                 params=dict(hori_fov_deg=90.0, vert_fov_deg=20.0, n_beams=128, frequency_hz=720e3,
                             min_range=0.2, max_range=50.0, working_range=20.0, range_res=0.008),
                 sources="Tritech Gemini 720im datasheet 0729-SOM-00001 issue 03 (VideoRay-hosted; "
                         "size read from the drawing)"))
_add(PayloadSpec("tritech_gemini_720is", "sonar", "Tritech Gemini 720is imaging sonar (720 kHz, aluminium)",
                 manufacturer="Tritech", mass=3.40, volume=_vol(3.40, 1.30), size=(0.107, 0.271, 0.135),
                 params=dict(hori_fov_deg=120.0, vert_fov_deg=20.0, n_beams=512, frequency_hz=720e3,
                             min_range=0.1, max_range=120.0, working_range=20.0, range_res=0.004),
                 sources="Tritech Gemini 720is datasheet 0703-SOM-00001 issues 11 and 15"))
_add(PayloadSpec("blueview_m900", "sonar", "Teledyne BlueView M900 Mk2 imaging sonar (900 kHz, 1000 m)",
                 manufacturer="Teledyne Marine", mass=2.5, volume=_vol(2.5, 1.2), size=(0.192, 0.102, 0.102),
                 params=dict(hori_fov_deg=130.0, vert_fov_deg=12.0, n_beams=768, frequency_hz=900e3,
                             min_range=0.5, max_range=100.0, working_range=20.0, range_res=0.013),
                 sources="Teledyne BlueView M900 Mk2 leaflet PLD20590-3 (minimum range not "
                         "published; 0.5 m assumed)"))
_add(PayloadSpec("ping360", "sonar", "Blue Robotics Ping360 mechanically scanning sonar (750 kHz). "
                 "Not simulated: the imaging-sonar model can't scan; mass / buoyancy / URDF only",
                 manufacturer="Blue Robotics", mass=0.51, volume=_vol(0.51, 0.175), size=(0.08, 0.08, 0.11),
                 params=dict(beamwidth_h_deg=2.0, beamwidth_v_deg=25.0, frequency_hz=750e3,
                             min_range=0.75, max_range=50.0, step_deg=0.9),
                 simulated=False,
                 sources="Blue Robotics Ping360 product page (housing size approximate)"))

_add(PayloadSpec("ping2", "altimeter", "Blue Robotics Ping2 sonar altimeter / echosounder (115 kHz)",
                 manufacturer="Blue Robotics", mass=0.187, volume=_vol(0.187, 0.100), size=(0.05, 0.06, 0.06),
                 params=dict(frequency_hz=115e3, beamwidth_deg=25.0, min_range=0.3, max_range=100.0,
                             rate_hz=10.0),
                 sources="Blue Robotics Ping2 product page (rate configurable; 10 Hz assumed; "
                         "housing size approximate)"))

_add(PayloadSpec("waterlinked_a50", "dvl", "Water Linked DVL A50 (1 MHz, 4-beam Janus)",
                 manufacturer="Water Linked", mass=0.17, volume=_vol(0.17, 0.105), size=(0.066, 0.066, 0.025),
                 params=dict(beam_angle_deg=22.5, min_range=0.05, max_range=50.0, rate_hz=15.0,
                             velocity_noise=None),
                 sources="waterlinked.com/datasheets/dvl-a50 (noise not published)"))
_add(PayloadSpec("waterlinked_a125", "dvl", "Water Linked DVL A125 (420 kHz, 4-beam Janus)",
                 manufacturer="Water Linked", mass=0.75, volume=_vol(0.75, 0.50), size=(0.125, 0.125, 0.030),
                 params=dict(beam_angle_deg=22.5, min_range=0.05, max_range=125.0, rate_hz=15.0,
                             velocity_noise=None),
                 sources="waterlinked.com/datasheets/dvl-a125 (noise not published)"))
_add(PayloadSpec("nortek_dvl500", "dvl", "Nortek DVL500 300 m (500 kHz, 4-beam Janus)",
                 manufacturer="Nortek", mass=3.5, volume=_vol(3.5, 0.5), size=(0.186, 0.186, 0.203),
                 params=dict(beam_angle_deg=25.0, min_range=0.1, max_range=200.0, rate_hz=8.0,
                             velocity_noise=0.008),
                 sources="nortekgroup.com DVL500-300m datasheet (single-ping std 0.8 cm/s at 1.5 m/s)"))
_add(PayloadSpec("teledyne_pathfinder", "dvl", "Teledyne RDI Pathfinder 600 kHz (ROV version)",
                 manufacturer="Teledyne Marine", mass=2.0, volume=_vol(2.0, 0.73),
                 size=(0.2286, 0.1016, 0.0711),
                 params=dict(beam_angle_deg=30.0, min_range=0.15, max_range=89.0, rate_hz=12.0,
                             velocity_noise=0.005),
                 sources="Teledyne RDI Pathfinder 600 datasheet and DVL guide (Apr 2022) outline drawing"))

_add(PayloadSpec("newton_gripper", "gripper", "Blue Robotics Newton Subsea Gripper (62 mm jaws)",
                 manufacturer="Blue Robotics", mass=0.524, volume=_vol(0.524, 0.267),
                 size=(0.303, 0.036, 0.036),
                 params=dict(opening_m=0.062, grip_force_n=97.0, close_time_s=1.6,
                             jaw_length=0.06, opening_deg=62.0),
                 sources="Blue Robotics Newton Gripper product page"))
_add(PayloadSpec("qysea_2finger_arm", "gripper", "QYSEA FIFISH 2-Finger Robotic Arm (V6 Expert / V-EVO)",
                 manufacturer="QYSEA", mass=0.8, volume=_vol(0.8, 0.0), size=(0.40, 0.11, 0.07),
                 params=dict(opening_m=0.12, grip_force_n=100.0, jaw_length=0.09, opening_deg=84.0),
                 sources="store.qysea.com 2-Finger Robotic Arm (published as neutrally buoyant)"))
_add(PayloadSpec("chasing_grabber_arm_2", "gripper", "Chasing Grabber Arm 2 (two-jaw clamp)",
                 manufacturer="Chasing", mass=0.446 + 0.076, volume=_vol(0.522, 0.0),
                 size=(0.38, 0.035, 0.035),
                 params=dict(opening_m=0.17, grip_force_n=7 * 9.80665, jaw_length=0.125,
                             opening_deg=86.0),
                 sources="Chasing Grabber Arm 2 user manual (reseller-hosted); in-water weight not "
                         "published -- neutral assumed"))
_add(PayloadSpec("videoray_pro5_manipulator", "gripper",
                 "VideoRay Pro 5 rotating manipulator (Blueprint Lab, 78 mm claw)",
                 manufacturer="VideoRay", mass=0.36, volume=_vol(0.36, 0.24), size=(0.195, 0.05, 0.05),
                 params=dict(opening_m=0.078, grip_force_n=600.0, jaw_length=0.06, opening_deg=80.0),
                 sources="VideoRay Pro 5 2-axis manipulator sheet (20_MANIPULATOR_2axis_Pro5)"))
_add(PayloadSpec("videoray_rotating_manipulator", "gripper",
                 "VideoRay rotating manipulator (Pro 4 / Defender, parallel jaws)",
                 manufacturer="VideoRay", mass=1.17, volume=_vol(1.17, 0.73), size=(0.47, 0.102, 0.05),
                 params=dict(opening_m=0.057, jaw_length=0.06, opening_deg=56.0),
                 sources="VideoRay Pro 4 rotating manipulator sheet 2018 (2015 sheet: 2.4 / 0.97 kg)"))
_add(PayloadSpec("lumen_light_pair", "light", "Two Blue Robotics Lumen subsea lights (1500 lm each)",
                 manufacturer="Blue Robotics", mass=2 * 0.118, volume=2 * _vol(0.118, 0.050),
                 size=(0.078, 0.20, 0.037),
                 params=dict(lumens=1500.0, beam_angle_deg=135.0, count=2),
                 sources="Blue Robotics Lumen product page"))

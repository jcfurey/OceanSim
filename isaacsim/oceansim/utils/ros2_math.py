"""Pure (numpy-only) math for the OceanSim ROS2 sensor publishers.

Deliberately free of ``rclpy`` / ROS message / Isaac Sim imports so it is unit
testable without ROS or Isaac Sim. ``ros2_sensors.py`` builds the actual ROS
messages from the values computed here.
"""

import math

import numpy as np


def quat_wxyz_to_xyzw(q):
    """Isaac scalar-first (w, x, y, z) -> ROS scalar-last (x, y, z, w)."""
    return (float(q[1]), float(q[2]), float(q[3]), float(q[0]))


def rot_from_quat_wxyz(q):
    """Body->world rotation matrix R from a (w, x, y, z) quaternion (v_world = R @ v_body)."""
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def world_to_body(rot_body_to_world, vec_world):
    """Rotate a world-frame vector into the body frame: R^T @ v."""
    return rot_body_to_world.T @ np.asarray(vec_world, dtype=float)


def specific_force_body(rot_body_to_world, a_world, gravity):
    """IMU specific force in the body frame: f = R^T (a_world - g_world).

    Gravity points down (g_world = (0, 0, -gravity)), so at rest (a_world = 0) the
    reading is +gravity on the axis opposing gravity (REP-145).
    """
    a = np.asarray(a_world, dtype=float)
    f_world = a - np.array([0.0, 0.0, -float(gravity)])
    return rot_body_to_world.T @ f_world


def diag3(a, b, c):
    """Row-major 3x3 covariance with the given diagonal."""
    return [a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, c]


def diag6(*vals):
    """Row-major 6x6 covariance with the given (<=6) diagonal entries."""
    m = [0.0] * 36
    for i, v in enumerate(vals):
        m[i * 6 + i] = v
    return m


def sim_time_to_sec_nanosec(sim_time):
    """Split a float sim time (seconds) into integer (sec, nanosec) with carry."""
    sec = int(sim_time)
    nanosec = int(round((sim_time - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return sec, nanosec


class RateGate:
    """Returns True at most every ``1/hz`` seconds of sim time (hz<=0 => always)."""

    def __init__(self, hz):
        self._period = (1.0 / hz) if hz and hz > 0 else 0.0
        self._last = None

    def ready(self, now):
        if self._period <= 0.0:
            return True
        # Reset if sim time jumped backward (world reset) so publishing resumes.
        if self._last is None or now < self._last or (now - self._last) >= self._period:
            self._last = now
            return True
        return False


def sonar_beam_directions(hori_fov_deg, n_beams):
    """Per-beam unit directions in the imaging-sonar optical convention.

    Azimuth lives in -y, range/forward in +z (so az = atan2(-y, z)) -- matches the
    real Oculus driver / sonar_image_proc / sonar_proc.
    """
    half = math.radians(hori_fov_deg) / 2.0
    bearings = np.linspace(-half, half, n_beams)
    return [(0.0, float(-math.sin(b)), float(math.cos(b))) for b in bearings]


def sonar_ranges(min_range, max_range, n_range):
    """Per-bin range CENTRES (metres) for a ProjectedSonarImage.

    Oculus convention (oculus_sonar_driver/ping_to_sonar_image.h):
    ``range(i) = (i + 0.5) * rangeResolution`` -- bin centres, not edges. We use
    ``rangeResolution = (max - min) / n_range`` and add the sensor's ``min_range``
    offset (the real driver assumes 0; our sensor has a non-zero near range).
    """
    res = (float(max_range) - float(min_range)) / max(int(n_range), 1)
    return (float(min_range) + (np.arange(int(n_range)) + 0.5) * res).astype(np.float32).tolist()


def oculus_beamwidths(frequency_hz):
    """(azimuth, elevation) beamwidths in RADIANS for the Oculus M-series, keyed by
    frequency band -- mirrors liboculus/Constants.h + oculus_sonar_driver's
    ping_to_sonar_image.h (rx_beamwidths = AzimuthBeamwidthRad, tx_beamwidths =
    ElevationBeamwidthRad). The beamwidth is the angular WIDTH of a beam, NOT the
    beam spacing (hori_fov / n_beams). Defaults to the 1.2 MHz (M3000d/M1200d
    low-freq) values for an unrecognised frequency.
    """
    f = float(frequency_hz)
    if 1.1e6 < f < 1.3e6:        # 1.2 MHz (M300d/M1200d LF): az 0.6 deg, el 20 deg
        return math.radians(0.6), math.radians(20.0)
    if 2.0e6 < f < 2.2e6:        # 2.1 MHz (M1200d): az 0.4 deg, el 20 deg
        return math.radians(0.4), math.radians(20.0)
    if 2.9e6 < f < 3.1e6:        # 3.0 MHz (M300d HF): az 0.4 deg, el 20 deg
        return math.radians(0.4), math.radians(20.0)
    return math.radians(0.6), math.radians(20.0)


def sonar_intensity_uint8(grid):
    """Map an OceanSim sonar_map grid to the row-major (range-major) uint8 image.

    grid is (n_range, n_azimuth, 3) with channel 2 the intensity in [0, 1].
    Returns (img8, n_range, n_beams); img8 is contiguous (n_range, n_beams) so
    ``img8.reshape(-1)`` is range-major (element (r,b) at r*n_beams + b), the
    marine_acoustic_msgs / Oculus driver convention.
    """
    intensity = np.ascontiguousarray(grid[:, :, 2])
    n_range, n_beams = intensity.shape
    img8 = np.clip(intensity * 255.0, 0.0, 255.0).astype(np.uint8)
    return img8, n_range, n_beams

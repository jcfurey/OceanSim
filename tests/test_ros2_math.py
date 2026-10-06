"""Unit tests for isaacsim/oceansim/utils/ros2_math.py (pure numpy, no ROS)."""

import importlib.util
import math
import os

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "ros2_math.py")


def _load():
    spec = importlib.util.spec_from_file_location("ros2_math", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load()


def _quat_yaw(deg):
    a = math.radians(deg) / 2.0
    return (math.cos(a), 0.0, 0.0, math.sin(a))  # wxyz, rotation about +z


def test_quat_wxyz_to_xyzw(m):
    assert m.quat_wxyz_to_xyzw([1, 2, 3, 4]) == (2.0, 3.0, 4.0, 1.0)


def test_rot_identity_and_yaw(m):
    assert np.allclose(m.rot_from_quat_wxyz((1, 0, 0, 0)), np.eye(3))
    R = m.rot_from_quat_wxyz(_quat_yaw(90))
    assert np.allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-9)  # body +x -> world +y


def test_world_to_body_is_R_transpose(m):
    R = m.rot_from_quat_wxyz(_quat_yaw(90))
    # world +y maps back to body +x
    assert np.allclose(m.world_to_body(R, [0.0, 1.0, 0.0]), [1, 0, 0], atol=1e-9)


def test_specific_force_rest_and_accel(m):
    I = np.eye(3)
    # level, at rest -> +g on z (REP-145)
    assert np.allclose(m.specific_force_body(I, [0, 0, 0], 9.81), [0, 0, 9.81])
    # level + 2 m/s^2 world x -> body x = 2, z = g
    assert np.allclose(m.specific_force_body(I, [2, 0, 0], 9.81), [2, 0, 9.81])
    # yaw 90 deg, world +x accel -> body -y
    R = m.rot_from_quat_wxyz(_quat_yaw(90))
    assert np.allclose(m.specific_force_body(R, [2, 0, 0], 9.81), [0, -2, 9.81], atol=1e-9)


def test_diag(m):
    assert m.diag3(1, 2, 3) == [1, 0, 0, 0, 2, 0, 0, 0, 3]
    d6 = m.diag6(1, 2, 3, 4, 5, 6)
    assert len(d6) == 36 and [d6[i * 6 + i] for i in range(6)] == [1, 2, 3, 4, 5, 6]


def test_sim_time_split_and_carry(m):
    assert m.sim_time_to_sec_nanosec(2.5) == (2, 500000000)
    # rounding that hits 1e9 must carry into sec
    sec, ns = m.sim_time_to_sec_nanosec(3.9999999999)
    assert sec == 4 and 0 <= ns < 1_000_000_000


def test_rate_gate(m):
    g = m.RateGate(10.0)  # period 0.1 s
    assert g.ready(0.0) is True
    assert g.ready(0.05) is False
    assert g.ready(0.10) is True
    assert g.ready(0.05) is True   # backward jump (reset) -> immediately ready
    g0 = m.RateGate(0.0)           # disabled -> always
    assert all(g0.ready(t) for t in (0.0, 0.0, 0.0))


def test_sonar_beam_directions(m):
    dirs = m.sonar_beam_directions(90.0, 3)
    assert len(dirs) == 3
    assert np.allclose(dirs[1], (0.0, 0.0, 1.0), atol=1e-9)   # centre beam: forward +z
    assert dirs[0][1] > 0 and dirs[2][1] < 0                  # azimuth spread in -y


def test_sonar_ranges_bin_centres(m):
    # Oculus convention: range(i) = min + (i+0.5)*res, res = (max-min)/n.
    r = m.sonar_ranges(0.2, 3.0, 5)
    res = (3.0 - 0.2) / 5            # 0.56
    assert len(r) == 5
    assert r[0] == pytest.approx(0.2 + 0.5 * res)   # first bin CENTRE, not min
    assert r[-1] == pytest.approx(0.2 + 4.5 * res)  # last bin centre, < max
    assert np.allclose(np.diff(r), res)             # uniform spacing == res


def test_oculus_beamwidths_by_frequency(m):
    az12, el12 = m.oculus_beamwidths(1.2e6)
    assert az12 == pytest.approx(math.radians(0.6))   # M300d LF azimuth beamwidth
    assert el12 == pytest.approx(math.radians(20.0))  # elevation aperture
    az30, _ = m.oculus_beamwidths(3.0e6)
    assert az30 == pytest.approx(math.radians(0.4))   # HF azimuth beamwidth
    # rx beamwidth (0.6deg) is WIDER than typical beam spacing (e.g. 130/520=0.25deg)
    assert az12 > math.radians(130.0 / 520.0)


def test_sonar_intensity_uint8_range_major(m):
    n_range, n_beams = 2, 3
    grid = np.zeros((n_range, n_beams, 3), dtype=np.float32)
    grid[:, :, 2] = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 1.0]])
    img8, nr, nb = m.sonar_intensity_uint8(grid)
    assert (nr, nb) == (n_range, n_beams)
    flat = img8.reshape(-1)
    # range-major: element (r,b) at r*n_beams + b
    assert flat[0 * n_beams + 2] == 255   # (range0, beam2) == 1.0
    assert flat[1 * n_beams + 0] == int(0.25 * 255)   # uint8 cast truncates (63)


def test_rategate_harmonic_rate_not_halved(m):
    # A 60 Hz gate driven by a 60 Hz sim: accumulating t += 1/60 lands the
    # elapsed interval a few ULPs below the period on some ticks; the strict >=
    # comparison then skipped every other publish (halving the rate). With the
    # tolerance, EVERY tick must fire.
    gate = m.RateGate(60.0)
    t, dt = 0.0, 1.0 / 60.0
    fired = 0
    for _ in range(600):
        if gate.ready(t):
            fired += 1
        t += dt
    assert fired == 600, f"gate fired {fired}/600 at the harmonic rate"


def test_rategate_still_throttles(m):
    # tolerance must not break sub-harmonic throttling: 10 Hz gate on a 60 Hz sim
    # fires on every 6th tick.
    gate = m.RateGate(10.0)
    t, dt = 0.0, 1.0 / 60.0
    fired = sum(1 for i in range(600) if gate.ready(t + i * dt))
    assert 99 <= fired <= 101


def test_single_beam_points_forward(m):
    # n_beams=1 used to put the sole beam at -fov/2 (the fan edge) via
    # linspace(-h, h, 1); a single beam must point straight ahead (+z).
    (d,) = m.sonar_beam_directions(130.0, 1)
    assert d[1] == pytest.approx(0.0)   # -sin(0)
    assert d[2] == pytest.approx(1.0)   # cos(0)


def test_oculus_beamwidths_bands(m):
    import math
    az, el = m.oculus_beamwidths(1.2e6)     # M300d/M1200d LF
    assert az == pytest.approx(math.radians(0.6))
    assert el == pytest.approx(math.radians(20.0))
    az, _ = m.oculus_beamwidths(2.1e6)      # M1200d HF
    assert az == pytest.approx(math.radians(0.4))
    az, _ = m.oculus_beamwidths(3.0e6)      # M300d HF
    assert az == pytest.approx(math.radians(0.4))
    az, el = m.oculus_beamwidths(999.0)     # unrecognised -> 1.2 MHz defaults
    assert az == pytest.approx(math.radians(0.6))


def test_sonar_beam_directions_bin_centres(m):
    """With the sensor's angular_res, beam j points at its bin centre
    -fov/2 + (j+0.5)*res (the grid bins azimuth from the fan edge)."""
    fov, res = 130.0, 0.25
    n = int(np.ceil(fov / res))
    dirs = m.sonar_beam_directions(fov, n, res)
    az = np.degrees(np.arctan2([-d[1] for d in dirs], [d[2] for d in dirs]))
    assert az[0] == pytest.approx(-65.0 + 0.125)
    assert np.allclose(np.diff(az), res)


def test_sonar_ranges_follow_sensor_grid(m):
    """The sensor grid is np.arange(min, max, range_res); published centres
    must follow that grid even when the span isn't a whole number of bins."""
    lo, hi, res = 0.2, 3.0, 0.03
    grid_edges = np.arange(lo, hi, res)
    r = m.sonar_ranges(lo, hi, len(grid_edges), res)
    assert np.allclose(r, grid_edges + res / 2, atol=1e-6)


@pytest.mark.parametrize("arr", [
    np.arange(24, dtype=np.uint8).reshape(2, 4, 3),               # rgb8
    np.linspace(0.5, 9.5, 12, dtype=np.float32).reshape(3, 4),     # 32FC1 depth
    np.arange(40, dtype=np.uint8).reshape(5, 8)[:, ::2],           # non-contiguous view
])
def test_uint8_payload_matches_tobytes(m, arr):
    """array('B') payload (rosidl fast path) carries exactly arr.tobytes()."""
    import array as _array
    out = m.uint8_payload(arr)
    assert isinstance(out, _array.array) and out.typecode == "B"
    assert out.tobytes() == np.ascontiguousarray(arr).tobytes()

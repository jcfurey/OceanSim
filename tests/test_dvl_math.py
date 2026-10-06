"""Unit tests for the pure DVL math (Janus transform + adaptive update rate).

Pure numpy -- no Isaac Sim. Loaded by file path to avoid the isaacsim.oceansim
namespace package.
"""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "dvl_math.py")


def _load():
    spec = importlib.util.spec_from_file_location("dvl_math", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load()


# --- beam_velocity_transform -----------------------------------------------

def test_transform_known_values(m):
    # elevation 30 deg: sin=0.5 -> 1/(2*sin)=1.0; cos=sqrt(3)/2 -> 1/(4*cos)=0.288675
    t = m.beam_velocity_transform(30.0)
    assert t.shape == (3, 4)
    assert np.allclose(t[0], [1.0, 0.0, -1.0, 0.0])
    assert np.allclose(t[1], [0.0, 1.0, 0.0, -1.0])
    assert np.allclose(t[2], [0.288675134] * 4)


def test_transform_matches_inline_formula(m):
    # Characterise against the closed form for the default 22.5 deg elevation.
    elev = 22.5
    s, c = np.sin(np.deg2rad(elev)), np.cos(np.deg2rad(elev))
    expected = np.array([[1 / (2 * s), 0, -1 / (2 * s), 0],
                         [0, 1 / (2 * s), 0, -1 / (2 * s)],
                         [1 / (4 * c)] * 4])
    assert np.allclose(m.beam_velocity_transform(elev), expected)


@pytest.mark.parametrize("bad", [0.0, 90.0, -90.0, 180.0])
def test_transform_rejects_degenerate_elevation(m, bad):
    with pytest.raises(ValueError):
        m.beam_velocity_transform(bad)


# --- adaptive_sensor_dt ----------------------------------------------------

FB = (5.0, 100.0)        # (min_freq, max_freq) Hz
RB = (7.5, 50.0)         # (near, far) m
SS = 1500.0


def test_dt_close_range_is_max_freq(m):
    assert m.adaptive_sensor_dt(5.0, FB, RB, SS) == pytest.approx(1 / 100.0)
    assert m.adaptive_sensor_dt(7.5, FB, RB, SS) == pytest.approx(1 / 100.0)  # at the near bound


def test_dt_far_range_is_min_freq(m):
    assert m.adaptive_sensor_dt(80.0, FB, RB, SS) == pytest.approx(1 / 5.0)
    assert m.adaptive_sensor_dt(50.0, FB, RB, SS) == pytest.approx(1 / 5.0)   # at the far bound


def test_dt_mid_range_ramp(m):
    # Ramp from hi_f at near down to the sound-speed-limited freq at FAR.
    mr = 20.0
    far_freq = min(100.0, SS / (2 * 50.0))            # 15 Hz at far=50 m
    freq = 100.0 + (far_freq - 100.0) * (mr - 7.5) / (50.0 - 7.5)
    assert m.adaptive_sensor_dt(mr, FB, RB, SS) == pytest.approx(1 / freq)
    # and it sits between the two fixed-rate extremes
    assert 1 / 100.0 < m.adaptive_sensor_dt(mr, FB, RB, SS) < 1 / 5.0


def test_dt_does_not_invert_for_low_max_freq(m):
    # When hi_f is below the sound-speed limit at the near bound, the old formula
    # made the frequency RISE with range (dt fall) -- physically backwards. The
    # ramp must be non-increasing in frequency (dt non-decreasing) with range.
    fb = (5.0, 20.0)                                   # hi_f 20 < c/(2*near)=100
    dts = [m.adaptive_sensor_dt(r, fb, RB, SS) for r in (8.0, 15.0, 25.0, 40.0, 49.0)]
    assert all(dts[k] <= dts[k + 1] + 1e-12 for k in range(len(dts) - 1)), dts
    assert all(dt >= 1 / 20.0 - 1e-12 for dt in dts)  # never faster than hi_f


def test_dt_continuous_at_near_bound(m):
    # the ramp meets the close-range branch at min_range == near (freq == max).
    eps = 1e-6
    assert m.adaptive_sensor_dt(7.5 + eps, FB, RB, SS) == pytest.approx(1 / 100.0, rel=1e-4)


def test_dt_nan_range_falls_back_to_min_freq(m):
    # all beams missed -> NaN closest range -> slowest safe rate, not a NaN dt.
    dt = m.adaptive_sensor_dt(float('nan'), FB, RB, SS)
    assert np.isfinite(dt)
    assert dt == pytest.approx(1 / 5.0)


# --- mount_point_velocity_body (lever arm) ---------------------------------

def _rot_z(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def test_lever_arm_zero_rotation_is_plain_body_velocity(m):
    rot = _rot_z(30.0)
    v_world = np.array([0.4, -0.2, 0.1])
    out = m.mount_point_velocity_body(v_world, np.zeros(3), rot, [0.3, 0.1, -0.2])
    assert np.allclose(out, rot.T @ v_world)


def test_lever_arm_yaw_rate_adds_sway(m):
    """0.5 m/s surge with 0.5 rad/s yaw; DVL 0.209 m behind the centre of mass
    (the case from the audit): the mount moves sideways at -omega * x."""
    out = m.mount_point_velocity_body([0.5, 0.0, 0.0], [0.0, 0.0, 0.5], np.eye(3),
                                      [-0.209, 0.0, -0.06])
    assert out == pytest.approx([0.5, -0.1045, 0.0])


def test_lever_arm_matches_finite_difference_of_mount_position(m):
    """Independent check: differentiate the mount's world position along a
    rigid motion (constant v, omega) and express it in the body frame."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        v = rng.normal(size=3)
        w = rng.normal(size=3)
        r_body = rng.normal(size=3)
        yaw0 = rng.uniform(0, 360)
        rot0 = _rot_z(yaw0) @ np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)],
                                        [0, np.sin(0.3), np.cos(0.3)]])
        dt = 1e-6

        def mount_world(t):
            # rotation R(t) = exp([w]x t) R0 (world-frame angular velocity)
            th = np.linalg.norm(w) * t
            k = w / np.linalg.norm(w)
            kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            r_t = np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx
            return v * t + r_t @ rot0 @ r_body

        fd = (mount_world(dt) - mount_world(-dt)) / (2 * dt)
        out = m.mount_point_velocity_body(v, w, rot0, r_body)
        assert np.allclose(out, rot0.T @ fd, atol=1e-6)


# --- velocity_covariance / altitude_from_beam_ranges -----------------------

def test_velocity_covariance_matches_sampled_noise(m):
    """T Sigma T^T must equal the empirical covariance of T @ beam_noise."""
    t = m.beam_velocity_transform(22.5)
    l = np.diag([0.02, 0.03, 0.025, 0.02])
    l[1, 0] = 0.01  # correlated beams
    rng = np.random.default_rng(7)
    samples = (t @ (l @ rng.standard_normal((4, 200000)))).T
    assert np.allclose(m.velocity_covariance(t, l), np.cov(samples.T), atol=2e-5)


def test_velocity_covariance_isotropic_scaling(m):
    """Isotropic beam variance s^2 -> s^2/(2 sin^2) in x/y and s^2/(4 cos^2) in z."""
    s2 = 0.01
    cov = m.velocity_covariance(m.beam_velocity_transform(22.5), np.sqrt(s2) * np.eye(4))
    e = np.deg2rad(22.5)
    assert cov[0, 0] == pytest.approx(s2 / (2 * np.sin(e) ** 2))
    assert cov[2, 2] == pytest.approx(s2 / (4 * np.cos(e) ** 2))
    assert cov[0, 1] == pytest.approx(0.0) and cov[0, 2] == pytest.approx(0.0)


def test_velocity_covariance_zero_for_noise_free(m):
    assert not np.any(m.velocity_covariance(m.beam_velocity_transform(22.5), np.zeros((4, 4))))


def test_altitude_is_vertical_not_slant_range(m):
    h = 4.0
    slant = h / np.cos(np.deg2rad(22.5))
    assert m.altitude_from_beam_ranges([slant] * 4, 22.5) == pytest.approx(h)


def test_altitude_ignores_missed_beams(m):
    h = 2.5
    slant = h / np.cos(np.deg2rad(30.0))
    out = m.altitude_from_beam_ranges([slant, float("nan"), slant, float("inf")], 30.0)
    assert out == pytest.approx(h)
    assert np.isnan(m.altitude_from_beam_ranges([float("nan")] * 4, 22.5))

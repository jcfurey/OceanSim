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

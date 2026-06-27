"""Unit tests for isaacsim/oceansim/utils/camera_math.py (pure numpy)."""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "camera_math.py")


def _load():
    spec = importlib.util.spec_from_file_location("camera_math", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cm():
    return _load()


def test_pinhole_intrinsics(cm):
    info = cm.pinhole_intrinsics(width=640, height=480, focal=1.0,
                                 h_aper=2.0, v_aper=1.5)
    fx = 640 * 1.0 / 2.0
    fy = 480 * 1.0 / 1.5
    assert info["fx"] == pytest.approx(fx)
    assert info["fy"] == pytest.approx(fy)
    assert info["cx"] == 320.0 and info["cy"] == 240.0
    # K row-major [fx 0 cx; 0 fy cy; 0 0 1]
    assert info["k"] == [fx, 0.0, 320.0, 0.0, fy, 240.0, 0.0, 0.0, 1.0]
    # P [fx 0 cx 0; 0 fy cy 0; 0 0 1 0]
    assert info["p"] == [fx, 0.0, 320.0, 0.0, 0.0, fy, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert info["distortion_model"] == "plumb_bob"
    assert info["d"] == [0.0] * 5


def test_pinhole_zero_aperture_fallback(cm):
    info = cm.pinhole_intrinsics(width=100, height=50, focal=1.0, h_aper=0, v_aper=0)
    assert info["fx"] == 100.0 and info["fy"] == 100.0  # fy falls back to fx


def test_radial_to_planar_factor(cm):
    fx = fy = 100.0
    cx, cy = 50.0, 25.0
    factor = cm.radial_to_planar_factor(50, 100, fx, fy, cx, cy)
    assert factor.shape == (50, 100)
    # At the principal point the ray is along the optical axis: planar == radial.
    assert factor[int(cy), int(cx)] == pytest.approx(1.0, abs=1e-6)
    # Off-axis pixels foreshorten: factor < 1.
    assert factor[0, 0] < 1.0
    assert np.all(factor > 0) and np.all(factor <= 1.0 + 1e-6)

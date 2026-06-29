"""Test the underwater render kernel's non-finite-depth guard (CPU Warp)."""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")
wp = pytest.importorskip("warp")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "UWrenderer_utils.py")

DEV = "cpu"


def _load():
    wp.init()
    spec = importlib.util.spec_from_file_location("UWrenderer_utils", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load()


def test_infinite_background_depth_with_zero_coeff_is_finite(m):
    """Background pixels come back as +inf depth from distance_to_camera. With a
    zero coefficient channel, -inf*0 = NaN used to corrupt the pixel via
    wp.uint8(NaN). The guard must keep the output well-defined."""
    raw = np.array([[[100, 150, 200, 255]]], dtype=np.uint8)   # 1x1 RGBA
    depth = np.array([[np.inf]], dtype=np.float32)
    out = wp.zeros((1, 1, 4), dtype=wp.uint8, device=DEV)
    wp.launch(m.UW_render, dim=(1, 1),
              inputs=[wp.array(raw, dtype=wp.uint8, device=DEV),
                      wp.array(depth, dtype=wp.float32, device=DEV),
                      wp.vec3(0.0, 0.0, 0.0),            # backscatter_value
                      wp.vec3(0.0, 0.1, 0.1),            # atten_coeff: ch0 == 0
                      wp.vec3(0.0, 0.0, 0.0)],           # backscatter_coeff
              outputs=[out], device=DEV)
    wp.synchronize()
    px = out.numpy()[0, 0]
    # ch0: zero atten coeff -> exp(0)=1 -> raw unchanged (100); no backscatter.
    assert px[0] == 100
    # ch1/ch2: positive coeff over a far (clamped) depth -> fully attenuated -> 0.
    assert px[1] == 0 and px[2] == 0
    assert px[3] == 255                                  # alpha passthrough


def test_finite_depth_unchanged_by_guard(m):
    """A finite depth must be unaffected by the non-finite guard."""
    raw = np.array([[[120, 120, 120, 255]]], dtype=np.uint8)
    depth = np.array([[2.0]], dtype=np.float32)
    out = wp.zeros((1, 1, 4), dtype=wp.uint8, device=DEV)
    wp.launch(m.UW_render, dim=(1, 1),
              inputs=[wp.array(raw, dtype=wp.uint8, device=DEV),
                      wp.array(depth, dtype=wp.float32, device=DEV),
                      wp.vec3(0.1, 0.1, 0.1),
                      wp.vec3(0.05, 0.05, 0.05),
                      wp.vec3(0.05, 0.05, 0.05)],
              outputs=[out], device=DEV)
    wp.synchronize()
    px = out.numpy()[0, 0]
    # exp(-2*0.05)=~0.905; raw 120*0.905 ~= 108.6, plus small backscatter.
    assert np.all(np.isfinite(px))
    assert 100 <= px[0] <= 130

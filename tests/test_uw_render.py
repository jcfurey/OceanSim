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


def test_uw_render_2_infinite_depth_is_finite(m):
    """UW_render_2 (caustics path) needs the same background guard as UW_render."""
    raw = np.array([[[100, 150, 200, 255]]], dtype=np.uint8)
    out = wp.zeros((1, 1, 4), dtype=wp.uint8, device=DEV)
    wp.launch(m.UW_render_2, dim=(1, 1),
              inputs=[wp.array(raw, dtype=wp.uint8, device=DEV),
                      wp.array(np.array([[np.inf]], np.float32), dtype=wp.float32, device=DEV),
                      1.0,
                      wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.1, 0.1), wp.vec3(0.0, 0.0, 0.0)],
              outputs=[out], device=DEV)
    wp.synchronize()
    px = out.numpy()[0, 0]
    assert px[0] == 100 and px[1] == 0 and px[2] == 0 and px[3] == 255


def _attenuate(m, depth, max_range=20.0, raw_val=255):
    h, w = depth.shape
    raw = np.full((h, w, 4), raw_val, dtype=np.uint8)
    out = wp.zeros((h, w), dtype=wp.float32, device=DEV)
    wp.launch(m.UW_depth_turbidity_attenuator, dim=(h, w),
              inputs=[wp.array(raw, dtype=wp.uint8, device=DEV),
                      wp.array(depth.astype(np.float32), dtype=wp.float32, device=DEV),
                      float(max_range),
                      wp.vec3(0.0, 0.31, 0.24), wp.vec3(0.05, 0.05, 0.05), wp.vec3(0.05, 0.2, 0.05),
                      0.0, 0],
              outputs=[out], device=DEV)
    wp.synchronize()
    return out.numpy()


def test_depth_attenuator_applies_max_range(m):
    """A bright surface beyond max_range must read as no return (0); the old
    kernel ignored max_range and kept white pixels out to ~27.7 m."""
    out = _attenuate(m, np.array([[5.0, 19.9, 25.0, 27.0]]))
    assert out[0, 0] == pytest.approx(5.0) and out[0, 1] == pytest.approx(19.9)
    assert out[0, 2] == 0.0 and out[0, 3] == 0.0


def test_depth_attenuator_non_finite_is_no_return(m):
    out = _attenuate(m, np.array([[np.inf, np.nan, -1.0, 2.0]]))
    assert out[0].tolist()[:3] == [0.0, 0.0, 0.0] and out[0, 3] == pytest.approx(2.0)


def test_caustics_uv_follow_horizontal_plane(m):
    """On a flat Z-up floor (constant z) caustics must vary across x AND y;
    projecting on XZ left v constant -> one texture row (stripes)."""
    h, w, th, tw = 4, 4, 8, 8
    yy, xx = np.mgrid[0:h, 0:w]
    pos = np.stack([xx * 0.37, yy * 0.53, np.full((h, w), -1.0)], axis=-1).astype(np.float32)
    nml = np.zeros((h, w, 3), np.float32); nml[..., 2] = 1.0
    tex = np.zeros((th, tw, 4), np.uint8)
    tex[..., 0] = np.arange(th, dtype=np.uint8)[:, None] * 30   # red encodes texture row
    tex[..., 3] = 255
    out = wp.zeros((h, w, 4), dtype=wp.uint8, device=DEV)
    wp.launch(m.blend_caustics, dim=(w, h),
              inputs=[wp.array(np.zeros((h, w, 4), np.uint8), dtype=wp.uint8, device=DEV),
                      wp.array(pos, dtype=wp.float32, device=DEV),
                      wp.array(nml, dtype=wp.float32, device=DEV),
                      wp.array(tex, dtype=wp.uint8, device=DEV),
                      wp.vec3(0.0, 0.0, 1.0), 1.0, 1.0, 1.0, -10.0, 10.0, tw, th],
              outputs=[out], device=DEV)
    wp.synchronize()
    red = out.numpy()[..., 0]
    assert len(np.unique(red[:, 0])) > 1      # moving in y changes the sampled row


def test_depth_to_world_pos_with_camera_to_world_matrix(m):
    """The callers now pass inv(cameraViewTransform.reshape(4,4).T); with it the
    centre pixel's point lands at camera position + d along the view axis (the
    raw view transform put it at -C, so caustics slid with the camera)."""
    w = h = 5
    c = np.array([2.0, -1.0, 3.0])
    world_from_cam = np.eye(4); world_from_cam[:3, 3] = c        # USD cam looks down -Z
    view_row = np.linalg.inv(world_from_cam).T.reshape(-1)       # CameraParams layout
    proj = np.diag([2.0, 2.0, 0.0, 0.0])
    depth = np.full((h, w), 4.0, dtype=np.float32)
    out = wp.zeros((h, w, 3), dtype=wp.float32, device=DEV)
    cam_to_world = np.linalg.inv(view_row.reshape(4, 4).T).astype(np.float32)
    wp.launch(m.depth_to_world_pos, dim=(w, h),
              inputs=[wp.array(depth, dtype=wp.float32, device=DEV),
                      wp.mat44f(proj.astype(np.float32)), wp.mat44f(cam_to_world), w, h],
              outputs=[out], device=DEV)
    wp.synchronize()
    centre = out.numpy()[h // 2, w // 2]
    assert np.allclose(centre, c + [0.0, 0.0, -4.0], atol=0.6)  # centre ray ~ -Z (pixel-corner offset)
    assert np.linalg.norm(centre - c) == pytest.approx(4.0, abs=1e-4)

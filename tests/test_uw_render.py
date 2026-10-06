"""Tests for the underwater camera model (CPU Warp): the linear-light
Akkaynak-Treibitz image formation, its non-finite-depth guard, the
visibility-gated degraded depth, caustics UVs and the water-table conversion."""

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


def _attenuate(m, depth, max_range=20.0, raw_val=255, backscatter=(0.0, 0.31, 0.24),
               atten=(0.05, 0.05, 0.05), back_coeff=(0.05, 0.2, 0.05), min_visibility=0.25):
    h, w = depth.shape
    raw = np.full((h, w, 4), raw_val, dtype=np.uint8)
    out = wp.zeros((h, w), dtype=wp.float32, device=DEV)
    wp.launch(m.UW_depth_turbidity_attenuator, dim=(h, w),
              inputs=[wp.array(raw, dtype=wp.uint8, device=DEV),
                      wp.array(depth.astype(np.float32), dtype=wp.float32, device=DEV),
                      float(max_range),
                      wp.vec3(*backscatter), wp.vec3(*atten), wp.vec3(*back_coeff),
                      0.0, float(min_visibility), 0],
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


# --- linear-light image formation -------------------------------------------

def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _reference(raw, depth, b_inf, atten, back):
    """numpy I = J exp(-beta_D z) + B_inf (1 - exp(-beta_B z)) in linear light."""
    j = _srgb_to_linear(raw[..., :3] / 255.0)
    z = depth[..., None]
    lin = (j * np.exp(-z * np.asarray(atten))
           + _srgb_to_linear(np.asarray(b_inf)) * (1.0 - np.exp(-z * np.asarray(back))))
    return np.floor(_linear_to_srgb(lin) * 255.0 + 0.5).astype(np.uint8)


def _render(m, raw, depth, b_inf, atten, back, scale=None):
    h, w = depth.shape
    out = wp.zeros((h, w, 4), dtype=wp.uint8, device=DEV)
    ins = [wp.array(raw, dtype=wp.uint8, device=DEV),
           wp.array(depth.astype(np.float32), dtype=wp.float32, device=DEV)]
    if scale is None:
        kern = m.UW_render
    else:
        kern = m.UW_render_2
        ins.append(float(scale))
    ins += [wp.vec3(*b_inf), wp.vec3(*atten), wp.vec3(*back)]
    wp.launch(kern, dim=(h, w), inputs=ins, outputs=[out], device=DEV)
    wp.synchronize()
    return out.numpy()


def test_clear_water_leaves_every_srgb_level_unchanged(m):
    """Zero coefficients: the sRGB decode -> encode round trip must be exact
    for all 256 levels (no banding introduced by the linear-light path)."""
    lv = np.arange(256, dtype=np.uint8)
    raw = np.stack([lv, lv[::-1], lv, np.full(256, 7, np.uint8)], axis=-1)[None]
    out = _render(m, raw, np.full((1, 256), 3.0), (0.4, 0.5, 0.6), (0, 0, 0), (0, 0, 0))
    assert np.array_equal(out, raw)


def test_distant_water_renders_the_veiling_colour(m):
    """Far enough away only the veil remains, and it must be B_inf as given
    (an sRGB colour) whatever the object colour."""
    b_inf = (0.1, 0.42, 0.52)
    raw = np.array([[[255, 0, 128, 255], [0, 255, 30, 255]]], np.uint8)
    out = _render(m, raw, np.full((1, 2), 500.0), b_inf, (0.2, 0.1, 0.05), (0.2, 0.1, 0.05))
    want = np.floor(np.asarray(b_inf) * 255.0 + 0.5)
    assert np.allclose(out[0, :, :3], want, atol=1)


def test_render_matches_linear_light_reference(m):
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 256, (6, 7, 4), dtype=np.uint8)
    depth = rng.uniform(0.1, 15.0, (6, 7))
    b_inf, atten, back = (0.05, 0.31, 0.24), (0.37, 0.044, 0.035), (0.05, 0.2, 0.05)
    out = _render(m, raw, depth, b_inf, atten, back)
    ref = _reference(raw.astype(np.float64), depth, b_inf, atten, back)
    assert np.abs(out[..., :3].astype(int) - ref).max() <= 1
    assert np.array_equal(out[..., 3], raw[..., 3])


def test_attenuation_is_beta_in_linear_light(m):
    """A mid-grey at z = ln2 / beta must carry exactly half its linear
    radiance; on the encoded values (the old model) it came out ~2x too dark
    in linear terms."""
    beta = 0.2
    z = np.log(2.0) / beta
    raw = np.array([[[188, 188, 188, 255]]], np.uint8)
    out = _render(m, raw, np.array([[z]]), (0, 0, 0), (beta,) * 3, (0, 0, 0))
    half = _srgb_to_linear(188 / 255.0) / 2.0
    assert _srgb_to_linear(out[0, 0, 0] / 255.0) == pytest.approx(half, rel=0.02)


def test_uw_render_2_scale_multiplies_both_coefficients(m):
    rng = np.random.default_rng(4)
    raw = rng.integers(0, 256, (3, 4, 4), dtype=np.uint8)
    depth = rng.uniform(0.5, 8.0, (3, 4))
    b_inf, atten, back = (0.14, 0.3, 0.5), (0.1, 0.05, 0.02), (0.2, 0.1, 0.05)
    scaled = _render(m, raw, depth, b_inf, atten, back, scale=1.7)
    direct = _render(m, raw, depth, b_inf, tuple(1.7 * a for a in atten),
                     tuple(1.7 * b for b in back))
    assert np.abs(scaled.astype(int) - direct.astype(int)).max() <= 1


# --- visibility-gated degraded depth ----------------------------------------

def test_depth_gate_keeps_dark_surfaces_up_close(m):
    """The old gate compared absolute brightness to 0.25, so a dark (sRGB 40)
    surface never returned depth even at 0.5 m; with clear signal over the
    veil it must."""
    out = _attenuate(m, np.array([[0.5, 1.0]]), raw_val=40)
    assert out[0].tolist() == pytest.approx([0.5, 1.0])


def test_depth_gate_drops_black_surfaces(m):
    out = _attenuate(m, np.array([[0.5, 5.0]]), raw_val=0)
    assert out[0].tolist() == [0.0, 0.0]


def test_depth_gate_depends_on_water_turbidity(m):
    """Same surface and range: kept in clear water, lost in the veil of murky
    water, where backscatter dominates what reaches the camera."""
    depth = np.array([[8.0]])
    clear = _attenuate(m, depth, raw_val=120, backscatter=(0.05, 0.3, 0.25),
                       atten=(0.02, 0.02, 0.02), back_coeff=(0.02, 0.02, 0.02))
    murky = _attenuate(m, depth, raw_val=120, backscatter=(0.3, 0.3, 0.1),
                       atten=(0.6, 0.6, 0.6), back_coeff=(0.6, 0.6, 0.6))
    assert clear[0, 0] == pytest.approx(8.0) and murky[0, 0] == 0.0


def test_depth_gate_threshold_matches_visibility_ratio(m):
    """visibility = D / (D + B) in linear light (channel means): the threshold
    straddling the analytic value flips the decision."""
    raw_val, z = 120, 6.0
    b_inf, atten, back = (0.1, 0.3, 0.3), (0.15, 0.1, 0.1), (0.2, 0.2, 0.2)
    j = _srgb_to_linear(raw_val / 255.0)
    d = np.mean(j * np.exp(-z * np.asarray(atten)))
    b = np.mean(_srgb_to_linear(np.asarray(b_inf)) * (1 - np.exp(-z * np.asarray(back))))
    vis = d / (d + b)
    kw = dict(raw_val=raw_val, backscatter=b_inf, atten=atten, back_coeff=back)
    assert _attenuate(m, np.array([[z]]), min_visibility=vis - 0.01, **kw)[0, 0] == pytest.approx(z)
    assert _attenuate(m, np.array([[z]]), min_visibility=vis + 0.01, **kw)[0, 0] == 0.0


# --- water tables -------------------------------------------------------------

def test_transmittance_tables_convert_to_beta(m):
    """Jerlov per-metre transmittance N -> beta = -ln N: clearer water (higher
    N) must give a SMALLER beta, and red the largest one (it was the reverse
    when N was used as beta directly)."""
    type_i = m.water_coefficients((0.905, 0.961, 0.982), "transmittance")
    type_9 = m.water_coefficients((0.550, 0.460, 0.290), "transmittance")
    assert type_i == pytest.approx((-np.log(0.905), -np.log(0.961), -np.log(0.982)))
    assert all(a < b for a, b in zip(type_i, type_9))
    assert type_i[0] == max(type_i)
    assert m.water_coefficients((1.0, 1.0, 1.0), "transmittance") == (0.0, 0.0, 0.0)


def test_beta_tables_pass_through_and_bad_input_is_rejected(m):
    assert m.water_coefficients((1.5, 0.2, 0.03)) == (1.5, 0.2, 0.03)
    with pytest.raises(ValueError):
        m.water_coefficients((1.2, 0.5, 0.5), "transmittance")
    with pytest.raises(ValueError):
        m.water_coefficients((0.0, 0.5, 0.5), "transmittance")
    with pytest.raises(ValueError):
        m.water_coefficients((0.1, 0.5, 0.5), "jerlov")

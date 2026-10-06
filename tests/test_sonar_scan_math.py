"""Characterisation tests for the imaging-sonar point selection (pure numpy).

The key test asserts the optimised two-stage ``select_in_range_points`` is exactly
equivalent to indexing by the readable ``valid_point_mask`` -- so the selection
can be optimised further (e.g. on-GPU) while staying behaviour-preserving.
"""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "sonar_scan_math.py")


def _load():
    spec = importlib.util.spec_from_file_location("sonar_scan_math", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s():
    return _load()


def test_known_case(s):
    # depth window (0.2, 3.0): keep finite depth strictly inside.
    depth = np.array([0.5, 0.1, 5.0, np.inf, 1.0, 2.0], dtype=np.float32)
    pcl = np.arange(18, dtype=np.float32).reshape(6, 3)
    pcl[4] = [np.nan, 0.0, 0.0]          # in-range depth but non-finite point -> dropped
    normals = (pcl + 100).astype(np.float32)
    sem = np.array([10, 11, 12, 13, 14, 15], dtype=np.uint32)

    pcl_v, n_v, s_v = s.select_in_range_points(depth, pcl, normals, sem, 0.2, 3.0)

    # depth in-range: indices 0, 4, 5; index 4 dropped (nan point) -> 0 and 5.
    assert s_v.tolist() == [10, 15]
    assert np.allclose(pcl_v, pcl[[0, 5]])
    assert np.allclose(n_v, normals[[0, 5]])
    assert pcl_v.dtype == np.float32 and n_v.dtype == np.float32 and s_v.dtype == np.uint32


def test_empty_when_none_in_range(s):
    depth = np.array([0.05, 9.0, np.nan], dtype=np.float32)
    pcl = np.zeros((3, 3), dtype=np.float32)
    pcl_v, n_v, s_v = s.select_in_range_points(depth, pcl, pcl.copy(),
                                               np.zeros(3, np.uint32), 0.2, 3.0)
    assert pcl_v.shape == (0, 3) and n_v.shape == (0, 3) and s_v.shape == (0,)
    assert pcl_v.dtype == np.float32 and s_v.dtype == np.uint32


@pytest.mark.parametrize("seed", range(8))
def test_optimised_equals_reference_mask(s, seed):
    """select_in_range_points must equal indexing by valid_point_mask."""
    rng = np.random.default_rng(seed)
    n = 5000
    depth = rng.uniform(-1.0, 6.0, n).astype(np.float32)
    depth[rng.random(n) < 0.05] = np.inf          # sprinkle non-finite depths
    pcl = rng.uniform(-5, 5, (n, 3)).astype(np.float32)
    pcl[rng.random(n) < 0.05] = np.nan            # sprinkle non-finite points
    normals = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    sem = rng.integers(0, 8, n).astype(np.uint32)
    mn, mx = 0.2, 3.0

    mask = s.valid_point_mask(depth, pcl, mn, mx)
    pcl_v, n_v, s_v = s.select_in_range_points(depth, pcl, normals, sem, mn, mx)

    assert pcl_v.shape[0] == int(mask.sum())
    assert np.array_equal(pcl_v, np.ascontiguousarray(pcl[mask], dtype=np.float32))
    assert np.array_equal(n_v, np.ascontiguousarray(normals[mask], dtype=np.float32))
    assert np.array_equal(s_v, sem[mask].astype(np.uint32))


# --- make_indexToProp_array (reflectivity lookup) --------------------------

def test_indexToProp_basic_mapping(s):
    # Typical OceanSim labels: 0/1 are BACKGROUND/UNLABELLED (no reflectivity ->
    # default 1.0); 2 and 3 carry reflectivity strings.
    idToLabels = {
        '0': {'class': 'BACKGROUND'},
        '1': {'class': 'UNLABELLED'},
        '2': {'reflectivity': '0.5'},
        '3': {'reflectivity': '2.0'},
    }
    arr = s.make_indexToProp_array(idToLabels, 'reflectivity')
    assert arr.shape == (4,)
    assert arr[0] == 1.0 and arr[1] == 1.0   # default for missing property
    assert arr[2] == 0.5 and arr[3] == 2.0


def test_indexToProp_numeric_key_ordering(s):
    # Id 10 must size the array to length 11 (numeric, not lexicographic, max).
    idToLabels = {'2': {'reflectivity': '0.3'}, '10': {'reflectivity': '0.9'}}
    arr = s.make_indexToProp_array(idToLabels, 'reflectivity')
    assert arr.shape == (11,)
    assert arr[10] == 0.9 and arr[2] == 0.3
    assert arr[5] == 1.0          # unlabelled gap keeps default


def test_indexToProp_non_numeric_value_keeps_default(s):
    # A non-numeric reflectivity (fallback label) must not raise -> stays 1.0.
    idToLabels = {'2': {'reflectivity': 'BACKGROUND'}, '3': {'reflectivity': '4.0'}}
    arr = s.make_indexToProp_array(idToLabels, 'reflectivity')
    assert arr[2] == 1.0 and arr[3] == 4.0


def test_indexToProp_empty(s):
    arr = s.make_indexToProp_array({}, 'reflectivity')
    assert arr.shape == (0,)


def test_indexToProp_query_other_property(s):
    # Querying a property no id carries -> all default.
    idToLabels = {'2': {'reflectivity': '0.5'}}
    arr = s.make_indexToProp_array(idToLabels, 'class')
    assert np.all(arr == 1.0)


def test_background_depth_is_invalid_without_runtime_warnings(s):
    depth = np.array([[1.0, np.inf, np.nan], [-np.inf, 2.0, 0.0]])
    pose = np.eye(4)
    pose[:3, 3] = [3, 4, 5]
    with np.errstate(all="raise"):
        points = s.depth_to_world_points(depth, pose, 2.0, 2.0)
    np.testing.assert_allclose(points[[0, 4, 5]], [[2.5, 4.25, 4], [3, 3.5, 3], [3, 4, 5]])
    assert np.isnan(points[[1, 2, 3]]).all()
    selected, _, _ = s.select_in_range_points(depth, points, np.zeros_like(points),
                                             np.zeros(6, np.uint32), 0.1, 5.0)
    np.testing.assert_array_equal(selected, points[[0, 4]].astype(np.float32))


def _fan(s, min_range, max_range, res, fov, beams, height=720):
    n_range = int(np.ceil((max_range - min_range) / res))
    rs, bs, scale = s.sonar_fan_lookup(min_range, max_range, res, np.deg2rad(90.0 - fov / 2.0),
                                       np.deg2rad(fov / beams), n_range, beams, fov, height)
    return rs, bs, scale, n_range


def _assert_mirror_symmetric(rs, bs, beams):
    valid = rs[..., 0] >= 0
    assert np.array_equal(valid, valid[:, ::-1])
    assert np.array_equal(rs, rs[:, ::-1])
    # Every left-hand beam span mirrors a right-hand one.
    np.testing.assert_array_equal(bs[:, ::-1, 0][valid], (beams - 1 - bs[..., 1])[valid])
    np.testing.assert_array_equal(bs[:, ::-1, 1][valid], (beams - 1 - bs[..., 0])[valid])


@pytest.mark.parametrize("fov,max_range", [(130.0, 12.0), (40.0, 5.0)])
def test_sonar_fan_has_metric_scale_and_correct_orientation(s, fov, max_range):
    """Analytical positions map to the expected bins on both M3000d fans."""
    min_range, res, beams, height = 0.1, 0.01, 512, 400
    rs, bs, scale, _ = _fan(s, min_range, max_range, res, fov, beams, height)
    assert scale == pytest.approx(max_range / height)
    assert rs.shape[:2] == bs.shape[:2] and rs.shape[0] == height and rs.shape[2] == 2
    assert rs.shape[1] / height == pytest.approx(2*np.sin(np.deg2rad(fov/2)), abs=2/height)
    _assert_mirror_symmetric(rs, bs, beams)
    azi_res = np.deg2rad(fov / beams)
    for bearing in (-fov/3, 0.0, fov/3):
        # Positive bearing is port (left), negative bearing starboard (right).
        radius = 0.6 * max_range
        right = -radius * np.sin(np.deg2rad(bearing))
        forward = radius * np.cos(np.deg2rad(bearing))
        i = int(np.floor(height - forward/scale))
        j = int(np.floor(right/scale + rs.shape[1]/2))
        # The pixel's bins lie within its own extent (half a diagonal, plus
        # half a bin for the bin-centre fallback) of the point.
        q_range = (radius - min_range) / res
        q_beam = (np.pi/2 + np.deg2rad(bearing) - np.deg2rad(90 - fov/2)) / azi_res
        reach_r = scale / np.sqrt(2) / res + 0.5
        reach_b = scale / np.sqrt(2) / radius / azi_res + 0.5
        assert np.all(np.abs(rs[i, j] + 0.5 - q_range) <= reach_r + 1e-9)
        assert np.all(np.abs(bs[i, j] + 0.5 - q_beam) <= reach_b + 1e-9)
        if bearing:
            assert np.all(np.sign(bs[i, j] + 0.5 - beams/2) == np.sign(bearing))
    # Above the arc / outside the fan / the near-range blind zone are black.
    assert np.all(rs[0, 0] == -1) and np.all(bs[0, 0] == -1)
    assert np.all(rs[-1, rs.shape[1]//2] == -1)


@pytest.mark.parametrize("min_range,max_range,res,fov,beams", [
    (0.1, 10.0, 10.0 / 1024, 130.0, 512),   # M3000d 1.2 MHz preset
    (0.1, 12.0, 12.0 / 1024, 130.0, 512),   # Revolution demo
    (0.1, 5.0, 5.0 / 1024, 40.0, 512),      # M3000d 3 MHz preset
    (0.5, 20.0, 20.0 / 1024, 130.0, 768),   # span not a whole number of bins
    (0.2, 20.0, 20.0 / 1024, 90.0, 128),    # fan edges at 45 deg through pixel corners
    (0.2, 20.0, 20.0 / 1024, 120.0, 256),   # a 45-degree beam edge
])
def test_every_polar_bin_reaches_the_fan(s, min_range, max_range, res, fov, beams):
    """Sampling the bin under each pixel centre skipped up to 2/3 of the
    bins; pooling must show every bin, symmetrically, without pooling much."""
    rs, bs, _, n_range = _fan(s, min_range, max_range, res, fov, beams)
    _assert_mirror_symmetric(rs, bs, beams)
    valid = rs[..., 0] >= 0
    r0, r1, b0, b1 = rs[valid, 0], rs[valid, 1], bs[valid, 0], bs[valid, 1]
    assert r0.min() >= 0 and r1.max() < n_range and b0.min() >= 0 and b1.max() < beams
    assert np.all(r1 >= r0) and np.all(b1 >= b0)
    # Rectangle coverage count per bin via a 2-D difference array.
    cover = np.zeros((n_range + 1, beams + 1), np.int64)
    np.add.at(cover, (r0, b0), 1)
    np.add.at(cover, (r0, b1 + 1), -1)
    np.add.at(cover, (r1 + 1, b0), -1)
    np.add.at(cover, (r1 + 1, b1 + 1), 1)
    cover = cover.cumsum(0).cumsum(1)[:n_range, :beams]
    assert np.count_nonzero(cover == 0) == 0
    assert np.median((r1 - r0 + 1) * (b1 - b0 + 1)) <= 3


def test_constant_slant_range_draws_circular_arc(s):
    rs, _, scale = s.sonar_fan_lookup(0.1, 10.0, 0.1, np.deg2rad(25),
                                      np.deg2rad(130/512), 99, 512, 130, 400)
    rows, cols = np.nonzero((rs[..., 0] <= 49) & (rs[..., 1] >= 49))  # [5.0, 5.1) m bin
    x = (cols + 0.5 - rs.shape[1]/2) * scale
    y = (400 - rows - 0.5) * scale
    radius = np.hypot(x, y)
    half_diagonal = scale / np.sqrt(2)
    assert radius.min() >= 5.0 - half_diagonal
    assert radius.max() < 5.1 + half_diagonal
    assert np.ptp(x) > 8.0


def test_fan_guides_remain_inside_the_aperture(s):
    rs, _, scale = s.sonar_fan_lookup(0.1, 10.0, 0.1, np.deg2rad(25),
                                      np.deg2rad(130/512), 99, 512, 130, 400)
    fan = rs[..., 0] >= 0
    guides = s.sonar_fan_guides(fan, 10.0, 130.0)
    assert np.all(guides[~fan] == 0)
    assert np.count_nonzero(guides) > 100
    # The 5 m ring intersects the 30 degree port ray at its metric position.
    x, y = -2.5, 5*np.cos(np.deg2rad(30))
    j = int(np.floor(x/scale + rs.shape[1]/2))
    i = int(np.floor(400 - y/scale))
    assert guides[i, j] > 0


@pytest.mark.parametrize("fov", [130.0, 40.0])
def test_query_camera_covers_m3000d_vertical_aperture_at_fan_edges(s, fov):
    width = 960
    height = s.sonar_render_height(width, fov, 20.0)
    fx = width / (2*np.tan(np.deg2rad(fov/2)))
    for az in (-fov/2, 0, fov/2):
        # A 10-degree elevation ray at any bearing fits the camera's image plane.
        up_per_forward = np.tan(np.deg2rad(10))/np.cos(np.deg2rad(az))
        assert fx * up_per_forward <= height/2

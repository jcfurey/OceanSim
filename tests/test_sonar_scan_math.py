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

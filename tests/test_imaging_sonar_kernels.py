"""Unit tests for the imaging-sonar Warp kernels.

These exercise the GPU kernels on the CPU device, so they need only
``warp-lang`` + ``numpy`` + ``pytest`` -- NO Isaac Sim. Run with:

    pip install warp-lang numpy pytest
    pytest tests/

or inside the OceanSim container:

    /isaac-sim/python.sh -m pytest tests/

The kernel module is loaded directly by file path so the tests do not depend on
the ``isaacsim.oceansim`` namespace package being importable.
"""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")
wp = pytest.importorskip("warp")

_KERNELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "isaacsim", "oceansim", "utils",
    "ImagingSonar_kernels.py")


def _load_kernels():
    spec = importlib.util.spec_from_file_location("ImagingSonar_kernels", _KERNELS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def kern():
    wp.init()
    return _load_kernels()


DEV = "cpu"


def test_bin_intensity_drops_out_of_grid_points(kern):
    """The bounds-check (PR #3) must drop points whose range/azimuth bin falls
    outside the grid instead of writing out of bounds."""
    n_range, n_beams = 2, 2
    bin_sum = wp.zeros(shape=(n_range, n_beams), dtype=wp.float32, device=DEV)
    bin_count = wp.zeros(shape=(n_range, n_beams), dtype=wp.int32, device=DEV)

    # offsets/res chosen so bin_idx = floor(coord). Points:
    #  (0.5, 0.5) -> bin (0,0)  in-grid
    #  (1.5, 0.5) -> bin (1,0)  in-grid
    #  (9.0, 0.5) -> bin (9,0)  OUT of range (>= n_range) -> must be dropped
    #  (0.5,-1.0) -> bin (0,-1) negative beam -> must be dropped
    pts = np.array([[0.5, 0.5, 0.0],
                    [1.5, 0.5, 0.0],
                    [9.0, 0.5, 0.0],
                    [0.5, -1.0, 0.0]], dtype=np.float32)
    pcl = wp.array(pts, dtype=wp.vec3, device=DEV)
    intensity = wp.array(np.array([1.0, 2.0, 99.0, 99.0], dtype=np.float32),
                         dtype=wp.float32, device=DEV)

    wp.launch(kern.bin_intensity, dim=4,
              inputs=[pcl, intensity, wp.float32(0.0), wp.float32(0.0),
                      wp.float32(1.0), wp.float32(1.0), bin_sum, bin_count],
              device=DEV)
    wp.synchronize()

    s = bin_sum.numpy()
    c = bin_count.numpy()
    assert s[0, 0] == pytest.approx(1.0)   # first point only
    assert s[1, 0] == pytest.approx(2.0)   # second point only
    assert c[0, 0] == 1 and c[1, 0] == 1
    # The two out-of-grid points contributed nothing anywhere.
    assert s.sum() == pytest.approx(3.0)
    assert int(c.sum()) == 2


def test_make_sonar_map_all_zero_guard_no_nan(kern):
    """Empty frame (global max 0) must not divide by zero -> no NaN (PR #4)."""
    n_range, n_beams = 3, 4
    shape = (n_range, n_beams)
    r = wp.array(np.ones(shape, dtype=np.float32), dtype=wp.float32, device=DEV)
    azi = wp.array(np.full(shape, np.pi / 2, dtype=np.float32), dtype=wp.float32, device=DEV)
    intensity = wp.zeros(shape, dtype=wp.float32, device=DEV)        # empty frame
    max_intensity = wp.zeros(shape=(1,), dtype=wp.float32, device=DEV)  # global max 0
    gau = wp.zeros(shape, dtype=wp.float32, device=DEV)
    ray = wp.zeros(shape, dtype=wp.float32, device=DEV)
    result = wp.zeros(shape, dtype=wp.vec3, device=DEV)

    wp.launch(kern.make_sonar_map_all, dim=shape,
              inputs=[r, azi, intensity, max_intensity, gau, ray,
                      wp.float32(0.0), wp.float32(1.0), result],
              device=DEV)
    wp.synchronize()

    out = result.numpy()
    assert np.all(np.isfinite(out)), "zero-guard failed: NaN/inf in sonar map"
    assert np.all(out[:, :, 2] == 0.0)  # intensity channel stays 0


def test_make_sonar_image_column_flip_and_bounds(kern):
    """make_sonar_image must mirror columns into [0, width-1] and write every
    pixel -- no out-of-bounds at j==0, column 0 written (PR #3)."""
    n_range, width = 2, 3
    inten = np.array([0.0, 0.5, 1.0], dtype=np.float32)       # per column
    grid = np.zeros((n_range, width, 3), dtype=np.float32)
    grid[:, :, 2] = inten                                     # channel 2 = intensity
    sonar_data = wp.array(grid, dtype=wp.vec3, device=DEV)
    sonar_image = wp.zeros(shape=(n_range, width, 4), dtype=wp.uint8, device=DEV)

    wp.launch(kern.make_sonar_image, dim=(n_range, width),
              inputs=[sonar_data, sonar_image], device=DEV)
    wp.synchronize()

    img = sonar_image.numpy()
    # column j maps to width-1-j; intensity*255 in channels 0..2, alpha=255.
    expected_cols = [int(round(v * 255)) for v in inten]      # [0,128,255] (approx)
    for j in range(width):
        col = width - 1 - j
        assert img[0, col, 0] == pytest.approx(expected_cols[j], abs=1)
        assert img[0, col, 3] == 255            # alpha written
    # Every column got an alpha (nothing left unwritten, no OOB skip of col 0).
    assert np.all(img[:, :, 3] == 255)

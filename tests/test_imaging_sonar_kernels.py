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


@pytest.mark.parametrize("seed", range(4))
def test_compact_in_range_matches_numpy_reference(kern, seed):
    """The GPU compaction kernel keeps exactly the same point set as the numpy
    reference selection (sonar_scan_math), modulo order (atomic append)."""
    n = 4000
    rng = np.random.default_rng(seed)
    depth = rng.uniform(-1.0, 6.0, n).astype(np.float32)
    depth[rng.random(n) < 0.05] = np.inf
    pcl = rng.uniform(-5, 5, (n, 3)).astype(np.float32)
    pcl[rng.random(n) < 0.05] = np.nan
    normals = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    sem = rng.integers(0, 100, n).astype(np.uint32)
    mn, mx = 0.2, 3.0

    # numpy reference
    valid = (np.isfinite(depth) & (depth > mn) & (depth < mx)
             & np.isfinite(pcl).all(axis=1))
    ref_pcl, ref_n, ref_sem = pcl[valid], normals[valid], sem[valid]

    d = wp.array(depth, dtype=wp.float32, device=DEV)
    p = wp.array(pcl, dtype=wp.float32, device=DEV)
    nm = wp.array(normals, dtype=wp.float32, device=DEV)
    s = wp.array(sem, dtype=wp.uint32, device=DEV)
    counter = wp.zeros(1, dtype=wp.int32, device=DEV)
    out_p = wp.zeros((n, 3), dtype=wp.float32, device=DEV)
    out_n = wp.zeros((n, 3), dtype=wp.float32, device=DEV)
    out_s = wp.zeros(n, dtype=wp.uint32, device=DEV)

    wp.launch(kern.compact_in_range, dim=n,
              inputs=[d, p, nm, s, wp.float32(mn), wp.float32(mx),
                      counter, out_p, out_n, out_s], device=DEV)
    wp.synchronize()

    m = int(counter.numpy()[0])
    assert m == int(valid.sum())
    got_p, got_n, got_s = out_p.numpy()[:m], out_n.numpy()[:m], out_s.numpy()[:m]

    def _key(pc, se):  # lexicographic order so set comparison ignores append order
        return np.lexsort((pc[:, 2], pc[:, 1], pc[:, 0], se))
    go, ro = _key(got_p, got_s), _key(ref_p := ref_pcl, ref_sem)
    assert np.array_equal(got_s[go], ref_sem[ro])
    assert np.allclose(got_p[go], ref_pcl[ro])
    assert np.allclose(got_n[go], ref_n[ro])


def _load_scan_math():
    path = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                        "utils", "sonar_scan_math.py")
    spec = importlib.util.spec_from_file_location("sonar_scan_math", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("seed", range(3))
def test_compact_in_range_wiring_matches_numpy_path(kern, seed):
    """Reproduce the exact array glue that ImagingSonarSensor._scan_gpu_compact
    performs around the kernel -- 2D (H,W) depth reshape, 4-channel normals
    sliced to [:, :3], reusable output buffers re-used across two frames, and the
    [:n_valid] prefix views -- and prove the prefix views equal the numpy
    reference scan path (sonar_scan_math.select_in_range_points). This covers the
    integration, not just the bare kernel."""
    scan_math = _load_scan_math()
    H, W = 48, 64
    n_px = H * W
    rng = np.random.default_rng(100 + seed)
    mn, mx = 0.2, 3.0

    # Reusable device buffers, allocated once (as the sensor does), used twice.
    out_p = wp.zeros((n_px, 3), dtype=wp.float32, device=DEV)
    out_n = wp.zeros((n_px, 3), dtype=wp.float32, device=DEV)
    out_s = wp.zeros(n_px, dtype=wp.uint32, device=DEV)
    counter = wp.zeros(1, dtype=wp.int32, device=DEV)

    for _ in range(2):  # second pass exercises buffer reuse / counter re-zero
        depth_img = rng.uniform(-1.0, 6.0, (H, W)).astype(np.float32)
        depth_img[rng.random((H, W)) < 0.05] = np.inf
        pcl = rng.uniform(-5, 5, (n_px, 3)).astype(np.float32)
        pcl[rng.random(n_px) < 0.05] = np.nan
        normals_img = rng.uniform(-1, 1, (H, W, 4)).astype(np.float32)  # 4-channel AOV
        sem_img = rng.integers(0, 100, (H, W)).astype(np.uint32)

        # numpy reference path (what _scan_numpy feeds to sonar_scan_math).
        normals_flat = normals_img.reshape(-1, 4)[:, :3]
        sem_flat = sem_img.reshape(-1).astype(np.uint32)
        ref_p, ref_n, ref_s = scan_math.select_in_range_points(
            depth_img.reshape(-1), pcl, normals_flat, sem_flat, mn, mx)

        # GPU wiring path: same reshape/slice as _scan_gpu_compact.
        d = wp.array(depth_img, dtype=wp.float32, device=DEV).reshape((-1,))
        p = wp.array(pcl, dtype=wp.float32, device=DEV).reshape((-1, 3))
        nm = wp.array(normals_img, dtype=wp.float32, device=DEV).reshape((-1, 4))[:, :3]
        s = wp.array(sem_img, dtype=wp.uint32, device=DEV).reshape((-1,))

        counter.zero_()
        wp.launch(kern.compact_in_range, dim=n_px,
                  inputs=[d, p, nm, s, wp.float32(mn), wp.float32(mx),
                          counter, out_p, out_n, out_s], device=DEV)
        wp.synchronize()

        m = int(counter.numpy()[0])
        assert m == ref_p.shape[0]
        got_p = out_p[:m].numpy()        # prefix views, as stored in scan_data
        got_n = out_n[:m].numpy()
        got_s = out_s[:m].numpy()

        def _key(pc, se):
            return np.lexsort((pc[:, 2], pc[:, 1], pc[:, 0], se))
        go, ro = _key(got_p, got_s), _key(ref_p, ref_s)
        assert np.array_equal(got_s[go], ref_s[ro])
        assert np.allclose(got_p[go], ref_p[ro])
        assert np.allclose(got_n[go], ref_n[ro])


# --- make_sonar_data work-buffer reuse -------------------------------------
# ImagingSonarSensor.make_sonar_data reuses grow-on-demand work buffers
# (intensity / pcl_local / pcl_spher) sliced to [:num_points] instead of
# allocating fresh arrays each frame, and re-zeroes a kept normalization-max
# buffer. The risk is stale data leaking when a later frame has fewer points
# than the buffer's high-water capacity. These run the real pipeline kernels and
# assert the reused-buffer path is bit-identical to fresh allocation across a
# grow-then-shrink-then-grow frame sequence.

def _pipeline_bins(kern, pcl, normals, sem, refl, vt, n_range, n_beams,
                   x_off, y_off, x_res, y_res, bufs):
    """Run compute_intensity -> world2local -> bin_intensity for one frame and
    return (bin_sum, bin_count) as numpy. ``bufs`` supplies the intensity /
    pcl_local / pcl_spher arrays (fresh or reused-and-sliced)."""
    n = pcl.shape[0]
    intensity, pcl_local, pcl_spher = bufs(n)
    p = wp.array(pcl, dtype=wp.float32, device=DEV)
    nm = wp.array(normals, dtype=wp.float32, device=DEV)
    s = wp.array(sem, dtype=wp.uint32, device=DEV)
    ir = wp.array(refl, dtype=wp.float32, device=DEV)
    wp.launch(kern.compute_intensity, dim=n,
              inputs=[p, nm, wp.mat44(vt), s, ir, wp.float32(0.1)],
              outputs=[intensity], device=DEV)
    wp.launch(kern.world2local, dim=n,
              inputs=[wp.mat44(vt), p], outputs=[pcl_local, pcl_spher], device=DEV)
    bin_sum = wp.zeros((n_range, n_beams), dtype=wp.float32, device=DEV)
    bin_count = wp.zeros((n_range, n_beams), dtype=wp.int32, device=DEV)
    wp.launch(kern.bin_intensity, dim=n,
              inputs=[pcl_spher, intensity, wp.float32(x_off), wp.float32(y_off),
                      wp.float32(x_res), wp.float32(y_res), bin_sum, bin_count],
              device=DEV)
    wp.synchronize()
    return bin_sum.numpy(), bin_count.numpy()


def _frame_inputs(seed, n):
    rng = np.random.default_rng(seed)
    # Points in front of the sensor so they fall within the binning grid.
    pcl = np.stack([rng.uniform(-0.6, 0.6, n),
                    rng.uniform(-0.6, 0.6, n),
                    rng.uniform(0.3, 1.2, n)], axis=1).astype(np.float32)
    normals = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    sem = rng.integers(0, 4, n).astype(np.uint32)
    return pcl, normals, sem


def test_make_sonar_data_buffer_reuse_matches_fresh(kern):
    n_range, n_beams = 32, 48
    min_range, min_azi = 0.2, np.deg2rad(90 - 130 / 2)
    range_res, azi_res = 0.03, np.deg2rad(130 / n_beams)
    refl = np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32)  # indexToProp
    # identity rotation, no translation -> sensor at origin (a valid extrinsic).
    vt = np.eye(4, dtype=np.float32).reshape(-1)

    # grow-on-demand reusable buffers, mirroring _ensure_point_buffers / the
    # [:num_points] views; capacity only ever grows.
    state = {"cap": 0, "i": None, "l": None, "s": None}

    def reused(n):
        if state["cap"] < n:
            state["cap"] = n
            state["i"] = wp.empty(n, dtype=wp.float32, device=DEV)
            state["l"] = wp.empty(n, dtype=wp.vec3, device=DEV)
            state["s"] = wp.empty(n, dtype=wp.vec3, device=DEV)
        return state["i"][:n], state["l"][:n], state["s"][:n]

    def fresh(n):
        return (wp.empty(n, dtype=wp.float32, device=DEV),
                wp.empty(n, dtype=wp.vec3, device=DEV),
                wp.empty(n, dtype=wp.vec3, device=DEV))

    # grow (4000) -> shrink (700, exercises stale residue) -> grow again (5000).
    for seed, n in [(1, 4000), (2, 700), (3, 5000), (4, 700)]:
        pcl, normals, sem = _frame_inputs(seed, n)
        args = (kern, pcl, normals, sem, refl, vt, n_range, n_beams,
                min_range, min_azi, range_res, azi_res)
        s_fresh, c_fresh = _pipeline_bins(*args, bufs=fresh)
        s_reuse, c_reuse = _pipeline_bins(*args, bufs=reused)
        assert np.array_equal(c_fresh, c_reuse), f"bin_count differs at n={n}"
        # bit-identical: same kernels, same inputs, only allocation differs.
        assert np.array_equal(s_fresh, s_reuse), f"bin_sum differs at n={n}"


# --- kernel hardening: NaN guards + uint8 clamp ----------------------------

def _rotz(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_compute_intensity_matches_numpy_reference(kern):
    """The dist-reuse refactor of compute_intensity must equal the closed-form
    intensity = reflectivity * cos_theta * exp(-att*dist) on non-degenerate
    inputs (sensor_loc = -(R^T T))."""
    n = 3000
    rng = np.random.default_rng(7)
    R = _rotz(30.0)
    T = np.array([1.0, -2.0, 0.5])
    vt = np.eye(4, dtype=np.float64)
    vt[:3, :3] = R
    vt[:3, 3] = T
    sensor_loc = -(R.T @ T)

    pcl = rng.uniform(-5, 5, (n, 3)).astype(np.float32)
    normals = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    K = 6
    sem = rng.integers(0, K, n).astype(np.uint32)
    refl = rng.uniform(0.1, 2.0, K).astype(np.float32)
    att = 0.1

    incidence = pcl.astype(np.float64) - sensor_loc
    dist = np.linalg.norm(incidence, axis=1)
    unit = incidence / dist[:, None]
    cos_theta = np.sum(-unit * normals.astype(np.float64), axis=1)
    ref = refl[sem].astype(np.float64) * cos_theta * np.exp(-att * dist)

    out = wp.zeros(n, dtype=wp.float32, device=DEV)
    wp.launch(kern.compute_intensity, dim=n,
              inputs=[wp.array(pcl, dtype=wp.float32, device=DEV),
                      wp.array(normals, dtype=wp.float32, device=DEV),
                      wp.mat44(vt.reshape(-1)),
                      wp.array(sem, dtype=wp.uint32, device=DEV),
                      wp.array(refl, dtype=wp.float32, device=DEV),
                      wp.float32(att)],
              outputs=[out], device=DEV)
    wp.synchronize()
    assert np.allclose(out.numpy(), ref, rtol=1e-3, atol=1e-4)


def test_compute_intensity_coincident_point_is_finite(kern):
    """A point exactly at the sensor location (dist == 0) must not divide by
    zero -> finite 0 intensity, not NaN (which would poison the bin sum)."""
    R = _rotz(15.0)
    T = np.array([0.3, 0.4, -0.5])
    vt = np.eye(4, dtype=np.float64)
    vt[:3, :3] = R
    vt[:3, 3] = T
    sensor_loc = -(R.T @ T)

    pcl = np.array([sensor_loc, [1.0, 1.0, 1.0]], dtype=np.float32)  # row 0 coincident
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    sem = np.array([1, 1], dtype=np.uint32)
    refl = np.array([0.0, 1.0], dtype=np.float32)

    out = wp.zeros(2, dtype=wp.float32, device=DEV)
    wp.launch(kern.compute_intensity, dim=2,
              inputs=[wp.array(pcl, dtype=wp.float32, device=DEV),
                      wp.array(normals, dtype=wp.float32, device=DEV),
                      wp.mat44(vt.reshape(-1)),
                      wp.array(sem, dtype=wp.uint32, device=DEV),
                      wp.array(refl, dtype=wp.float32, device=DEV),
                      wp.float32(0.1)],
              outputs=[out], device=DEV)
    wp.synchronize()
    o = out.numpy()
    assert np.all(np.isfinite(o))
    assert o[0] == 0.0   # coincident point -> zero direction -> zero intensity


def test_cartesian_to_spherical_origin_is_finite(kern):
    """world2local feeds cartesian_to_spherical; a point mapping to the local
    origin (r == 0) must yield a finite elevation, not acos(nan)."""
    pcl = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)  # row 0 -> origin
    vt = np.eye(4, dtype=np.float64).reshape(-1)
    local = wp.zeros(2, dtype=wp.vec3, device=DEV)
    spher = wp.zeros(2, dtype=wp.vec3, device=DEV)
    wp.launch(kern.world2local, dim=2,
              inputs=[wp.mat44(vt), wp.array(pcl, dtype=wp.float32, device=DEV)],
              outputs=[local, spher], device=DEV)
    wp.synchronize()
    assert np.all(np.isfinite(spher.numpy())), "origin produced NaN/inf in spherical coords"


def test_make_sonar_image_clamps_overflow(kern):
    """An intensity > 1 must saturate to 255, not wrap modulo 256."""
    n_range, width = 1, 3
    grid = np.zeros((n_range, width, 3), dtype=np.float32)
    grid[0, :, 2] = [0.5, 1.0, 1.5]            # last column overflows pre-clamp
    sonar_data = wp.array(grid, dtype=wp.vec3, device=DEV)
    sonar_image = wp.zeros((n_range, width, 4), dtype=wp.uint8, device=DEV)
    wp.launch(kern.make_sonar_image, dim=(n_range, width),
              inputs=[sonar_data, sonar_image], device=DEV)
    wp.synchronize()
    img = sonar_image.numpy()
    # column j -> width-1-j; intensity 1.5 is column 2 -> output column 0.
    assert img[0, 0, 0] == 255          # saturated, NOT 382 % 256 == 126
    assert img[0, 1, 0] == 255          # intensity 1.0 -> 255
    assert img[0, 2, 0] == 127          # intensity 0.5 -> 127 (unchanged)

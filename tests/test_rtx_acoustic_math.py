"""Tests for the RTX acoustic GMO -> grid folding (pure numpy).

The acoustic GMO is signal-way A-scans: ``amp`` is ``numSgws * num_samples_per_sgw``
amplitude samples laid out as numSgws contiguous A-scan blocks. Sample ``k`` is at
range ``range_offset + k * meters_per_sample``; signal way ``s`` at beam position
``s / (numSgws-1) * (n_beams-1)``. Both axes are resampled as the bin mean of the
linear interpolant of ``|amp|`` (zero where a bin has no samples), then
peak-normalised.
"""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim",
                     "utils", "rtx_acoustic_math.py")

# Clean constants: min_range 0, mps == range_res == 0.01 and range_offset half a
# bin, so sample k sits on range bin k's centre.
NSPG = 50
MPS, R0 = 0.01, 0.005
MN, RES = 0.0, 0.01
NR, NB = 100, 8

# Runner defaults (oceansim_ros2 rtx_acoustic): air-medium sample pitch vs 5 mm
# bins, 63 signal ways (64 elements) over 130 / 0.25 = 520 beams.
PROD = dict(mps=343 * 1.024e-4 / 2, r0=343 * 2.5e-3 / 2, mn=0.1, res=0.005,
            nr=1220, nb=520, nspg=320, n_sgw=63)


def _load():
    spec = importlib.util.spec_from_file_location("rtx_acoustic_math", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load()


def _fold(m, amp, nspg=NSPG, mps=MPS, r0=R0, nb=NB):
    return m.fold_gmo_to_grid(np.asarray(amp, float), nspg, mps, r0, MN, RES, NR, nb)


def _ascan(peak_k, val=1.0, nspg=NSPG):
    a = np.zeros(nspg)
    a[peak_k] = val
    return a


def _reference_bin_mean(x, y, edges, sub=400):
    """Brute force: average np.interp over `sub` points per bin, counting only
    points inside the knot span (independent of the cumulative-integral code)."""
    out = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        pts = edges[i] + (np.arange(sub) + 0.5) / sub * (edges[i + 1] - edges[i])
        inside = (pts >= x[0]) & (pts <= x[-1])
        if inside.any():
            out[i] = np.interp(pts[inside], x, y).mean()
    return out


# --- degenerate input -------------------------------------------------------

def test_empty(m):
    g = _fold(m, [])
    assert g.shape == (NR, NB) and g.dtype == np.float32 and not g.any()


def test_zero_nspg_returns_empty(m):
    g = _fold(m, np.ones(NSPG), nspg=0)
    assert not g.any()


def test_degenerate_grid_returns_empty_not_crash(m):
    # n_beams == 0 used to raise ValueError from grid.max() on a zero-size array.
    g = m.fold_gmo_to_grid(np.ones(NSPG), NSPG, MPS, R0, MN, RES, NR, 0)
    assert g.shape == (NR, 0) and not g.size
    g = m.fold_gmo_to_grid(np.ones(NSPG), NSPG, MPS, R0, MN, RES, 0, NB)
    assert g.shape == (0, NB) and not g.size


def test_single_beam_averages_signal_ways(m):
    amp = np.concatenate([_ascan(10, 1.0), _ascan(10, 3.0)])
    g = _fold(m, amp, nb=1)
    assert g.shape == (NR, 1) and int(np.argmax(g[:, 0])) == 10


def test_single_sample_ascan_is_point_binned(m):
    g = m.fold_gmo_to_grid(np.array([2.0, 5.0]), 1, MPS, 0.123, MN, RES, NR, NB)
    assert np.nonzero(g.max(axis=1))[0].tolist() == [12]
    # beams 0 / NB-1 average the (2 -> 5) interpolant over their inner half beam
    edge = 3.0 * 0.25 / (NB - 1)
    assert g[12, 0] == pytest.approx((2.0 + edge) / (5.0 - edge))
    assert g[12, NB - 1] == pytest.approx(1.0)


# --- range axis ---------------------------------------------------------------

def test_single_sgw_peak_maps_to_sample_bin(m):
    # one signal way, peak on bin 30's centre -> bin 30 is the maximum and the
    # interpolation blur is symmetric around it; sole sgw -> beam 0 only.
    g = _fold(m, _ascan(30, 2.0))
    col = g[:, 0]
    assert int(np.argmax(col)) == 30 and col[30] == pytest.approx(1.0)
    assert col[29] == pytest.approx(col[31]) and col[29] < 0.5
    assert g[:, 1:].sum() == 0.0
    assert np.nonzero(col)[0].tolist() == [29, 30, 31]


def test_range_matches_bruteforce_linear_interpolant(m):
    rng = np.random.default_rng(0)
    nspg, mps, r0 = 40, 0.0237, 0.051
    a = rng.uniform(-3, 3, nspg)
    g = m.fold_gmo_to_grid(a, nspg, mps, r0, MN, RES, NR, 1)
    ref = _reference_bin_mean(r0 + np.arange(nspg) * mps, np.abs(a),
                              MN + np.arange(NR + 1) * RES)
    assert np.allclose(g[:, 0], ref / ref.max(), atol=2e-3)


def test_sparse_samples_leave_no_black_rows(m):
    """Production pitch: 17.6 mm samples into 5 mm bins. The old point scatter
    lit only ~28% of the rows inside the sample span; every one must be lit,
    and a constant A-scan must come out flat."""
    p = PROD
    g = m.fold_gmo_to_grid(np.ones(p["nspg"]), p["nspg"], p["mps"], p["r0"],
                           p["mn"], p["res"], p["nr"], 1)
    first = int(np.floor((p["r0"] - p["mn"]) / p["res"]))
    last = int(np.floor((p["r0"] + (p["nspg"] - 1) * p["mps"] - p["mn"]) / p["res"]))
    lit = np.nonzero(g[:, 0])[0]
    assert lit.min() == first and lit.max() == last and lit.size == last - first + 1
    assert np.allclose(g[first:last + 1, 0], 1.0, atol=1e-5)


def test_dense_samples_average_without_banding(m):
    """Samples finer than a bin (2.85 per bin here): the old sum alternated 2 / 3
    samples per bin, a +-20% banding on a flat return; the mean is flat."""
    nspg, mps = 300, RES / 2.85
    g = m.fold_gmo_to_grid(np.ones(nspg), nspg, mps, 0.0, MN, RES, NR, 1)
    full = g[1:int((nspg - 1) * mps / RES) - 1, 0]
    assert full.size > 50 and np.allclose(full, 1.0, atol=1e-5)


def test_range_offset_shifts_bins(m):
    # sample 0 at 0.505 -> bin 50 holds the peak, nothing below it.
    g = _fold(m, _ascan(0, 1.0), r0=0.505)
    assert int(np.argmax(g[:, 0])) == 50 and g[:50].sum() == 0.0


def test_out_of_range_samples_dropped(m):
    # nspg 150 > window: a peak well past the last bin contributes nothing.
    a = np.zeros(150)
    a[120] = 9.0          # range 1.205 m, window ends at 1.0 m
    a[40] = 3.0           # in range
    g = _fold(m, a, nspg=150)
    assert int(np.argmax(g[:, 0])) == 40 and g[40, 0] == pytest.approx(1.0)
    assert g[90:].sum() == 0.0


def test_boundary_return_splits_evenly(m):
    # A sample exactly on a bin boundary (0.20 m between bins 19 and 20) is
    # shared equally: no systematic half-bin bias either way.
    g = _fold(m, _ascan(20, 1.0), r0=0.0)
    assert g[19, 0] == pytest.approx(g[20, 0]) and g[19, 0] == pytest.approx(1.0)


def test_range_centroid_is_unbiased(m):
    """Isolated returns at arbitrary sub-bin ranges, production pitch: the
    intensity centroid over the published bin centres (i + 0.5) * res lands on
    the true range (floor/round point binning was off by up to res/2)."""
    p = PROD
    rng = np.random.default_rng(1)
    for k in rng.integers(20, p["nspg"] - 20, 12):
        g = m.fold_gmo_to_grid(_ascan(k, 1.0, p["nspg"]), p["nspg"], p["mps"], p["r0"],
                               p["mn"], p["res"], p["nr"], 1)[:, 0]
        centres = p["mn"] + (np.arange(p["nr"]) + 0.5) * p["res"]
        centroid = (g * centres).sum() / g.sum()
        assert centroid == pytest.approx(p["r0"] + k * p["mps"], abs=p["res"] * 0.05)


def test_relative_normalisation(m):
    # two isolated samples |amp| 2 and 4 -> 0.5 and 1.0 after peak-norm.
    a = np.zeros(NSPG)
    a[10], a[20] = 2.0, -4.0
    g = _fold(m, a)
    assert g[10, 0] == pytest.approx(0.5)
    assert g[20, 0] == pytest.approx(1.0)


def test_abs_amplitude(m):
    g = _fold(m, _ascan(30, -3.0))
    assert int(np.argmax(g[:, 0])) == 30 and g[30, 0] == pytest.approx(1.0)


# --- azimuth axis ------------------------------------------------------------

def test_two_sgw_map_to_extreme_beams(m):
    # 2 signal ways sit on beams 0 and NB-1 and are interpolated between.
    amp = np.concatenate([_ascan(10, 1.0), _ascan(20, 1.0)])
    g = _fold(m, amp)
    assert g[10, 0] == pytest.approx(1.0) and g[20, NB - 1] == pytest.approx(1.0)
    assert g[10, NB - 1] < 0.05 and g[20, 0] < 0.05
    assert np.all(np.diff(g[10]) < 0) and np.all(np.diff(g[20]) > 0)


def test_few_sgw_fill_every_beam(m):
    """Production: 63 signal ways over 520 beams left 457 beam columns black."""
    p = PROD
    amp = np.tile(np.ones(p["nspg"]), p["n_sgw"])
    g = m.fold_gmo_to_grid(amp, p["nspg"], p["mps"], p["r0"], p["mn"], p["res"],
                           p["nr"], p["nb"])
    assert np.all(g.max(axis=0) > 0)
    row = g[int((2.0 - p["mn"]) / p["res"])]
    assert np.allclose(row, 1.0, atol=1e-5)


def test_azimuth_matches_bruteforce_linear_interpolant(m):
    rng = np.random.default_rng(2)
    n_sgw, nb = 7, 30
    vals = rng.uniform(0.1, 2.0, n_sgw)
    amp = np.concatenate([_ascan(25, v) for v in vals])
    g = _fold(m, amp, nb=nb)
    knots = np.arange(n_sgw) * (nb - 1) / (n_sgw - 1)
    ref = _reference_bin_mean(knots, vals, np.arange(nb + 1) - 0.5)
    assert np.allclose(g[25] / g[25].max(), ref / ref.max(), atol=2e-3)


def test_many_sgw_share_beams_average(m):
    # More signal ways than beams: each beam averages the ways it covers, so
    # identical A-scans give equal beams (the old sum banded by sgw count).
    n_sgw, nspg, nb = 16, 10, 4
    amp = np.tile(_ascan(3, 1.0, nspg=nspg), n_sgw)
    g = m.fold_gmo_to_grid(amp, nspg, MPS, R0, MN, RES, NR, nb)
    assert np.allclose(g[3, :], 1.0)
    assert g.sum() == pytest.approx(g[2:5].sum())


def test_weights_cached_and_read_only(m):
    _fold(m, _ascan(5, 1.0))
    w = m._linear_bin_weights(R0, MPS, NSPG, MN, RES, NR)
    assert m._linear_bin_weights(R0, MPS, NSPG, MN, RES, NR) is w
    with pytest.raises(ValueError):
        w[0, 0] = 1.0

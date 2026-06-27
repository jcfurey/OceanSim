"""Unit tests for the MultivariateNormal / MultivariateUniform noise samplers.

Pure numpy -- no Isaac Sim, no Warp. Run with:

    pip install numpy pytest && pytest tests/

Modules are loaded by file path to avoid the isaacsim.oceansim namespace.
"""

import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")

_UTILS = os.path.join(os.path.dirname(__file__), "..", "isaacsim", "oceansim", "utils")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_UTILS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def MVN():
    return _load("MultivariateNormal").MultivariateNormal


@pytest.fixture(scope="module")
def MVU():
    return _load("MultivariateUniform").MultivariateUniform


def test_cholesky_reconstructs_covariance(MVN):
    """init_cov with a full SPD matrix must yield L with L @ L.T == cov."""
    cov = np.array([[4.0, 2.0, 0.6],
                    [2.0, 3.0, 0.5],
                    [0.6, 0.5, 2.0]])
    m = MVN(3)
    m.init_cov(cov)
    L = m.get_sqrt_cov()
    assert np.allclose(L, np.tril(L)), "sqrt_cov must be lower-triangular"
    assert np.allclose(L @ L.T, cov, atol=1e-9)


def test_cholesky_rejects_non_pd(MVN):
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])  # indefinite
    m = MVN(2)
    assert m.cholesky(cov.copy()) is False


def test_init_sigma_and_cov_treat_sigma_as_stddev(MVN):
    m = MVN(2)
    m.init_sigma(0.5)                       # sigma = std dev
    assert np.allclose(np.diag(m.get_sqrt_cov()), [0.5, 0.5])
    m2 = MVN(2)
    m2.init_cov(0.25)                       # variance -> std = 0.5
    assert np.allclose(np.diag(m2.get_sqrt_cov()), [0.5, 0.5])


def test_sample_covariance_matches(MVN):
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    m = MVN(2)
    m.init_cov(cov)
    m.gen = np.random.default_rng(0)        # determinism
    samples = np.array([m.sample_array() for _ in range(200000)])
    emp = np.cov(samples, rowvar=False)
    assert np.allclose(emp, cov, atol=0.05)


def test_not_uncertain_returns_zeros(MVN):
    m = MVN(3)                              # never initialized
    assert not m.is_uncertain()
    assert np.allclose(m.sample_array(), np.zeros(3))


def test_uniform_bounds_and_exponential_mean(MVU):
    u = MVU(1)
    u.init_bounds(2.0)
    u.rng = np.random.default_rng(0)
    s = np.array([u.sample_float() for _ in range(50000)])
    assert s.min() >= 0.0 and s.max() <= 2.0
    # exponential with -max*log(U) has mean == max
    u.rng = np.random.default_rng(1)
    e = np.array([u.sample_exponential() for _ in range(200000)])
    assert e.mean() == pytest.approx(2.0, rel=0.05)

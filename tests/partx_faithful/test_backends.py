import numpy as np
import pytest

from adaptive_roa.partx_faithful.backends import GPCBackend, SklearnGPRBackend, backend_factory


def _labels(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y_pm = np.where(X[:, 0] < 0, 1.0, -1.0)          # pool convention: +1 success
    return X, y_pm


def test_falsify_success_negates_the_mean_only():
    X, y = _labels()
    a = GPCBackend(falsify="failure", seed=0).fit(X, y)
    b = GPCBackend(falsify="success", seed=0).fit(X, y)
    ma, sa = a.latent(X)
    mb, sb = b.latent(X)
    np.testing.assert_allclose(mb, -ma)
    np.testing.assert_allclose(sb, sa)
    np.testing.assert_allclose(a.p_success(X), b.p_success(X))


def test_gpc_backend_reads_pool_labels():
    X, y = _labels()
    m = GPCBackend(seed=0).fit(X, y)
    p = m.p_success(np.array([[-1.5, 0.0], [1.5, 0.0]]))
    assert p[0] > 0.5 > p[1]
    g, _ = m.latent(np.array([[-1.5, 0.0], [1.5, 0.0]]))
    assert g[0] > 0 > g[1]                             # g > 0 = satisfying under falsify=failure


def test_gpr_backend_interpolates_robustness():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(30, 2))
    y = X[:, 0] ** 2 - X[:, 1]
    mean, sd = SklearnGPRBackend().fit(X, y).latent(X)
    np.testing.assert_allclose(mean, y, atol=1e-3)
    assert np.all(sd < 1e-2)


def test_factory():
    make = backend_factory("gpc", falsify="failure", n_restarts=1)
    assert isinstance(make(3), GPCBackend)
    assert isinstance(backend_factory("gpr")(3), SklearnGPRBackend)
    with pytest.raises(ValueError):
        backend_factory("nope")
    with pytest.raises(ValueError):
        GPCBackend(falsify="maybe")

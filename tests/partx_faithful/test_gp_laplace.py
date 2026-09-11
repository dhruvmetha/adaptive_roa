import numpy as np
import pytest
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from adaptive_roa.partx_faithful.gp_laplace import LaplaceGPClassifier


def _data(n=40, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_logit_latent_matches_sklearn_with_fixed_kernel():
    X, y = _data()
    Xt = np.random.default_rng(2).normal(size=(10, 2))
    ref = GaussianProcessClassifier(
        kernel=ConstantKernel(1.5, "fixed") * Matern(0.8, "fixed", nu=2.5), optimizer=None
    ).fit(X, y)
    ours = LaplaceGPClassifier(link="logit", mean="zero", standardize=False, fixed=(1.5, 0.8)).fit(X, y)
    m_ref, v_ref = ref.latent_mean_and_variance(Xt)
    m, s = ours.latent(Xt)
    np.testing.assert_allclose(m, m_ref, atol=1e-6)
    np.testing.assert_allclose(s**2, v_ref, atol=1e-6)
    assert ours.lml_ == pytest.approx(ref.log_marginal_likelihood_value_, abs=1e-6)


def test_logit_lml_gradient_matches_sklearn():
    X, y = _data()
    ref = GaussianProcessClassifier(kernel=ConstantKernel(1.5) * Matern(0.8, nu=2.5), optimizer=None).fit(X, y)
    theta = ref.kernel_.theta                       # [log 1.5, log 0.8]
    lml_ref, g_ref = ref.log_marginal_likelihood(theta, eval_gradient=True)
    ours = LaplaceGPClassifier(link="logit", mean="zero", standardize=False)
    lml, g = ours.log_marginal_likelihood(theta, X, np.where(y > 0, 1.0, -1.0), 0.0, eval_gradient=True)
    assert lml == pytest.approx(lml_ref, abs=1e-6)
    np.testing.assert_allclose(g, g_ref, rtol=1e-6, atol=1e-8)


def test_probit_gradient_matches_finite_differences():
    X, y = _data()
    gp = LaplaceGPClassifier(link="probit", standardize=False)
    y_pm = np.where(y > 0, 1.0, -1.0)
    mu = float(norm.ppf((y.sum() + 1.0) / (len(y) + 2.0)))
    theta = np.log([2.0, 0.5])
    _, g = gp.log_marginal_likelihood(theta, X, y_pm, mu, eval_gradient=True)
    eps, fd = 1e-5, []
    for j in range(2):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += eps
        tm[j] -= eps
        fd.append((gp.log_marginal_likelihood(tp, X, y_pm, mu)
                   - gp.log_marginal_likelihood(tm, X, y_pm, mu)) / (2 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-4)


def test_single_class_region_fits_with_smoothed_rate_mean():
    X = np.random.default_rng(3).uniform(-1, 1, size=(30, 2))
    gp = LaplaceGPClassifier(seed=0).fit(X, np.zeros(30, dtype=int))
    assert gp.mu_ == pytest.approx(norm.ppf(1.0 / 32.0))
    assert np.all(gp.predict_proba(X) < 0.5)


def test_cap_fits_on_a_seeded_subsample():
    X, y = _data(n=50)
    a = LaplaceGPClassifier(max_points=20, seed=4).fit(X, y)
    b = LaplaceGPClassifier(max_points=20, seed=4).fit(X, y)
    assert a.capped_ and a.n_fit_ == 20
    np.testing.assert_array_equal(a.Z_, b.Z_)


def test_probabilities_follow_the_labels():
    rng = np.random.default_rng(5)
    X = rng.uniform(-2, 2, size=(80, 2))
    y = (X[:, 0] < 0).astype(int)
    gp = LaplaceGPClassifier(seed=0).fit(X, y)
    p = gp.predict_proba(np.array([[-1.5, 0.0], [1.5, 0.0]]))
    assert 0.0 < p[1] < 0.5 < p[0] < 1.0


def test_seeded_fits_are_reproducible():
    X, y = _data()
    a = LaplaceGPClassifier(seed=7).fit(X, y)
    b = LaplaceGPClassifier(seed=7).fit(X, y)
    assert a.lengthscale_ == b.lengthscale_ and a.sigma_f2_ == b.sigma_f2_
    np.testing.assert_array_equal(a.latent(X)[0], b.latent(X)[0])


def test_latent_of_empty_input_is_empty():
    X, y = _data()
    m, s = LaplaceGPClassifier(seed=0).fit(X, y).latent(np.empty((0, 2)))
    assert m.shape == (0,) and s.shape == (0,)

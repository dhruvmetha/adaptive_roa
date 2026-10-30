"""Within-iteration hyperparameter reuse (partx_faithful_traj) and the mean-only plug-in f*."""
import numpy as np

from adaptive_roa.partx_faithful.algorithm import PartX, PartXConfig
from adaptive_roa.partx_faithful.backends import GPCBackend
from adaptive_roa.partx_faithful.gp_laplace import LaplaceGPClassifier
from adaptive_roa.partx_faithful.sampling import PoolSampler
from adaptive_roa.partx_faithful.state import load_state, save_state


class _Recorder:
    """A model factory that records the `fixed` argument of every model it builds."""

    THETA = (1.5, 0.25)

    def __init__(self):
        self.calls = []

    def __call__(self, seed, fixed=None):
        self.calls.append(fixed)
        return _ThetaStub(fixed)


class _ThetaStub:
    """g(x) = x0 - 0.5, sd 0.1/sqrt(n), with hyperparameters to hand back."""

    def __init__(self, fixed):
        self.fixed = fixed

    def fit(self, X, y):
        self.n = max(len(X), 1)
        return self

    def latent(self, X):
        X = np.atleast_2d(X)
        return X[:, 0] - 0.5, np.full(len(X), 0.1 / np.sqrt(self.n))

    def hyperparameters(self):
        return _Recorder.THETA if self.fixed is None else self.fixed


def _px(budget, factory, reuse, n=4000, seed=0, initial=40):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, 1, size=(n, 2))
    labels = np.where(starts[:, 0] < 0.5, 1.0, -1.0)
    used = np.zeros(n, dtype=bool)
    used[:initial] = True
    ps = PoolSampler(starts, labels, used, scale=np.ones(2))
    cfg = PartXConfig(n0=4, n_bo=3, n_c=6, R=3, M=40, seed=seed)
    px = PartX(cfg, np.zeros(2), np.ones(2), ps, factory, budget, init_X=starts[:initial],
               init_y=labels[:initial], init_pool_idx=np.arange(initial), reuse_hyperparameters=reuse)
    return px, ps


def test_reuse_is_off_by_default():
    rec = _Recorder()
    px, _ = _px(budget=150, factory=rec, reuse=False)
    px.run()
    assert rec.calls and all(f is None for f in rec.calls)
    assert px.n_fixed_fits == 0


def test_later_fits_in_the_same_iteration_reuse_the_first_fits_hyperparameters():
    rec = _Recorder()
    px, _ = _px(budget=150, factory=rec, reuse=True)
    px.run_root()                       # 3 EI-step fits, then the classification fit
    assert rec.calls == [None, _Recorder.THETA, _Recorder.THETA, _Recorder.THETA]
    assert px.n_fixed_fits == 3


def test_each_region_re_optimizes_once_per_iteration():
    rec = _Recorder()
    px, _ = _px(budget=150, factory=rec, reuse=True)
    px.run_root()
    rec.calls.clear()
    px.step()
    optimized_this_iteration = sum(1 for v in px._theta.values() if v[0] == px.iteration)
    assert optimized_this_iteration >= 2                 # both children
    assert rec.calls.count(None) == optimized_this_iteration


def test_a_pickled_run_with_reuse_resumes_exactly(tmp_path):
    full, _ = _px(budget=150, factory=_Recorder(), reuse=True)
    full.run()
    part, _ = _px(budget=150, factory=_Recorder(), reuse=True)
    part.run_root()
    part.step()
    save_state(part, tmp_path / "s.pkl", "h")
    resumed = load_state(tmp_path / "s.pkl", "h")
    _, fresh = _px(budget=150, factory=_Recorder(), reuse=True)
    resumed.attach(fresh, _Recorder())
    resumed.run()
    assert resumed.log == full.log
    assert resumed.n_fixed_fits == full.n_fixed_fits
    assert resumed._theta == full._theta


def _small_gp():
    rng = np.random.default_rng(3)
    X = rng.random((200, 3))
    y = (X @ np.array([1.0, -2.0, 0.5]) > 0.0).astype(int)
    gp = LaplaceGPClassifier(seed=0, n_restarts=0, max_points=500).fit(X, y)
    return gp, rng.random((1000, 3))


def test_latent_mean_is_identical_to_the_latent_mean():
    gp, Q = _small_gp()
    np.testing.assert_array_equal(gp.latent_mean(Q, chunk=64), gp.latent(Q, chunk=64)[0])
    np.testing.assert_array_equal(gp.latent_mean(Q), gp.latent(Q)[0])


def test_backend_latent_mean_keeps_the_falsify_sign():
    rng = np.random.default_rng(4)
    X = rng.random((150, 2))
    y = np.where(X[:, 0] > 0.5, 1.0, -1.0)
    for falsify in ("failure", "success"):
        b = GPCBackend(falsify=falsify, seed=0, n_restarts=0, max_points=500).fit(X, y)
        np.testing.assert_array_equal(b.latent_mean(X), b.latent(X)[0])


def test_a_fixed_fit_uses_exactly_the_given_hyperparameters():
    rng = np.random.default_rng(5)
    X = rng.random((150, 2))
    y = np.where(X[:, 0] > 0.5, 1.0, -1.0)
    opt = GPCBackend(seed=0, n_restarts=1, max_points=500).fit(X, y)
    fixed = GPCBackend(seed=0, fixed=opt.hyperparameters(), n_restarts=1, max_points=500).fit(X, y)
    assert fixed.hyperparameters() == opt.hyperparameters()
    assert not fixed.at_bound
    np.testing.assert_allclose(fixed.latent(X)[0], opt.latent(X)[0])

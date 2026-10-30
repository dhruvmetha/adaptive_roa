"""Local models Part-X fits per region (spec §5.5).

Every backend exposes latent(X) -> (m, s), the posterior of the
ROBUSTNESS-oriented function g that Part-X classifies and runs EI on. g < 0
means falsifying. Under falsify="failure" (D2) g is the success latent f, so
failures are the falsifiers; falsify="success" negates it.
"""
from __future__ import annotations

import warnings

import numpy as np

from adaptive_roa.partx_faithful.gp_laplace import LaplaceGPClassifier


class GPCBackend:
    def __init__(self, falsify: str = "failure", seed: int = 0, fixed=None, **gp_kwargs):
        if falsify not in ("failure", "success"):
            raise ValueError(f"falsify must be 'failure' or 'success', got {falsify!r}")
        self.sign = 1.0 if falsify == "failure" else -1.0
        # fixed = (sigma_f2, lengthscale) skips the hyperparameter optimizer (partx_faithful_traj's
        # within-iteration reuse); None optimizes as Part-X* does.
        self.gp = LaplaceGPClassifier(seed=seed, fixed=fixed, **gp_kwargs)

    def fit(self, X, y):
        """y uses the pool convention: +1 success, -1 failure."""
        self.gp.fit(X, (np.asarray(y) > 0).astype(int))
        return self

    def latent(self, X):
        m, s = self.gp.latent(X)
        return self.sign * m, s

    def latent_mean(self, X):
        return self.sign * self.gp.latent_mean(X)

    def hyperparameters(self):
        return (self.gp.sigma_f2_, self.gp.lengthscale_)

    def p_success(self, X):
        return self.gp.predict_proba(X)

    @property
    def capped(self) -> bool:
        return bool(self.gp.capped_)

    @property
    def at_bound(self) -> bool:
        return bool(self.gp.at_bound_)


class SklearnGPRBackend:
    """cpslab InternalGPR, cloned for the reference check only (spec §8.2).

    Matern(nu=2.5), alpha=1e-6, normalize_y, 5 restarts, random_state=12345, and
    a StandardScaler on the inputs. y is real-valued robustness.
    """

    capped = False
    at_bound = False

    def __init__(self, seed: int = 12345):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        from sklearn.preprocessing import StandardScaler

        self.model = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True,
                                              n_restarts_optimizer=5, random_state=seed)
        self.scale = StandardScaler()

    def fit(self, X, y):
        Xs = self.scale.fit_transform(np.asarray(X, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(Xs, np.asarray(y, dtype=float))
        return self

    def latent(self, X):
        Xs = self.scale.transform(np.atleast_2d(np.asarray(X, dtype=float)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = self.model.predict(Xs, return_std=True)
        return m, np.maximum(s, 1e-12)

    def p_success(self, X):
        raise NotImplementedError("the GPR reference backend models robustness, not p_success")


def backend_factory(kind: str = "gpc", falsify: str = "failure", **gp_kwargs):
    """(seed, fixed=None) -> fresh local model. cpslab pins every GPR to random_state=12345."""
    if kind == "gpc":
        return lambda seed, fixed=None: GPCBackend(falsify=falsify, seed=seed, fixed=fixed, **gp_kwargs)
    if kind == "gpr":
        return lambda seed, fixed=None: SklearnGPRBackend(seed=12345)
    raise ValueError(f"unknown backend kind {kind!r}")

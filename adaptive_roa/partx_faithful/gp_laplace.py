"""Exact GP classifier with the Laplace approximation (Rasmussen & Williams 2006, Alg. 3.1, 3.2, 5.1).

The runs use the probit likelihood p(y=1|f) = Phi(f). The logistic link exists
only so tests can compare against sklearn's GaussianProcessClassifier, which
refuses single-class data and has no probit link (spec §5.1).

Prior f ~ GP(mu, k) with k = sigma_f2 * Matern-5/2 (one isotropic lengthscale)
on inputs standardized within the region. mu is fixed from the data, the
smoothed success rate, which is the analog of cpslab's normalize_y (spec §5.2).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit, log_ndtr
from scipy.stats import norm

_SQRT5 = np.sqrt(5.0)


def matern52(XA: np.ndarray, XB: np.ndarray, sigma_f2: float, lengthscale: float) -> np.ndarray:
    u = _SQRT5 * cdist(XA, XB) / lengthscale
    return sigma_f2 * (1.0 + u + u * u / 3.0) * np.exp(-u)


def matern52_with_grads(X: np.ndarray, sigma_f2: float, lengthscale: float):
    """K(X, X) and its derivatives w.r.t. log(sigma_f2) and log(lengthscale)."""
    u = _SQRT5 * cdist(X, X) / lengthscale
    e = np.exp(-u)
    K = sigma_f2 * (1.0 + u + u * u / 3.0) * e
    dK_log_sigma_f2 = K
    dK_log_lengthscale = sigma_f2 * (u * u / 3.0) * (1.0 + u) * e
    return K, (dK_log_sigma_f2, dK_log_lengthscale)


def likelihood_terms(f: np.ndarray, y_pm: np.ndarray, link: str):
    """log p(y|f), d/df log p, W = -d2/df2 log p, and dW/df, for y in {-1, +1}.

    dW/df is what enters the Alg. 5.1 implicit term as
    s2 = -0.5 * diag((K^-1 + W)^-1) * dW/df. That sign matches sklearn's
    implementation, which agrees with finite differences.
    """
    if link == "probit":
        z = y_pm * f
        logp = log_ndtr(z)
        r = np.exp(norm.logpdf(z) - logp)          # phi(z)/Phi(z), stable in both tails
        grad = y_pm * r
        W = r * r + z * r
        dW = -y_pm * (2.0 * r**3 + 3.0 * z * r**2 + (z * z - 1.0) * r)
        return logp, grad, W, dW
    if link == "logit":
        pi = expit(f)
        logp = -np.logaddexp(0.0, -y_pm * f)
        grad = (y_pm + 1.0) / 2.0 - pi
        W = pi * (1.0 - pi)
        dW = W * (1.0 - 2.0 * pi)
        return logp, grad, W, dW
    raise ValueError(f"unknown link {link!r}")


def posterior_mode(K: np.ndarray, y_pm: np.ndarray, mu: float, link: str,
                   max_iter: int = 200, tol: float = 1e-10) -> dict:
    """Newton iterations for the Laplace mode (Alg. 3.1 with prior mean mu).

    Mirrors sklearn's _posterior_mode: the returned sW, L, grad and dW belong to
    the f at the start of the final iteration, which is where the Newton step
    and the Alg. 5.1 gradient are evaluated.
    """
    n = len(y_pm)
    f = np.full(n, float(mu))
    lml_prev = -np.inf
    out: dict = {}
    for _ in range(max_iter):
        _, grad, W, dW = likelihood_terms(f, y_pm, link)
        sW = np.sqrt(W)
        sWK = sW[:, None] * K
        L = cholesky(np.eye(n) + sWK * sW[None, :], lower=True)
        b = W * (f - mu) + grad
        a = b - sW * cho_solve((L, True), sWK @ b)
        f = K @ a + mu
        logp = likelihood_terms(f, y_pm, link)[0]
        lml = float(-0.5 * a @ (f - mu) + logp.sum() - np.log(np.diag(L)).sum())
        out = dict(f=f, a=a, grad=grad, sW=sW, dW=dW, L=L, lml=lml)
        if lml - lml_prev < tol:
            break
        lml_prev = lml
    return out


class LaplaceGPClassifier:
    def __init__(self, link: str = "probit", mean: str = "smoothed_rate", standardize: bool = True,
                 n_restarts: int = 5, max_points: int = 2000,
                 lengthscale_bounds=(0.01, 100.0), sigma_f2_bounds=(0.01, 25.0),
                 seed: int = 0, fixed=None):
        if link not in ("probit", "logit"):
            raise ValueError(f"unknown link {link!r}")
        if mean not in ("smoothed_rate", "zero"):
            raise ValueError(f"unknown mean {mean!r}")
        self.link, self.mean, self.standardize = link, mean, bool(standardize)
        self.n_restarts, self.max_points, self.seed = int(n_restarts), int(max_points), int(seed)
        self.lengthscale_bounds = tuple(float(v) for v in lengthscale_bounds)
        self.sigma_f2_bounds = tuple(float(v) for v in sigma_f2_bounds)
        self.fixed = fixed                  # (sigma_f2, lengthscale): skip the optimizer
        self.capped_ = False
        self.at_bound_ = False

    def _log_bounds(self) -> np.ndarray:
        return np.log(np.array([self.sigma_f2_bounds, self.lengthscale_bounds], dtype=float))

    def _prep(self, X) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.x_mean_) / self.x_scale_

    def log_marginal_likelihood(self, theta, Z, y_pm, mu, eval_gradient=False):
        """Laplace log marginal likelihood at theta = [log sigma_f2, log lengthscale]."""
        sigma_f2, lengthscale = np.exp(np.asarray(theta, dtype=float))
        if not eval_gradient:
            K = matern52(Z, Z, sigma_f2, lengthscale)
            return posterior_mode(K, y_pm, mu, self.link)["lml"]
        K, dKs = matern52_with_grads(Z, sigma_f2, lengthscale)
        m = posterior_mode(K, y_pm, mu, self.link)
        sW, L, a, grad, dW = m["sW"], m["L"], m["a"], m["grad"], m["dW"]
        R = sW[:, None] * cho_solve((L, True), np.diag(sW))
        C = solve_triangular(L, sW[:, None] * K, lower=True)
        s2 = -0.5 * (np.diag(K) - np.einsum("ij,ij->j", C, C)) * dW
        g = np.empty(len(dKs))
        for j, dK in enumerate(dKs):
            s1 = 0.5 * a @ dK @ a - 0.5 * np.sum(R * dK)
            bj = dK @ grad
            s3 = bj - K @ (R @ bj)
            g[j] = s1 + s2 @ s3
        return m["lml"], g

    def fit(self, X, y01):
        X = np.asarray(X, dtype=float)
        y01 = (np.asarray(y01) > 0).astype(int)
        if len(X) == 0:
            raise ValueError("LaplaceGPClassifier.fit needs at least one point")
        rng = np.random.default_rng(self.seed)
        self.capped_ = len(X) > self.max_points
        if self.capped_:
            keep = np.sort(rng.choice(len(X), self.max_points, replace=False))
            X, y01 = X[keep], y01[keep]
        if self.standardize:
            self.x_mean_ = X.mean(axis=0)
            scale = X.std(axis=0)
            scale[scale == 0.0] = 1.0
            self.x_scale_ = scale
        else:
            self.x_mean_ = np.zeros(X.shape[1])
            self.x_scale_ = np.ones(X.shape[1])
        Z = self._prep(X)
        y_pm = np.where(y01 > 0, 1.0, -1.0)
        if self.mean == "smoothed_rate":
            self.mu_ = float(norm.ppf((y01.sum() + 1.0) / (len(y01) + 2.0)))
        else:
            self.mu_ = 0.0
        bounds = self._log_bounds()
        if self.fixed is not None:
            theta = np.log(np.asarray(self.fixed, dtype=float))
            self.at_bound_ = False
        else:
            starts = [np.clip(np.zeros(2), bounds[:, 0], bounds[:, 1])]
            starts += [rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(self.n_restarts)]

            def objective(t):
                lml, g = self.log_marginal_likelihood(t, Z, y_pm, self.mu_, eval_gradient=True)
                return -lml, -g

            best = None
            for t0 in starts:
                res = minimize(objective, t0, jac=True, method="L-BFGS-B", bounds=bounds)
                if best is None or res.fun < best.fun:
                    best = res
            theta = best.x
            self.at_bound_ = bool(np.any(np.isclose(theta, bounds[:, 0], atol=1e-6)
                                         | np.isclose(theta, bounds[:, 1], atol=1e-6)))
        self.sigma_f2_, self.lengthscale_ = (float(v) for v in np.exp(theta))
        K = matern52(Z, Z, self.sigma_f2_, self.lengthscale_)
        m = posterior_mode(K, y_pm, self.mu_, self.link)
        self.Z_, self.grad_, self.sW_, self.L_, self.lml_ = Z, m["grad"], m["sW"], m["L"], m["lml"]
        self.n_fit_ = len(Z)
        return self

    def latent(self, X, chunk: int = 4096):
        Zs = self._prep(np.atleast_2d(np.asarray(X, dtype=float)))
        if len(Zs) == 0:
            return np.empty(0), np.empty(0)
        means, variances = [], []
        for i in range(0, len(Zs), chunk):
            Ks = matern52(Zs[i:i + chunk], self.Z_, self.sigma_f2_, self.lengthscale_)
            means.append(self.mu_ + Ks @ self.grad_)
            v = solve_triangular(self.L_, self.sW_[:, None] * Ks.T, lower=True)
            variances.append(self.sigma_f2_ - np.einsum("ij,ij->j", v, v))
        return np.concatenate(means), np.sqrt(np.maximum(np.concatenate(variances), 1e-12))

    def predict_proba(self, X):
        if self.link != "probit":
            raise NotImplementedError("predict_proba is probit-only; the logistic link exists for tests")
        m, s = self.latent(X)
        return norm.cdf(m / np.sqrt(1.0 + s * s))

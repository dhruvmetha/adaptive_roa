# Faithful Part-X (`partx_faithful`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Part-X arm that implements Alg. 1 to 4 of Pedrielli et al. (arXiv 2110.10729) on our stochastic, pool-based RoA benchmarks, validate it against the cpslab reference code and a pendulum gate, then run it in the 12 paper cells.

**Architecture:** A new package, `adaptive_roa/partx_faithful/`. Its core modules (GP classifier, backends, regions, samplers, MCstep/Classify, SampleBO, the Alg. 4 state machine, volumes, state) depend only on numpy and scipy. A runner drives the loop over the trajectory pool and writes engine-format epoch directories at matched trajectory budgets, using the existing `evaluate_full_roa_classifier`. A Hydra entry point reuses the `configs/adaptive_v2` tree, so launch overrides match `run_adaptive.py`.

**Tech Stack:** Python 3, numpy, scipy (linalg, optimize, stats), scikit-learn (tests and the reference backend only), Hydra/OmegaConf, pytest. The project env is `/common/home/st1122/Projects/adaptive_roa/env`.

**Spec:** `docs/superpowers/specs/2026-09-11-partx-faithful-stochastic-design.md` (commit f9a7be8). Read it before starting. Section numbers below (§3, §6.4, …) refer to it.

## Global Constraints

- Run every Python command as `PYTHONNOUSERSITE=1 ./env/bin/python …` from the repo (or worktree) root. A user-site numpy shadows the env's otherwise.
- Every SCRIPT invocation from the worktree (`scripts/*.py`, including the SLURM jobs) must also set `PYTHONPATH=<worktree root>`. The env installs `adaptive_roa` in editable mode pointing at the MAIN tree. A script run from `scripts/` without it imports the main tree's package, which has no `partx_faithful` (verified 2026-09-11). pytest is unaffected, because `tests/conftest.py` puts the root first.
- Test command: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful -q`.
- Core modules (`gp_laplace`, `backends`, `region`, `sampling`, `mcstep`, `bo`, `algorithm`, `volume`, `state`) import only numpy, scipy and the standard library at module level. `backends.SklearnGPRBackend` imports sklearn lazily. The reference check runs these modules under Python 3.10 and numpy < 2.
- Do not modify anything under `adaptive_roa/partx/`, since live jobs import it. Importing `adaptive_roa.partx.tree.build_root` is fine.
- Arm name `partx_faithful`. Run directories are `<campaign prefix>_partx_faithful`.
- Hyperparameters (spec §6.5), with no tuning on our metrics:
  - pendulum (d = 2): n0 = 10.
  - cartpole (4), quad2D (6), quad3D (13): n0 = 30.
  - Every system: nBO = 10, nc = 100, R × M = 20 × 500, B = 2, δC = 0.05, δv = 0.001.
- GP defaults (spec §5.2 and §5.3): 5 restarts, lengthscale ∈ [0.01, 100], σ_f² ∈ [0.01, 25], size cap 2,000 points. The EI candidate cap is 20,000.
- Falsifier = failure (D2). The `falsify: success` flag exists but stays off.
- Early termination records unspent budget and never tops up (D8).
- Commit messages carry no AI or tool attribution lines of any kind (user rule).
- Development happens in a worktree on branch `feat/partx-faithful`, created from `feat/quadrotor-stoch` at or after f9a7be8. The main tree is shared with other sessions, so don't commit there except where Task 13 says so.

## File map

| File | Responsibility |
|---|---|
| `adaptive_roa/partx_faithful/__init__.py` | Package docstring only |
| `adaptive_roa/partx_faithful/gp_laplace.py` | Exact Laplace GP classifier (probit; logit for tests) |
| `adaptive_roa/partx_faithful/backends.py` | `GPCBackend` (falsify sign), `SklearnGPRBackend` (cpslab clone), `backend_factory` |
| `adaptive_roa/partx_faithful/region.py` | `Region`, region types, `child_masks` |
| `adaptive_roa/partx_faithful/sampling.py` | `SampleStore`, `lhs`, `PoolSampler`, `ContinuousSampler` |
| `adaptive_roa/partx_faithful/mcstep.py` | `mc_step`, `classify`, `falsification_indicator` (Alg. 2, Alg. 3, eq. 4) |
| `adaptive_roa/partx_faithful/bo.py` | `expected_improvement`, `sample_bo` (Alg. 1) |
| `adaptive_roa/partx_faithful/algorithm.py` | `PartXConfig`, `PartX` (Alg. 4 state machine), `multinomial_counts`, `derive_seed` |
| `adaptive_roa/partx_faithful/volume.py` | Falsification-volume estimates |
| `adaptive_roa/partx_faithful/state.py` | `config_hash`, atomic `save_state`, `load_state` |
| `adaptive_roa/partx_faithful/readout.py` | `PiecewiseModel`, `PiecewiseHandle` (torch; not core) |
| `adaptive_roa/partx_faithful/runner.py` | `schedule`, `resolve_n0`, `CheckpointWriter`, `run_partx_faithful`, `run_from_cfg` |
| `configs/adaptive_v2/partx_faithful.yaml` | Hydra config: `defaults: [default, _self_]` + `partx_faithful:` block |
| `scripts/run_partx_faithful.py` | Hydra entry point |
| `scripts/partx_reference_check.py` | One-off cpslab comparison (scratch env) |
| `scripts/partx_faithful_gate.py` | Pendulum gate criteria (spec §8.3) |
| `tests/partx_faithful/…` | One test file per module, plus runner and config tests |

---

### Task 0: Worktree

- [ ] **Step 1: Create the worktree with the using-git-worktrees skill**

Invoke `superpowers:using-git-worktrees`. Branch `feat/partx-faithful` from the current `feat/quadrotor-stoch` HEAD, which must contain commit f9a7be8 (the spec). All later tasks run inside the worktree unless they say otherwise.

- [ ] **Step 2: Confirm the env runs from the worktree**

The conda env lives in the main tree, so point at it by absolute path:

```bash
PYTHONNOUSERSITE=1 /common/home/st1122/Projects/adaptive_roa/env/bin/python -c "import numpy, scipy, sklearn; print(numpy.__version__, scipy.__version__, sklearn.__version__)"
```

Expected: `2.2.6 1.15.2 1.7.2`. In every later command, `./env/bin/python` means this absolute path when you run it from the worktree.

---

### Task 1: Laplace GP classifier

**Files:**
- Create: `adaptive_roa/partx_faithful/__init__.py`
- Create: `adaptive_roa/partx_faithful/gp_laplace.py`
- Create: `tests/partx_faithful/__init__.py` (empty)
- Test: `tests/partx_faithful/test_gp_laplace.py`

**Interfaces:**
- Produces:
  - `LaplaceGPClassifier(link="probit", mean="smoothed_rate", standardize=True, n_restarts=5, max_points=2000, lengthscale_bounds=(0.01, 100.0), sigma_f2_bounds=(0.01, 25.0), seed=0, fixed=None)`
  - `.fit(X, y01) -> self`, where y01 ∈ {0, 1}.
  - `.latent(X) -> (m, s)`: the posterior mean and standard deviation of the success latent f.
  - `.predict_proba(X) -> p`, computed as Φ(m/√(1+s²)). Probit only.
  - `.log_marginal_likelihood(theta, Z, y_pm, mu, eval_gradient=False)` with theta = [log σ_f², log ℓ].
  - Attributes: `capped_`, `at_bound_`, `mu_`, `sigma_f2_`, `lengthscale_`, `lml_`, `n_fit_`, `Z_`.

- [ ] **Step 1: Write the failing tests**

`tests/partx_faithful/test_gp_laplace.py`:

```python
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_gp_laplace.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'adaptive_roa.partx_faithful'`.

- [ ] **Step 3: Write the package init and the classifier**

`adaptive_roa/partx_faithful/__init__.py`:

```python
"""Faithful Part-X (Pedrielli et al., arXiv 2110.10729) for the stochastic RoA benchmarks.

Spec: docs/superpowers/specs/2026-09-11-partx-faithful-stochastic-design.md.
The core modules (gp_laplace, backends, region, sampling, mcstep, bo, algorithm,
volume, state) import numpy/scipy only, so the cpslab reference check can run
them under numpy 1.x. readout and runner need the project env.
"""
```

`adaptive_roa/partx_faithful/gp_laplace.py`:

```python
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
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_gp_laplace.py -q`
Expected: `8 passed`. If the sklearn comparison is off by more than 1e-6, check that `posterior_mode` returns the start-of-final-iteration `grad`, `sW` and `L`, as sklearn does.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx_faithful/__init__.py adaptive_roa/partx_faithful/gp_laplace.py tests/partx_faithful/__init__.py tests/partx_faithful/test_gp_laplace.py
git commit -m "feat(partx_faithful): exact Laplace GP classifier with probit link"
```

---
### Task 2: Local-model backends

**Files:**
- Create: `adaptive_roa/partx_faithful/backends.py`
- Test: `tests/partx_faithful/test_backends.py`

**Interfaces:**
- Consumes: `LaplaceGPClassifier` (Task 1).
- Produces:
  - `GPCBackend(falsify="failure", seed=0, **gp_kwargs)` with `.fit(X, y_pm) -> self` (y ∈ {−1, +1}, +1 = success), `.latent(X) -> (m, s)` for the robustness-oriented g (g = f under failure, −f under success), `.p_success(X) -> p` (always about success), and properties `.capped` and `.at_bound`.
  - `SklearnGPRBackend(seed=12345)` with `.fit(X, y_real)`, `.latent(X)`, `.capped=False`, `.at_bound=False`.
  - `backend_factory(kind="gpc", falsify="failure", **gp_kwargs) -> Callable[[int], model]`.

- [ ] **Step 1: Write the failing tests**

`tests/partx_faithful/test_backends.py`:

```python
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_backends.py -q`
Expected: `ModuleNotFoundError: No module named 'adaptive_roa.partx_faithful.backends'`.

- [ ] **Step 3: Implement**

`adaptive_roa/partx_faithful/backends.py`:

```python
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
    def __init__(self, falsify: str = "failure", seed: int = 0, **gp_kwargs):
        if falsify not in ("failure", "success"):
            raise ValueError(f"falsify must be 'failure' or 'success', got {falsify!r}")
        self.sign = 1.0 if falsify == "failure" else -1.0
        self.gp = LaplaceGPClassifier(seed=seed, **gp_kwargs)

    def fit(self, X, y):
        """y uses the pool convention: +1 success, -1 failure."""
        self.gp.fit(X, (np.asarray(y) > 0).astype(int))
        return self

    def latent(self, X):
        m, s = self.gp.latent(X)
        return self.sign * m, s

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
    """seed -> fresh local model. cpslab pins every GPR to random_state=12345."""
    if kind == "gpc":
        return lambda seed: GPCBackend(falsify=falsify, seed=seed, **gp_kwargs)
    if kind == "gpr":
        return lambda seed: SklearnGPRBackend(seed=12345)
    raise ValueError(f"unknown backend kind {kind!r}")
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_backends.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx_faithful/backends.py tests/partx_faithful/test_backends.py
git commit -m "feat(partx_faithful): GPC and cpslab-GPR local-model backends"
```

---

### Task 3: Regions, sample store and samplers

**Files:**
- Create: `adaptive_roa/partx_faithful/region.py`
- Create: `adaptive_roa/partx_faithful/sampling.py`
- Test: `tests/partx_faithful/test_region_sampling.py`

**Interfaces:**
- Produces:
  - `region.py`:
    - `RTYPES`, `REMAINING = ("r", "r+", "r-")`, `CLASSIFIED = ("+", "-")`.
    - `Region(rid, parent, low, high, branch_dir, depth, rtype="r", samples=[], children=[], pool_candidates=None)` with `.volume()`, `.contains(X)` and `.split_bounds(dim, B)`.
    - `child_masks(x, bounds, dim) -> list[bool mask]`.
  - `sampling.py`:
    - `SampleStore(dim)` with `.add(X, y, pool_idx) -> list[int]`, `.X(ids)`, `.y(ids)`, `.pool_idx(ids)` and `.n`.
    - `lhs(n, low, high, rng) -> (n, d)`.
    - `PoolSampler(starts, labels, used, scale)` and `ContinuousSampler(fn)`. Both share the same methods:
      - `mark_used(idx)`
      - `init_root(region)`
      - `split(parent, children, dim)`
      - `n_available(region) -> int`
      - `snap_design(design, region) -> handles`
      - `candidates(region, cap, rng) -> (X, handles)`
      - `take(handles) -> (X, y, pool_idx)`
    - `PoolSampler` handles are int pool indices. `ContinuousSampler` handles are the points themselves.

- [ ] **Step 1: Write the failing tests**

`tests/partx_faithful/test_region_sampling.py`:

```python
import numpy as np
import pytest

from adaptive_roa.partx_faithful.region import Region, child_masks
from adaptive_roa.partx_faithful.sampling import ContinuousSampler, PoolSampler, SampleStore, lhs


def _region(low, high, rid=0):
    return Region(rid=rid, parent=-1, low=np.asarray(low, float), high=np.asarray(high, float),
                  branch_dir=0, depth=0)


def test_split_bounds_halves_one_dimension():
    (lo0, hi0), (lo1, hi1) = _region([0, 0], [2, 4]).split_bounds(1)
    assert hi0[1] == 2.0 and lo1[1] == 2.0
    assert lo0[0] == 0.0 and hi1[0] == 2.0


def test_child_masks_assign_every_point_exactly_once():
    bounds = _region([0.0], [2.0]).split_bounds(0)
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    masks = child_masks(x, bounds, 0)
    assert np.all(sum(m.astype(int) for m in masks) == 1)
    assert masks[0].tolist() == [True, True, False, False, False]
    assert masks[1].tolist() == [False, False, True, True, True]


def test_lhs_puts_one_point_in_each_stratum():
    X = lhs(8, [0.0, 0.0], [1.0, 2.0], np.random.default_rng(0))
    for d, width in enumerate([1.0, 2.0]):
        assert sorted(np.floor(X[:, d] / width * 8).astype(int).tolist()) == list(range(8))


def test_sample_store_grows_and_round_trips():
    s = SampleStore(2)
    assert s.add(np.ones((100, 2)), np.arange(100), np.arange(100)) == list(range(100))
    assert s.add([[5.0, 6.0]], [7.0], [-1]) == [100]
    np.testing.assert_array_equal(s.X([100]), [[5.0, 6.0]])
    assert s.y([3])[0] == 3.0 and s.pool_idx([100])[0] == -1 and s.n == 101
    assert s.X([]).shape == (0, 2)


def _pool(n=400, seed=0, n_used=10):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, 1, size=(n, 2))
    labels = np.where(starts[:, 0] < 0.5, 1.0, -1.0)
    used = np.zeros(n, dtype=bool)
    used[:n_used] = True
    return PoolSampler(starts, labels, used, scale=np.ones(2)), starts


def _split_root(ps):
    root = _region([0, 0], [1, 1])
    ps.init_root(root)
    kids = [Region(rid=i + 1, parent=0, low=lo, high=hi, branch_dir=1, depth=1)
            for i, (lo, hi) in enumerate(root.split_bounds(0))]
    all_cand = root.pool_candidates.copy()
    ps.split(root, kids, 0)
    return root, kids, all_cand


def test_split_partitions_the_candidates():
    ps, _ = _pool()
    root, kids, all_cand = _split_root(ps)
    assert root.pool_candidates is None
    both = np.concatenate([k.pool_candidates for k in kids])
    assert sorted(both.tolist()) == sorted(all_cand.tolist())


def test_snap_returns_unused_unique_in_box_points():
    ps, starts = _pool()
    _, kids, _ = _split_root(ps)
    h = ps.snap_design(lhs(20, kids[0].low, kids[0].high, np.random.default_rng(1)), kids[0])
    assert len(h) == 20 and len(set(h.tolist())) == 20
    assert not ps.used[h].any()
    assert np.all(kids[0].contains(starts[h]))
    assert np.all(h >= 10)                                     # the initial set is excluded


def test_take_marks_used_and_refuses_reuse():
    ps, _ = _pool()
    _, kids, _ = _split_root(ps)
    h = ps.snap_design(lhs(3, kids[0].low, kids[0].high, np.random.default_rng(1)), kids[0])
    X, y, p = ps.take(h)
    assert ps.used[p].all() and X.shape == (3, 2) and set(y.tolist()) <= {1.0, -1.0}
    with pytest.raises(RuntimeError):
        ps.take(h[:1])


def test_candidates_respect_the_cap_and_empty_boxes():
    ps, _ = _pool()
    root = _region([0, 0], [1, 1])
    ps.init_root(root)
    X, h = ps.candidates(root, 5, np.random.default_rng(0))
    assert len(h) == 5 and X.shape == (5, 2)
    empty = _region([0, 0], [1, 1], rid=9)
    empty.pool_candidates = np.empty(0, dtype=np.int64)
    assert ps.n_available(empty) == 0
    assert len(ps.snap_design(np.array([[0.5, 0.5]]), empty)) == 0


def test_continuous_sampler_evaluates_the_function():
    cs = ContinuousSampler(lambda x: x[0] + x[1])
    X, y, p = cs.take(np.array([[1.0, 2.0]]))
    assert y[0] == 3.0 and p[0] == -1
    U, h = cs.candidates(_region([0, 0], [1, 1]), 7, np.random.default_rng(0))
    assert U.shape == (7, 2) and np.all((U >= 0) & (U <= 1))
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_region_sampling.py -q`
Expected: `ModuleNotFoundError` for `adaptive_roa.partx_faithful.region`.

- [ ] **Step 3: Implement `region.py`**

```python
"""Region boxes and the Part-X region types (spec §3)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RTYPES = ("r", "r+", "r-", "+", "-", "u", "i")
REMAINING = ("r", "r+", "r-")
CLASSIFIED = ("+", "-")


@dataclass
class Region:
    rid: int
    parent: int
    low: np.ndarray
    high: np.ndarray
    branch_dir: int
    depth: int
    rtype: str = "r"
    samples: list = field(default_factory=list)       # sample ids into the SampleStore
    children: list = field(default_factory=list)      # child region ids
    pool_candidates: np.ndarray | None = None          # pool indices inside the box (PoolSampler)

    def volume(self) -> float:
        return float(np.prod(self.high - self.low))

    def contains(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.all((X >= self.low) & (X <= self.high), axis=1)

    def split_bounds(self, dim: int, B: int = 2) -> list:
        """B equal cuts along `dim` (cpslab branch_region with uniform=True)."""
        edges = self.low[dim] + (np.arange(B + 1) / B) * (self.high[dim] - self.low[dim])
        out = []
        for k in range(B):
            lo, hi = self.low.copy(), self.high.copy()
            lo[dim], hi[dim] = edges[k], edges[k + 1]
            out.append((lo, hi))
        return out


def child_masks(x: np.ndarray, bounds: list, dim: int) -> list:
    """Which child each coordinate x (along `dim`) belongs to.

    Child k takes lo_k <= x < hi_k, except that the first child has no lower
    limit and the last no upper limit. So every point lands in exactly one
    child, including points on an edge or just outside the parent box.
    """
    x = np.asarray(x, dtype=float)
    B = len(bounds)
    masks = []
    for k, (lo, hi) in enumerate(bounds):
        m = np.ones(len(x), dtype=bool)
        if k > 0:
            m &= x >= lo[dim]
        if k < B - 1:
            m &= x < hi[dim]
        masks.append(m)
    return masks
```

- [ ] **Step 4: Implement `sampling.py`**

```python
"""Sample store, LHS, and the two samplers Part-X draws evaluations from (spec §3, §6.3)."""
from __future__ import annotations

import numpy as np

from adaptive_roa.partx_faithful.region import child_masks


class SampleStore:
    """Append-only store of evaluated points. Sample ids are row numbers."""

    def __init__(self, dim: int):
        self._X = np.empty((64, dim))
        self._y = np.empty(64)
        self._pool = np.empty(64, dtype=np.int64)
        self.n = 0

    def _grow(self, need: int) -> None:
        cap = len(self._y)
        while cap < need:
            cap *= 2
        if cap == len(self._y):
            return
        X = np.empty((cap, self._X.shape[1]))
        X[:self.n] = self._X[:self.n]
        y = np.empty(cap)
        y[:self.n] = self._y[:self.n]
        p = np.empty(cap, dtype=np.int64)
        p[:self.n] = self._pool[:self.n]
        self._X, self._y, self._pool = X, y, p

    def add(self, X, y, pool_idx) -> list:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        p = np.asarray(pool_idx, dtype=np.int64).reshape(-1)
        k = len(X)
        self._grow(self.n + k)
        self._X[self.n:self.n + k] = X
        self._y[self.n:self.n + k] = y
        self._pool[self.n:self.n + k] = p
        ids = list(range(self.n, self.n + k))
        self.n += k
        return ids

    def X(self, ids) -> np.ndarray:
        return self._X[np.asarray(ids, dtype=np.int64)]

    def y(self, ids) -> np.ndarray:
        return self._y[np.asarray(ids, dtype=np.int64)]

    def pool_idx(self, ids) -> np.ndarray:
        return self._pool[np.asarray(ids, dtype=np.int64)]


def lhs(n: int, low, high, rng) -> np.ndarray:
    """Latin hypercube design of n points in the box [low, high]."""
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    d = len(low)
    if n <= 0:
        return np.empty((0, d))
    strata = np.stack([rng.permutation(n) for _ in range(d)], axis=1)
    u = (strata + rng.random((n, d))) / n
    return low + u * (high - low)


class PoolSampler:
    """Part-X evaluations drawn from the pre-simulated pool (spec §3, "Pool sampling").

    starts [N, d] and labels [N] (+1 success, -1 failure) are indexed by pool
    index, and `used` marks indices already evaluated. Each region carries the
    pool indices inside its box (region.pool_candidates), which are split along
    with the region, so a query never rescans the whole pool. Distances are
    Euclidean after dividing by `scale`, the root box's side lengths.
    """

    def __init__(self, starts, labels, used, scale):
        self.starts = np.asarray(starts, dtype=float)
        self.labels = np.asarray(labels, dtype=float)
        self.used = np.asarray(used, dtype=bool).copy()
        self.scale = np.asarray(scale, dtype=float)

    def mark_used(self, pool_idx) -> None:
        self.used[np.asarray(pool_idx, dtype=np.int64)] = True

    def init_root(self, region) -> None:
        inside = np.all((self.starts >= region.low) & (self.starts <= region.high), axis=1)
        region.pool_candidates = np.flatnonzero(inside)

    def split(self, parent, children, dim) -> None:
        cand = parent.pool_candidates
        masks = child_masks(self.starts[cand, dim], [(c.low, c.high) for c in children], dim)
        for child, m in zip(children, masks):
            child.pool_candidates = cand[m]
        parent.pool_candidates = None

    def _available(self, region) -> np.ndarray:
        cand = region.pool_candidates
        return cand[~self.used[cand]]

    def n_available(self, region) -> int:
        return int(len(self._available(region)))

    def snap_design(self, design, region) -> np.ndarray:
        """For each design point in order, the nearest unused pool start in the box. No repeats."""
        avail = self._available(region)
        design = np.atleast_2d(np.asarray(design, dtype=float))
        if len(avail) == 0 or len(design) == 0:
            return np.empty(0, dtype=np.int64)
        Z = self.starts[avail] / self.scale
        taken = np.zeros(len(avail), dtype=bool)
        chosen = []
        for q in design / self.scale:
            d2 = np.sum((Z - q) ** 2, axis=1)
            d2[taken] = np.inf
            j = int(np.argmin(d2))
            if not np.isfinite(d2[j]):
                break
            taken[j] = True
            chosen.append(int(avail[j]))
        return np.asarray(chosen, dtype=np.int64)

    def candidates(self, region, cap, rng):
        avail = self._available(region)
        if len(avail) > cap:
            avail = np.sort(rng.choice(avail, size=int(cap), replace=False))
        return self.starts[avail], avail

    def take(self, handles):
        idx = np.asarray(handles, dtype=np.int64).reshape(-1)
        if len(np.unique(idx)) != len(idx):
            raise RuntimeError("PoolSampler.take: duplicate pool index")
        if self.used[idx].any():
            raise RuntimeError("PoolSampler.take: pool index already used")
        self.used[idx] = True
        return self.starts[idx], self.labels[idx], idx


class ContinuousSampler:
    """Reference-check sampler (spec §8.2): evaluates fn anywhere in the box."""

    def __init__(self, fn):
        self.fn = fn

    def mark_used(self, pool_idx) -> None:
        pass

    def init_root(self, region) -> None:
        pass

    def split(self, parent, children, dim) -> None:
        pass

    def n_available(self, region) -> int:
        return int(np.iinfo(np.int64).max)

    def snap_design(self, design, region) -> np.ndarray:
        return np.atleast_2d(np.asarray(design, dtype=float))

    def candidates(self, region, cap, rng):
        U = region.low + rng.random((int(cap), len(region.low))) * (region.high - region.low)
        return U, U

    def take(self, handles):
        X = np.atleast_2d(np.asarray(handles, dtype=float))
        y = np.array([float(self.fn(x)) for x in X])
        return X, y, np.full(len(X), -1, dtype=np.int64)
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_region_sampling.py -q`
Expected: `9 passed`.

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/partx_faithful/region.py adaptive_roa/partx_faithful/sampling.py tests/partx_faithful/test_region_sampling.py
git commit -m "feat(partx_faithful): regions, sample store, LHS, pool and continuous samplers"
```

---
### Task 4: MCstep, Classify and the eq. 4 indicator

**Files:**
- Create: `adaptive_roa/partx_faithful/mcstep.py`
- Test: `tests/partx_faithful/test_mcstep.py`

**Interfaces:**
- Consumes: `lhs` (Task 3). A model is any object with `latent(X) -> (m, s)`.
- Produces:
  - `z_value(delta_c) -> float`, computed as Φ⁻¹(1 − δC/2).
  - `Quantiles(q_max, var_q_max, q_min, var_q_min)`.
  - `mc_step(model, low, high, R, M, delta_c, rng) -> Quantiles`.
  - `classify(rtype, volume, min_volume, q, delta_c, compat_ci_var=False) -> str`.
  - `falsification_indicator(model, low, high, R, M, rng, times_volume=False) -> float`.

- [ ] **Step 1: Write the failing tests**

`tests/partx_faithful/test_mcstep.py`:

```python
import numpy as np
import pytest

from adaptive_roa.partx_faithful.mcstep import (Quantiles, classify, falsification_indicator,
                                                mc_step, z_value)


class _Const:
    def __init__(self, m, s):
        self.m, self.s = m, s

    def latent(self, X):
        n = len(X)
        return np.full(n, self.m), np.full(n, self.s)


def _q(q_max, q_min, var=0.0):
    return Quantiles(q_max=q_max, var_q_max=var, q_min=q_min, var_q_min=var)


def test_z_value():
    assert z_value(0.05) == pytest.approx(1.959964, abs=1e-6)


def test_mc_step_on_a_constant_model():
    q = mc_step(_Const(0.3, 0.1), np.zeros(2), np.ones(2), R=5, M=20, delta_c=0.05,
                rng=np.random.default_rng(0))
    z = z_value(0.05)
    assert q.q_max == pytest.approx(0.3 + z * 0.1)
    assert q.q_min == pytest.approx(0.3 - z * 0.1)
    assert q.var_q_max == pytest.approx(0.0) and q.var_q_min == pytest.approx(0.0)


@pytest.mark.parametrize("rtype,q,expected", [
    ("r", _q(-0.1, -0.5), "-"),
    ("r", _q(0.5, 0.1), "+"),
    ("r", _q(0.5, -0.5), "r"),
    ("+", _q(0.5, 0.1), "+"),
    ("+", _q(0.5, 0.0), "r+"),
    ("+", _q(0.5, -0.1), "r+"),
    ("-", _q(-0.1, -0.5), "-"),
    ("-", _q(0.0, -0.5), "r-"),
    ("r+", _q(-1.0, -2.0), "r+"),
    ("r-", _q(2.0, 1.0), "r-"),
])
def test_classify_truth_table(rtype, q, expected):
    assert classify(rtype, 1.0, 0.0, q, 0.05) == expected


def test_small_regions_become_u():
    assert classify("r", 1e-9, 1e-6, _q(5.0, 4.0), 0.05) == "u"


def test_terminal_types_cannot_be_classified():
    with pytest.raises(ValueError):
        classify("u", 1.0, 0.0, _q(1.0, 0.0), 0.05)


def test_compat_ci_var_uses_the_variance_instead_of_the_sd():
    # var 0.04 -> sd 0.2: z*0.2 = 0.39 > 0.3 keeps 'r'; z*0.04 = 0.078 < 0.3 gives '+'
    q = _q(1.0, 0.3, var=0.04)
    assert classify("r", 1.0, 0.0, q, 0.05) == "r"
    assert classify("r", 1.0, 0.0, q, 0.05, compat_ci_var=True) == "+"


def test_falsification_indicator():
    rng = np.random.default_rng(0)
    low, high = np.zeros(2), np.array([2.0, 3.0])
    assert falsification_indicator(_Const(0.0, 1.0), low, high, 2, 10, rng) == pytest.approx(0.5)
    assert falsification_indicator(_Const(0.0, 1.0), low, high, 2, 10, rng,
                                   times_volume=True) == pytest.approx(3.0)
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_mcstep.py -q`
Expected: `ModuleNotFoundError` for `adaptive_roa.partx_faithful.mcstep`.

- [ ] **Step 3: Implement**

`adaptive_roa/partx_faithful/mcstep.py`:

```python
"""MCstep (Alg. 2), Classify (Alg. 3) and the continued-sampling indicator (eq. 4).

All three work on g, the robustness-oriented latent (g < 0 = falsifying).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from adaptive_roa.partx_faithful.sampling import lhs


def z_value(delta_c: float) -> float:
    return float(norm.ppf(1.0 - delta_c / 2.0))


@dataclass
class Quantiles:
    q_max: float        # paper Q-bar: mean over R of max_m (m + z s)
    var_q_max: float    # var over R of that max, divided by R
    q_min: float        # paper Q-underbar: mean over R of min_m (m - z s)
    var_q_min: float


def mc_step(model, low, high, R: int, M: int, delta_c: float, rng) -> Quantiles:
    """Alg. 2 with one GP per region (spec §4 row 3)."""
    z = z_value(delta_c)
    X = np.vstack([lhs(M, low, high, rng) for _ in range(R)])
    m, s = model.latent(X)
    m, s = m.reshape(R, M), s.reshape(R, M)
    q_max = np.max(m + z * s, axis=1)
    q_min = np.min(m - z * s, axis=1)
    return Quantiles(float(q_max.mean()), float(q_max.var() / R),
                     float(q_min.mean()), float(q_min.var() / R))


def classify(rtype: str, volume: float, min_volume: float, q: Quantiles, delta_c: float,
             compat_ci_var: bool = False) -> str:
    """Alg. 3, as in cpslab classification.py. compat_ci_var reproduces cpslab's Var-for-sd (spec §4 row 1)."""
    if volume <= min_volume:
        return "u"
    z = z_value(delta_c)
    w_max = q.var_q_max if compat_ci_var else np.sqrt(q.var_q_max)
    w_min = q.var_q_min if compat_ci_var else np.sqrt(q.var_q_min)
    upper = q.q_max + z * w_max       # confident upper bound on the region's max of g
    lower = q.q_min - z * w_min       # confident lower bound on the region's min of g
    if rtype == "+":
        return "r+" if lower <= 0.0 else "+"
    if rtype == "-":
        return "r-" if upper >= 0.0 else "-"
    if rtype == "r":
        if upper < 0.0:
            return "-"
        if lower > 0.0:
            return "+"
        return "r"
    if rtype in ("r+", "r-"):
        return rtype                  # cpslab leaves reclassified regions unchanged
    raise ValueError(f"cannot classify a region of type {rtype!r}")


def falsification_indicator(model, low, high, R: int, M: int, rng, times_volume: bool = False) -> float:
    """Paper eq. (4): mean over the region of P(g(x) < 0) = Phi(-m/s).

    times_volume reproduces cpslab's calculate_mc_integral, which multiplies
    by the volume instead of normalizing (spec §4 row 2).
    """
    X = np.vstack([lhs(M, low, high, rng) for _ in range(R)])
    m, s = model.latent(X)
    value = float(np.mean(norm.cdf(-m / s)))
    return value * float(np.prod(np.asarray(high) - np.asarray(low))) if times_volume else value
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_mcstep.py -q`
Expected: `16 passed`.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx_faithful/mcstep.py tests/partx_faithful/test_mcstep.py
git commit -m "feat(partx_faithful): MCstep, Alg. 3 classification and the eq. 4 indicator"
```

---

### Task 5: SampleBO (Alg. 1)

**Files:**
- Create: `adaptive_roa/partx_faithful/bo.py`
- Test: `tests/partx_faithful/test_bo.py`

**Interfaces:**
- Consumes: `lhs`, `SampleStore`, and a sampler (Task 3).
- Produces:
  - `expected_improvement(m, s, f_star) -> ei`, for minimization.
  - `sample_bo(region, store, sampler, fit_model, take, n0, n_bo, cand_cap, rng, f_star_mode="plugin") -> {"new_ids", "shortfall", "infeasible"}`.
  - `fit_model(region)` returns a model fitted on `region.samples`.
  - `take(handles)` adds the handles' evaluations to `region.samples` and returns their new ids. The algorithm supplies it.

- [ ] **Step 1: Write the failing tests**

`tests/partx_faithful/test_bo.py`:

```python
import numpy as np
import pytest
from scipy.stats import norm

from adaptive_roa.partx_faithful.bo import expected_improvement, sample_bo
from adaptive_roa.partx_faithful.region import Region
from adaptive_roa.partx_faithful.sampling import PoolSampler, SampleStore


def test_ei_at_the_incumbent_is_phi0_times_sd():
    assert expected_improvement(np.array([1.0]), np.array([2.0]), 1.0)[0] == pytest.approx(2.0 * norm.pdf(0.0))


def test_ei_prefers_a_lower_mean_and_a_larger_sd():
    e = expected_improvement(np.array([0.0, 1.0]), np.array([1.0, 1.0]), 0.5)
    assert e[0] > e[1]
    e = expected_improvement(np.array([1.0, 1.0]), np.array([0.1, 1.0]), 0.5)
    assert e[1] > e[0]


class _Linear:
    """g(x) = x0 with a constant sd. At fixed sd, EI falls as the mean rises, so
    each EI step takes the lowest available x0. The sd is 0.05, not tiny, so EI
    stays well above underflow for every candidate and argmax never breaks ties
    by index."""

    def fit(self, X, y):
        return self

    def latent(self, X):
        X = np.atleast_2d(X)
        return X[:, 0].copy(), np.full(len(X), 0.05)


def _setup(n=300, all_used=False):
    rng = np.random.default_rng(0)
    starts = rng.uniform(0, 1, size=(n, 2))
    used = np.full(n, all_used)
    ps = PoolSampler(starts, np.ones(n), used, scale=np.ones(2))
    region = Region(rid=0, parent=-1, low=np.zeros(2), high=np.ones(2), branch_dir=0, depth=0)
    ps.init_root(region)
    store = SampleStore(2)

    def take(handles):
        if len(handles) == 0:
            return []
        ids = store.add(*ps.take(handles))
        region.samples.extend(ids)
        return ids

    return ps, region, store, take, starts


def test_tops_up_by_lhs_then_takes_the_ei_argmax():
    ps, region, store, take, starts = _setup()
    res = sample_bo(region, store, ps, lambda r: _Linear(), take, n0=6, n_bo=3,
                    cand_cap=10_000, rng=np.random.default_rng(0))
    assert not res["infeasible"] and res["shortfall"] == 0 and len(res["new_ids"]) == 9
    topped_up = store.pool_idx(res["new_ids"][:6])
    rest = np.delete(starts[:, 0], topped_up)
    np.testing.assert_allclose(np.sort(store.X(res["new_ids"][6:])[:, 0]), np.sort(rest)[:3])


def test_skips_the_top_up_when_the_region_already_has_n0():
    ps, region, store, take, _ = _setup()
    take(np.arange(6))
    res = sample_bo(region, store, ps, lambda r: _Linear(), take, n0=6, n_bo=2,
                    cand_cap=100, rng=np.random.default_rng(0))
    assert len(res["new_ids"]) == 2 and len(region.samples) == 8


def test_marks_an_empty_box_infeasible():
    ps, region, store, take, _ = _setup(all_used=True)
    res = sample_bo(region, store, ps, lambda r: _Linear(), take, n0=6, n_bo=3,
                    cand_cap=100, rng=np.random.default_rng(0))
    assert res == {"new_ids": [], "shortfall": 9, "infeasible": True}
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_bo.py -q`
Expected: `ModuleNotFoundError` for `adaptive_roa.partx_faithful.bo`.

- [ ] **Step 3: Implement**

`adaptive_roa/partx_faithful/bo.py`:

```python
"""SampleBO (Alg. 1): LHS top-up to n0, then nBO sequential expected-improvement steps.

EI minimizes g, so it hunts falsifiers (paper eq. 3). With binary outcomes
there is no observed best value, so f* is the plug-in minimum of the posterior
mean at the region's sampled points (spec §3, "EI on binary data"). The
"observed" mode (min of y) exists for the deterministic reference check.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from adaptive_roa.partx_faithful.sampling import lhs


def expected_improvement(m, s, f_star: float) -> np.ndarray:
    s = np.maximum(np.asarray(s, dtype=float), 1e-12)
    imp = f_star - np.asarray(m, dtype=float)
    u = imp / s
    return imp * norm.cdf(u) + s * norm.pdf(u)


def sample_bo(region, store, sampler, fit_model, take, n0: int, n_bo: int, cand_cap: int, rng,
              f_star_mode: str = "plugin") -> dict:
    """Alg. 1 on one region. Always takes n_bo EI steps, as cpslab does (spec §4 row 4)."""
    new_ids: list = []
    shortfall = 0
    need = max(int(n0) - len(region.samples), 0)
    if need > 0:
        if sampler.n_available(region) == 0:
            return {"new_ids": new_ids, "shortfall": need + int(n_bo), "infeasible": True}
        handles = sampler.snap_design(lhs(need, region.low, region.high, rng), region)
        shortfall += need - len(handles)
        new_ids += take(handles)
    for t in range(int(n_bo)):
        Xc, hc = sampler.candidates(region, cand_cap, rng)
        if len(Xc) == 0 or not region.samples:
            return {"new_ids": new_ids, "shortfall": shortfall + int(n_bo) - t, "infeasible": True}
        model = fit_model(region)
        m, s = model.latent(Xc)
        if f_star_mode == "plugin":
            f_star = float(np.min(model.latent(store.X(region.samples))[0]))
        elif f_star_mode == "observed":
            f_star = float(np.min(store.y(region.samples)))
        else:
            raise ValueError(f"unknown f_star_mode {f_star_mode!r}")
        j = int(np.argmax(expected_improvement(m, s, f_star)))
        new_ids += take(hc[j:j + 1])
    return {"new_ids": new_ids, "shortfall": shortfall, "infeasible": False}
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_bo.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/partx_faithful/bo.py tests/partx_faithful/test_bo.py
git commit -m "feat(partx_faithful): SampleBO with plug-in EI"
```

---

### Task 6: The Alg. 4 state machine and resumable state

**Files:**
- Create: `adaptive_roa/partx_faithful/algorithm.py`
- Create: `adaptive_roa/partx_faithful/state.py`
- Test: `tests/partx_faithful/test_algorithm.py`

**Interfaces:**
- Consumes: Tasks 3 to 5.
- Produces:
  - `PartXConfig(n0, n_bo=10, n_c=100, R=20, M=500, B=2, delta_c=0.05, delta_v=0.001, ei_cand_cap=20000, ei_f_star="plugin", compat_ci_var=False, compat_eq4_times_volume=False, seed=42)`.
  - `derive_seed(*parts) -> int` and `multinomial_counts(weights, total, rng) -> int array`.
  - `PartX(cfg, low, high, sampler, model_factory, budget, init_X=None, init_y=None, init_pool_idx=None)`:
    - Methods: `.run_root()`, `.step()`, `.run()`, `.leaves()`, `.fit(region)` and `.attach(sampler, model_factory)`.
    - Public attributes: `regions`, `store`, `log`, `used`, `budget`, `n_init`, `iteration`, `phase`, `done`, `remaining`, `classified`, `root_volume`, `shortfall`, `n_fits`, `n_capped`, `n_at_bound`, `cfg` and `on_sample`. `on_sample` is a callable(px) or None.
  - `state.py`: `config_hash(obj) -> str`, `save_state(px, path, chash)`, `load_state(path, chash) -> PartX`.

- [ ] **Step 1: Write the failing tests**

`tests/partx_faithful/test_algorithm.py`:

```python
import numpy as np
import pytest

from adaptive_roa.partx_faithful.algorithm import PartX, PartXConfig, multinomial_counts
from adaptive_roa.partx_faithful.sampling import PoolSampler
from adaptive_roa.partx_faithful.state import load_state, save_state


class _Stub:
    """g(x) = x0 - 0.5, sd 0.1/sqrt(n). Deterministic and instant."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        self.n = max(len(X), 1)
        return self

    def latent(self, X):
        X = np.atleast_2d(X)
        return X[:, 0] - 0.5, np.full(len(X), 0.1 / np.sqrt(self.n))


def _px(budget, n=4000, n0=4, n_bo=2, n_c=6, seed=0, initial=40, x0_max=1.0):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, 1, size=(n, 2))
    starts[:, 0] *= x0_max
    labels = np.where(starts[:, 0] < 0.5, 1.0, -1.0)
    used = np.zeros(n, dtype=bool)
    used[:initial] = True
    ps = PoolSampler(starts, labels, used, scale=np.ones(2))
    cfg = PartXConfig(n0=n0, n_bo=n_bo, n_c=n_c, R=3, M=40, seed=seed)
    px = PartX(cfg, np.zeros(2), np.ones(2), ps, _Stub, budget,
               init_X=starts[:initial], init_y=labels[:initial], init_pool_idx=np.arange(initial))
    return px, ps


def test_the_budget_is_spent_exactly_and_logged_once():
    px, _ = _px(budget=120)
    px.run()
    assert px.done and px.used == 120 == len(px.log) == len(set(px.log))


def test_a_straight_boundary_gets_classified_on_the_right_sides():
    px, _ = _px(budget=400)
    px.run()
    leaves = px.leaves()
    plus = [r for r in leaves if r.rtype == "+"]
    minus = [r for r in leaves if r.rtype == "-"]
    assert plus and minus
    assert all(r.low[0] >= 0.5 - 1e-12 for r in plus)       # g > 0 only where x0 > 0.5
    assert all(r.high[0] <= 0.5 + 1e-12 for r in minus)


def test_branch_directions_follow_the_permutation():
    px, _ = _px(budget=400)
    px.run_root()
    px.step()
    root = px.regions[0]
    d0 = int(px.perm[0])
    kids = [px.regions[k] for k in root.children]
    assert kids[0].high[d0] == pytest.approx(0.5) and kids[1].low[d0] == pytest.approx(0.5)
    px.step()
    grandkids = [px.regions[k] for k in kids[0].children]
    assert grandkids
    d1 = int(px.perm[1 % 2])
    assert grandkids[0].high[d1] == pytest.approx((kids[0].low[d1] + kids[0].high[d1]) / 2)


def test_children_inherit_the_parent_samples_exactly():
    px, _ = _px(budget=400)
    px.run_root()
    before = set(px.regions[0].samples)
    px.step()
    kids = [px.regions[k] for k in px.regions[0].children]
    inherited = [s for k in kids for s in k.samples if s in before]
    assert sorted(inherited) == sorted(before)


def test_the_gate_falls_through_to_the_final_phase():
    px, _ = _px(budget=5, n0=4, n_bo=2)
    px.run_root()
    assert px.used == 2
    px.step()
    assert px.phase == "final" and px.used == 5
    px.step()
    assert px.done


def test_on_sample_fires_after_every_batch():
    px, _ = _px(budget=60)
    seen = []
    px.on_sample = lambda p: seen.append(p.used)
    px.run()
    assert seen[-1] == 60 and seen == sorted(seen)


def test_a_box_without_pool_points_becomes_infeasible():
    px, _ = _px(budget=200, x0_max=0.5)          # the pool covers only x0 < 0.5
    px.run()
    assert any(r.rtype == "i" and r.low[0] >= 0.5 for r in px.leaves())


def test_the_same_seed_gives_the_same_run():
    a, _ = _px(budget=150)
    a.run()
    b, _ = _px(budget=150)
    b.run()
    assert a.log == b.log
    assert [r.rtype for r in a.leaves()] == [r.rtype for r in b.leaves()]


def test_a_pickled_run_resumes_exactly(tmp_path):
    full, _ = _px(budget=150)
    full.run()
    part, _ = _px(budget=150)
    part.run_root()
    part.step()
    part.step()
    save_state(part, tmp_path / "s.pkl", "h")
    resumed = load_state(tmp_path / "s.pkl", "h")
    _, fresh = _px(budget=150)                   # same pool, only the initial set used
    resumed.attach(fresh, _Stub)
    resumed.run()
    assert resumed.log == full.log
    with pytest.raises(RuntimeError):
        load_state(tmp_path / "s.pkl", "other")


def test_multinomial_counts():
    rng = np.random.default_rng(0)
    c = multinomial_counts([0.0, 1.0, 3.0], 400, rng)
    assert c[0] == 0 and c.sum() == 400 and c[2] > c[1]
    assert multinomial_counts([0.0, 0.0], 10, rng).tolist() == [0, 0]
    assert multinomial_counts([1.0], 0, rng).tolist() == [0]


def test_the_budget_must_cover_the_root():
    with pytest.raises(ValueError):
        _px(budget=1, n0=4, n_bo=2)
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_algorithm.py -q`
Expected: `ModuleNotFoundError` for `adaptive_roa.partx_faithful.algorithm`.

- [ ] **Step 3: Implement `state.py`**

```python
"""Atomic pickle of the Part-X state (spec §6.6)."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_state(px, path, chash: str) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump({"config_hash": chash, "px": px}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_state(path, chash: str):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if blob["config_hash"] != chash:
        raise RuntimeError(f"{path}: state was saved under config {blob['config_hash']}, "
                           f"this run is {chash}; refusing to resume")
    return blob["px"]
```

- [ ] **Step 4: Implement `algorithm.py`**

```python
"""Alg. 4 of Pedrielli et al. (Part-X) as a resumable state machine (spec §3, §6.4, §6.6).

The flow follows cpslab's singlereplication.py. The root gets SampleBO and a
classification. Each later iteration either branches every remaining region
(when the budget covers the n0 top-up plus nBO for every child) and then runs
continued sampling on classified regions, or runs the final volume-proportional
phase.

The budget counts NEW evaluations. The initial samples are shared with every
other arm and are free. `log` lists new sample ids in canonical order
(ascending region id within each phase). Every random draw comes from a
generator seeded by (seed, iteration, region id, stage), and every fit is
seeded by (seed, region id, number of samples), so a resumed run replays
exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_roa.partx_faithful.bo import sample_bo
from adaptive_roa.partx_faithful.mcstep import classify, falsification_indicator, mc_step
from adaptive_roa.partx_faithful.region import CLASSIFIED, REMAINING, Region, child_masks
from adaptive_roa.partx_faithful.sampling import SampleStore, lhs


@dataclass
class PartXConfig:
    n0: int
    n_bo: int = 10
    n_c: int = 100
    R: int = 20
    M: int = 500
    B: int = 2
    delta_c: float = 0.05
    delta_v: float = 0.001
    ei_cand_cap: int = 20000
    ei_f_star: str = "plugin"
    compat_ci_var: bool = False
    compat_eq4_times_volume: bool = False
    seed: int = 42


def derive_seed(*parts) -> int:
    return int(np.random.SeedSequence([int(p) % (2**63) for p in parts]).generate_state(1)[0])


def multinomial_counts(weights, total, rng) -> np.ndarray:
    """cpslab assign_budgets, drawn from a seeded generator (spec §4 row 7)."""
    w = np.asarray(weights, dtype=float)
    if total <= 0 or len(w) == 0 or w.sum() <= 0.0:
        return np.zeros(len(w), dtype=int)
    return rng.multinomial(int(total), w / w.sum())


_UNCLASSIFIED, _INDICATOR, _CONTINUED, _FINAL = 1, 2, 3, 4
_ITERATION = -1          # region-id slot for draws that belong to the whole iteration


class PartX:
    def __init__(self, cfg: PartXConfig, low, high, sampler, model_factory, budget: int,
                 init_X=None, init_y=None, init_pool_idx=None):
        low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        self.cfg, self.d, self.budget, self.used = cfg, len(low), int(budget), 0
        self.sampler, self.model_factory = sampler, model_factory
        self.on_sample = None
        self.model_cache: dict = {}
        self.store = SampleStore(self.d)
        root = Region(rid=0, parent=-1, low=low, high=high, branch_dir=0, depth=0)
        sampler.init_root(root)
        if init_X is not None and len(init_X):
            root.samples = self.store.add(init_X, init_y, init_pool_idx)
        self.n_init = len(root.samples)
        if self.budget < max(cfg.n0 - self.n_init, 0) + cfg.n_bo:
            raise ValueError("budget must cover the root's top-up and nBO "
                             "(cpslab singlereplication.py:58-60)")
        self.regions = {0: root}
        self.next_rid = 1
        self.root_volume = root.volume()
        self.min_volume = cfg.delta_v ** self.d * self.root_volume
        self.perm = np.random.default_rng(cfg.seed).permutation(self.d)
        self.remaining, self.classified, self.unclassified, self.infeasible = [], [], [], []
        self.log: list = []
        self.iteration, self.phase, self.done = 0, "root", False
        self.shortfall = self.n_fits = self.n_capped = self.n_at_bound = 0

    # ---- bookkeeping ------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(sampler=None, model_factory=None, on_sample=None, model_cache={})
        return state

    def attach(self, sampler, model_factory) -> None:
        """Reattach after unpickling. Rebuilds the sampler's used-mask from the store."""
        self.sampler, self.model_factory, self.model_cache = sampler, model_factory, {}
        pool = self.store.pool_idx(np.arange(self.store.n))
        sampler.mark_used(pool[pool >= 0])

    def leaves(self) -> list:
        return [self.regions[k] for k in sorted(self.regions) if not self.regions[k].children]

    def _rng(self, rid: int, stage: int):
        return np.random.default_rng(derive_seed(self.cfg.seed, self.iteration, rid, stage))

    def _take(self, region, handles) -> list:
        if len(handles) == 0:
            return []
        ids = self.store.add(*self.sampler.take(handles))
        region.samples.extend(ids)
        self.log.extend(ids)
        self.used += len(ids)
        if self.on_sample is not None:
            self.on_sample(self)
        return ids

    def fit(self, region):
        n = len(region.samples)
        cached = self.model_cache.get(region.rid)
        if cached is not None and cached[0] == n:
            return cached[1]
        model = self.model_factory(derive_seed(self.cfg.seed, region.rid, n))
        model.fit(self.store.X(region.samples), self.store.y(region.samples))
        self.n_fits += 1
        self.n_capped += int(bool(getattr(model, "capped", False)))
        self.n_at_bound += int(bool(getattr(model, "at_bound", False)))
        self.model_cache[region.rid] = (n, model)
        return model

    def _classify(self, region, rng) -> None:
        q = mc_step(self.fit(region), region.low, region.high, self.cfg.R, self.cfg.M,
                    self.cfg.delta_c, rng)
        region.rtype = classify(region.rtype, region.volume(), self.min_volume, q,
                                self.cfg.delta_c, self.cfg.compat_ci_var)

    def _file(self, region, remaining: list, classified: list) -> None:
        if region.rtype in REMAINING:
            remaining.append(region.rid)
        elif region.rtype in CLASSIFIED:
            classified.append(region.rid)
        elif region.rtype == "u":
            self.unclassified.append(region.rid)
        elif region.rtype == "i":
            self.infeasible.append(region.rid)
        else:
            raise ValueError(f"unknown region type {region.rtype!r}")

    def _sample_unclassified(self, region) -> None:
        """cpslab samples_management_unclassified: SampleBO, then MCstep and Classify."""
        rng = self._rng(region.rid, _UNCLASSIFIED)
        res = sample_bo(region, self.store, self.sampler, self.fit,
                        lambda h: self._take(region, h), self.cfg.n0, self.cfg.n_bo,
                        self.cfg.ei_cand_cap, rng, self.cfg.ei_f_star)
        self.shortfall += res["shortfall"]
        if res["infeasible"] or not region.samples:
            region.rtype = "i"
            return
        self._classify(region, rng)

    def _sample_more(self, region, count: int, rng) -> None:
        """cpslab samples_management_classified: `count` LHS points, then reclassify."""
        if self.sampler.n_available(region) == 0:
            self.shortfall += count
            region.rtype = "i"
            return
        handles = self.sampler.snap_design(lhs(count, region.low, region.high, rng), region)
        self.shortfall += count - len(handles)
        self._take(region, handles)
        self._classify(region, rng)

    # ---- Alg. 4 -------------------------------------------------------------
    def run_root(self) -> None:
        if self.phase != "root":
            raise RuntimeError("run_root was already called")
        root = self.regions[0]
        self._sample_unclassified(root)
        self._file(root, self.remaining, self.classified)
        self.phase = "main"

    def step(self) -> None:
        if self.done:
            return
        if self.phase == "root":
            raise RuntimeError("call run_root first")
        left = self.budget - self.used
        if left <= 0 or not (self.remaining or self.classified):
            self.done = True
            return
        self.iteration += 1
        cfg = self.cfg
        plan, need = [], 0
        for rid in sorted(self.remaining):
            parent = self.regions[rid]
            dim = int(self.perm[parent.branch_dir % self.d])
            bounds = parent.split_bounds(dim, cfg.B)
            masks = child_masks(self.store.X(parent.samples)[:, dim], bounds, dim)
            parts = [[s for s, keep in zip(parent.samples, m) if keep] for m in masks]
            need += sum(max(cfg.n0 - len(p), 0) + cfg.n_bo for p in parts)
            plan.append((parent, dim, bounds, parts))
        if self.remaining and need <= left:
            self._branch(plan)
            self._continued_sampling()
        else:
            self._final_phase()

    def _branch(self, plan) -> None:
        self.phase = "branch"
        children = []
        for parent, dim, bounds, parts in plan:
            kids = []
            for (lo, hi), ids in zip(bounds, parts):
                kid = Region(rid=self.next_rid, parent=parent.rid, low=lo, high=hi,
                             branch_dir=parent.branch_dir + 1, depth=parent.depth + 1,
                             rtype="r", samples=list(ids))
                self.regions[kid.rid] = kid
                self.next_rid += 1
                kids.append(kid)
            parent.children = [k.rid for k in kids]
            self.sampler.split(parent, kids, dim)
            children.extend(kids)
        remaining = []
        for kid in children:                       # ascending rid (spec §6.4)
            self._sample_unclassified(kid)
            self._file(kid, remaining, self.classified)
        self.remaining = remaining

    def _continued_sampling(self) -> None:
        if not self.classified:
            return
        self.phase = "continued"
        cfg = self.cfg
        rids = sorted(self.classified)
        weights = []
        for rid in rids:
            r = self.regions[rid]
            weights.append(falsification_indicator(self.fit(r), r.low, r.high, cfg.R, cfg.M,
                                                   self._rng(rid, _INDICATOR),
                                                   times_volume=cfg.compat_eq4_times_volume))
        counts = multinomial_counts(weights, min(cfg.n_c, self.budget - self.used),
                                    self._rng(_ITERATION, _CONTINUED))
        classified = []
        for rid, cnt in zip(rids, counts):
            r = self.regions[rid]
            if cnt > 0:
                self._sample_more(r, int(cnt), self._rng(rid, _CONTINUED))
            self._file(r, self.remaining, classified)
        self.classified = classified

    def _final_phase(self) -> None:
        self.phase = "final"
        rids = sorted(self.remaining + self.classified)
        counts = multinomial_counts([self.regions[r].volume() for r in rids],
                                    self.budget - self.used, self._rng(_ITERATION, _FINAL))
        remaining, classified = [], []
        for rid, cnt in zip(rids, counts):
            r = self.regions[rid]
            if cnt > 0:
                self._sample_more(r, int(cnt), self._rng(rid, _FINAL))
            self._file(r, remaining, classified)
        self.remaining, self.classified = remaining, classified

    def run(self) -> None:
        if self.phase == "root":
            self.run_root()
        while not self.done:
            self.step()
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_algorithm.py -q`
Expected: `11 passed`. If `test_a_straight_boundary_gets_classified_on_the_right_sides` finds no '+' or '−' leaves, print `Counter(r.rtype for r in px.leaves())` and `px.iteration` before touching thresholds. With budget 400 the tree should reach depth 4 or more.

- [ ] **Step 6: Run the whole suite so far, then commit**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful -q`
Expected: all tests pass.

```bash
git add adaptive_roa/partx_faithful/algorithm.py adaptive_roa/partx_faithful/state.py tests/partx_faithful/test_algorithm.py
git commit -m "feat(partx_faithful): Alg. 4 state machine with exact replay and atomic state"
```

---
### Task 7: Volume estimates and the piecewise readout

**Files:**
- Create: `adaptive_roa/partx_faithful/volume.py`
- Create: `adaptive_roa/partx_faithful/readout.py`
- Test: `tests/partx_faithful/test_volume_readout.py`

**Interfaces:**
- Consumes: `Region`, `lhs` (Task 3).
- Produces:
  - `classified_falsification_volume(leaves, root_volume) -> float`, computed as v(Θ⁻)/v(S).
  - `remaining_plus_violating_volume(leaves, root_volume) -> float`. This is the paper's "P-X" row: the volume of r, r+, r− and − leaves.
  - `gp_quantile_falsification_volume(leaves, models, R, M, quantiles, rng, root_volume) -> {q: fraction}`, as in cpslab `fv_using_gp`.
  - `PiecewiseModel(leaves, models, root_low, root_high)` with `.p_success(X)`, where `models` maps region id to an object with `p_success`.
  - `PiecewiseHandle(piecewise)` with `.eval()`, `.to(device)` and `__call__(states) -> logits tensor`.

- [ ] **Step 1: Write the failing tests**

`tests/partx_faithful/test_volume_readout.py`:

```python
import numpy as np
import pytest
import torch

from adaptive_roa.partx_faithful.readout import PiecewiseHandle, PiecewiseModel
from adaptive_roa.partx_faithful.region import Region
from adaptive_roa.partx_faithful.volume import (classified_falsification_volume,
                                                gp_quantile_falsification_volume,
                                                remaining_plus_violating_volume)


def _leaf(rid, low, high, rtype):
    return Region(rid=rid, parent=-1, low=np.array(low, float), high=np.array(high, float),
                  branch_dir=0, depth=0, rtype=rtype)


class _G:
    def __init__(self, m, s=0.01, p=0.5):
        self.m, self.s, self.p = m, s, p

    def latent(self, X):
        return np.full(len(X), self.m), np.full(len(X), self.s)

    def p_success(self, X):
        return np.full(len(np.atleast_2d(X)), self.p)


def test_classified_and_px_volumes():
    leaves = [_leaf(1, [0, 0], [0.5, 1], "-"), _leaf(2, [0.5, 0], [0.75, 1], "r"),
              _leaf(3, [0.75, 0], [1, 1], "+")]
    assert classified_falsification_volume(leaves, 1.0) == pytest.approx(0.5)
    assert remaining_plus_violating_volume(leaves, 1.0) == pytest.approx(0.75)


def test_gp_quantile_volume_counts_confident_negatives():
    leaves = [_leaf(1, [0, 0], [0.5, 1], "-"), _leaf(2, [0.5, 0], [1, 1], "+"),
              _leaf(3, [0, 0], [0, 0], "u")]
    models = {1: _G(-1.0), 2: _G(1.0), 3: _G(-1.0)}
    fv = gp_quantile_falsification_volume(leaves, models, 2, 20, [0.5, 0.05], np.random.default_rng(0), 1.0)
    assert fv[0.5] == pytest.approx(0.5) and fv[0.05] == pytest.approx(0.5)


def test_readout_uses_each_leafs_model_and_clips_outside_points():
    leaves = [_leaf(1, [0, 0], [0.5, 1], "-"), _leaf(2, [0.5, 0], [1, 1], "+")]
    pw = PiecewiseModel(leaves, {1: _G(0, p=0.2), 2: _G(0, p=0.9)}, np.zeros(2), np.ones(2))
    np.testing.assert_allclose(pw.p_success(np.array([[0.25, 0.5], [0.75, 0.5], [5.0, 0.5]])),
                               [0.2, 0.9, 0.9])


def test_handle_returns_logits_that_sigmoid_back_to_p():
    leaves = [_leaf(1, [0, 0], [1, 1], "r")]
    handle = PiecewiseHandle(PiecewiseModel(leaves, {1: _G(0, p=0.3)}, np.zeros(2), np.ones(2)))
    out = handle.eval().to("cpu")(torch.tensor([[0.5, 0.5]], dtype=torch.float32))
    assert torch.sigmoid(out).item() == pytest.approx(0.3, abs=1e-6)
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_volume_readout.py -q`
Expected: `ModuleNotFoundError` for `adaptive_roa.partx_faithful.readout`.

- [ ] **Step 3: Implement `volume.py`**

```python
"""Falsification-volume estimates (spec §3 "Output"; cpslab fv_without_gp / fv_using_gp)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from adaptive_roa.partx_faithful.sampling import lhs


def classified_falsification_volume(leaves, root_volume: float) -> float:
    """v(Theta^-)/v(S), the return value of Alg. 4."""
    return float(sum(r.volume() for r in leaves if r.rtype == "-") / root_volume)


def remaining_plus_violating_volume(leaves, root_volume: float) -> float:
    """The paper's "P-X" row in Tables 1-3: r, r+, r- and - leaves."""
    return float(sum(r.volume() for r in leaves if r.rtype in ("r", "r+", "r-", "-")) / root_volume)


def gp_quantile_falsification_volume(leaves, models: dict, R: int, M: int, quantiles, rng,
                                     root_volume: float) -> dict:
    """cpslab fv_using_gp: per leaf (skipping u and i), the share of R*M LHS points
    whose q-quantile of g is below 0, times the leaf volume, summed over leaves."""
    out = {float(q): 0.0 for q in quantiles}
    for r in leaves:
        model = models.get(r.rid)
        if r.rtype in ("u", "i") or model is None:
            continue
        X = np.vstack([lhs(M, r.low, r.high, rng) for _ in range(R)])
        m, s = model.latent(X)
        for q in out:
            out[q] += float(np.mean(norm.ppf(q, m, s) < 0.0)) * r.volume()
    return {q: v / root_volume for q, v in out.items()}
```

- [ ] **Step 4: Implement `readout.py`**

```python
"""Part-X's scored readout (spec §5.4) in the evaluator's classifier contract.

evaluate_full_roa_classifier calls model(raw_states_tensor) -> logits and takes
sigmoid(logits) as p_success, the same contract adaptive_roa/partx/model_handle.py
satisfies.
"""
from __future__ import annotations

import numpy as np
import torch


class PiecewiseModel:
    def __init__(self, leaves, models: dict, root_low, root_high):
        self.leaves = list(leaves)
        self.models = models
        self.low = np.asarray(root_low, dtype=float)
        self.high = np.asarray(root_high, dtype=float)

    def p_success(self, X) -> np.ndarray:
        X = np.clip(np.atleast_2d(np.asarray(X, dtype=float)), self.low, self.high)
        p = np.full(len(X), np.nan)
        todo = np.ones(len(X), dtype=bool)
        for leaf in self.leaves:
            m = todo & leaf.contains(X)
            if m.any():
                p[m] = self.models[leaf.rid].p_success(X[m])
                todo &= ~m
        if todo.any():
            raise RuntimeError(f"readout: {int(todo.sum())} points fall in no leaf")
        return p


class PiecewiseHandle:
    def __init__(self, piecewise: PiecewiseModel):
        self.pw = piecewise

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, states):
        X = states.detach().cpu().numpy() if torch.is_tensor(states) else np.asarray(states)
        p = np.clip(self.pw.p_success(X), 1e-12, 1.0 - 1e-12)
        device = states.device if torch.is_tensor(states) else "cpu"
        return torch.as_tensor(np.log(p / (1.0 - p)), dtype=torch.float32, device=device)
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_volume_readout.py -q`
Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/partx_faithful/volume.py adaptive_roa/partx_faithful/readout.py tests/partx_faithful/test_volume_readout.py
git commit -m "feat(partx_faithful): falsification volumes and the piecewise readout"
```

---

### Task 8: The runner, checkpoints and resume

**Files:**
- Create: `adaptive_roa/partx_faithful/runner.py`
- Test: `tests/partx_faithful/test_runner.py`

**Interfaces:**
- Consumes: everything above, plus:
  - `adaptive_roa.partx.tree.build_root`, read-only.
  - `adaptive_roa.adaptive_v2.eval.full_roa.evaluate_full_roa_classifier`.
  - `adaptive_roa.adaptive_v2.engine._atomic_write_json` and `_convert_numpy`.
- Produces:
  - `schedule(initial, samples_per_epoch, n_epochs) -> (budget, checkpoints)`.
  - `resolve_n0(n0, state_dim) -> int`.
  - `CheckpointWriter`.
  - `run_partx_faithful(*, system, starts, labels, eval_states_file, output_dir, initial, samples_per_epoch, n_epochs, pcfg, gp_kwargs, falsify, attractor_radius, extra) -> dict`.
  - `run_from_cfg(cfg) -> dict`, used by Task 9.
  - Each epoch directory holds `artifacts_v2.json`, `full_roa_per_point.npz` and `partx_leaves.npz`. The leaves file has keys `low`, `high`, `rtype`, `rid`, `n_samples`, `root_low` and `root_high`.
  - The run directory also holds `partx_state.pkl` and `final_results.json`.

- [ ] **Step 1: Write the failing tests**

`tests/partx_faithful/test_runner.py`:

```python
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from adaptive_roa.partx.tree import build_root
from adaptive_roa.partx_faithful.algorithm import PartX, PartXConfig
from adaptive_roa.partx_faithful.runner import resolve_n0, run_partx_faithful, schedule
from adaptive_roa.systems.pendulum import PendulumSystem

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
spm = importlib.import_module("stoch_prob_metrics")


def _p_true(X):
    return 1.0 / (1.0 + np.exp(-4.0 * (1.0 - np.abs(X[:, 0]))))


@pytest.fixture
def toy(tmp_path):
    system = PendulumSystem()
    root = build_root(system)
    rng = np.random.default_rng(0)
    starts = rng.uniform(root.low, root.high, size=(3000, 2))
    labels = np.where(rng.random(3000) < _p_true(starts), 1.0, -1.0)
    g0, g1 = np.meshgrid(np.linspace(-3.0, 3.0, 15), np.linspace(-6.0, 6.0, 15))
    grid = np.round(np.column_stack([g0.ravel(), g1.ravel()]), 6)
    p = _p_true(grid)
    droot = tmp_path / "data"
    droot.mkdir()
    np.savetxt(droot / "test_set.txt", np.column_stack([grid, p]), delimiter=",", fmt="%.6f")
    np.savez(droot / "eval_success_prob.npz", starts=grid, p_success=p,
             successes=np.round(50 * p), trials=np.full(len(p), 50.0))
    return system, starts, labels, droot


def _go(toy, out):
    system, starts, labels, droot = toy
    return run_partx_faithful(
        system=system, starts=starts, labels=labels, eval_states_file=str(droot / "test_set.txt"),
        output_dir=str(out), initial=60, samples_per_epoch=30, n_epochs=4,
        pcfg=PartXConfig(n0=5, n_bo=2, n_c=10, R=4, M=50, seed=0),
        gp_kwargs=dict(n_restarts=1, max_points=500), falsify="failure",
        attractor_radius=0.1, extra={"test": True})


def test_schedule_and_n0():
    assert schedule(100, 100, 20) == (1900, [100 + 100 * e for e in range(20)])
    assert resolve_n0("auto", 2) == 10 and resolve_n0("auto", 13) == 30
    assert resolve_n0(None, 4) == 30 and resolve_n0(30, 2) == 30


def test_every_epoch_is_written_and_scoreable(toy, tmp_path):
    out = tmp_path / "run"
    _go(toy, out)
    tt, d2 = [], []
    for e in range(4):
        d = out / f"epoch_{e:03d}"
        for f in ("full_roa_per_point.npz", "artifacts_v2.json", "partx_leaves.npz"):
            assert (d / f).exists(), f"{d.name}/{f}"
        a = json.loads((d / "artifacts_v2.json").read_text())
        assert a["sampling_mode"] == "partx_faithful"
        tt.append(a["train_trajectories"])
        d2.append(a["acquisition"]["d2_indices"])
    assert tt == [60, 90, 120, 150]
    assert [len(x) for x in d2] == [30, 30, 30, 0]
    flat = [i for x in d2 for i in x]
    assert len(set(flat)) == 90 and min(flat) >= 60          # never the initial set
    with np.load(out / "epoch_003" / "full_roa_per_point.npz") as z:
        assert np.all((z["p_success"] > 0) & (z["p_success"] < 1))
    row, _ = spm.score_epoch_full(out / "epoch_003", spm.load_ground_truth(toy[3]), None, None)
    assert np.isfinite(row["KL"]) and row["train_trajectories"] == 150
    assert json.loads((out / "final_results.json").read_text())["used"] == 90


def test_a_crash_and_resume_matches_an_uninterrupted_run(toy, tmp_path, monkeypatch):
    _go(toy, tmp_path / "a")
    real_step = PartX.step
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated preemption")
        return real_step(self)

    monkeypatch.setattr(PartX, "step", flaky)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        _go(toy, tmp_path / "b")
    monkeypatch.setattr(PartX, "step", real_step)
    _go(toy, tmp_path / "b")
    for e in range(4):
        da, db = tmp_path / "a" / f"epoch_{e:03d}", tmp_path / "b" / f"epoch_{e:03d}"
        with np.load(da / "full_roa_per_point.npz") as za, np.load(db / "full_roa_per_point.npz") as zb:
            np.testing.assert_array_equal(za["p_success"], zb["p_success"])
        ja = json.loads((da / "artifacts_v2.json").read_text())
        jb = json.loads((db / "artifacts_v2.json").read_text())
        assert ja["acquisition"]["d2_indices"] == jb["acquisition"]["d2_indices"]
        assert ja["train_trajectories"] == jb["train_trajectories"]
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_runner.py -q`
Expected: `ModuleNotFoundError` for `adaptive_roa.partx_faithful.runner`.

- [ ] **Step 3: Implement**

`adaptive_roa/partx_faithful/runner.py`:

```python
"""Standalone Part-X runner (spec §6).

Runs Alg. 4 over the trajectory pool and writes engine-format epoch directories
whenever the sample count reaches initial + e * samples_per_epoch. The snapshot
for epoch e is the current leaf partition, with each leaf's GP fit on the
initial set plus the first (b_e - initial) entries of the acquisition log
(spec §6.4). Checkpoints are written the moment the count crosses them. A run
can be killed at any point and restarted with the same command.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from adaptive_roa.adaptive_v2.engine import _atomic_write_json, _convert_numpy
from adaptive_roa.adaptive_v2.eval.full_roa import evaluate_full_roa_classifier
from adaptive_roa.partx.tree import build_root
from adaptive_roa.partx_faithful.algorithm import PartX, PartXConfig, derive_seed
from adaptive_roa.partx_faithful.backends import backend_factory
from adaptive_roa.partx_faithful.readout import PiecewiseHandle, PiecewiseModel
from adaptive_roa.partx_faithful.region import RTYPES
from adaptive_roa.partx_faithful.sampling import PoolSampler
from adaptive_roa.partx_faithful.state import config_hash, load_state, save_state
from adaptive_roa.partx_faithful.volume import (classified_falsification_volume,
                                                gp_quantile_falsification_volume,
                                                remaining_plus_violating_volume)

FV_QUANTILES = (0.5, 0.05, 0.01)
_SNAPSHOT_STAGE = 7        # seed slot for readout fits, separate from the algorithm's


def schedule(initial: int, samples_per_epoch: int, n_epochs: int):
    """T = (n_epochs - 1) * samples_per_epoch new evaluations; one checkpoint per epoch."""
    budget = (int(n_epochs) - 1) * int(samples_per_epoch)
    return budget, [int(initial) + e * int(samples_per_epoch) for e in range(int(n_epochs))]


def resolve_n0(n0, state_dim: int) -> int:
    """Spec §6.5: the paper's 2-D setting (10) for d = 2, the F16 setting (30) otherwise."""
    if n0 is None or n0 == "auto":
        return 10 if int(state_dim) == 2 else 30
    return int(n0)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class CheckpointWriter:
    def __init__(self, *, output_dir, system, eval_states_file, attractor_radius, checkpoints,
                 initial, model_factory, pcfg, extra):
        self.out = Path(output_dir)
        self.system, self.eval_states_file = system, str(eval_states_file)
        self.attractor_radius = float(attractor_radius)
        self.checkpoints, self.initial = list(checkpoints), int(initial)
        self.model_factory, self.pcfg, self.extra = model_factory, pcfg, extra
        self.next_e = 0
        self._cache: dict = {}

    def epoch_dir(self, e: int) -> Path:
        return self.out / f"epoch_{e:03d}"

    def complete(self, e: int) -> bool:
        d = self.epoch_dir(e)
        return (d / "artifacts_v2.json").exists() and (d / "full_roa_per_point.npz").exists()

    def n_new(self, e: int) -> int:
        return self.checkpoints[e] - self.initial

    def on_sample(self, px) -> None:
        """Write every checkpoint the sample count has reached (called after each batch)."""
        while self.next_e < len(self.checkpoints) and self.initial + px.used >= self.checkpoints[self.next_e]:
            e = self.next_e
            self.next_e += 1
            if not self.complete(e):
                self.write(px, e)
            else:
                self._fill_prev_d2(px, e)      # a crash may have landed between the two writes

    def finish(self, px) -> None:
        """After termination: epochs Part-X never reached carry the final state (D8)."""
        for e in range(self.next_e, len(self.checkpoints)):
            if not self.complete(e):
                self.write(px, e)
        self.next_e = len(self.checkpoints)

    def _fill_prev_d2(self, px, e: int) -> None:
        """Match the other arms: epoch e-1 records the pool indices bought between e-1 and e."""
        if e == 0 or not self.complete(e - 1):
            return
        prev = self.epoch_dir(e - 1) / "artifacts_v2.json"
        a = json.loads(prev.read_text())
        lo, hi = min(self.n_new(e - 1), px.used), min(self.n_new(e), px.used)
        d2 = [int(i) for i in px.store.pool_idx(px.log[lo:hi])]
        if a["acquisition"].get("d2_indices") != d2:
            a["acquisition"]["d2_indices"] = d2
            _atomic_write_json(prev, a, indent=2)

    def _models(self, px, leaves, allowed: set) -> dict:
        models = {}
        for leaf in leaves:
            node = leaf
            ids = [s for s in node.samples if s in allowed]
            while not ids and node.parent >= 0:
                node = px.regions[node.parent]
                ids = [s for s in node.samples if s in allowed]
            if not ids:
                raise RuntimeError(f"leaf {leaf.rid}: no samples on the path to the root")
            key = (node.rid, len(ids))
            if key not in self._cache:
                model = self.model_factory(derive_seed(self.pcfg.seed, _SNAPSHOT_STAGE, node.rid, len(ids)))
                model.fit(px.store.X(ids), px.store.y(ids))
                self._cache = {k: v for k, v in self._cache.items() if k[0] != node.rid}
                self._cache[key] = model
            models[leaf.rid] = self._cache[key]
        return models

    def write(self, px, e: int) -> None:
        n_new = min(self.n_new(e), px.used)
        allowed = set(range(px.n_init)) | set(px.log[:n_new])
        leaves = px.leaves()
        models = self._models(px, leaves, allowed)
        root = px.regions[0]
        d = self.epoch_dir(e)
        d.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_full_roa_classifier(
            classifier=PiecewiseHandle(PiecewiseModel(leaves, models, root.low, root.high)),
            system=self.system, eval_states_file=self.eval_states_file, lambda_star=0.5, delta=0.1,
            attractor_radius=self.attractor_radius, device="cpu", output_dir=str(d), verbose=False,
            decision_rule="one_sided")
        np.savez(d / "partx_leaves.npz",
                 low=np.array([r.low for r in leaves]), high=np.array([r.high for r in leaves]),
                 rtype=np.array([r.rtype for r in leaves]), rid=np.array([r.rid for r in leaves]),
                 n_samples=np.array([sum(s in allowed for s in r.samples) for r in leaves]),
                 root_low=root.low, root_high=root.high)
        types = Counter(r.rtype for r in leaves)
        fv = gp_quantile_falsification_volume(
            leaves, models, self.pcfg.R, self.pcfg.M, FV_QUANTILES,
            np.random.default_rng(derive_seed(self.pcfg.seed, _SNAPSHOT_STAGE, e)), px.root_volume)
        diag = {
            "iteration": px.iteration, "phase": px.phase, "n_leaves": len(leaves),
            "leaf_types": {t: int(types.get(t, 0)) for t in RTYPES},
            "falsification_volume_classified": classified_falsification_volume(leaves, px.root_volume),
            "falsification_volume_px": remaining_plus_violating_volume(leaves, px.root_volume),
            "falsification_volume_gp": {str(q): v for q, v in fv.items()},
            "budget": px.budget, "used": int(n_new),
            "unspent": int(px.budget - px.used) if px.done else 0,
            "shortfall": int(px.shortfall), "n_fits": int(px.n_fits),
            "n_capped_fits": int(px.n_capped), "n_at_bound_fits": int(px.n_at_bound),
            "done": bool(px.done),
        }
        art = {
            "epoch": int(e), "train_trajectories": int(self.initial + n_new),
            "sampling_mode": "partx_faithful",
            "threshold_state": {"lambda_star": 0.5, "delta_star": 0.1, "q_hat": None, "q_hat_eval": None,
                                "q_hat_success_eval": None, "q_hat_failure_eval": None},
            "acquisition": {"d1_indices": [], "d2_indices": [], "n_candidates_evaluated": None,
                            "n_certain_discarded": 0, "n_invalid_added": 0, "diagnostics": diag},
            "endpoint_error": {"skipped": "partx_faithful predicts no endpoints"},
            "eval_metrics": metrics, "d1_eval_metrics": None, "conformal_state": None,
            "extra": self.extra,
        }
        _atomic_write_json(d / "artifacts_v2.json", _convert_numpy(art), indent=2)
        self._fill_prev_d2(px, e)


def run_partx_faithful(*, system, starts, labels, eval_states_file, output_dir, initial: int,
                       samples_per_epoch: int, n_epochs: int, pcfg: PartXConfig, gp_kwargs: dict,
                       falsify: str, attractor_radius: float, extra: dict) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    starts = np.asarray(starts, dtype=float)
    labels = np.asarray(labels, dtype=float)
    budget, checkpoints = schedule(initial, samples_per_epoch, n_epochs)
    root = build_root(system)
    used = np.zeros(len(starts), dtype=bool)
    used[:initial] = True
    sampler = PoolSampler(starts, labels, used, scale=root.high - root.low)
    factory = backend_factory("gpc", falsify=falsify, **gp_kwargs)
    chash = config_hash({"pcfg": asdict(pcfg), "gp": gp_kwargs, "falsify": falsify, "initial": initial,
                         "samples_per_epoch": samples_per_epoch, "n_epochs": n_epochs,
                         "n_pool": len(starts)})
    state_path = out / "partx_state.pkl"
    if state_path.exists():
        px = load_state(state_path, chash)
        px.attach(sampler, factory)
        print(f"[partx_faithful] resuming at iteration {px.iteration}, used {px.used}/{px.budget}", flush=True)
    else:
        px = PartX(pcfg, root.low, root.high, sampler, factory, budget, init_X=starts[:initial],
                   init_y=labels[:initial], init_pool_idx=np.arange(initial))
    writer = CheckpointWriter(output_dir=out, system=system, eval_states_file=eval_states_file,
                              attractor_radius=attractor_radius, checkpoints=checkpoints,
                              initial=initial, model_factory=factory, pcfg=pcfg,
                              extra={**extra, "commit": _git_commit()})
    px.on_sample = writer.on_sample
    writer.on_sample(px)                     # epoch 0, and anything a crash left unwritten
    t0 = time.time()
    if px.phase == "root":
        px.run_root()
        save_state(px, state_path, chash)
    while not px.done:
        px.step()
        save_state(px, state_path, chash)
        if px.done:
            break
        leaves = px.leaves()
        types = Counter(r.rtype for r in leaves)
        print(f"[partx_faithful] iter {px.iteration} phase={px.phase} used={px.used}/{px.budget} "
              f"leaves={len(leaves)} " + " ".join(f"{t}={types[t]}" for t in RTYPES if types.get(t))
              + f" fits={px.n_fits} capped={px.n_capped} elapsed={time.time() - t0:.0f}s", flush=True)
    writer.finish(px)
    leaves = px.leaves()
    summary = {"budget": px.budget, "used": px.used, "unspent": px.budget - px.used,
               "iterations": px.iteration, "n_leaves": len(leaves),
               "leaf_types": dict(Counter(r.rtype for r in leaves)), "shortfall": px.shortfall,
               "n_fits": px.n_fits, "n_capped_fits": px.n_capped, "n_at_bound_fits": px.n_at_bound,
               "elapsed_s_this_process": time.time() - t0}
    _atomic_write_json(out / "final_results.json", _convert_numpy(summary), indent=2)
    return summary


def run_from_cfg(cfg) -> dict:
    """Build the inputs from a composed configs/adaptive_v2/partx_faithful.yaml."""
    import hydra
    from omegaconf import OmegaConf

    from adaptive_roa.adaptive.data_source import TrajectoryDataSourceConfig
    from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource

    if str(cfg.data_source.get("pool_format", "text")) != "npz":
        raise ValueError("partx_faithful needs an npz pool (data_source.pool_format=npz)")
    system = hydra.utils.instantiate(cfg.system)
    ds = NpzTrajectoryDataSource(TrajectoryDataSourceConfig(
        trajectories_dir=cfg.data_source.trajectories_dir,
        shuffled_indices_file=cfg.data_source.shuffled_indices_file,
        shuffled_labels_file=cfg.data_source.get("shuffled_labels_file", None),
        eval_states_file=cfg.data_source.get("eval_states_file", None)))
    if ds.labels is None:
        raise ValueError("the pool has no shuffled_labels_file; Part-X needs outcomes")
    starts = np.asarray(ds.start_states, dtype=float)
    alg = OmegaConf.to_container(cfg.partx_faithful.algorithm, resolve=True)
    alg["n0"] = resolve_n0(alg.get("n0"), starts.shape[1])
    alg["seed"] = int(cfg.get("seed", 42))
    pcfg = PartXConfig(**alg)
    gp = OmegaConf.to_container(cfg.partx_faithful.gp, resolve=True)
    gp_kwargs = dict(n_restarts=int(gp["n_restarts"]), max_points=int(gp["max_points"]),
                     lengthscale_bounds=tuple(gp["lengthscale_bounds"]),
                     sigma_f2_bounds=tuple(gp["sigma_f2_bounds"]))
    extra = {"partx_faithful": OmegaConf.to_container(cfg.partx_faithful, resolve=True),
             "n0_resolved": pcfg.n0}
    return run_partx_faithful(
        system=system, starts=starts, labels=np.asarray(ds.labels, dtype=float),
        eval_states_file=cfg.data_source.test_set_file, output_dir=cfg.output_dir,
        initial=int(cfg.initial_train_size), samples_per_epoch=int(cfg.samples_per_epoch),
        n_epochs=int(cfg.n_epochs), pcfg=pcfg, gp_kwargs=gp_kwargs,
        falsify=str(cfg.partx_faithful.falsify), attractor_radius=float(cfg.attractor_radius),
        extra=extra)
```

Note: the engine scores against `data_source.test_set_file`, not `eval_states_file` (`engine.py:338`). The runner does the same, so its eval states match the scorer's ground-truth grid.

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_runner.py -q`
Expected: `3 passed`. If `score_epoch_full` raises "eval states do not match the ground-truth grid", check that the fixture's `test_set.txt` and `eval_success_prob.npz` hold the same rounded grid.

- [ ] **Step 5: Run the full suite, then commit**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful -q`
Expected: all tests pass.

```bash
git add adaptive_roa/partx_faithful/runner.py tests/partx_faithful/test_runner.py
git commit -m "feat(partx_faithful): resumable runner writing engine-format checkpoints"
```

---

### Task 9: Hydra config, entry point and a real-data smoke run

**Files:**
- Create: `configs/adaptive_v2/partx_faithful.yaml`
- Create: `scripts/run_partx_faithful.py`
- Test: `tests/partx_faithful/test_config.py`

**Interfaces:**
- Consumes: `run_from_cfg` (Task 8).
- Produces: the launch command `scripts/run_partx_faithful.py <overrides>`. It takes the same overrides as `run_adaptive.py`, plus `partx_faithful.*`.

- [ ] **Step 1: Write the failing test**

`tests/partx_faithful/test_config.py`:

```python
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from adaptive_roa.utils.env_config import get_data_dir


def test_config_composes_with_the_engine_overrides():
    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
    cfgdir = str(Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2")
    with initialize_config_dir(config_dir=cfgdir, version_base=None):
        cfg = compose(config_name="partx_faithful",
                      overrides=["system=pendulum_stoch", "noise_level=low", "output_dir=/tmp/unused"])
    a = cfg.partx_faithful.algorithm
    assert a.n0 == "auto" and a.n_bo == 10 and a.n_c == 100 and a.R == 20 and a.M == 500
    assert a.B == 2 and a.delta_c == 0.05 and a.delta_v == 0.001 and a.ei_cand_cap == 20000
    assert not a.compat_ci_var and not a.compat_eq4_times_volume and a.ei_f_star == "plugin"
    g = cfg.partx_faithful.gp
    assert g.n_restarts == 5 and g.max_points == 2000
    assert list(g.lengthscale_bounds) == [0.01, 100.0] and list(g.sigma_f2_bounds) == [0.01, 25.0]
    assert cfg.partx_faithful.falsify == "failure"
    assert cfg.data_source.test_set_file.endswith("pendulum/gaussian_signal/lqr/low/test_set.txt")
    assert str(cfg.data_source.pool_format) == "npz"
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_config.py -q`
Expected: FAIL with a Hydra `MissingConfigException` for `partx_faithful`.

- [ ] **Step 3: Write the config**

`configs/adaptive_v2/partx_faithful.yaml`:

```yaml
# Faithful Part-X runner config (spec docs/superpowers/specs/2026-09-11-partx-faithful-stochastic-design.md).
# Composes `default` so system / controller / noise_family / noise_level overrides
# resolve the data paths exactly as run_adaptive.py does. The predictor,
# acquisition and eval groups that `default` pulls in are unused here.
defaults:
  - default
  - _self_

partx_faithful:
  algorithm:
    n0: auto             # spec §6.5: 10 when the state is 2-D, else 30
    n_bo: 10
    n_c: 100
    R: 20
    M: 500
    B: 2
    delta_c: 0.05
    delta_v: 0.001
    ei_cand_cap: 20000
    ei_f_star: plugin
    compat_ci_var: false             # reference check only (spec §4 row 1)
    compat_eq4_times_volume: false   # reference check only (spec §4 row 2)
  gp:
    n_restarts: 5
    max_points: 2000
    lengthscale_bounds: [0.01, 100.0]
    sigma_f2_bounds: [0.01, 25.0]
  falsify: failure       # D2; `success` exists for a later sensitivity run only
```

- [ ] **Step 4: Write the entry point**

`scripts/run_partx_faithful.py`:

```python
"""Faithful Part-X runner (spec docs/superpowers/specs/2026-09-11-partx-faithful-stochastic-design.md).

Takes the same overrides as run_adaptive.py, e.g.

    python scripts/run_partx_faithful.py system=pendulum_stoch noise_level=low \
        initial_train_size=100 samples_per_epoch=100 n_epochs=20 seed=42 \
        output_dir=/common/users/shared/pracsys/adaptive_roa_experiments/gaussian_torque/pen_low_partx_faithful

Re-running the same command resumes from output_dir/partx_state.pkl.
"""
from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from adaptive_roa.utils.env_config import (
    get_data_dir,
    get_env_config,
    get_exp_dir,
    get_net_id,
    get_shared_data_base,
)

# Same resolvers as scripts/run_adaptive.py, so every shared config resolves.
if not OmegaConf.has_resolver("net_id"):
    OmegaConf.register_new_resolver("net_id", lambda: get_net_id())
if not OmegaConf.has_resolver("exp_dir"):
    OmegaConf.register_new_resolver("exp_dir", lambda: get_exp_dir())
if not OmegaConf.has_resolver("data_dir"):
    OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
if not OmegaConf.has_resolver("shared_data_base"):
    OmegaConf.register_new_resolver("shared_data_base", lambda default="": get_shared_data_base() or default)
if not OmegaConf.has_resolver("env"):
    OmegaConf.register_new_resolver(
        "env", lambda key, default="": os.environ.get(key, get_env_config().get(key, default)))


@hydra.main(config_path="../configs/adaptive_v2", config_name="partx_faithful", version_base=None)
def main(cfg: DictConfig):
    from adaptive_roa.partx_faithful.runner import run_from_cfg

    print(OmegaConf.to_yaml(cfg.partx_faithful))
    run_from_cfg(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the config test to confirm it passes**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_config.py -q`
Expected: `1 passed`.

- [ ] **Step 6: Smoke-run on the real pendulum pool with a tiny schedule**

Write to the session scratchpad, not the shared experiment tree (`$SCRATCH` is that directory):

```bash
PYTHONPATH=$PWD PYTHONNOUSERSITE=1 ./env/bin/python scripts/run_partx_faithful.py system=pendulum_stoch noise_level=low \
  initial_train_size=100 samples_per_epoch=20 n_epochs=3 seed=42 \
  output_dir=$SCRATCH/partx_faithful_smoke
```

When this plan was written, a draft of this exact code ran in about 25 s on the real pendulum pool. It produced 35 fits, all of them at the σ_f² = 25 ceiling. That is expected on pendulum low: its outcomes are nearly deterministic, so maximum likelihood wants σ_f² in the hundreds. On pendulum high the fitted σ_f² is 1.2 to 1.8, well inside the bound. The diagnostics record this per epoch as `n_at_bound_fits`, and it is not a bug.

Expected:
- `[partx_faithful] iter …` lines, ending with `epoch_000`, `epoch_001` and `epoch_002` each holding all three files.
- `final_results.json` with `"used": 40`.
- `artifacts_v2.json` files with `train_trajectories` 100, 120 and 140, and `extra.n0_resolved` = 10.

Then rerun the identical command. It must print `resuming …`, write nothing new, and exit cleanly.

- [ ] **Step 7: Commit**

```bash
git add configs/adaptive_v2/partx_faithful.yaml scripts/run_partx_faithful.py tests/partx_faithful/test_config.py
git commit -m "feat(partx_faithful): Hydra config and entry point"
```

---
### Task 10: Reference check against cpslab

**Files:**
- Create: `scripts/partx_reference_check.py`
- Create: `docs/partx_faithful_reference_check.md` (the script writes it)

**Interfaces:**
- Consumes: `PartX`, `PartXConfig`, `backend_factory("gpr")`, `ContinuousSampler` and `gp_quantile_falsification_volume`. The cpslab package comes from a clone.
- Produces: a markdown report with a pass or fail per function (spec §8.2).

- [ ] **Step 1: Build the scratch env and clone cpslab**

`$SCRATCH` is the executing session's scratchpad directory.

```bash
git clone --depth 1 https://github.com/cpslab-asu/part-x.git $SCRATCH/part-x
git -C $SCRATCH/part-x rev-parse --short HEAD          # record this in the report (expected f296c40)
/common/home/st1122/miniforge3/bin/conda create -y -p $SCRATCH/partx_ref_env python=3.10
$SCRATCH/partx_ref_env/bin/pip install "numpy<2" scipy "scikit-learn<1.6" treelib pathos matplotlib scikit-optimize
PYTHONPATH=$SCRATCH/part-x:$PWD $SCRATCH/partx_ref_env/bin/python -c \
  "import partx.coreAlgorithm, adaptive_roa.partx_faithful.algorithm; print('ok')"
```

Expected: `ok`. If the second import fails, a core module is importing something beyond numpy and scipy. Fix it before continuing, because that breaks the Global Constraints.

- [ ] **Step 2: Write the script**

`scripts/partx_reference_check.py`:

```python
#!/usr/bin/env python
"""One-off reference check (spec §8.2): our Part-X loop in deterministic mode vs cpslab.

Runs in a scratch env, because cpslab needs numpy<2, treelib and pathos:
  PYTHONPATH=<cpslab clone>:<repo> <env>/bin/python scripts/partx_reference_check.py \
      --reps 10 --budget 5000 --jobs 16 --cpslab-commit <sha> --out docs/partx_faithful_reference_check.md
"""
from __future__ import annotations

import argparse
import pathlib
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2 - 40.0


def goldstein(x):
    a = 1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
    b = 30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)
    return a * b - 50.0


FUNCS = {"himmelblau": (himmelblau, np.array([[-5.0, 5.0], [-5.0, 5.0]])),
         "goldstein": (goldstein, np.array([[-1.0, 1.0], [-1.0, 1.0]]))}
PAPER_T1 = {"himmelblau": 17.671, "goldstein": 0.304}     # P-X (dq = 0.5), R = 10, M = 100
PAPER_MC = {"himmelblau": 17.030, "goldstein": 0.302}
S = dict(n0=10, n_bo=10, n_c=100, R=10, M=100, B=2, delta_c=0.05, delta_v=0.001)   # paper §5.1
Q = [0.5, 0.05, 0.01]
GROUPS = {"+": ("+",), "-": ("-",), "remaining": ("r", "r+", "r-"), "u": ("u",)}


def area(sup):
    return float(np.prod(sup[:, 1] - sup[:, 0]))


def monte_carlo(fn, sup, n=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    U = sup[:, 0] + rng.random((n, 2)) * (sup[:, 1] - sup[:, 0])
    return float(np.mean(fn(U.T) < 0.0)) * area(sup)


def run_cpslab(args):
    name, rep, budget = args
    from partx.bayesianOptimization import InternalBO
    from partx.coreAlgorithm import PartXOptions, run_single_replication
    from partx.gprInterface import InternalGPR
    from partx.results import fv_using_gp
    from partx.utils import OracleCreator

    fn, sup = FUNCS[name]
    opts = PartXOptions("ref", sup, 2, budget, S["n0"], S["n_bo"], S["n_c"], S["delta_c"], S["R"], S["M"],
                        S["delta_v"], Q, S["B"], True, 12345, InternalGPR(), InternalBO())
    oracle = OracleCreator(None, 1, 1)
    with tempfile.TemporaryDirectory() as work:
        ftree = run_single_replication((rep, opts, fn, oracle, pathlib.Path(work)))["ftree"]
        fv = np.asarray(fv_using_gp(ftree, opts, oracle, Q, np.random.default_rng(12345 + rep))).sum(axis=0)
    return {"fv": (fv * area(sup)).tolist(), "types": [leaf.data.region_class for leaf in ftree.leaves()]}


def run_ours(args):
    name, rep, budget = args
    from adaptive_roa.partx_faithful.algorithm import PartX, PartXConfig
    from adaptive_roa.partx_faithful.backends import backend_factory
    from adaptive_roa.partx_faithful.sampling import ContinuousSampler
    from adaptive_roa.partx_faithful.volume import gp_quantile_falsification_volume

    fn, sup = FUNCS[name]
    cfg = PartXConfig(**S, ei_cand_cap=10_000, ei_f_star="observed", compat_ci_var=True,
                      compat_eq4_times_volume=True, seed=12345 + rep)
    px = PartX(cfg, sup[:, 0], sup[:, 1], ContinuousSampler(fn), backend_factory("gpr"), budget)
    px.run()
    leaves = px.leaves()
    models = {r.rid: px.fit(r) for r in leaves if r.rtype not in ("u", "i") and r.samples}
    fv = gp_quantile_falsification_volume(leaves, models, S["R"], S["M"], Q,
                                          np.random.default_rng(12345 + rep), px.root_volume)
    return {"fv": [fv[q] * px.root_volume for q in Q], "types": [r.rtype for r in leaves]}


def stats(results):
    fv = np.array([r["fv"][0] for r in results])
    counts = {g: np.array([sum(t in ts for t in r["types"]) for r in results], dtype=float)
              for g, ts in GROUPS.items()}
    return fv, counts


def within_2se(a, b):
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    return abs(a.mean() - b.mean()) <= 2.0 * se + 1e-12, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--cpslab-commit", default="unknown")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lines = ["# partx_faithful reference check", "",
             f"cpslab commit `{args.cpslab_commit}`; paper §5.1 settings {S}; T = {args.budget}; "
             f"{args.reps} replications per function per implementation.", "",
             "Volumes are raw areas, as in the paper's tables. Pass = within 2 SE (spec §8.2).", ""]
    overall = True
    for name, (fn, sup) in FUNCS.items():
        jobs = [(name, rep, args.budget) for rep in range(args.reps)]
        with ProcessPoolExecutor(args.jobs) as ex:
            ref = list(ex.map(run_cpslab, jobs))
            ours = list(ex.map(run_ours, jobs))
        fv_ref, c_ref = stats(ref)
        fv_our, c_our = stats(ours)
        ok_fv, se_fv = within_2se(fv_our, fv_ref)
        mc = monte_carlo(fn, sup)
        lines += [f"## {name}", "",
                  "| quantity | cpslab mean | ours mean | 2·SE | pass |", "|---|---:|---:|---:|---|",
                  f"| GP-quantile volume (q = 0.5) | {fv_ref.mean():.4f} | {fv_our.mean():.4f} | "
                  f"{2 * se_fv:.4f} | {'yes' if ok_fv else 'NO'} |"]
        ok_all = ok_fv
        for g in GROUPS:
            ok, se = within_2se(c_our[g], c_ref[g])
            ok_all &= ok
            lines.append(f"| leaves of type {g} | {c_ref[g].mean():.1f} | {c_our[g].mean():.1f} | "
                         f"{2 * se:.1f} | {'yes' if ok else 'NO'} |")
        rel = abs(fv_our.mean() - fv_ref.mean()) / mc
        lines += ["", f"Monte Carlo truth (1e6 points): {mc:.4f}. Paper Table 1 P-X (q = 0.5): "
                  f"{PAPER_T1[name]}, paper Monte Carlo: {PAPER_MC[name]}. "
                  f"|ours − cpslab| / truth = {rel:.2%}.", "",
                  f"**{name}: {'PASS' if ok_all else 'FAIL'}**", ""]
        overall &= ok_all
    lines.append(f"**Overall: {'PASS' if overall else 'FAIL'}**")
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke it on a small budget**

```bash
PYTHONPATH=$SCRATCH/part-x:$PWD $SCRATCH/partx_ref_env/bin/python scripts/partx_reference_check.py \
  --reps 2 --budget 300 --jobs 4 --out $SCRATCH/refcheck_smoke.md
```

Expected: a report with both functions and no exceptions. Pass or fail doesn't matter at this size. If cpslab raises because the budget is too small for the root, raise `--budget`.

- [ ] **Step 4: The real run**

Launch through the compute skill as a CPU job with 16 cores, using the same command with `--reps 10 --budget 5000 --jobs 16 --cpslab-commit <sha from Step 1> --out docs/partx_faithful_reference_check.md`. If a single 5,000-evaluation cpslab replication takes more than 2 hours in the smoke timing, use `--budget 2000` for both implementations, as spec §8.2 allows, and say so in the report.

- [ ] **Step 5: Read the result and decide**

- **PASS:** continue.
- **FAIL on region counts:** debug our control flow against `singlereplication.py` before anything else. That is the bug the check exists to catch.
- **FAIL only on the volume row, while the report's |ours − cpslab| / truth is under 1%:** the 2-SE rule may be too strict. The paper's SEs are around 1e-7, so tiny differences between our EI argmax (a random candidate set) and cpslab's L-BFGS can fail it. Stop and report to the user. Changing the pass rule is their call, not the executor's.

- [ ] **Step 6: Commit**

```bash
git add scripts/partx_reference_check.py docs/partx_faithful_reference_check.md
git commit -m "test(partx_faithful): reference check against the cpslab implementation"
```

---

### Task 11: Pendulum gate

**Files:**
- Create: `scripts/partx_faithful_gate.py`
- Test: `tests/partx_faithful/test_gate.py`
- Create: `docs/partx_faithful_pendulum_gate.md` (the script writes it)

**Interfaces:**
- Consumes: run directories from Task 9's entry point, `scripts/stoch_prob_metrics.py` (`load_ground_truth`, `match_to_truth`, `score_epoch_full`), and `partx_state.pkl`.
- Produces: the gate report with criteria C1 to C5 (spec §8.3).

- [ ] **Step 1: Write the failing tests for the criterion helpers**

`tests/partx_faithful/test_gate.py`:

```python
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
gate = importlib.import_module("partx_faithful_gate")


def test_leaf_types_at_clips_and_assigns():
    lows = np.array([[0.0, 0.0], [0.5, 0.0]])
    highs = np.array([[0.5, 1.0], [1.0, 1.0]])
    t = gate.leaf_types_at(np.array([[0.2, 0.5], [0.8, 0.5], [9.0, 0.5]]), lows, highs,
                           np.array(["-", "+"]), np.zeros(2), np.ones(2))
    assert t.tolist() == ["-", "+", "+"]


def test_coverage_and_wrong_side():
    types = np.array(["+", "+", "-", "r"])
    p = np.array([0.9, 0.4, 0.1, 0.5])
    cov, wrong, n_c = gate.coverage_and_wrong_side(types, p)
    assert cov == pytest.approx(0.75) and n_c == 3 and wrong == pytest.approx(1 / 3)


def test_ambiguous_fraction():
    assert gate.ambiguous_fraction(np.array([0.1, 0.2, 0.5, 0.8, 0.9])) == pytest.approx(0.6)
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_gate.py -q`
Expected: `ModuleNotFoundError: No module named 'partx_faithful_gate'`.

- [ ] **Step 3: Write the gate script**

`scripts/partx_faithful_gate.py`:

```python
#!/usr/bin/env python
"""Pendulum gate for the partx_faithful arm (spec §8.3). Criteria were fixed in the spec before any run:

  C1  every epoch directory exists and hits its budget, or the run records unspent budget
  C2  (low only) classified leaves hold >= 50% of the eval grid at the last checkpoint
  C3  <= 5% of eval-grid points in classified leaves are on the wrong side of p = 0.5 (oracle)
  C4  acquired points are ambiguous (0.2 <= p <= 0.8) more often than the initial set
  C5  n0 = 10 and n0 = 30 reported side by side (no pass/fail)

Usage:
  python scripts/partx_faithful_gate.py --low-key low --out docs/partx_faithful_pendulum_gate.md \
      --run low=<run_dir>:<dataset_root> --run med=... --run high=... --run med_n030=...
"""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spm = importlib.import_module("stoch_prob_metrics")


def leaf_types_at(X, lows, highs, rtypes, root_low, root_high) -> np.ndarray:
    X = np.clip(np.asarray(X, dtype=float), root_low, root_high)
    out = np.full(len(X), "", dtype=object)
    todo = np.ones(len(X), dtype=bool)
    for lo, hi, t in zip(lows, highs, rtypes):
        m = todo & np.all((X >= lo) & (X <= hi), axis=1)
        out[m] = t
        todo &= ~m
    return out


def coverage_and_wrong_side(types, p):
    types = np.asarray(types, dtype=object)
    classified = np.isin(types, ["+", "-"])
    wrong = ((types == "+") & (p < 0.5)) | ((types == "-") & (p > 0.5))
    n_c = int(classified.sum())
    return float(classified.mean()), (float(wrong.sum() / n_c) if n_c else 0.0), n_c


def ambiguous_fraction(p, lo=0.2, hi=0.8) -> float:
    p = np.asarray(p, dtype=float)
    return float(np.mean((p >= lo) & (p <= hi))) if len(p) else float("nan")


def check_run(name: str, run_dir: Path, dataset_root: Path, is_low: bool) -> dict:
    cfg = yaml.safe_load((run_dir / ".hydra" / "config.yaml").read_text())
    initial, spe, n_epochs = int(cfg["initial_train_size"]), int(cfg["samples_per_epoch"]), int(cfg["n_epochs"])
    eps = [run_dir / f"epoch_{e:03d}" for e in range(n_epochs)]
    complete = all((d / "artifacts_v2.json").exists() and (d / "full_roa_per_point.npz").exists() for d in eps)
    arts = [json.loads((d / "artifacts_v2.json").read_text()) for d in eps] if complete else []
    on_budget = complete and all(a["train_trajectories"] == initial + e * spe for e, a in enumerate(arts))
    unspent = arts[-1]["acquisition"]["diagnostics"]["unspent"] if complete else None
    c1 = bool(complete and (on_budget or (unspent or 0) > 0))
    if not complete:
        return dict(name=name, c1=False, note="incomplete run")
    last = eps[-1]
    leaves = np.load(last / "partx_leaves.npz")
    with np.load(last / "full_roa_per_point.npz") as z:
        X = z["start_states"].astype(float)
    gt = spm.load_ground_truth(dataset_root)
    p = gt[1][spm.match_to_truth(X, gt[0])]
    types = leaf_types_at(X, leaves["low"], leaves["high"], leaves["rtype"].astype(str),
                          leaves["root_low"], leaves["root_high"])
    cov, wrong, n_c = coverage_and_wrong_side(types, p)
    with open(run_dir / "partx_state.pkl", "rb") as f:
        px = pickle.load(f)["px"]
    tree = cKDTree(gt[0])
    p_init = gt[1][tree.query(px.store.X(np.arange(px.n_init)))[1]]
    p_acq = gt[1][tree.query(px.store.X(px.log))[1]]
    a_init, a_acq = ambiguous_fraction(p_init), ambiguous_fraction(p_acq)
    row, _ = spm.score_epoch_full(last, gt, None, None)
    return dict(name=name, n0=arts[-1]["extra"].get("n0_resolved"), c1=c1, coverage=cov,
                c2=(cov >= 0.50) if is_low else None, wrong=wrong, n_classified=n_c,
                c3=wrong <= 0.05, amb_init=a_init, amb_acq=a_acq, c4=a_acq > a_init,
                unspent=unspent, KL=row["KL"], sAUROC=row.get("sAUROC"))


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "pass" if v else "FAIL"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=run_dir:dataset_root")
    ap.add_argument("--low-key", default="low")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for spec in args.run:
        name, rest = spec.split("=", 1)
        run_dir, droot = rest.split(":", 1)
        rows.append(check_run(name, Path(run_dir), Path(droot), name == args.low_key))
    cols = ["name", "n0", "c1", "coverage", "c2", "wrong", "n_classified", "c3", "amb_init", "amb_acq",
            "c4", "unspent", "KL", "sAUROC"]
    lines = ["# partx_faithful pendulum gate", "", "Criteria from spec §8.3, fixed before any run.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    gated = [r for r in rows if r.get("n0") in (10, None) and not r["name"].endswith("_n030")]
    ok = all(r.get("c1") and r.get("c3") and r.get("c4") and r.get("c2") in (True, None) for r in gated)
    lines += ["", f"**Gate (n0 = 10 runs): {'PASS' if ok else 'FAIL'}**",
              "", "C5 is the n0 = 10 vs n0 = 30 comparison in the table above; it has no pass/fail."]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the helper tests to confirm they pass**

Run: `PYTHONNOUSERSITE=1 ./env/bin/python -m pytest tests/partx_faithful/test_gate.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit the script**

```bash
git add scripts/partx_faithful_gate.py tests/partx_faithful/test_gate.py
git commit -m "feat(partx_faithful): pendulum gate criteria script"
```

- [ ] **Step 6: Launch the four pendulum runs through the compute skill**

These are CPU jobs: 16 cores, the default memory, and `PYTHONNOUSERSITE=1`, run from the worktree.

```bash
PY="env PYTHONPATH=$PWD PYTHONNOUSERSITE=1 ./env/bin/python"   # $PWD = the worktree root
GT=/common/users/shared/pracsys/adaptive_roa_experiments/gaussian_torque
COMMON="system=pendulum_stoch initial_train_size=100 samples_per_epoch=100 n_epochs=20 seed=42"
$PY scripts/run_partx_faithful.py $COMMON noise_level=low  output_dir=$GT/pen_low_partx_faithful
$PY scripts/run_partx_faithful.py $COMMON noise_level=med  output_dir=$GT/pen_partx_faithful
$PY scripts/run_partx_faithful.py $COMMON noise_level=high output_dir=$GT/pen_high_partx_faithful
$PY scripts/run_partx_faithful.py $COMMON noise_level=med  partx_faithful.algorithm.n0=30 \
    output_dir=$GT/pen_partx_faithful_n030
```

The med prefix is `pen`, not `pen_med` (see `pend_med` in `scripts/score_stoch_incremental.py`). The `_n030` directory is not a registered arm, so the scorer ignores it. Start an hourly poller as the compute skill describes, and keep it until all four finish.

- [ ] **Step 7: Run the gate and write the report**

```bash
D=/common/users/shared/pracsys/genMoPlan/data_trajectories/stochastic/pendulum/gaussian_signal/lqr
PYTHONPATH=$PWD PYTHONNOUSERSITE=1 ./env/bin/python scripts/partx_faithful_gate.py --low-key low \
  --run low=$GT/pen_low_partx_faithful:$D/low --run med=$GT/pen_partx_faithful:$D/med \
  --run high=$GT/pen_high_partx_faithful:$D/high --run med_n030=$GT/pen_partx_faithful_n030:$D/med \
  --out docs/partx_faithful_pendulum_gate.md
git add docs/partx_faithful_pendulum_gate.md
git commit -m "docs(partx_faithful): pendulum gate results"
```

- [ ] **Step 8: STOP and review with the user**

Show the user the gate table and the reference-check result. Do not start Task 12 without their explicit go-ahead. If the gate fails, report which criterion failed and what the diagnostics show (leaf types, capped or at-bound fits, shortfall). Bugs get fixed. Design changes and hyperparameter changes need sign-off (spec §8.3).

---

### Task 12: The remaining nine cells

**Files:** none. This task launches and monitors jobs.

- [ ] **Step 1: Time a quad3D probe**

Through the compute skill, run quad3D f_0.20 into its real output directory with a 2-hour wall limit. The runner resumes, so the probe's work is kept.

```bash
PY="env PYTHONPATH=$PWD PYTHONNOUSERSITE=1 ./env/bin/python"   # $PWD = the worktree root
EXP=/common/users/shared/pracsys/adaptive_roa_experiments/quadrotor_stoch
$PY scripts/run_partx_faithful.py system=quadrotor3d_ppo_stoch noise_family=corridor_sine_ambient \
  +controller=ppo_800k noise_level=f_0.20 seed=42 output_dir=$EXP/q3d800k_cs020_partx_faithful
```

From the `[partx_faithful] iter …` lines, read the seconds per iteration and `capped` counts. Extrapolate to 28,500 evaluations, and allow for later iterations having more regions. Report the estimate to the user before launching the quad3D fleet. If one run would exceed Amarel's 3-day wall several times over, propose a process pool over regions within an iteration (spec §6.6). That is a new task with its own tests, not an ad-hoc edit.

- [ ] **Step 2: Launch cartpole and quad2D through the compute skill**

```bash
PY="env PYTHONPATH=$PWD PYTHONNOUSERSITE=1 ./env/bin/python"   # $PWD = the worktree root
GT=/common/users/shared/pracsys/adaptive_roa_experiments/gaussian_torque
EXP=/common/users/shared/pracsys/adaptive_roa_experiments/quadrotor_stoch
CP="system=cartpole_stoch controller=safe_explorer_ppo initial_train_size=300 samples_per_epoch=150 n_epochs=12 seed=42"
$PY scripts/run_partx_faithful.py $CP noise_level=baseline output_dir=$GT/cprl_base_partx_faithful
$PY scripts/run_partx_faithful.py $CP noise_level=low      output_dir=$GT/cprl_low_partx_faithful
$PY scripts/run_partx_faithful.py $CP noise_level=med      output_dir=$GT/cprl_med_partx_faithful
$PY scripts/run_partx_faithful.py $CP noise_level=high     output_dir=$GT/cprl_high_partx_faithful
$PY scripts/run_partx_faithful.py system=quadrotor2d_stoch noise_family=corridor_sine_ambient \
  noise_level=smooth initial_train_size=2000 samples_per_epoch=500 n_epochs=24 seed=42 \
  output_dir=$EXP/q2d_cs_partx_faithful
```

- [ ] **Step 3: Launch quad3D f_0.00, f_0.12 and f_0.40, and resume f_0.20, after the user approves the Step 1 estimate**

```bash
PY="env PYTHONPATH=$PWD PYTHONNOUSERSITE=1 ./env/bin/python"   # $PWD = the worktree root
EXP=/common/users/shared/pracsys/adaptive_roa_experiments/quadrotor_stoch
Q3="system=quadrotor3d_ppo_stoch noise_family=corridor_sine_ambient +controller=ppo_800k seed=42"
$PY scripts/run_partx_faithful.py $Q3 noise_level=f_0.00 output_dir=$EXP/q3d800k_cs000_partx_faithful
$PY scripts/run_partx_faithful.py $Q3 noise_level=f_0.12 output_dir=$EXP/q3d800k_cs012_partx_faithful
$PY scripts/run_partx_faithful.py $Q3 noise_level=f_0.20 output_dir=$EXP/q3d800k_cs020_partx_faithful
$PY scripts/run_partx_faithful.py $Q3 noise_level=f_0.40 output_dir=$EXP/q3d800k_cs040_partx_faithful
```

The quad3D schedule (10,000 initial, 1,500 per epoch, 20 epochs) comes from the system config. The 800k pool loads fully into memory, as it does for the engine, so request at least the memory the `q3d800k_*_partx_fix` jobs used. On Amarel, outputs go to `/scratch/st1122/adaptive_roa/experiments/…`, and results come back by rsync. Sync the `.hydra` directories too (memory note "Amarel result sync needs .hydra").

- [ ] **Step 4: Monitor until every run writes `final_results.json`**

Keep the hourly poller running. A preempted or timed-out job restarts with the identical command; never relaunch into a fresh directory. Check liveness with `sacct` and `squeue`, not by watching log growth.

---

### Task 13: Scoring, paper scripts and docs

Run this in the MAIN working tree (`/common/home/st1122/Projects/adaptive_roa`), not the worktree. `scripts/paper/` is untracked, so it doesn't exist in the worktree, and `scripts/score_stoch_incremental.py` carries other sessions' uncommitted edits. Leave these edits uncommitted and report the diff to the user.

**Files:**
- Modify: `scripts/score_stoch_incremental.py` (the `ARMS` list)
- Modify: `scripts/paper/final_epoch_table.py:61,76`
- Modify: `scripts/paper/plot_levelsets_paper.py:49,59,81`
- Modify: `scripts/paper/plot_prob_metrics_paper.py:133`
- Modify: `scripts/paper/rule_gap_vs_baselines.py:42`
- Modify: `scripts/paper/score_levelsets_hi.py:75`
- Modify: `scripts/paper/plot_learning_curves_paper.py:625`
- Modify: `scripts/plot_stoch_all_levels.py:58,117`
- Modify: `/common/users/shared/pracsys/genMoPlan/docs/stochastic/METHODS.md` (§2 `gp`, §5 `partx`, §8 table)
- Modify: `…/docs/stochastic/pendulum/lqr/README.md:65` and `…/docs/stochastic/quadrotor2d/rl/README.md:378-381`

- [ ] **Step 1: Register the arm with the scorer**

In `scripts/score_stoch_incremental.py`, in `ARMS`, change `"yield_a1", "yield_mlp", "partx_fix", "clf_dir00", …` to `"yield_a1", "yield_mlp", "partx_fix", "partx_faithful", "clf_dir00", …`. That's one added name and nothing else. `predictor_of()` already maps `partx*` to `"gp"`.

- [ ] **Step 2: Run the scorer once and confirm the rows land**

```bash
PYTHONNOUSERSITE=1 ./env/bin/python scripts/score_stoch_incremental.py
grep -c ",partx_faithful," /common/users/shared/pracsys/genMoPlan/docs/stochastic/pendulum/lqr/gaussian_all_levels.csv
```

Expected: 60 rows (3 levels × 20 epochs) once the pendulum runs are complete.

- [ ] **Step 3: Switch the paper's Part-X column**

Make these replacements. Each is a key or label swap. Don't change anything else on those lines.

- `final_epoch_table.py:61`: `("partx_fix", "Part-X (GP)")` → `("partx_faithful", "Part-X")`
- `final_epoch_table.py:76`: `"Part-X (GP)"` → `"Part-X"`
- `plot_levelsets_paper.py:49`: key `"partx_fix"` → `"partx_faithful"`, label `"Part-X (GP)"` → `"Part-X"`
- `plot_levelsets_paper.py:59,81`: `"partx_fix"` → `"partx_faithful"`
- `plot_prob_metrics_paper.py:133`: key `"partx_fix"` → `"partx_faithful"`
- `rule_gap_vs_baselines.py:42`: `("partx_fix", False)` → `("partx_faithful", False)`
- `score_levelsets_hi.py:75`: `"partx_fix"` → `"partx_faithful"`
- `plot_learning_curves_paper.py:625`: `Part-X (GP)` → `Part-X`
- `plot_stoch_all_levels.py:58`: key `"partx_fix"` → `"partx_faithful"`, label → `"Part-X"`
- `plot_stoch_all_levels.py:117`: `"partx_fix"` → `"partx_faithful"`

Leave `scripts/paper/pool_size_ablation.py` unchanged. `partx_faithful` runs only on the 800k pool, so it has no 100k twin. Flag this to the user.

- [ ] **Step 4: Regenerate the tables and check the column**

```bash
PYTHONNOUSERSITE=1 ./env/bin/python scripts/paper/final_epoch_table.py
grep -n "Part-X" /common/users/shared/pracsys/genMoPlan/docs/stochastic/paper/tables/final_epoch_prob_metrics_main_15k.md | head
```

Expected: rows labelled `Part-X`, with values that differ from the old `Part-X (GP)` rows.

- [ ] **Step 5: Fix the docs**

- **METHODS.md §5**: replace the `### \`partx\` …` subsection (the heading at line 220 and its body, up to the next `---`) with:

  ```markdown
  ### `partx_faithful`: Part-X (Pedrielli et al., arXiv 2110.10729)
  Alg. 1-4 as published: one GP classifier per region (probit Laplace, Matérn-5/2),
  n0 LHS plus nBO expected-improvement samples in every new region, min/max Monte
  Carlo classification with r+/r-/u types, continued sampling of classified regions
  per eq. 4, and branching gated by budget with a final volume-proportional phase.
  Forced adaptations: samples snap to unused pool starts inside the region, and
  binary outcomes replace robustness (plug-in EI, level set p = 0.5). Settings:
  n0 = 10 on pendulum and 30 elsewhere; nBO = 10, nc = 100, R × M = 20 × 500,
  B = 2, δC = 0.05, δv = 0.001. The scored p field comes from the leaf GPs.
  Spec: `docs/superpowers/specs/2026-09-11-partx-faithful-stochastic-design.md`;
  validation: `docs/partx_faithful_reference_check.md`,
  `docs/partx_faithful_pendulum_gate.md` (both in the adaptive_roa repo).

  ### `partx_fix`: GP straddle LSE with partition stratification (formerly labelled Part-X)
  One global sparse GP classifier with straddle sampling, stratified over a
  partition tree. The 2026-09-11 audit found it is not Part-X, so it is no longer
  reported under that name. Rows stay in the CSVs.
  ```

- **METHODS.md §8 table**: after the `partx` row, add `| \`partx_faithful\` | partx (standalone runner) | — | **local GPC** | ✓ | ✓ | ✓ |`. In the existing `partx` row, change the arm cell to `` `partx_fix` ``.
- **`pendulum/lqr/README.md:65`**: `| \`partx\` | partx | gp_reg |` → `| \`partx\` | partx | gp |`.
- **`quadrotor2d/rl/README.md:378-381`**: replace the sentence about Part-X under-spending with: "This under-spending belonged to the pre-fix `partx` arm. `partx_fix` spent its full 500 per epoch. Its `cs` epoch-23 row is a single-epoch GP fit spike (KL 0.407 against 0.047 to 0.072 over epochs 18 to 22), so the 'worse ×4' verdict rests on one epoch."

- [ ] **Step 6: Report to the user**

List the uncommitted main-tree diffs (`git -C /common/home/st1122/Projects/adaptive_roa diff --stat`, plus the untracked `scripts/paper` edits) and the doc edits. Ask whether to commit the `score_stoch_incremental.py` line separately from the other sessions' changes in that file.

---

## Spec coverage

| Spec section | Task |
|---|---|
| §2 D1 (GP classifier), D2 (falsify flag), D3 (readout), D8 (unspent budget) | 1, 2, 7, 8 |
| §2 D4 (scope), D5 (standalone runner), D6 (n0 by dimension), D7 (paper over code) | 11, 12, 8, 9, 4 |
| §3 algorithm mapping | 3, 4, 5, 6 |
| §4 discrepancies and compat switches | 4, 6, 10 |
| §5 local GP classifier | 1, 2 |
| §6 runner, config, checkpoints, resume | 6, 8, 9 |
| §7 scoring and reporting | 13 |
| §8.1 unit tests | 1 to 8, 11 |
| §8.2 reference check | 10 |
| §8.3 pendulum gate | 11 |
| §9 rollout, including the quad3D timing probe | 0, 11, 12, 13 |

## Notes for the executor

- The one deliberate addition to the spec is `partx_leaves.npz` in each epoch directory, which gate criteria C2 and C3 need. The spec's diagnostics list only covers counts.
- Resume equivalence is tested on array contents and JSON fields, not bytes. `np.savez` zip headers can differ between writes.
- `evaluate_full_roa_classifier` also writes plots into each epoch directory, as it does for every other classifier-family arm.

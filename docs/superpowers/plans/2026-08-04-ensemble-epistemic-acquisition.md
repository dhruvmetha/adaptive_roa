# Ensemble Epistemic Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Acquire training data on the *epistemic* part of label entropy, separated from the aleatoric part by a deep ensemble.

**Architecture:** Pure scoring functions turn per-member success probabilities into four scores (total / aleatoric / epistemic-BALD / epistemic-variance). Two new probability backends expose per-member predictions (`estimate_members`) alongside the existing ensemble-marginal `estimate`. One new acquisition strategy consumes them. Flow-matching ensembles train one member per GPU in parallel. Every single-model path is untouched.

**Tech Stack:** Python 3.10, PyTorch, PyTorch Lightning, Hydra/OmegaConf, NumPy, pytest.

## Global Constraints

- M = 5 members. K = 20 endpoint samples per member at acquisition, K = 100 at eval.
- d2_ratio = 1.0, 19 epochs, seed 42 for all campaign runs.
- Prediction labels: 1 = success, 0 = failure, −1 = uncertain, −2 = invalid.
- `classify_attractor` returns 1 = success, −1 = failure, 0 = invalid (note: *different* from prediction labels).
- Run tests with `./env/bin/python -m pytest`, not bare `pytest`.
- New params to existing functions default to `None` for backward compatibility.
- Never break the single-model path: new files where possible, gated branches where not.
- Commit messages: no AI/tool attribution of any kind.

---

### Task 1: Uncertainty score functions

The highest-risk component: a biased estimator here silently produces a null result. Pure functions, no I/O, fully testable.

**Files:**
- Create: `adaptive_roa/adaptive_v2/strategy/uncertainty_scores.py`
- Test: `tests/adaptive_v2/test_uncertainty_scores.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `binary_entropy(p: np.ndarray) -> np.ndarray`
  - `total_uncertainty(p_members: np.ndarray) -> np.ndarray` — `[M,N] -> [N]`
  - `aleatoric_uncertainty(p_members: np.ndarray) -> np.ndarray`
  - `epistemic_bald(p_members: np.ndarray) -> np.ndarray`
  - `epistemic_variance(p_members: np.ndarray, k: float | None) -> np.ndarray`
  - `score_by_mode(mode: str, p_members: np.ndarray, k: float | None) -> np.ndarray`
  - `SCORE_MODES: tuple[str, ...] = ("total", "aleatoric", "epistemic_bald", "epistemic_var")`

- [ ] **Step 1: Write the failing tests**

```python
# tests/adaptive_v2/test_uncertainty_scores.py
import numpy as np
import pytest

from adaptive_roa.adaptive_v2.strategy.uncertainty_scores import (
    SCORE_MODES, aleatoric_uncertainty, binary_entropy, epistemic_bald,
    epistemic_variance, score_by_mode, total_uncertainty,
)


def test_binary_entropy_endpoints_and_peak():
    e = binary_entropy(np.array([0.0, 1.0, 0.5]))
    assert e[0] == pytest.approx(0.0)
    assert e[1] == pytest.approx(0.0)
    assert e[2] == pytest.approx(np.log(2))


def test_decomposition_identity_is_exact_when_members_are_noise_free():
    # total = aleatoric + epistemic_bald must hold exactly (classifier case)
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, size=(5, 200))
    lhs = total_uncertainty(p)
    rhs = aleatoric_uncertainty(p) + epistemic_bald(p)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_agreeing_members_give_zero_epistemic():
    p = np.tile(np.array([0.1, 0.5, 0.9]), (5, 1))       # all members identical
    np.testing.assert_allclose(epistemic_bald(p), 0.0, atol=1e-12)
    np.testing.assert_allclose(epistemic_variance(p, k=None), 0.0, atol=1e-12)


def test_disagreeing_members_give_large_epistemic():
    p = np.array([[0.2, 0.2], [0.2, 0.2], [0.8, 0.8], [0.8, 0.8], [0.5, 0.5]])
    assert epistemic_bald(p).min() > 0.1
    assert epistemic_variance(p, k=None).min() > 0.05


def test_epistemic_variance_debias_removes_finite_k_noise():
    # Members agree in TRUTH; each p_hat is a K-sample binomial estimate.
    # Debiased variance must be ~0; raw variance must not be.
    rng = np.random.default_rng(0)
    M, K, N = 5, 20, 20000
    for p_true in (0.5, 0.3, 0.1):
        ph = rng.binomial(K, p_true, size=(M, N)) / K
        assert abs(epistemic_variance(ph, k=K).mean()) < 5e-4
        assert epistemic_variance(ph, k=None).mean() > 3e-3   # undebiased is inflated


def test_naive_bald_bias_is_the_documented_magnitude():
    # Asserts the bias that motivates epistemic_var, rather than assuming it.
    rng = np.random.default_rng(0)
    M, K, N = 5, 20, 20000
    ph = rng.binomial(K, 0.5, size=(M, N)) / K
    expected = (1.0 / (2 * K)) * (1 - 1.0 / M)             # ~0.020 nats
    assert epistemic_bald(ph).mean() == pytest.approx(expected, rel=0.25)


def test_bald_bias_vanishes_at_a_deterministic_state():
    ph = np.zeros((5, 100))
    assert epistemic_bald(ph).mean() == pytest.approx(0.0, abs=1e-12)


def test_score_by_mode_dispatches_and_rejects_unknown():
    p = np.tile(np.array([0.3, 0.7]), (5, 1))
    for mode in SCORE_MODES:
        assert score_by_mode(mode, p, k=20).shape == (2,)
    with pytest.raises(ValueError, match="unknown score mode"):
        score_by_mode("nope", p, k=20)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_uncertainty_scores.py -x -q`
Expected: FAIL — `ModuleNotFoundError: adaptive_roa.adaptive_v2.strategy.uncertainty_scores`

- [ ] **Step 3: Write the implementation**

```python
# adaptive_roa/adaptive_v2/strategy/uncertainty_scores.py
"""Split an ensemble's predictive uncertainty into aleatoric and epistemic parts.

For M members with per-member success probabilities p_1..p_M:

    H(p_bar)  =  E_m[H(p_m)]  +  I(y ; m | x)
    total        aleatoric       epistemic

Acquiring on the epistemic term targets what more data can fix; acquiring on the
total confuses that with irreducible outcome randomness, which is how entropy
acquisition degenerates under heavy process noise.

FINITE-SAMPLE WARNING. When each p_m is itself estimated from K Monte-Carlo
samples (flow matching), `epistemic_bald` is biased UPWARD by roughly
(1/2K)(1 - 1/M) -- ~0.021 nats at K=20, M=5. That bias is roughly flat across
the interior and vanishes only at p = 0 and 1, so it lifts every non-deterministic
state above every decided one regardless of whether members actually disagree.
It does NOT shrink as M grows. `epistemic_variance` with `k` set removes exactly
this term and is the safe default; pass `k=None` when p_m carries no sampling
noise (a classifier forward pass).
"""
from __future__ import annotations

import numpy as np

SCORE_MODES: tuple[str, ...] = ("total", "aleatoric", "epistemic_bald", "epistemic_var")
_EPS = 1e-12


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy of Bernoulli(p) in nats; exact 0 at p = 0 and p = 1."""
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _check(p_members: np.ndarray) -> np.ndarray:
    p = np.asarray(p_members, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"p_members must be [M, N], got shape {p.shape}")
    if p.shape[0] < 2:
        raise ValueError(f"need >= 2 members to separate epistemic, got {p.shape[0]}")
    return p


def total_uncertainty(p_members: np.ndarray) -> np.ndarray:
    """H of the ensemble marginal: the score the current entropy strategy uses."""
    return binary_entropy(_check(p_members).mean(axis=0))


def aleatoric_uncertainty(p_members: np.ndarray) -> np.ndarray:
    """Mean per-member entropy: the irreducible part. Used as a negative control."""
    return binary_entropy(_check(p_members)).mean(axis=0)


def epistemic_bald(p_members: np.ndarray) -> np.ndarray:
    """Mutual information between the label and the member index (BALD).

    Unbiased only when p_m carries no sampling noise. See the module warning.
    """
    p = _check(p_members)
    return binary_entropy(p.mean(axis=0)) - binary_entropy(p).mean(axis=0)


def epistemic_variance(p_members: np.ndarray, k: float | None) -> np.ndarray:
    """Between-member variance, debiased for each member's K-sample MC noise.

    Var_m[p_m] - mean_m[p_m(1-p_m)/(K-1)]. The subtracted term is the unbiased
    estimate of a proportion's sampling variance, so the result estimates true
    member disagreement and goes to 0 when members agree, at any K. Pass k=None
    when p_m is exact (no sampling), which skips the correction.
    """
    p = _check(p_members)
    var = p.var(axis=0, ddof=1)
    if k is None or not np.isfinite(k) or k <= 1:
        return var
    return var - (p * (1.0 - p) / (float(k) - 1.0)).mean(axis=0)


def score_by_mode(mode: str, p_members: np.ndarray, k: float | None) -> np.ndarray:
    """Dispatch to one score. `k` is ignored by every mode except epistemic_var."""
    if mode == "total":
        return total_uncertainty(p_members)
    if mode == "aleatoric":
        return aleatoric_uncertainty(p_members)
    if mode == "epistemic_bald":
        return epistemic_bald(p_members)
    if mode == "epistemic_var":
        return epistemic_variance(p_members, k)
    raise ValueError(f"unknown score mode {mode!r}; expected one of {SCORE_MODES}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_uncertainty_scores.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/adaptive_v2/strategy/uncertainty_scores.py tests/adaptive_v2/test_uncertainty_scores.py
git commit -m "feat(acquisition): aleatoric/epistemic decomposition of label entropy

Adds total/aleatoric/BALD/debiased-variance scores over per-member
probabilities. Tests assert the naive BALD finite-K bias magnitude rather
than assuming it, and that the debiased variance removes it."
```

---

### Task 2: Ensemble probability backends

**Files:**
- Create: `adaptive_roa/adaptive_v2/probability/ensemble_prob.py`
- Test: `tests/adaptive_v2/test_ensemble_prob.py`

**Interfaces:**
- Consumes: `OutcomeProbabilities` from `adaptive_roa.adaptive_v2.types`.
- Produces, on both backends:
  - `estimate(start_states: np.ndarray) -> OutcomeProbabilities` (ensemble marginal)
  - `estimate_members(start_states: np.ndarray, verbose: bool = False) -> np.ndarray` `[M, N]`
  - attributes `system`, `attractor_radius`, `device`, `n_members: int`, `member_sample_size: int | None`
  - `bind_model(model_handle) -> None`
- `member_sample_size` is the `k` later passed to `epistemic_variance`: `None` for the classifier (exact), `num_mc_samples` for flow matching.

- [ ] **Step 1: Write the failing tests**

```python
# tests/adaptive_v2/test_ensemble_prob.py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.ensemble_prob import (
    EnsembleClassifierProbabilityBackend, EnsembleEndpointMCProbabilityBackend,
)


class _FakePosterior:
    """Two members with fixed, different logits."""
    n_members = 2

    def forward_all_members(self, x):
        n = x.shape[0]
        m0 = torch.full((n, 1), 2.0)       # sigmoid ~0.881
        m1 = torch.full((n, 1), -2.0)      # sigmoid ~0.119
        return torch.stack([m0, m1], dim=0)


class _FakeHandle:
    def __init__(self):
        self.posterior = _FakePosterior()
    def eval(self):
        return self


def test_classifier_backend_members_shape_and_values():
    b = EnsembleClassifierProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.1}), system=None, device="cpu")
    b.bind_model(_FakeHandle())
    pm = b.estimate_members(np.zeros((4, 2), dtype=np.float32))
    assert pm.shape == (2, 4)
    assert pm[0].mean() == pytest.approx(0.8808, abs=1e-3)
    assert pm[1].mean() == pytest.approx(0.1192, abs=1e-3)


def test_classifier_backend_marginal_is_mean_of_members():
    b = EnsembleClassifierProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.1}), system=None, device="cpu")
    b.bind_model(_FakeHandle())
    x = np.zeros((4, 2), dtype=np.float32)
    pm = b.estimate_members(x)
    out = b.estimate(x)
    np.testing.assert_allclose(out.p_success, pm.mean(axis=0), atol=1e-9)
    np.testing.assert_allclose(out.p_success + out.p_failure + out.p_invalid, 1.0, atol=1e-9)


def test_classifier_member_sample_size_is_none_because_no_sampling():
    b = EnsembleClassifierProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.1}), system=None, device="cpu")
    assert b.member_sample_size is None


def test_fm_backend_member_sample_size_is_k():
    cfg = OmegaConf.create({"attractor_radius": 0.1, "num_mc_samples": 20})
    b = EnsembleEndpointMCProbabilityBackend(cfg, system=None, device="cpu")
    assert b.member_sample_size == 20


def test_backend_rejects_a_single_member_ensemble():
    class _One:
        n_members = 1
        def forward_all_members(self, x):
            return torch.zeros((1, x.shape[0], 1))
    class _H:
        posterior = _One()
        def eval(self): return self
    b = EnsembleClassifierProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.1}), system=None, device="cpu")
    with pytest.raises(ValueError, match="at least 2 members"):
        b.bind_model(_H())


def test_estimate_before_bind_model_raises():
    b = EnsembleClassifierProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.1}), system=None, device="cpu")
    with pytest.raises(RuntimeError, match="before bind_model"):
        b.estimate_members(np.zeros((2, 2), dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_ensemble_prob.py -x -q`
Expected: FAIL — `ModuleNotFoundError: ...probability.ensemble_prob`

- [ ] **Step 3: Write the implementation**

```python
# adaptive_roa/adaptive_v2/probability/ensemble_prob.py
"""Probability backends that expose PER-MEMBER predictions, not just the marginal.

`estimate()` keeps the existing contract and returns the ensemble marginal, so
threshold, calibration and eval code needs no changes. `estimate_members()` is
the new surface the decomposition strategy needs: without per-member values the
aleatoric and epistemic parts cannot be separated at all.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class _EnsembleBackendBase:
    def __init__(self, cfg: Any, system: Any, device: str):
        self.attractor_radius = float(cfg.attractor_radius)
        self.system = system
        self.device = device
        self.model_handle: Any = None
        self.n_members: int = 0

    def bind_model(self, model_handle: Any) -> None:
        self.model_handle = model_handle
        n = int(getattr(self._posterior(), "n_members", 0))
        if n < 2:
            raise ValueError(
                f"ensemble backend needs at least 2 members, got {n}. A 1-member "
                "'ensemble' has no epistemic signal and would silently score 0."
            )
        self.n_members = n

    def _posterior(self) -> Any:
        if self.model_handle is None:
            raise RuntimeError("ensemble backend used before bind_model")
        return getattr(self.model_handle, "posterior", self.model_handle)

    def estimate(self, start_states: np.ndarray) -> OutcomeProbabilities:
        """Ensemble marginal. p_invalid is 0 for both backends' label space."""
        p = self.estimate_members(start_states).mean(axis=0)
        return OutcomeProbabilities(
            p_success=p, p_failure=1.0 - p, p_invalid=np.zeros_like(p),
        )


class EnsembleClassifierProbabilityBackend(_EnsembleBackendBase):
    """Per-member probabilities from one exact forward pass per member.

    No sampling anywhere, so `member_sample_size` is None and BALD is unbiased.
    """

    member_sample_size: int | None = None

    @torch.no_grad()
    def estimate_members(self, start_states: np.ndarray, verbose: bool = False) -> np.ndarray:
        post = self._posterior()
        x = torch.as_tensor(np.asarray(start_states), dtype=torch.float32,
                            device=self.device)
        logits = post.forward_all_members(x)           # [M, N, 1]
        p = torch.sigmoid(logits).squeeze(-1)          # [M, N]
        return p.detach().cpu().numpy().astype(np.float64)


class EnsembleEndpointMCProbabilityBackend(_EnsembleBackendBase):
    """Per-member probabilities from K endpoint samples per member.

    Each p_m is a K-sample binomial estimate, so it carries sampling noise;
    `member_sample_size` reports K so the epistemic score can debias it.
    """

    def __init__(self, cfg: Any, system: Any, device: str):
        super().__init__(cfg, system, device)
        self.num_mc_samples = int(cfg.num_mc_samples)

    @property
    def member_sample_size(self) -> int:
        return self.num_mc_samples

    @torch.no_grad()
    def estimate_members(self, start_states: np.ndarray, verbose: bool = False) -> np.ndarray:
        handle = self.model_handle
        if handle is None:
            raise RuntimeError("ensemble backend used before bind_model")
        x = torch.as_tensor(np.asarray(start_states), dtype=torch.float32,
                            device=self.device)
        out = np.empty((self.n_members, x.shape[0]), dtype=np.float64)
        for m in range(self.n_members):
            hits = torch.zeros(x.shape[0], dtype=torch.float64)
            for _ in range(self.num_mc_samples):
                pred = handle.predict_endpoint_member(m, x)
                lab = self.system.classify_attractor(pred, self.attractor_radius)
                hits += (lab == 1).double().cpu()
            out[m] = (hits / float(self.num_mc_samples)).numpy()
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_ensemble_prob.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/adaptive_v2/probability/ensemble_prob.py tests/adaptive_v2/test_ensemble_prob.py
git commit -m "feat(probability): ensemble backends exposing per-member predictions

estimate() keeps returning the ensemble marginal so existing threshold and
eval code is unchanged; estimate_members() adds the [M, N] surface the
decomposition needs. member_sample_size reports K (None when exact) so the
epistemic score knows whether to debias."
```

---

### Task 3: Decomposition acquisition strategy

**Files:**
- Create: `adaptive_roa/adaptive_v2/strategy/decomposition.py`
- Test: `tests/adaptive_v2/test_decomposition_strategy.py`

**Interfaces:**
- Consumes: `score_by_mode`, `SCORE_MODES` (Task 1); a backend with `estimate_members`, `n_members`, `member_sample_size`, `system` (Task 2); `select_greedy`/`select_greedy_diverse` from `adaptive_roa.adaptive_v2.strategy.dispersion_score`; `AcquisitionResult`, `ThresholdState` from `adaptive_roa.adaptive_v2.types`.
- Produces: `DecompositionAcquisitionStrategy(cfg)` with `mode = "decomposition"` and `.select(pool, probability_backend, threshold_backend, threshold_state, target_count, exclude=None) -> AcquisitionResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/adaptive_v2/test_decomposition_strategy.py
import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.strategy.decomposition import DecompositionAcquisitionStrategy
from adaptive_roa.adaptive_v2.types import ThresholdState
from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent


class _FakeSystem(DynamicalSystem):
    def define_manifold_structure(self):
        return [ManifoldComponent("SO2", 1, "angle"),
                ManifoldComponent("Real", 1, "angular_velocity")]

    def define_state_bounds(self):
        return {"angle": (-math.pi, math.pi), "angular_velocity": (-8.0, 8.0)}

    def classify_attractor(self, state, radius=0.1):
        lab = torch.zeros(state.shape[0], dtype=torch.int64)
        lab[state[:, 0].abs() < radius] = 1
        return lab


class _FakePool:
    def __init__(self, states):
        self.states = np.asarray(states, dtype=np.float32)
        self.last_exclude = None

    def sample_candidates_without_marking(self, n, exclude=None):
        self.last_exclude = exclude
        take = min(n, len(self.states))
        return self.states[:take], list(range(100, 100 + take))


class _FakeMemberBackend:
    """p_members fixed per candidate: [M, N]."""

    def __init__(self, p_members, system, k=None):
        self.p_members = np.asarray(p_members, dtype=np.float64)
        self.system, self.device, self.attractor_radius = system, "cpu", 0.2
        self.n_members = self.p_members.shape[0]
        self.member_sample_size = k

    def estimate_members(self, states, verbose=False):
        return self.p_members[:, : len(states)]


def _cfg(**over):
    base = dict(score="epistemic_var", d2_ratio=1.0, n_candidates=50,
                selection_rule="greedy", diversity_pool_multiplier=5, verbose=False)
    base.update(over)
    return OmegaConf.create(base)


def _ts():
    return ThresholdState(lambda_star=0.5, delta_star=0.1)


def test_epistemic_mode_prefers_disagreement_over_ambiguity():
    # candidate 0: all members say 0.5 -> max ALEATORIC, zero epistemic
    # candidate 1: members split 0/1  -> max EPISTEMIC
    p = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 1.0], [0.5, 1.0]])
    states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    sys_ = _FakeSystem()
    strat = DecompositionAcquisitionStrategy(_cfg(score="epistemic_var"))
    res = strat.select(_FakePool(states), _FakeMemberBackend(p, sys_), None, _ts(), 1)
    assert res.d2_indices == [101]          # the disagreement candidate


def test_aleatoric_mode_prefers_the_opposite_candidate():
    p = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 1.0], [0.5, 1.0]])
    states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    strat = DecompositionAcquisitionStrategy(_cfg(score="aleatoric"))
    res = strat.select(_FakePool(states), _FakeMemberBackend(p, _FakeSystem()), None, _ts(), 1)
    assert res.d2_indices == [100]          # the genuinely-ambiguous candidate


def test_total_mode_reproduces_plain_entropy_ranking():
    # marginals 0.5 and 0.5 -> tie on total, though epistemic differs sharply
    p = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 1.0], [0.5, 1.0]])
    strat = DecompositionAcquisitionStrategy(_cfg(score="total"))
    states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    res = strat.select(_FakePool(states), _FakeMemberBackend(p, _FakeSystem()), None, _ts(), 2)
    assert sorted(res.d2_indices) == [100, 101]


def test_diagnostics_report_score_mode_and_members():
    p = np.array([[0.2, 0.6], [0.8, 0.6]])
    strat = DecompositionAcquisitionStrategy(_cfg(score="epistemic_bald"))
    states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    res = strat.select(_FakePool(states), _FakeMemberBackend(p, _FakeSystem()), None, _ts(), 1)
    assert res.diagnostics["score_mode"] == "epistemic_bald"
    assert res.diagnostics["n_members"] == 2
    assert "epistemic_mean" in res.diagnostics and "aleatoric_mean" in res.diagnostics


def test_backend_without_estimate_members_raises():
    class _NoMembers:
        system, device, attractor_radius = _FakeSystem(), "cpu", 0.2
    strat = DecompositionAcquisitionStrategy(_cfg())
    with pytest.raises(RuntimeError, match="estimate_members"):
        strat.select(_FakePool(np.zeros((2, 2), np.float32)), _NoMembers(), None, _ts(), 1)


def test_unknown_score_mode_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown score mode"):
        DecompositionAcquisitionStrategy(_cfg(score="bogus"))


def test_zero_target_count_skips_cleanly():
    strat = DecompositionAcquisitionStrategy(_cfg())
    res = strat.select(_FakePool(np.zeros((2, 2), np.float32)),
                       _FakeMemberBackend(np.full((2, 2), 0.5), _FakeSystem()),
                       None, _ts(), 0)
    assert res.d2_indices == [] and res.diagnostics["skipped_reason"] == "target_count_zero"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_decomposition_strategy.py -x -q`
Expected: FAIL — `ModuleNotFoundError: ...strategy.decomposition`

- [ ] **Step 3: Write the implementation**

```python
# adaptive_roa/adaptive_v2/strategy/decomposition.py
"""Acquisition on one component of an ensemble's uncertainty.

Threshold-free, like the entropy strategy: no lambda*, delta*, q_hat or decision
rule is read, so the training loop never depends on a fitted boundary. The only
difference between arms is which component of the uncertainty is scored, which
is what makes `aleatoric` a usable negative control -- it is the same code path
with one word changed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    select_greedy, select_greedy_diverse,
)
from adaptive_roa.adaptive_v2.strategy.uncertainty_scores import (
    SCORE_MODES, aleatoric_uncertainty, epistemic_bald, score_by_mode,
)
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState

_SELECTION_RULES = ("greedy", "greedy_diverse")


class DecompositionAcquisitionStrategy:
    mode = "decomposition"

    def __init__(self, cfg: Any):
        self.score_mode = str(cfg.score)
        if self.score_mode not in SCORE_MODES:
            raise ValueError(
                f"unknown score mode {self.score_mode!r}; expected one of {SCORE_MODES}")
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_candidates = int(cfg.n_candidates)
        self.selection_rule = str(cfg.selection_rule)
        if self.selection_rule not in _SELECTION_RULES:
            raise ValueError(
                f"selection_rule must be one of {_SELECTION_RULES}, got {self.selection_rule!r}")
        self.diversity_pool_multiplier = int(cfg.diversity_pool_multiplier)
        self.verbose = bool(cfg.verbose)

    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        # threshold_state is deliberately unread; see module docstring.
        if target_count <= 0:
            return self._skip("target_count_zero")

        system = getattr(probability_backend, "system", None)
        if system is None or not hasattr(probability_backend, "estimate_members"):
            raise RuntimeError(
                "DecompositionAcquisitionStrategy needs a backend exposing system and "
                f"estimate_members(); got {type(probability_backend).__name__}. A "
                "non-ensemble backend cannot separate aleatoric from epistemic."
            )

        states, indices = pool.sample_candidates_without_marking(
            self.n_candidates, exclude=exclude)
        n_actual = len(indices)
        if n_actual == 0:
            return self._skip("pool_exhausted")

        p_members = probability_backend.estimate_members(states, verbose=self.verbose)
        k = getattr(probability_backend, "member_sample_size", None)
        score = np.asarray(score_by_mode(self.score_mode, p_members, k), dtype=np.float64)
        score = np.where(np.isfinite(score), score, np.nan)

        positions = self._apply_selection_rule(score, np.asarray(states), system, target_count)
        selected = [indices[int(p)] for p in positions]

        diagnostics = self._diagnostics(p_members, score, positions, n_actual, k)
        if self.verbose:
            print(f"    [Decomposition/{self.score_mode}] {n_actual} candidates, "
                  f"M={diagnostics['n_members']}, selected {len(selected)}; "
                  f"epistemic_mean={diagnostics['epistemic_mean']:.5f} "
                  f"aleatoric_mean={diagnostics['aleatoric_mean']:.5f}")

        return AcquisitionResult(
            d1_indices=[], d2_indices=selected,
            n_candidates_evaluated=n_actual,
            n_certain_discarded=n_actual - len(selected),
            n_invalid_added=0, diagnostics=diagnostics,
        )

    @staticmethod
    def _circular_mask(system) -> np.ndarray:
        n = int(system.state_dim)
        mask = np.zeros(n, dtype=bool)
        idx = system.get_circular_indices()
        if idx:
            mask[np.asarray(idx, dtype=int)] = True
        return mask

    def _apply_selection_rule(self, score, states, system, target_count) -> np.ndarray:
        if self.selection_rule == "greedy":
            return select_greedy(score, target_count)
        return select_greedy_diverse(
            score, states,
            system.get_normalization_scales().cpu().numpy().astype(np.float64),
            self._circular_mask(system), target_count,
            pool_multiplier=self.diversity_pool_multiplier,
        )

    def _diagnostics(self, p_members, score, positions, n_actual, k) -> dict[str, Any]:
        finite = score[np.isfinite(score)]
        # Both components are recorded on every arm, not just the one being
        # scored: that is what lets the report show WHY an arm chose what it did.
        epi = epistemic_bald(p_members)
        ale = aleatoric_uncertainty(p_members)
        return {
            "score_mode": self.score_mode,
            "n_members": int(np.asarray(p_members).shape[0]),
            "member_sample_size": k,
            "score_min": float(finite.min()) if len(finite) else None,
            "score_max": float(finite.max()) if len(finite) else None,
            "score_mean": float(finite.mean()) if len(finite) else None,
            "score_mean_selected": float(np.nanmean(score[positions])) if len(positions) else None,
            "epistemic_mean": float(np.nanmean(epi)),
            "aleatoric_mean": float(np.nanmean(ale)),
            "epistemic_mean_selected": float(np.nanmean(epi[positions])) if len(positions) else None,
            "aleatoric_mean_selected": float(np.nanmean(ale[positions])) if len(positions) else None,
            "n_candidates_evaluated": int(n_actual),
            "selection_rule": self.selection_rule,
        }

    @staticmethod
    def _skip(reason: str) -> AcquisitionResult:
        return AcquisitionResult(
            d1_indices=[], d2_indices=[], n_candidates_evaluated=0,
            n_certain_discarded=0, n_invalid_added=0,
            diagnostics={"skipped_reason": reason},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_decomposition_strategy.py -q`
Expected: 7 passed

- [ ] **Step 5: Run the whole adaptive_v2 suite for regressions**

Run: `./env/bin/python -m pytest tests/adaptive_v2/ -q`
Expected: all pass; no existing test changed behaviour.

- [ ] **Step 6: Commit**

```bash
git add adaptive_roa/adaptive_v2/strategy/decomposition.py tests/adaptive_v2/test_decomposition_strategy.py
git commit -m "feat(acquisition): decomposition strategy with four score modes

One strategy, one word of config difference between arms, so aleatoric is a
true negative control. Diagnostics record BOTH components on every arm so the
report can show why an arm selected what it did."
```

---

### Task 4: Configs for the new arms

**Files:**
- Create: `configs/adaptive_v2/predictor/clf_ensemble.yaml`
- Create: `configs/adaptive_v2/probability/ensemble_classifier_prob.yaml`
- Create: `configs/adaptive_v2/acquisition/decomp_total.yaml`
- Create: `configs/adaptive_v2/acquisition/decomp_epi_var.yaml`
- Create: `configs/adaptive_v2/acquisition/decomp_epi_bald.yaml`
- Create: `configs/adaptive_v2/acquisition/decomp_aleat.yaml`
- Test: `tests/adaptive_v2/test_decomposition_configs.py`

**Interfaces:**
- Consumes: `DecompositionAcquisitionStrategy` (Task 3), `EnsembleClassifierProbabilityBackend` (Task 2).
- Produces: composable Hydra groups `predictor=clf_ensemble` and `acquisition=decomp_{total,epi_var,epi_bald,aleat}`.

FM predictor/probability configs are deferred to Task 6, where the trainer they depend on exists.

- [ ] **Step 1: Write the failing test**

```python
# tests/adaptive_v2/test_decomposition_configs.py
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_DIR = "/common/home/st1122/Projects/adaptive_roa/configs/adaptive_v2"
MODES = {"decomp_total": "total", "decomp_epi_var": "epistemic_var",
         "decomp_epi_bald": "epistemic_bald", "decomp_aleat": "aleatoric"}


@pytest.mark.parametrize("arm,mode", sorted(MODES.items()))
def test_each_arm_composes_and_selects_its_score(arm, mode):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default",
                      overrides=["system=pendulum_stoch", "noise_level=high",
                                 f"acquisition={arm}", "predictor=clf_ensemble"])
    assert cfg.acquisition.score == mode
    assert cfg.acquisition._target_.endswith("decomposition.DecompositionAcquisitionStrategy")
    assert cfg.sampling_mode == arm


def test_clf_ensemble_uses_the_ensemble_probability_backend():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default",
                      overrides=["system=pendulum_stoch", "noise_level=high",
                                 "acquisition=decomp_epi_var", "predictor=clf_ensemble"])
    assert cfg.probability._target_.endswith(
        "ensemble_prob.EnsembleClassifierProbabilityBackend")
    assert cfg.predictor.bnn.posterior == "ensemble"
    assert cfg.predictor.bnn.n_members == 5


def test_strategy_instantiates_from_config():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default",
                      overrides=["system=pendulum_stoch", "noise_level=high",
                                 "acquisition=decomp_epi_var", "predictor=clf_ensemble"])
    strat = instantiate(cfg.acquisition)
    assert strat.score_mode == "epistemic_var"
    assert strat.d2_ratio == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_decomposition_configs.py -x -q`
Expected: FAIL — Hydra `Could not find 'acquisition/decomp_total'`

- [ ] **Step 3: Write the configs**

```yaml
# configs/adaptive_v2/probability/ensemble_classifier_prob.yaml
# @package _global_
probability:
  _target_: adaptive_roa.adaptive_v2.probability.ensemble_prob.EnsembleClassifierProbabilityBackend
  attractor_radius: ${attractor_radius}
```

```yaml
# configs/adaptive_v2/predictor/clf_ensemble.yaml
# @package _global_
# Deep-ensemble classifier. Same trainer as bnn_ensemble; the difference is the
# probability backend, which exposes per-member predictions for the decomposition.
defaults:
  - /probability: ensemble_classifier_prob
threshold:
  decision_rule: one_sided
calibration:
  decision_rule: one_sided
predictor:
  type: classifier
  name: clf_ensemble
  trainer_target: adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer.BayesianMLPTrainer
  batch_size: 1024
  val_batch_size: 2048
  bnn:
    posterior: ensemble
    hidden_dims: [256, 512, 256]
    lr: 1.0e-3
    weight_decay: 1.0e-5
    max_epochs: 200
    patience: 20
    prior_sigma: 1.0
    n_members: 5
    n_marginal_samples: 64
  lightning_trainer:
    gradient_clip_val: 1.0
    log_every_n_steps: 10
```

```yaml
# configs/adaptive_v2/acquisition/decomp_total.yaml
# @package _global_
# Ensemble entropy of the marginal -- the current method, re-run on the ensemble
# so it is comparable to the epistemic arms without an ensembling confound.
sampling_mode: decomp_total
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.decomposition.DecompositionAcquisitionStrategy
  score: total
  d2_ratio: 1.0
  n_candidates: 50000
  selection_rule: greedy_diverse
  diversity_pool_multiplier: 5
  verbose: true
```

```yaml
# configs/adaptive_v2/acquisition/decomp_epi_var.yaml
# @package _global_
# Debiased between-member variance: unbiased at any K.
sampling_mode: decomp_epi_var
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.decomposition.DecompositionAcquisitionStrategy
  score: epistemic_var
  d2_ratio: 1.0
  n_candidates: 50000
  selection_rule: greedy_diverse
  diversity_pool_multiplier: 5
  verbose: true
```

```yaml
# configs/adaptive_v2/acquisition/decomp_epi_bald.yaml
# @package _global_
# Naive mutual information. Carries a ~(1/2K)(1-1/M) upward bias when p_m is
# MC-estimated; run as its own arm to measure whether that bias matters.
sampling_mode: decomp_epi_bald
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.decomposition.DecompositionAcquisitionStrategy
  score: epistemic_bald
  d2_ratio: 1.0
  n_candidates: 50000
  selection_rule: greedy_diverse
  diversity_pool_multiplier: 5
  verbose: true
```

```yaml
# configs/adaptive_v2/acquisition/decomp_aleat.yaml
# @package _global_
# NEGATIVE CONTROL: deliberately targets irreducible randomness. Expected to be
# the worst arm at high noise; if it is not, the decomposition is not working.
sampling_mode: decomp_aleat
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.decomposition.DecompositionAcquisitionStrategy
  score: aleatoric
  d2_ratio: 1.0
  n_candidates: 50000
  selection_rule: greedy_diverse
  diversity_pool_multiplier: 5
  verbose: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_decomposition_configs.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add configs/adaptive_v2/predictor/clf_ensemble.yaml \
        configs/adaptive_v2/probability/ensemble_classifier_prob.yaml \
        configs/adaptive_v2/acquisition/decomp_*.yaml \
        tests/adaptive_v2/test_decomposition_configs.py
git commit -m "feat(config): classifier-ensemble predictor and four decomposition arms"
```

---

### Task 5: Classifier end-to-end smoke test

Proves the classifier half works before any FM infrastructure exists — the cheap half of phase 1 becomes launchable at the end of this task.

**Files:**
- Test: `tests/adaptive_v2/test_decomposition_smoke.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: nothing importable; a launch gate.

- [ ] **Step 1: Write the test**

```python
# tests/adaptive_v2/test_decomposition_smoke.py
"""One tiny adaptive epoch per score mode, on the real engine.

Guards the failure that cost hours last campaign: an acquisition strategy that
raises on a predictor family is only discovered when every arm has burned GPU
time. This runs the real AdaptiveEngine on a 200-trajectory pool.
"""
import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = "/common/home/st1122/Projects/adaptive_roa/configs/adaptive_v2"
ARMS = ["decomp_total", "decomp_epi_var", "decomp_epi_bald", "decomp_aleat"]


@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_one_epoch_runs_for_each_score_mode(arm, tmp_path):
    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default", overrides=[
            "system=pendulum_stoch", "noise_level=high",
            f"acquisition={arm}", "predictor=clf_ensemble",
            "n_epochs=1", "initial_train_size=200", "samples_per_epoch=200",
            "acquisition.n_candidates=500",
            "predictor.bnn.n_members=2", "predictor.bnn.max_epochs=2",
            "eval.max_eval_rows=200", "eval.num_mc_samples_eval=4",
            f"output_dir={tmp_path}/{arm}",
        ])
        AdaptiveEngine(cfg).run()
    assert (tmp_path / arm / "epoch_000" / "full_roa_per_point.npz").exists()
```

- [ ] **Step 2: Run it**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_decomposition_smoke.py -q -m slow`
Expected: 4 passed. If a mode raises, fix the strategy or backend before continuing — do not proceed to FM.

- [ ] **Step 3: Verify the arms actually differ**

Run:
```bash
./env/bin/python - <<'EOF'
import json, glob
for f in sorted(glob.glob("/tmp/pytest-*/**/epoch_000/artifacts_v2.json", recursive=True)):
    d = json.load(open(f))["acquisition"]["diagnostics"]
    print(d.get("score_mode"), "epi=%.5f" % d.get("epistemic_mean", float("nan")),
          "ale=%.5f" % d.get("aleatoric_mean", float("nan")))
EOF
```
Expected: four different `score_mode` values; `epistemic_mean` and `aleatoric_mean` populated on all of them.

- [ ] **Step 4: Commit**

```bash
git add tests/adaptive_v2/test_decomposition_smoke.py
git commit -m "test(acquisition): end-to-end smoke for all four score modes"
```

---

### Task 6: Ensemble flow-matching handle and parallel-member trainer

The expensive half. FM members train **in parallel, one per GPU** — sequential members would be ~250h per arm.

**Files:**
- Create: `adaptive_roa/adaptive_v2/trainers/ensemble_flow_matching_trainer.py`
- Create: `configs/adaptive_v2/predictor/fm_ensemble.yaml`
- Create: `configs/adaptive_v2/probability/ensemble_endpoint_mc.yaml`
- Test: `tests/adaptive_v2/test_ensemble_fm_handle.py`

**Interfaces:**
- Consumes: `EnsembleEndpointMCProbabilityBackend` (Task 2), the existing `FlowMatchingTrainer` at `adaptive_roa/adaptive_v2/trainers/flow_matching_trainer.py`.
- Produces:
  - `EnsembleFlowMatcherHandle(members: list)` with `n_members: int`, `predict_endpoint_member(m: int, x: torch.Tensor) -> torch.Tensor`, and `predict_endpoint(x)` (round-robin over members, used only by legacy callers)
  - `EnsembleFlowMatchingTrainer(cfg, system, system_name)` with `fit(dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None)` returning a handle

- [ ] **Step 1: Write the failing handle tests**

```python
# tests/adaptive_v2/test_ensemble_fm_handle.py
import numpy as np
import pytest
import torch

from adaptive_roa.adaptive_v2.trainers.ensemble_flow_matching_trainer import (
    EnsembleFlowMatcherHandle,
)


class _ConstMember:
    def __init__(self, value):
        self.value = value
    def predict_endpoint(self, x, **kw):
        return torch.full((x.shape[0], 2), float(self.value))
    def eval(self):
        return self
    def to(self, device):
        return self


def test_handle_reports_member_count():
    h = EnsembleFlowMatcherHandle([_ConstMember(i) for i in range(5)])
    assert h.n_members == 5


def test_predict_endpoint_member_selects_that_member():
    h = EnsembleFlowMatcherHandle([_ConstMember(i) for i in range(3)])
    x = torch.zeros((4, 2))
    for m in range(3):
        assert torch.allclose(h.predict_endpoint_member(m, x), torch.full((4, 2), float(m)))


def test_predict_endpoint_round_robins_exactly_not_randomly():
    # Exact enumeration, matching EnsemblePosterior.predictive_logit_samples:
    # sampling members with a seeded generator gives a FIXED skewed weight vector.
    h = EnsembleFlowMatcherHandle([_ConstMember(i) for i in range(5)])
    x = torch.zeros((1, 2))
    seen = [float(h.predict_endpoint(x)[0, 0]) for _ in range(10)]
    assert seen == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


def test_single_member_handle_rejected():
    with pytest.raises(ValueError, match="at least 2 members"):
        EnsembleFlowMatcherHandle([_ConstMember(0)])


def test_out_of_range_member_raises():
    h = EnsembleFlowMatcherHandle([_ConstMember(i) for i in range(2)])
    with pytest.raises(IndexError):
        h.predict_endpoint_member(5, torch.zeros((1, 2)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_ensemble_fm_handle.py -x -q`
Expected: FAIL — `ModuleNotFoundError: ...ensemble_flow_matching_trainer`

- [ ] **Step 3: Implement the handle and trainer**

```python
# adaptive_roa/adaptive_v2/trainers/ensemble_flow_matching_trainer.py
"""Deep ensemble of flow matchers, trained one member per GPU in parallel.

Members are statistically independent, so training them is embarrassingly
parallel: M processes, member m pinned to device m, no communication. Wall-clock
equals a SINGLE member. The sequential loop used for the classifier ensemble
would be ~M x 50h per arm here, which is why this exists.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

from adaptive_roa.adaptive_v2.trainers.flow_matching_trainer import FlowMatchingTrainer


class EnsembleFlowMatcherHandle:
    """M flow matchers behind one object.

    `predict_endpoint` round-robins members EXACTLY rather than sampling one at
    random. EnsemblePosterior.predictive_logit_samples documents why: a seeded
    generator produced fixed weights [.125, .281, .109, .234, .250] against an
    exact .2, "enough to flip decisions near lambda*". A biased marginal is a
    systematic error, not noise that averages out.
    """

    def __init__(self, members: list):
        members = list(members)
        if len(members) < 2:
            raise ValueError(
                f"EnsembleFlowMatcherHandle needs at least 2 members, got {len(members)}")
        self.members = members
        self._cursor = 0

    @property
    def n_members(self) -> int:
        return len(self.members)

    def predict_endpoint_member(self, m: int, x: torch.Tensor, **kw) -> torch.Tensor:
        if not 0 <= m < len(self.members):
            raise IndexError(f"member {m} out of range for {len(self.members)} members")
        return self.members[m].predict_endpoint(x, **kw)

    def predict_endpoint(self, x: torch.Tensor, **kw) -> torch.Tensor:
        m = self._cursor % len(self.members)
        self._cursor += 1
        return self.predict_endpoint_member(m, x, **kw)

    def eval(self):
        for mem in self.members:
            if hasattr(mem, "eval"):
                mem.eval()
        return self


def _train_one_member(rank: int, cfg_blob, dataset_files, out_dir, resume, seed_base):
    """Child process: train member `rank` on device `rank`."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.manual_seed(seed_base + rank)
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg_blob)
    # Each child sees exactly one GPU, so it must address it as device 0.
    OmegaConf.update(cfg, "predictor.lightning_trainer.devices", 1, force_add=True)
    trainer = FlowMatchingTrainer(cfg, None, None)
    member_dir = Path(out_dir) / f"member_{rank}"
    member_dir.mkdir(parents=True, exist_ok=True)
    trainer.fit(dataset_files, str(member_dir), resume_checkpoint=resume)


class EnsembleFlowMatchingTrainer:
    """Trains M flow matchers concurrently and assembles them into one handle."""

    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name
        self.n_members = int(cfg.predictor.ensemble.n_members)
        self.seed_base = int(cfg.get("seed", 42))

    def fit(self, dataset_files: dict, output_dir: str,
            resume_checkpoint: str | None = None):
        from omegaconf import OmegaConf
        blob = OmegaConf.to_container(self.cfg, resolve=True)
        mp.spawn(
            _train_one_member,
            args=(blob, dataset_files, output_dir, resume_checkpoint, self.seed_base),
            nprocs=self.n_members, join=True,
        )
        members = [self._load_member(output_dir, m) for m in range(self.n_members)]
        return EnsembleFlowMatcherHandle(members)

    def _load_member(self, output_dir: str, m: int):
        """Reload member m from disk.

        The child process already reloaded its own best checkpoint, but that
        object cannot cross the process boundary, so the parent reloads from
        disk using the same path FlowMatchingTrainer.fit uses: glob best*.ckpt
        and let Lightning reconstruct from the saved hyperparameters.
        """
        import glob

        from hydra.utils import get_class

        member_dir = Path(output_dir) / f"member_{m}"
        ckpts = sorted(glob.glob(str(member_dir / "**" / "best*.ckpt"), recursive=True))
        if not ckpts:
            raise FileNotFoundError(
                f"no best*.ckpt under {member_dir}; member {m} did not finish training. "
                "Assembling an ensemble from a partially-trained member would report a "
                "silently wrong epistemic estimate."
            )
        cls = get_class(self.cfg.flow_matcher._target_)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            return cls.load_from_checkpoint(ckpts[0], device=device)
        except Exception as exc:  # mirrors the fallback in FlowMatchingTrainer.fit
            print(f"Warning: load_from_checkpoint failed for member {m} ({exc}); "
                  "loading state dict directly")
            import hydra

            model = hydra.utils.instantiate(self.cfg.model)
            ckpt = torch.load(ckpts[0], map_location="cpu", weights_only=False)
            model.load_state_dict({k.replace("model.", ""): v
                                   for k, v in ckpt["state_dict"].items()
                                   if k.startswith("model.")})
            member = cls(system=self.system, model=model, optimizer=self.cfg.optimizer)
            return member
```

- [ ] **Step 4: Run handle tests to verify they pass**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_ensemble_fm_handle.py -q`
Expected: 5 passed

- [ ] **Step 5: Write the FM configs**

```yaml
# configs/adaptive_v2/probability/ensemble_endpoint_mc.yaml
# @package _global_
probability:
  _target_: adaptive_roa.adaptive_v2.probability.ensemble_prob.EnsembleEndpointMCProbabilityBackend
  attractor_radius: ${attractor_radius}
  num_mc_samples: 20
```

```yaml
# configs/adaptive_v2/predictor/fm_ensemble.yaml
# @package _global_
# Deep ensemble of flow matchers, members trained one per GPU in parallel.
# Requires n_members GPUs on the node.
defaults:
  - /probability: ensemble_endpoint_mc
predictor:
  type: generative
  name: fm_ensemble
  trainer_target: adaptive_roa.adaptive_v2.trainers.ensemble_flow_matching_trainer.EnsembleFlowMatchingTrainer
  batch_size: 1024
  val_batch_size: 2048
  ensemble:
    n_members: 5
  flow_matching:
    latent_dim: ${flow_matching.latent_dim}
    num_integration_steps: 100
    mae_val_frequency: 10
    use_loss_weights: false
    use_log_loss_weights: false
    use_manifold: false
    clamp_noise: false
    zero_latent: true
    noise_scale: 1.0
    quat_loss_weight: 1.0
  lightning_trainer:
    max_epochs: 200
    accelerator: gpu
    devices: 1
    precision: 32
    gradient_clip_val: 1.0
    log_every_n_steps: 10
    check_val_every_n_epoch: 1
    enable_progress_bar: true
    enable_model_summary: true
    callbacks:
      - _target_: lightning.pytorch.callbacks.ModelCheckpoint
        monitor: val_loss
        mode: min
        save_top_k: 1
        save_last: true
        filename: "best-{epoch:02d}-{val_loss:.4f}"
        auto_insert_metric_name: false
      - _target_: lightning.pytorch.callbacks.EarlyStopping
        monitor: val_loss
        mode: min
        patience: 100
        verbose: true
        min_delta: 0.0001
```

- [ ] **Step 6: Verify the FM configs compose**

Run:
```bash
./env/bin/python scripts/run_adaptive.py --cfg job --resolve \
  system=pendulum_stoch noise_level=high acquisition=decomp_epi_var \
  predictor=fm_ensemble 2>&1 | grep -E "n_members|_target_.*Ensemble|score:"
```
Expected: shows `n_members: 5`, both `Ensemble*` targets, and `score: epistemic_var`.

- [ ] **Step 7: Commit**

```bash
git add adaptive_roa/adaptive_v2/trainers/ensemble_flow_matching_trainer.py \
        configs/adaptive_v2/predictor/fm_ensemble.yaml \
        configs/adaptive_v2/probability/ensemble_endpoint_mc.yaml \
        tests/adaptive_v2/test_ensemble_fm_handle.py
git commit -m "feat(fm): parallel-member ensemble trainer and handle

Members train one per GPU concurrently; wall-clock equals a single member.
predict_endpoint round-robins members exactly rather than sampling, matching
the reasoning already documented on EnsemblePosterior."
```

---

### Task 7: Member-stratified eval in full_roa

**Files:**
- Modify: `adaptive_roa/adaptive_v2/eval/full_roa.py` (the MC sampling loop, around lines 735–760)
- Test: `tests/adaptive_v2/test_full_roa_ensemble_weighting.py`

**Interfaces:**
- Consumes: `EnsembleFlowMatcherHandle` (Task 6).
- Produces: no new public API; behaviour gated on `hasattr(flow_matcher, "n_members")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/adaptive_v2/test_full_roa_ensemble_weighting.py
"""K eval samples must be split EVENLY across members.

Sampling members instead gives a fixed skewed weight vector under a seeded
generator, which biases the marginal systematically rather than averaging out.
"""
import numpy as np
import torch

from adaptive_roa.adaptive_v2.trainers.ensemble_flow_matching_trainer import (
    EnsembleFlowMatcherHandle,
)


class _CountingMember:
    def __init__(self, idx, counter):
        self.idx, self.counter = idx, counter
    def predict_endpoint(self, x, **kw):
        self.counter[self.idx] += 1
        return torch.zeros((x.shape[0], 2))
    def eval(self):
        return self


def test_each_member_is_used_exactly_k_over_m_times():
    counter = {i: 0 for i in range(5)}
    h = EnsembleFlowMatcherHandle([_CountingMember(i, counter) for i in range(5)])
    for _ in range(100):                       # K = 100
        h.predict_endpoint(torch.zeros((3, 2)))
    assert set(counter.values()) == {20}       # exactly K/M each, no skew


def test_uneven_k_distributes_within_one_call_of_balanced():
    counter = {i: 0 for i in range(3)}
    h = EnsembleFlowMatcherHandle([_CountingMember(i, counter) for i in range(3)])
    for _ in range(10):
        h.predict_endpoint(torch.zeros((1, 2)))
    assert max(counter.values()) - min(counter.values()) <= 1
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_full_roa_ensemble_weighting.py -q`
Expected: PASS if Task 6's round-robin is correct. If it fails, the handle samples rather than enumerates — fix the handle, not the test.

- [ ] **Step 3: Add the gated branch in full_roa.py**

Read the MC loop near line 735. The current inner loop is:

```python
for sample_idx in range(num_mc_samples):
    pred = flow_matcher.predict_endpoint(batch_inputs)
```

Replace with a version that pins the member explicitly when the handle is an ensemble, so member usage does not depend on call-ordering state:

```python
n_members = getattr(flow_matcher, "n_members", None)
for sample_idx in range(num_mc_samples):
    if n_members:
        # Exact enumeration: sample_idx % M gives each member floor(K/M) or
        # ceil(K/M) draws. Relying on the handle's internal cursor instead
        # would make the weighting depend on how many times anything else
        # happened to call predict_endpoint first.
        pred = flow_matcher.predict_endpoint_member(sample_idx % n_members, batch_inputs)
    else:
        pred = flow_matcher.predict_endpoint(batch_inputs)
```

- [ ] **Step 4: Verify single-model eval is unchanged**

Run: `./env/bin/python -m pytest tests/adaptive_v2/test_evaluator.py -q`
Expected: all pass — the `n_members` attribute is absent on single models, so the `else` branch runs exactly as before.

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/adaptive_v2/eval/full_roa.py tests/adaptive_v2/test_full_roa_ensemble_weighting.py
git commit -m "feat(eval): split K eval samples evenly across ensemble members

Pins the member by sample index rather than relying on handle state, so the
marginal weighting cannot depend on prior call ordering. Gated on n_members;
single-model eval is byte-identical."
```

---

### Task 8: Experiment log

**Files:**
- Create: `scripts/exp_log.py`
- Create: `docs/experiments/ensemble_epistemic/LOG.md`
- Test: `tests/test_exp_log.py`

**Interfaces:**
- Consumes: nothing.
- Produces: CLI `append` / `update-status` / `report`, and `code_hash(paths) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exp_log.py
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = "/common/home/st1122/Projects/adaptive_roa/scripts/exp_log.py"
PY = "/common/home/st1122/Projects/adaptive_roa/env/bin/python"


def _run(*args, cwd):
    return subprocess.run([PY, SCRIPT, *args], cwd=cwd, capture_output=True, text=True)


def test_append_then_report_roundtrip(tmp_path):
    log = tmp_path / "runs.jsonl"
    r = _run("append", "--log", str(log), "--run-id", "fm_high_epi_var",
             "--system", "pendulum_stoch", "--level", "high", "--predictor", "fm",
             "--arm", "epi_var", "--cluster", "amarel", "--job-id", "123",
             "--output-dir", "/scratch/x", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    rec = json.loads(log.read_text().strip())
    assert rec["run_id"] == "fm_high_epi_var"
    assert rec["status"] == "launched"
    assert len(rec["code_hash"]) == 12          # short content hash
    assert rec["launched_at"].startswith("20")


def test_update_status_rewrites_only_that_run(tmp_path):
    log = tmp_path / "runs.jsonl"
    for rid in ("a", "b"):
        _run("append", "--log", str(log), "--run-id", rid, "--system", "s",
             "--level", "high", "--predictor", "clf", "--arm", "total",
             "--cluster", "ilab", "--job-id", "1", "--output-dir", "/x", cwd=tmp_path)
    _run("update-status", "--log", str(log), "--run-id", "a", "--status", "preempted",
         cwd=tmp_path)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert {r["run_id"]: r["status"] for r in recs} == {"a": "preempted", "b": "launched"}


def test_report_groups_by_status(tmp_path):
    log = tmp_path / "runs.jsonl"
    _run("append", "--log", str(log), "--run-id", "a", "--system", "s", "--level",
         "high", "--predictor", "clf", "--arm", "total", "--cluster", "ilab",
         "--job-id", "1", "--output-dir", "/x", cwd=tmp_path)
    out = _run("report", "--log", str(log), cwd=tmp_path).stdout
    assert "launched" in out and "a" in out


def test_missing_run_id_on_update_is_an_error(tmp_path):
    log = tmp_path / "runs.jsonl"
    log.write_text("")
    r = _run("update-status", "--log", str(log), "--run-id", "ghost",
             "--status", "done", cwd=tmp_path)
    assert r.returncode != 0
    assert "ghost" in (r.stderr + r.stdout)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/test_exp_log.py -x -q`
Expected: FAIL — script does not exist

- [ ] **Step 3: Write the script**

```python
#!/usr/bin/env python
"""Append-only experiment log for the ensemble-epistemic campaign.

Every launched run gets one JSON line. `code_hash` is a content hash over the
source files that determine a run's behaviour: the previous campaign rsynced
UNCOMMITTED working-tree changes to a second cluster, and there is now no way to
reconstruct which code any given run used. A git SHA would not have helped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HASHED = [
    "adaptive_roa/adaptive_v2/strategy/decomposition.py",
    "adaptive_roa/adaptive_v2/strategy/uncertainty_scores.py",
    "adaptive_roa/adaptive_v2/probability/ensemble_prob.py",
    "adaptive_roa/adaptive_v2/trainers/ensemble_flow_matching_trainer.py",
    "adaptive_roa/adaptive_v2/eval/full_roa.py",
]


def code_hash(paths: list[str] | None = None) -> str:
    h = hashlib.sha256()
    for rel in sorted(paths or HASHED):
        p = REPO / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:12]


def _read(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(l) for l in log.read_text().splitlines() if l.strip()]


def _write(log: Path, recs: list[dict]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("".join(json.dumps(r) + "\n" for r in recs))


def cmd_append(a) -> int:
    log = Path(a.log)
    recs = _read(log)
    recs.append({
        "run_id": a.run_id,
        "launched_at": datetime.now().isoformat(timespec="seconds"),
        "system": a.system, "level": a.level, "predictor": a.predictor, "arm": a.arm,
        "score_mode": a.score_mode, "seed": a.seed, "n_members": a.n_members,
        "k_acq": a.k_acq, "d2_ratio": a.d2_ratio, "cluster": a.cluster,
        "job_id": a.job_id, "output_dir": a.output_dir,
        "code_hash": code_hash(), "status": "launched", "notes": a.notes or "",
    })
    _write(log, recs)
    print(f"appended {a.run_id} ({recs[-1]['code_hash']})")
    return 0


def cmd_update(a) -> int:
    log = Path(a.log)
    recs = _read(log)
    hit = [r for r in recs if r["run_id"] == a.run_id]
    if not hit:
        print(f"no run with run_id {a.run_id!r} in {log}", file=sys.stderr)
        return 1
    for r in hit:
        r["status"] = a.status
        if a.notes:
            r["notes"] = (r.get("notes", "") + " | " + a.notes).strip(" |")
    _write(log, recs)
    print(f"{a.run_id} -> {a.status}")
    return 0


def cmd_report(a) -> int:
    recs = _read(Path(a.log))
    if not recs:
        print("(empty log)")
        return 0
    by: dict[str, list[dict]] = {}
    for r in recs:
        by.setdefault(r["status"], []).append(r)
    for status in sorted(by):
        print(f"\n== {status} ({len(by[status])}) ==")
        for r in sorted(by[status], key=lambda x: x["run_id"]):
            print(f"  {r['run_id']:<28} {r['predictor']:<4} {r['level']:<6} "
                  f"{r['arm']:<10} {r['cluster']:<8} job={r['job_id']:<10} "
                  f"code={r['code_hash']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_a = sub.add_parser("append")
    ap_a.add_argument("--log", required=True)
    for f in ("run-id", "system", "level", "predictor", "arm", "cluster",
              "job-id", "output-dir"):
        ap_a.add_argument(f"--{f}", required=True)
    ap_a.add_argument("--score-mode", default="")
    ap_a.add_argument("--seed", type=int, default=42)
    ap_a.add_argument("--n-members", type=int, default=5)
    ap_a.add_argument("--k-acq", type=int, default=20)
    ap_a.add_argument("--d2-ratio", type=float, default=1.0)
    ap_a.add_argument("--notes", default="")
    ap_a.set_defaults(func=cmd_append)

    ap_u = sub.add_parser("update-status")
    ap_u.add_argument("--log", required=True)
    ap_u.add_argument("--run-id", required=True)
    ap_u.add_argument("--status", required=True)
    ap_u.add_argument("--notes", default="")
    ap_u.set_defaults(func=cmd_update)

    ap_r = sub.add_parser("report")
    ap_r.add_argument("--log", required=True)
    ap_r.set_defaults(func=cmd_report)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/test_exp_log.py -q`
Expected: 4 passed

- [ ] **Step 5: Seed the narrative log**

```bash
mkdir -p docs/experiments/ensemble_epistemic
cat > docs/experiments/ensemble_epistemic/LOG.md <<'EOF'
# Ensemble Epistemic Acquisition — Experiment Log

Design: `docs/superpowers/specs/2026-08-04-ensemble-epistemic-acquisition-design.md`
Machine-readable run records: `runs.jsonl` (see `scripts/exp_log.py`).

Append newest entries at the top. Record what was launched, what broke, what was
decided, and why — the "why" is the part that is impossible to reconstruct later.

## 2026-08-04 — plan approved, implementation started
EOF
```

- [ ] **Step 6: Commit**

```bash
git add scripts/exp_log.py tests/test_exp_log.py docs/experiments/ensemble_epistemic/LOG.md
git commit -m "feat(exp): append-only experiment log with code content hashing

Records a content hash of the behaviour-determining source files per run. The
previous campaign rsynced uncommitted changes to a second cluster, leaving no
way to reconstruct which code a run used."
```

---

## Self-Review

**Spec coverage.** Decomposition and both estimators → Task 1. Ensemble backends with `estimate_members` → Task 2. Decomposition strategy with four score modes → Task 3. Configs → Tasks 4 and 6. Parallel-member FM trainer and handle → Task 6. Gated `full_roa` member weighting → Task 7. Experiment log with `code_hash` → Task 8. Pre-launch smoke tests → Task 5. Epoch-0 identity and the deterministic null test are campaign-time analyses using the existing validated machinery (`stoch_prob_metrics.py`, `stoch_compare_report.py`), not new code, so they are correctly absent from the build plan.

**Type consistency.** `estimate_members(states, verbose=False) -> [M, N]` and `member_sample_size` are defined in Task 2 and consumed unchanged in Task 3. `n_members` and `predict_endpoint_member(m, x)` are defined in Task 6 and consumed in Task 7. `score_by_mode(mode, p_members, k)` is defined in Task 1 and consumed in Task 3.

**Member reload.** `FlowMatchingTrainer` has no checkpoint-loading method — `fit` reloads inline by globbing `best*.ckpt` and calling `load_from_checkpoint`, with a state-dict fallback. Task 6's `_load_member` mirrors that exact path (including the fallback) rather than inventing a format, and raises if a member produced no checkpoint: assembling an ensemble from a partially-trained member would report a silently wrong epistemic estimate.

**Ordering.** Tasks 1–5 deliver the entire classifier half and are independently launchable — the cheap phase-1 runs can start after Task 5, before any FM work exists.

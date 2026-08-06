# Final-State Predictor Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add final-state (endpoint-predicting) predictor arms — deterministic MLP plus MFVI/ensemble/Laplace Bayesian variants — that emit a manifold-aware predictive *distribution* over `x_T` and plug into the existing endpoint-MC probability backend unchanged.

**Architecture:** Reuses the `(backbone, posterior, head)` triple from the outcome family. The posteriors already exist; this plan adds the *head* — a per-manifold-component likelihood (`Real` Gaussian, `SO2` wrapped normal, `SO3` tangent-space) built generically from `system.manifold_components` — and a `FinalStateModelHandle` satisfying the endpoint-MC contract. Probability comes from sampling K endpoints and counting `system.classify_attractor` labels, exactly as flow matching already does.

**Tech Stack:** PyTorch 2.5.1, Lightning, Hydra, pytest. No new pip dependencies.

**Design spec:** `docs/superpowers/specs/2026-07-27-probabilistic-predictors-design.md` (sections "Final-state head", "Manifold-aware likelihood", "Two head refinements").

**Prior plan:** `docs/superpowers/plans/2026-07-28-probabilistic-predictors-foundation.md` built `adaptive_roa/predictors/{posteriors,bayesian_mlp,handles}.py` and `OutcomeModelHandle`. This plan extends that package.

## Global Constraints

- Branch: `predictors-final-state`. Repo root: `/common/home/st1122/Projects/adaptive_roa`.
- Python interpreter is the repo-local env: **`./env/bin/python`**. Never plain `python`.
- Run tests as `./env/bin/python -m pytest <path> -v` from the repo root.
- Import root is `adaptive_roa` (editable install).
- **No new pip dependencies.**
- **Commit messages must contain zero AI/tool attribution** — no `Co-Authored-By`, no `Claude-Session`, no tool or model names. Hard user preference.
- **Another session commits to this repo concurrently.** Stage explicit paths only; never `git add -A` or `git commit -a`. Never run `git checkout` or switch branches.
- The 64 tests under `tests/predictors/` must keep passing.
- `predictor.type` is the FAMILY tag and must be `generative` for every arm here (that is what routes the endpoint dataset kind, the endpoint-MC backend, and the `evaluate_full_roa_fast` eval branch). `predictor.name` is the unique ARM name.
- Prior for Bayesian arms: isotropic Gaussian, `prior_sigma` identical across arms. MFVI headline runs use `kl_weight: 1.0` (untempered).

## The endpoint-MC contract (verified against `main`)

A bound model handle **must** implement these, all called **unguarded**:

| Requirement | Call site | Contract |
|---|---|---|
| `eval()`, `to(device)` | `engine.py:137-138`, `probability_estimator.py:46` | nn.Module-shaped |
| `predict_endpoint(states)` | `probability_estimator.py:143`, `:233`; `full_roa.py:600`; `endpoint_evaluation.py:114` | `[B, state_dim] -> [B, state_dim]`, **RAW** coordinates, **single positional arg**, **fresh sample every call** |
| `get_manifold_component_names()` | `endpoint_evaluation.py:100` | `list[str]` |
| `compute_manifold_distance_per_component(pred, true)` | `endpoint_evaluation.py:116` | `[B,D],[B,D] -> [B, len(names)]` |
| `distance_manifold.dist(x, y)` | `full_roa.py:621,649` (hasattr-guarded, but `get_manifold_component_names` at `:643` is guarded by the **same** check — supply both or neither) | `[B,D],[B,D] -> [B, C]` |

`refine_endpoints` is only needed when `refine_invalids: true`, which the default `endpoint_mc.yaml` leaves false.

**Two semantic obligations that silently corrupt results if missed:**

1. `predict_endpoint` output must be in the exact space `system.classify_attractor` consumes. Angles wrapped to `[-pi, pi]` for pendulum/cartpole/quad2d.
2. **Quadrotor3D quaternions must be unit-norm AND canonicalized to `qw >= 0`.** `Quadrotor3DSystem.classify_attractor` (`systems/quadrotor3d.py:260`) uses plain L2 over the raw 13-vector against an identity-quaternion goal, so `-q` — the same rotation — sits at L2 distance 2 and every near-goal endpoint is misclassified as separatrix. The flow matcher survives only because it calls `project_to_manifold` before returning (`quadrotor_3d/latent_conditional/flow_matcher.py:412-415`).

**Component-name cardinality.** Generating one name per `Real` dim, one per `SO2`, and one per `SO3` reproduces the flow matcher's hand-written counts exactly: pendulum 2, cartpole 4, quad2d 6, quad3d 10. Verified against the four `get_manifold_component_names` overrides.

## File Structure

**Create:**
- `adaptive_roa/predictors/manifold_likelihood.py` — `RealLikelihood`, `SO2Likelihood`, `SO3Likelihood`; each owns its parameter count, NLL, sampling, and geodesic distance
- `adaptive_roa/predictors/heads.py` — `FinalStateHead` assembling per-component likelihoods from `system.manifold_components`
- `adaptive_roa/predictors/final_state_handle.py` — `FinalStateModelHandle`
- `adaptive_roa/adaptive_v2/trainers/final_state_trainer.py` — `FinalStateTrainer`
- `adaptive_roa/probabilistic_classifier/bayesian_final_state.py` — export wrappers
- `configs/adaptive_v2/predictor/{mlp_det,bnn_mfvi_reg,bnn_ensemble_reg,bnn_laplace_reg}.yaml`
- `tests/predictors/{test_manifold_likelihood.py,test_final_state_head.py,test_final_state_handle.py,test_final_state_trainer.py,test_final_state_e2e.py}`

**Modify:**
- `adaptive_roa/predictors/bayesian_mlp.py` — `build_from_cfg` gains a head-aware output width

---

### Task 1: `RealLikelihood` — Gaussian component with beta-NLL

**Files:**
- Create: `adaptive_roa/predictors/manifold_likelihood.py`
- Test: `tests/predictors/test_manifold_likelihood.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ComponentLikelihood` ABC with `n_params(dim) -> int`, `nll(params, target, beta) -> Tensor[B]`, `sample(params, generator) -> Tensor[B, dim]`, `mean(params) -> Tensor[B, dim]`, `distance(a, b) -> Tensor[B, n_dist]`, `n_dist(dim) -> int`, `names(dim, base) -> list[str]`; and `RealLikelihood`.

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_manifold_likelihood.py`:

```python
import math

import pytest
import torch

from adaptive_roa.predictors.manifold_likelihood import RealLikelihood


def test_real_needs_two_params_per_dim():
    assert RealLikelihood().n_params(3) == 6


def test_real_nll_matches_the_closed_form_gaussian():
    """Closed form: 0.5*log(2*pi*sigma^2) + (y-mu)^2/(2*sigma^2), summed over dims."""
    lik = RealLikelihood()
    mu = torch.tensor([[1.0, -2.0]])
    log_sigma = torch.tensor([[0.0, math.log(2.0)]])  # sigma = 1.0, 2.0
    params = torch.cat([mu, log_sigma], dim=-1)
    y = torch.tensor([[1.5, -1.0]])

    expected = 0.0
    for m, s, t in ((1.0, 1.0, 1.5), (-2.0, 2.0, -1.0)):
        expected += 0.5 * math.log(2 * math.pi * s ** 2) + (t - m) ** 2 / (2 * s ** 2)

    assert lik.nll(params, y, beta=0.0).item() == pytest.approx(expected, rel=1e-5)


def test_beta_nll_reweights_by_sigma_but_beta_zero_is_plain_nll():
    """beta-NLL multiplies each dim's loss by sigma^(2*beta), detached (Seitzer 2022).
    At beta=0 the weight is 1 and it must reduce exactly to plain Gaussian NLL."""
    lik = RealLikelihood()
    params = torch.cat([torch.zeros(1, 2), torch.tensor([[0.0, math.log(3.0)]])], dim=-1)
    y = torch.ones(1, 2)

    plain = lik.nll(params, y, beta=0.0)
    weighted = lik.nll(params, y, beta=0.5)
    assert not torch.allclose(plain, weighted)

    # Reconstruct the beta=0.5 value from per-dim plain terms times sigma^(2*beta).
    per_dim = []
    for s, t in ((1.0, 1.0), (3.0, 1.0)):
        per_dim.append(0.5 * math.log(2 * math.pi * s ** 2) + t ** 2 / (2 * s ** 2))
    expected = per_dim[0] * (1.0 ** 1.0) + per_dim[1] * (3.0 ** 1.0)
    assert weighted.item() == pytest.approx(expected, rel=1e-5)


def test_real_sample_is_stochastic_and_reproducible_under_a_generator():
    lik = RealLikelihood()
    params = torch.cat([torch.zeros(4, 2), torch.zeros(4, 2)], dim=-1)
    assert not torch.allclose(lik.sample(params), lik.sample(params))
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    assert torch.allclose(lik.sample(params, generator=g1), lik.sample(params, generator=g2))


def test_real_sample_recovers_the_parameters_in_expectation():
    lik = RealLikelihood()
    mu = torch.tensor([[2.0, -1.0]])
    params = torch.cat([mu, torch.log(torch.tensor([[0.5, 1.5]]))], dim=-1)
    draws = torch.stack([lik.sample(params.expand(2000, -1)) for _ in range(1)], dim=0)[0]
    assert torch.allclose(draws.mean(0), mu[0], atol=0.1)
    assert torch.allclose(draws.std(0), torch.tensor([0.5, 1.5]), rtol=0.15)


def test_real_distance_is_absolute_difference_per_dim():
    lik = RealLikelihood()
    a = torch.tensor([[1.0, 5.0]])
    b = torch.tensor([[3.0, 1.0]])
    assert lik.n_dist(2) == 2
    assert torch.allclose(lik.distance(a, b), torch.tensor([[2.0, 4.0]]))


def test_real_names_are_one_per_dim():
    assert RealLikelihood().names(3, "velocity") == ["velocity_0", "velocity_1", "velocity_2"]
    assert RealLikelihood().names(1, "cart_position") == ["cart_position"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_manifold_likelihood.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.predictors.manifold_likelihood'`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/manifold_likelihood.py`:

```python
"""Per-manifold-component predictive likelihoods for final-state prediction.

A final-state predictor emits a DISTRIBUTION over x_T, not a point. The
distribution factorizes over the system's manifold components, and each
component type needs its own likelihood: a Euclidean Gaussian will not do for
an angle (pendulum's failure attractors sit at theta = +/-pi, exactly the wrap
seam) nor for a quaternion (where -q is the same rotation as q).

Each likelihood owns four things: how many parameters it consumes, its NLL, how
to sample from it, and its geodesic distance (used for endpoint-error reporting).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List

import torch


def _split_mu_logsigma(params: torch.Tensor, dim: int):
    return params[..., :dim], params[..., dim:2 * dim]


class ComponentLikelihood(ABC):
    """One manifold component's predictive distribution."""

    @abstractmethod
    def n_params(self, dim: int) -> int:
        """How many head outputs this component consumes for a `dim`-wide slice."""

    @abstractmethod
    def nll(self, params: torch.Tensor, target: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
        """Negative log-likelihood per batch row: [B, n_params], [B, dim] -> [B]."""

    @abstractmethod
    def sample(self, params: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """One draw: [B, n_params] -> [B, dim], in RAW state coordinates."""

    @abstractmethod
    def mean(self, params: torch.Tensor) -> torch.Tensor:
        """Distribution mean: [B, n_params] -> [B, dim], RAW coordinates."""

    @abstractmethod
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Geodesic distance per reported sub-component: [B,dim],[B,dim] -> [B, n_dist]."""

    @abstractmethod
    def n_dist(self, dim: int) -> int:
        """How many distance values `distance` returns."""

    @abstractmethod
    def names(self, dim: int, base: str) -> List[str]:
        """Reported names, one per value `distance` returns."""


class RealLikelihood(ComponentLikelihood):
    """Diagonal Gaussian over a Euclidean slice, with optional beta-NLL weighting.

    beta-NLL (Seitzer et al., ICLR 2022) multiplies each dimension's loss by a
    DETACHED sigma^(2*beta). Plain Gaussian NLL shrinks the gradient on
    high-error points as sigma grows there, starving exactly the points that
    need fitting; beta = 0.5 restores most of that signal. beta = 0 is plain NLL.
    """

    LOG_SIGMA_MIN = -7.0
    LOG_SIGMA_MAX = 7.0

    def n_params(self, dim: int) -> int:
        return 2 * int(dim)

    def _sigma(self, params: torch.Tensor, dim: int) -> torch.Tensor:
        _, log_sigma = _split_mu_logsigma(params, dim)
        return log_sigma.clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX).exp()

    def nll(self, params, target, beta: float = 0.0):
        dim = target.shape[-1]
        mu, _ = _split_mu_logsigma(params, dim)
        sigma = self._sigma(params, dim)
        per_dim = 0.5 * torch.log(2 * math.pi * sigma ** 2) + (target - mu) ** 2 / (2 * sigma ** 2)
        if beta > 0.0:
            # Detached: the weight must not create a gradient path of its own.
            per_dim = per_dim * (sigma.detach() ** (2.0 * beta))
        return per_dim.sum(dim=-1)

    def sample(self, params, generator=None):
        dim = params.shape[-1] // 2
        mu, _ = _split_mu_logsigma(params, dim)
        sigma = self._sigma(params, dim)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        return mu + sigma * eps

    def mean(self, params):
        dim = params.shape[-1] // 2
        return _split_mu_logsigma(params, dim)[0]

    def distance(self, a, b):
        return (a - b).abs()

    def n_dist(self, dim: int) -> int:
        return int(dim)

    def names(self, dim: int, base: str) -> List[str]:
        if int(dim) == 1:
            return [base]
        return [f"{base}_{i}" for i in range(int(dim))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_manifold_likelihood.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/manifold_likelihood.py tests/predictors/test_manifold_likelihood.py
git commit -m "feat(predictors): Euclidean component likelihood with beta-NLL"
```

---

### Task 2: `SO2Likelihood` — wrapped normal over an angle

The seam matters: pendulum's failure attractors sit at `theta = +/-pi`. A Euclidean Gaussian on `theta` puts mass in a false mode there.

**Files:**
- Modify: `adaptive_roa/predictors/manifold_likelihood.py`
- Test: `tests/predictors/test_manifold_likelihood.py`

**Interfaces:**
- Consumes: `ComponentLikelihood` from Task 1
- Produces: `SO2Likelihood`, plus the module-level helper `wrap_angle(x) -> Tensor`

- [ ] **Step 1: Write the failing test**

Append to `tests/predictors/test_manifold_likelihood.py`:

```python
def test_so2_params_are_sin_cos_and_log_sigma():
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood
    assert SO2Likelihood().n_params(1) == 3


def test_so2_mean_recovers_the_angle_via_atan2():
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood

    lik = SO2Likelihood()
    for theta in (0.0, 1.0, -1.0, 3.0, -3.0, math.pi - 1e-3):
        params = torch.tensor([[math.sin(theta), math.cos(theta), 0.0]])
        assert lik.mean(params).item() == pytest.approx(theta, abs=1e-5)


def test_so2_mean_is_unaffected_by_the_magnitude_of_sin_cos():
    """Only the direction matters; the head is not required to emit a unit vector."""
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood

    lik = SO2Likelihood()
    small = torch.tensor([[0.1 * math.sin(2.0), 0.1 * math.cos(2.0), 0.0]])
    large = torch.tensor([[9.0 * math.sin(2.0), 9.0 * math.cos(2.0), 0.0]])
    assert lik.mean(small).item() == pytest.approx(2.0, abs=1e-5)
    assert lik.mean(large).item() == pytest.approx(2.0, abs=1e-5)


def test_so2_samples_stay_wrapped_and_straddle_the_seam():
    """A mean near +pi with real spread must produce samples on BOTH sides of the
    seam, all within [-pi, pi]. This is the property a Euclidean Gaussian lacks."""
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood

    lik = SO2Likelihood()
    theta = math.pi - 0.05
    params = torch.tensor([[math.sin(theta), math.cos(theta), math.log(0.5)]]).expand(4000, -1)
    draws = lik.sample(params)
    assert draws.min() >= -math.pi - 1e-5 and draws.max() <= math.pi + 1e-5
    assert (draws > 0).any() and (draws < 0).any(), "samples never crossed the seam"


def test_so2_distance_is_the_short_way_round():
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood

    lik = SO2Likelihood()
    a = torch.tensor([[math.pi - 0.1]])
    b = torch.tensor([[-math.pi + 0.1]])
    assert lik.n_dist(1) == 1
    assert lik.distance(a, b).item() == pytest.approx(0.2, abs=1e-5)


def test_so2_nll_is_lower_for_a_target_at_the_predicted_angle():
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood

    lik = SO2Likelihood()
    theta = math.pi - 0.05
    params = torch.tensor([[math.sin(theta), math.cos(theta), math.log(0.3)]])
    on = lik.nll(params, torch.tensor([[theta]]))
    # Just across the seam, geodesically 0.1 away.
    near = lik.nll(params, torch.tensor([[-math.pi + 0.05]]))
    far = lik.nll(params, torch.tensor([[0.0]]))
    assert on.item() < near.item() < far.item()


def test_so2_names_mark_the_geodesic():
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood
    assert SO2Likelihood().names(1, "pole_angle") == ["pole_angle_geodesic"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_manifold_likelihood.py -k so2 -v`
Expected: FAIL — `ImportError: cannot import name 'SO2Likelihood'`

- [ ] **Step 3: Write the implementation**

Append to `adaptive_roa/predictors/manifold_likelihood.py`:

```python
def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Wrap to [-pi, pi]. The codebase-wide idiom (systems/pendulum.py:131 etc.)."""
    return torch.atan2(torch.sin(x), torch.cos(x))


class SO2Likelihood(ComponentLikelihood):
    """Wrapped normal over one angle.

    The head emits an UNNORMALIZED (sin, cos) direction plus log sigma; the mean
    angle is atan2(sin, cos), which is well-defined regardless of magnitude and
    has no seam. Sampling adds Gaussian noise in the tangent (angle) space and
    wraps, so mass near +/-pi correctly appears on both sides -- the property a
    Euclidean Gaussian on theta lacks, and the reason this class exists:
    pendulum's FAILURE attractors sit at theta = +/-pi.

    NLL uses the wrapped geodesic residual. This is the wrapped-normal density
    truncated to its principal term, which is accurate for sigma well under pi
    and is what makes the loss seam-aware.
    """

    LOG_SIGMA_MIN = -7.0
    LOG_SIGMA_MAX = math.log(math.pi)

    def n_params(self, dim: int) -> int:
        if int(dim) != 1:
            raise ValueError(f"SO2 component must be 1-dimensional, got {dim}")
        return 3

    def _sigma(self, params: torch.Tensor) -> torch.Tensor:
        return params[..., 2:3].clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX).exp()

    def mean(self, params):
        return torch.atan2(params[..., 0:1], params[..., 1:2])

    def nll(self, params, target, beta: float = 0.0):
        mu = self.mean(params)
        sigma = self._sigma(params)
        resid = wrap_angle(target - mu)
        per_dim = 0.5 * torch.log(2 * math.pi * sigma ** 2) + resid ** 2 / (2 * sigma ** 2)
        if beta > 0.0:
            per_dim = per_dim * (sigma.detach() ** (2.0 * beta))
        return per_dim.sum(dim=-1)

    def sample(self, params, generator=None):
        mu = self.mean(params)
        sigma = self._sigma(params)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        return wrap_angle(mu + sigma * eps)

    def distance(self, a, b):
        return wrap_angle(a - b).abs()

    def n_dist(self, dim: int) -> int:
        return 1

    def names(self, dim: int, base: str) -> List[str]:
        return [f"{base}_geodesic"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_manifold_likelihood.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/manifold_likelihood.py tests/predictors/test_manifold_likelihood.py
git commit -m "feat(predictors): wrapped-normal likelihood for SO2 components"
```

---

### Task 3: `SO3Likelihood` — tangent-space distribution over a unit quaternion

**Files:**
- Modify: `adaptive_roa/predictors/manifold_likelihood.py`
- Test: `tests/predictors/test_manifold_likelihood.py`

**Interfaces:**
- Consumes: `ComponentLikelihood` from Task 1
- Produces: `SO3Likelihood`, plus module-level `canonicalize_quaternion(q) -> Tensor` and `quaternion_multiply(a, b) -> Tensor`

- [ ] **Step 1: Write the failing test**

Append to `tests/predictors/test_manifold_likelihood.py`:

```python
def test_so3_params_are_a_quaternion_plus_three_log_sigmas():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood
    assert SO3Likelihood().n_params(4) == 7


def test_so3_mean_is_unit_norm_and_canonical():
    """qw >= 0 is REQUIRED: Quadrotor3DSystem.classify_attractor uses plain L2
    against an identity-quaternion goal, so -q (the same rotation) sits at
    distance 2 and every near-goal endpoint would be misclassified."""
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    params = torch.tensor([[-0.9, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0]])  # qw < 0, unnormalized
    q = lik.mean(params)
    assert torch.linalg.norm(q, dim=-1).item() == pytest.approx(1.0, abs=1e-6)
    assert q[0, 0].item() >= 0.0


def test_so3_samples_are_unit_norm_and_canonical():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    params = torch.tensor([[1.0, 0.0, 0.0, 0.0, math.log(0.3), math.log(0.3), math.log(0.3)]])
    draws = lik.sample(params.expand(500, -1))
    assert torch.allclose(torch.linalg.norm(draws, dim=-1), torch.ones(500), atol=1e-5)
    assert (draws[:, 0] >= 0).all(), "samples must be canonicalized to qw >= 0"


def test_so3_sample_concentrates_as_sigma_shrinks():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    spreads = []
    for log_sigma in (math.log(0.5), math.log(0.01)):
        params = torch.cat([q, torch.full((1, 3), log_sigma)], dim=-1).expand(800, -1)
        spreads.append(lik.distance(lik.sample(params), q.expand(800, -1)).mean().item())
    assert spreads[1] < spreads[0]


def test_so3_distance_is_the_rotation_angle_and_ignores_double_cover():
    """q and -q are the same rotation, so their geodesic distance must be 0."""
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert lik.n_dist(4) == 1
    assert lik.distance(q, q).item() == pytest.approx(0.0, abs=1e-6)
    assert lik.distance(q, -q).item() == pytest.approx(0.0, abs=1e-5)
    # 180 degrees about x.
    half_turn = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    assert lik.distance(q, half_turn).item() == pytest.approx(math.pi, abs=1e-4)


def test_so3_nll_is_lowest_at_the_predicted_rotation():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    params = torch.tensor([[1.0, 0.0, 0.0, 0.0, math.log(0.3), math.log(0.3), math.log(0.3)]])
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    tilted = torch.tensor([[math.cos(0.4), math.sin(0.4), 0.0, 0.0]])
    assert lik.nll(params, q).item() < lik.nll(params, tilted).item()


def test_so3_nll_is_invariant_to_target_sign():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    params = torch.tensor([[1.0, 0.0, 0.0, 0.0, math.log(0.3), math.log(0.3), math.log(0.3)]])
    t = torch.tensor([[math.cos(0.4), math.sin(0.4), 0.0, 0.0]])
    assert lik.nll(params, t).item() == pytest.approx(lik.nll(params, -t).item(), rel=1e-5)


def test_so3_names_mark_the_geodesic():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood
    assert SO3Likelihood().names(4, "orientation") == ["orientation_geodesic"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_manifold_likelihood.py -k so3 -v`
Expected: FAIL — `ImportError: cannot import name 'SO3Likelihood'`

- [ ] **Step 3: Write the implementation**

Append to `adaptive_roa/predictors/manifold_likelihood.py`:

```python
def canonicalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Unit-normalize and force qw >= 0 (mirrors systems/quadrotor3d.py:425-443).

    NOT cosmetic: Quadrotor3DSystem.classify_attractor compares raw 13-vectors by
    L2 against an identity-quaternion goal, so -q -- the same rotation -- lands at
    distance 2 and is misclassified.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.where(q[..., 0:1] < 0, -q, q)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, (w, x, y, z) convention."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


class SO3Likelihood(ComponentLikelihood):
    """Concentrated Gaussian in the so(3) tangent space at a mean rotation.

    The head emits an unnormalized 4-vector (canonicalized to a unit quaternion
    with qw >= 0) plus three log sigmas for the tangent axes. Sampling draws
    xi ~ N(0, diag(sigma^2)) in so(3) and applies q = q_bar * exp(xi/2), which
    keeps every sample exactly on the unit sphere -- something a Gaussian in R^4
    plus renormalization does not do faithfully.

    NLL uses the tangent-space residual of the target relative to the mean, so it
    is invariant to the target's sign (double cover).
    """

    LOG_SIGMA_MIN = -7.0
    LOG_SIGMA_MAX = 2.0

    def n_params(self, dim: int) -> int:
        if int(dim) != 4:
            raise ValueError(f"SO3 component must be 4-dimensional, got {dim}")
        return 7

    def _sigma(self, params: torch.Tensor) -> torch.Tensor:
        return params[..., 4:7].clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX).exp()

    def mean(self, params):
        return canonicalize_quaternion(params[..., 0:4])

    @staticmethod
    def _exp_map(xi: torch.Tensor) -> torch.Tensor:
        """so(3) tangent vector -> unit quaternion. xi is a rotation vector."""
        theta = xi.norm(dim=-1, keepdim=True)
        half = 0.5 * theta
        # sinc-style guard so the theta -> 0 limit is finite and differentiable.
        scale = torch.where(theta > 1e-6, torch.sin(half) / theta.clamp_min(1e-8),
                            torch.full_like(theta, 0.5))
        return torch.cat([torch.cos(half), xi * scale], dim=-1)

    @staticmethod
    def _log_map(q: torch.Tensor) -> torch.Tensor:
        """Unit quaternion -> so(3) rotation vector, sign-canonicalized first."""
        q = canonicalize_quaternion(q)
        w = q[..., 0:1].clamp(-1.0, 1.0)
        v = q[..., 1:4]
        v_norm = v.norm(dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(v_norm, w)
        scale = torch.where(v_norm > 1e-6, angle / v_norm.clamp_min(1e-8),
                            torch.full_like(v_norm, 2.0))
        return v * scale

    def _residual(self, params, target):
        q_bar = self.mean(params)
        q_t = canonicalize_quaternion(target)
        q_bar_inv = q_bar * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q_bar.device,
                                         dtype=q_bar.dtype)
        return self._log_map(quaternion_multiply(q_bar_inv, q_t))

    def nll(self, params, target, beta: float = 0.0):
        sigma = self._sigma(params)
        xi = self._residual(params, target)
        per_dim = 0.5 * torch.log(2 * math.pi * sigma ** 2) + xi ** 2 / (2 * sigma ** 2)
        if beta > 0.0:
            per_dim = per_dim * (sigma.detach() ** (2.0 * beta))
        return per_dim.sum(dim=-1)

    def sample(self, params, generator=None):
        q_bar = self.mean(params)
        sigma = self._sigma(params)
        eps = torch.randn(sigma.shape, generator=generator, device=sigma.device, dtype=sigma.dtype)
        return canonicalize_quaternion(quaternion_multiply(q_bar, self._exp_map(sigma * eps)))

    def distance(self, a, b):
        a = canonicalize_quaternion(a)
        b = canonicalize_quaternion(b)
        dot = (a * b).sum(dim=-1, keepdim=True).abs().clamp(max=1.0)
        return 2.0 * torch.acos(dot)

    def n_dist(self, dim: int) -> int:
        return 1

    def names(self, dim: int, base: str) -> List[str]:
        return [f"{base}_geodesic"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_manifold_likelihood.py -v`
Expected: PASS (22 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/manifold_likelihood.py tests/predictors/test_manifold_likelihood.py
git commit -m "feat(predictors): tangent-space SO3 likelihood with quaternion canonicalization"
```

---

### Task 4: `FinalStateHead` — assemble component likelihoods from the system

**Files:**
- Create: `adaptive_roa/predictors/heads.py`
- Test: `tests/predictors/test_final_state_head.py`

**Interfaces:**
- Consumes: `RealLikelihood`, `SO2Likelihood`, `SO3Likelihood` from Tasks 1-3
- Produces: `FinalStateHead(system, beta=0.5)` with `.n_params -> int`, `.nll(params, target) -> Tensor[B]`, `.sample(params, generator=None) -> Tensor[B, state_dim]`, `.mean(params) -> Tensor[B, state_dim]`, `.distance_per_component(a, b) -> Tensor[B, n_dist]`, `.component_names -> list[str]`

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_final_state_head.py`:

```python
import math

import pytest
import torch

from adaptive_roa.predictors.heads import FinalStateHead
from adaptive_roa.systems.cartpole import CartPoleSystem
from adaptive_roa.systems.quadrotor2d import Quadrotor2DSystem
from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem

SYSTEMS = [
    (CartPoleSystem, 4, 4, ["cart_position", "pole_angle_geodesic", "cart_velocity",
                            "pole_angular_velocity"]),
    (Quadrotor2DSystem, 6, 6, ["position_0", "position_1", "pitch_angle_geodesic",
                               "velocity_0", "velocity_1", "velocity_2"]),
    (Quadrotor3DSystem, 13, 10, None),
]


@pytest.mark.parametrize("cls,state_dim,n_names,expected_names", SYSTEMS)
def test_head_shapes_and_names_match_the_system(cls, state_dim, n_names, expected_names):
    """Component-name cardinality must match the flow matcher's hand-written
    counts (pendulum 2, cartpole 4, quad2d 6, quad3d 10) so endpoint-error
    reporting lines up across arms."""
    head = FinalStateHead(cls())
    assert len(head.component_names) == n_names
    if expected_names is not None:
        assert head.component_names == expected_names

    params = torch.randn(5, head.n_params)
    assert head.sample(params).shape == (5, state_dim)
    assert head.mean(params).shape == (5, state_dim)
    assert head.nll(params, head.sample(params)).shape == (5,)
    a, b = head.sample(params), head.sample(params)
    assert head.distance_per_component(a, b).shape == (5, n_names)


@pytest.mark.parametrize("cls,state_dim,_n,_e", SYSTEMS)
def test_head_samples_land_in_the_space_classify_attractor_expects(cls, state_dim, _n, _e):
    """A sample must be directly consumable by system.classify_attractor: angles
    wrapped, quaternions unit-norm with qw >= 0."""
    system = cls()
    head = FinalStateHead(system)
    draws = head.sample(torch.randn(64, head.n_params))

    for idx in system.get_circular_indices():
        assert draws[:, idx].abs().max().item() <= math.pi + 1e-5

    offset = 0
    for comp in system.manifold_components:
        if comp.manifold_type == "SO3":
            q = draws[:, offset:offset + 4]
            assert torch.allclose(torch.linalg.norm(q, dim=-1), torch.ones(64), atol=1e-5)
            assert (q[:, 0] >= 0).all()
        offset += comp.dim

    labels = system.classify_attractor(draws, radius=0.3)
    assert labels.shape == (64,)


def test_head_is_stochastic_but_reproducible_under_a_generator():
    head = FinalStateHead(CartPoleSystem())
    params = torch.randn(8, head.n_params)
    assert not torch.allclose(head.sample(params), head.sample(params))
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    assert torch.allclose(head.sample(params, generator=g1), head.sample(params, generator=g2))


def test_head_nll_decreases_when_the_target_is_the_predicted_mean():
    head = FinalStateHead(CartPoleSystem())
    params = torch.randn(16, head.n_params)
    at_mean = head.nll(params, head.mean(params))
    displaced = head.nll(params, head.mean(params) + 1.0)
    assert (at_mean < displaced).all()


def test_head_beta_is_threaded_into_the_component_likelihoods():
    params = torch.randn(4, FinalStateHead(CartPoleSystem()).n_params)
    target = FinalStateHead(CartPoleSystem()).mean(params) + 0.5
    plain = FinalStateHead(CartPoleSystem(), beta=0.0).nll(params, target)
    weighted = FinalStateHead(CartPoleSystem(), beta=0.5).nll(params, target)
    assert not torch.allclose(plain, weighted)


def test_unknown_component_type_is_rejected():
    """Use a stub rather than subclassing a real system: overriding
    define_manifold_structure on a concrete system can fail inside that system's
    own __init__ for unrelated reasons, which would make this pass for the wrong
    reason. FinalStateHead only reads `.manifold_components`."""
    from adaptive_roa.systems.base import ManifoldComponent

    class StubSystem:
        manifold_components = [ManifoldComponent("Hyperbolic", 2, "weird")]

    with pytest.raises(ValueError, match="Hyperbolic"):
        FinalStateHead(StubSystem())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_final_state_head.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.predictors.heads'`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/heads.py`:

```python
"""Output heads for predictor arms.

``FinalStateHead`` turns a flat parameter vector into a predictive distribution
over the final state, factorized across the system's manifold components. It is
built generically from ``system.manifold_components`` so every system is handled
by the same code path.
"""
from __future__ import annotations

from typing import List

import torch

from adaptive_roa.predictors.manifold_likelihood import (
    ComponentLikelihood,
    RealLikelihood,
    SO2Likelihood,
    SO3Likelihood,
)

_LIKELIHOODS = {
    "Real": RealLikelihood,
    "SO2": SO2Likelihood,
    "SO3": SO3Likelihood,
}


class FinalStateHead:
    """Predictive distribution over x_T, factorized by manifold component.

    ``beta`` is the beta-NLL exponent (Seitzer et al., ICLR 2022); 0.5 is the
    recommended default and 0.0 recovers plain Gaussian NLL.
    """

    def __init__(self, system, beta: float = 0.5):
        self.system = system
        self.beta = float(beta)
        self._parts: List[tuple[ComponentLikelihood, int, int, int, int]] = []

        param_offset = 0
        state_offset = 0
        names: List[str] = []
        for comp in system.manifold_components:
            lik_cls = _LIKELIHOODS.get(comp.manifold_type)
            if lik_cls is None:
                raise ValueError(
                    f"no likelihood registered for manifold component "
                    f"{comp.manifold_type!r}; expected one of {sorted(_LIKELIHOODS)}"
                )
            lik = lik_cls()
            n_p = lik.n_params(comp.dim)
            self._parts.append((lik, param_offset, n_p, state_offset, comp.dim))
            names.extend(lik.names(comp.dim, comp.name))
            param_offset += n_p
            state_offset += comp.dim

        self.n_params = param_offset
        self.state_dim = state_offset
        self.component_names = names

    def _iter(self, params: torch.Tensor, target: torch.Tensor | None = None):
        for lik, p0, n_p, s0, dim in self._parts:
            p = params[..., p0:p0 + n_p]
            t = None if target is None else target[..., s0:s0 + dim]
            yield lik, p, t, dim

    def nll(self, params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = None
        for lik, p, t, _dim in self._iter(params, target):
            term = lik.nll(p, t, beta=self.beta)
            total = term if total is None else total + term
        return total

    def sample(self, params: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.cat(
            [lik.sample(p, generator=generator) for lik, p, _t, _d in self._iter(params)], dim=-1
        )

    def mean(self, params: torch.Tensor) -> torch.Tensor:
        return torch.cat([lik.mean(p) for lik, p, _t, _d in self._iter(params)], dim=-1)

    def distance_per_component(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = [
            lik.distance(a[..., s0:s0 + dim], b[..., s0:s0 + dim])
            for lik, _p0, _n_p, s0, dim in self._parts
        ]
        return torch.cat(out, dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_final_state_head.py -v`
Expected: PASS (11 tests — 3 parametrized x2, plus 5 singles)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/heads.py tests/predictors/test_final_state_head.py
git commit -m "feat(predictors): manifold-aware final-state head"
```

---

### Task 5: `FinalStateModelHandle` — satisfy the endpoint-MC contract

This is the contract-critical task. The determinism requirement is the **opposite** of `OutcomeModelHandle`'s.

**Files:**
- Create: `adaptive_roa/predictors/final_state_handle.py`
- Test: `tests/predictors/test_final_state_handle.py`

**Interfaces:**
- Consumes: `FinalStateHead` (Task 4), `build_bayesian_mlp` / `Posterior` from `adaptive_roa.predictors.bayesian_mlp` and `.posteriors`
- Produces: `FinalStateModelHandle(posterior, head, system, device="cpu")` with `.eval()`, `.to(device)`, `.training`, `.predict_endpoint(states)`, `.get_manifold_component_names()`, `.compute_manifold_distance_per_component(pred, true)`, `.distance_manifold`

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_final_state_handle.py`:

```python
import numpy as np
import pytest
import torch

from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.final_state_handle import FinalStateModelHandle
from adaptive_roa.predictors.heads import FinalStateHead
from adaptive_roa.systems.cartpole import CartPoleSystem
from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem


def _handle(cls=CartPoleSystem, kind="mfvi"):
    system = cls()
    head = FinalStateHead(system)
    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
    posterior = build_bayesian_mlp(
        input_dim=input_dim, hidden_dims=[16, 16], output_dim=head.n_params, posterior=kind
    )
    return FinalStateModelHandle(posterior, head, system), system


def test_predict_endpoint_draws_a_fresh_sample_every_call():
    """THE contract. ProbabilityEstimator calls this K times on the SAME batch and
    the spread across calls IS the outcome probability. A seeded/deterministic
    handle collapses every arm to p in {0, 1}."""
    handle, system = _handle()
    x = torch.randn(16, int(system.state_dim))
    assert not torch.allclose(handle.predict_endpoint(x), handle.predict_endpoint(x))


def test_predict_endpoint_returns_raw_states_of_the_right_shape():
    handle, system = _handle()
    out = handle.predict_endpoint(torch.randn(16, int(system.state_dim)))
    assert out.shape == (16, int(system.state_dim))
    assert torch.isfinite(out).all()


def test_predict_endpoint_takes_a_single_positional_arg():
    """probability_estimator.py:143 calls it with exactly one positional arg."""
    import inspect

    sig = inspect.signature(FinalStateModelHandle.predict_endpoint)
    required = [p for n, p in list(sig.parameters.items())[1:]
                if p.default is inspect.Parameter.empty]
    assert len(required) == 1


def test_component_names_and_distance_shapes_agree():
    """endpoint_evaluation.py:119 assigns the distance array into a slice sized by
    len(names); a mismatch raises there, after training is already paid for."""
    handle, system = _handle()
    names = handle.get_manifold_component_names()
    a = handle.predict_endpoint(torch.randn(8, int(system.state_dim)))
    b = handle.predict_endpoint(torch.randn(8, int(system.state_dim)))
    d = handle.compute_manifold_distance_per_component(a, b)
    assert d.shape == (8, len(names))


def test_distance_manifold_is_present_and_agrees_with_the_method():
    """full_roa.py:642 guards get_manifold_component_names() on hasattr(
    'distance_manifold'), so a handle with one but not the other crashes."""
    handle, system = _handle()
    assert hasattr(handle, "distance_manifold")
    a = handle.predict_endpoint(torch.randn(8, int(system.state_dim)))
    b = handle.predict_endpoint(torch.randn(8, int(system.state_dim)))
    assert torch.allclose(handle.distance_manifold.dist(a, b),
                          handle.compute_manifold_distance_per_component(a, b))


def test_quadrotor3d_endpoints_are_canonicalized_for_classify_attractor():
    """Quadrotor3DSystem.classify_attractor uses plain L2 against an identity
    quaternion, so an uncanonicalized -q lands 2.0 away and every near-goal
    endpoint is misclassified."""
    handle, system = _handle(cls=Quadrotor3DSystem)
    out = handle.predict_endpoint(torch.randn(64, int(system.state_dim)))
    q = out[:, 3:7]
    assert torch.allclose(torch.linalg.norm(q, dim=-1), torch.ones(64), atol=1e-5)
    assert (q[:, 0] >= 0).all()
    assert system.classify_attractor(out, radius=0.3).shape == (64,)


def test_handle_is_module_shaped_for_the_engine():
    handle, _ = _handle()
    assert handle.eval() is not None
    assert handle.to("cpu") is not None
    assert isinstance(handle.training, bool)


@pytest.mark.parametrize("kind", ["deterministic", "mfvi", "ensemble", "laplace"])
def test_every_posterior_kind_still_produces_spread(kind):
    """The deterministic posterior has no weight spread, but the HEAD is still a
    distribution, so endpoints must still vary across calls. If they do not, the
    arm reports p in {0, 1} and the benchmark records a method result for a bug."""
    handle, system = _handle(kind=kind)
    x = torch.randn(16, int(system.state_dim))
    assert not torch.allclose(handle.predict_endpoint(x), handle.predict_endpoint(x))


def test_probabilities_are_not_degenerate_end_to_end():
    """The degeneracy guard from the spec: an accidentally-seeded handle looks
    like a plausible-but-wrong benchmark result rather than a bug."""
    handle, system = _handle()
    x = torch.randn(32, int(system.state_dim))
    draws = torch.stack([handle.predict_endpoint(x) for _ in range(20)], dim=0)
    labels = torch.stack([system.classify_attractor(d, radius=5.0) for d in draws], dim=0)
    p_success = (labels == 1).float().mean(dim=0)
    assert p_success.std().item() > 0.0 or (0.0 < p_success.mean().item() < 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_final_state_handle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.predictors.final_state_handle'`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/final_state_handle.py`:

```python
"""Model handle binding a final-state predictor to the endpoint-MC backend.

Satisfies the contract ``ProbabilityEstimator`` and ``compute_endpoint_prediction_error``
already rely on, so the Bayesian final-state arms route through the existing
conformal, threshold, and evaluation machinery unchanged.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import torch


class _ManifoldDistanceShim:
    """Minimal ``distance_manifold`` stand-in exposing ``dist``.

    full_roa.py guards get_manifold_component_names() behind
    hasattr(model, "distance_manifold"), so supplying only one of the pair means
    the per-component error stats are silently skipped (or crash). This shim
    delegates to the head so both consumers agree.
    """

    def __init__(self, head):
        self._head = head

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._head.distance_per_component(x, y)


class FinalStateModelHandle:
    """Predicts a distribution over x_T and samples ONE endpoint per call.

    Determinism here is the OPPOSITE of ``OutcomeModelHandle``: the estimator
    calls ``predict_endpoint`` K times on the same batch and the spread across
    those calls IS the outcome probability. Every call therefore draws a fresh
    weight sample AND a fresh head sample. Seeding this handle collapses every
    arm to p in {0, 1}.
    """

    def __init__(self, posterior, head, system: Any, device: str = "cpu"):
        self.posterior = posterior
        self.head = head
        self.system = system
        self.device = device
        self.training = False
        self.distance_manifold = _ManifoldDistanceShim(head)

    def eval(self):
        self.posterior.eval()
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.posterior.train(mode)
        self.training = bool(mode)
        return self

    def to(self, device):
        self.device = device
        self.posterior.to(device)
        return self

    def _params(self, states: torch.Tensor) -> torch.Tensor:
        embedded = self.system.embed_state_for_model(self.system.normalize_state(states))
        return self.posterior.forward_sample(embedded)

    def predict_endpoint(self, states) -> torch.Tensor:
        """[B, state_dim] raw -> [B, state_dim] raw. Fresh sample every call."""
        if torch.is_tensor(states):
            x = states.detach().to(dtype=torch.float32)
            out_device = states.device
        else:
            x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            out_device = self.device
        x = x.to(next(self.posterior.parameters()).device)
        with torch.no_grad():
            endpoints = self.head.sample(self._params(x))
        return endpoints.to(out_device)

    def get_manifold_component_names(self) -> List[str]:
        return list(self.head.component_names)

    def compute_manifold_distance_per_component(self, predicted, true) -> torch.Tensor:
        return self.head.distance_per_component(predicted, true)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/test_final_state_handle.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/final_state_handle.py tests/predictors/test_final_state_handle.py
git commit -m "feat(predictors): final-state model handle for the endpoint-MC contract"
```

---

### Task 6: `FinalStateTrainer` and the four arm configs

**Files:**
- Create: `adaptive_roa/adaptive_v2/trainers/final_state_trainer.py`
- Create: `configs/adaptive_v2/predictor/{mlp_det,bnn_mfvi_reg,bnn_ensemble_reg,bnn_laplace_reg}.yaml`
- Modify: `adaptive_roa/predictors/bayesian_mlp.py` (extend `build_from_cfg` with an explicit `output_dim`)
- Test: `tests/predictors/test_final_state_trainer.py`

**Interfaces:**
- Consumes: `FinalStateHead` (Task 4), `FinalStateModelHandle` (Task 5), `build_from_cfg` / `build_bayesian_mlp`
- Produces: `FinalStateTrainer(cfg, system, system_name)` with `fit(dataset_files, output_dir, resume_checkpoint=None) -> FinalStateModelHandle`

**Read first:** `adaptive_roa/adaptive_v2/trainers/bayesian_mlp_trainer.py` in full. This
trainer mirrors it closely — same `_run_lightning`, `_load_best`, ensemble-member
loop, post-hoc Laplace fit, and `best*.ckpt` conventions. The differences are
listed below; everything else should follow that file's established patterns.

**Differences from `BayesianMLPTrainer`:**

1. **Datamodule.** Endpoint data, not classification. Select the per-system class
   from the same map `FlowMatchingTrainer` uses (`trainers/flow_matching_trainer.py:25-31`):
   `pendulum -> PendulumEndpointDataModule`, `cartpole_pybullet -> CartPoleEndpointDataModule`,
   `quadrotor2d -> Quadrotor2DEndpointDataModule`, `quadrotor3d -> Quadrotor3DEndpointDataModule`.
   Construct with `data_file=dataset_files["train"]`, `validation_file=dataset_files["val"]`,
   `test_file=dataset_files["val"]`, `batch_size`, `val_batch_size`, `num_workers=0`.
   Batches are `{"start_state": [B, D], "end_state": [B, D]}` in RAW coordinates with
   angles pre-wrapped; the model owns normalization and embedding.

2. **Loss.** The Lightning module's `_step` is
   `head.nll(posterior.forward_sample(embedded_start), end_state).mean()`, plus
   `kl_weight * posterior.kl_divergence() / n_train` for MFVI. Log it as `val_nll`
   (and `val_loss` for the ELBO diagnostic) so `ModelCheckpoint` and `EarlyStopping`
   can monitor `val_nll` — same fix as `BayesianMLPTrainer`, for the same reason:
   the KL term is data-independent and monotone, so monitoring the total makes
   early stopping inert and always keeps the last epoch.

3. **Output width.** `output_dim` is `FinalStateHead(system, beta).n_params`, not 1.

4. **Laplace GGN.** Fit with `task="final_state"`, which **requires an explicit
   `sigma`**. Derive it from the fitted model: compute the RMS of the head's
   predicted sigma over the training set, or the RMS residual between
   `head.mean(params)` and the true endpoints. Pass that value. Do NOT let it
   default — `LastLayerLaplacePosterior.fit` raises if `sigma` is None, and that
   raise exists precisely because a default of 1.0 silently rescales the whole
   posterior covariance. Fit the GGN AFTER `_load_best`, and persist the
   covariance to `checkpoints/laplace_cov.pt` (it is a non-persistent buffer).

5. **`mlp_det`.** `posterior: deterministic`. It still produces a distribution,
   because the HEAD is a distribution — only the weights are a point estimate.
   Its config sets `d2_ratio: 0.0` so no conformal acquisition runs, per the spec's
   decision that the deterministic arm is a fixed-dataset baseline.

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_final_state_trainer.py`:

```python
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.trainers.final_state_trainer import FinalStateTrainer
from adaptive_roa.systems.cartpole import CartPoleSystem


def _write_endpoints(path, n=256, seed=0):
    """8-column cartpole endpoint pairs: x th xd thd | x th xd thd."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(-1.0, 1.0, size=(n, 4))
    end = start * 0.5  # a learnable contraction toward the origin
    np.savetxt(path, np.column_stack([start, end]))
    return str(path)


def _cfg(posterior, **overrides):
    fs = {
        "posterior": posterior, "hidden_dims": [16, 16], "lr": 1e-2,
        "weight_decay": 1e-5, "max_epochs": 3, "patience": 5, "prior_sigma": 1.0,
        "n_members": 2, "kl_weight": 1.0, "beta_nll": 0.5,
    }
    fs.update(overrides)
    return OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "generative", "name": f"fs_{posterior}",
                      "batch_size": 64, "val_batch_size": 64, "final_state": fs},
    })


@pytest.fixture
def files(tmp_path):
    return {"train": _write_endpoints(tmp_path / "train.txt"),
            "val": _write_endpoints(tmp_path / "val.txt", n=128, seed=1)}


@pytest.mark.parametrize("posterior", ["deterministic", "mfvi", "ensemble", "laplace"])
def test_trainer_returns_a_working_final_state_handle(posterior, files, tmp_path):
    trainer = FinalStateTrainer(_cfg(posterior), CartPoleSystem(), "cartpole_pybullet")
    handle = trainer.fit(files, str(tmp_path / f"out_{posterior}"))

    x = torch.randn(16, 4)
    out = handle.predict_endpoint(x)
    assert out.shape == (16, 4)
    assert torch.isfinite(out).all()
    # THE contract: fresh sample per call.
    assert not torch.allclose(handle.predict_endpoint(x), handle.predict_endpoint(x))


@pytest.mark.parametrize("posterior", ["deterministic", "mfvi", "ensemble", "laplace"])
def test_trainer_writes_a_warm_start_checkpoint(posterior, files, tmp_path):
    out = tmp_path / f"out_{posterior}"
    FinalStateTrainer(_cfg(posterior), CartPoleSystem(), "cartpole_pybullet").fit(files, str(out))
    assert list((out / "checkpoints").glob("best*.ckpt")), "engine warm start needs best*.ckpt"


def test_laplace_fits_its_ggn_with_an_explicit_sigma(files, tmp_path):
    """task='final_state' REQUIRES sigma; letting it default to 1.0 silently
    rescales the entire posterior covariance."""
    trainer = FinalStateTrainer(_cfg("laplace"), CartPoleSystem(), "cartpole_pybullet")
    out = tmp_path / "out_laplace"
    handle = trainer.fit(files, str(out))
    assert handle.posterior.is_fitted
    assert (out / "checkpoints" / "laplace_cov.pt").exists()


def test_the_arm_actually_learns_the_contraction(files, tmp_path):
    """Guards against shipping an untrained network: on end = 0.5*start the fitted
    head's mean must beat the identity map by a clear margin."""
    trainer = FinalStateTrainer(
        _cfg("deterministic", max_epochs=60), CartPoleSystem(), "cartpole_pybullet"
    )
    handle = trainer.fit(files, str(tmp_path / "out_learn"))

    raw = np.loadtxt(files["val"])
    x = torch.as_tensor(raw[:, :4], dtype=torch.float32)
    y = torch.as_tensor(raw[:, 4:], dtype=torch.float32)
    with torch.no_grad():
        pred = handle.head.mean(handle._params(x))
    assert (pred - y).pow(2).mean().item() < 0.5 * (x - y).pow(2).mean().item()


def test_ensemble_requires_enough_mc_samples_to_resolve_its_members(files, tmp_path):
    """K >= 2M: an M-atom empirical posterior sampled K times needs K >= 2M, or the
    reported probability is a sampling artifact of the member draw."""
    cfg = _cfg("ensemble", n_members=8)
    cfg.predictor.num_mc_samples = 10
    with pytest.raises(ValueError, match="num_mc_samples"):
        FinalStateTrainer(cfg, CartPoleSystem(), "cartpole_pybullet").fit(
            files, str(tmp_path / "out_guard")
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_final_state_trainer.py -v`
Expected: FAIL — `ModuleNotFoundError` for `final_state_trainer`

- [ ] **Step 3: Write the trainer**

Create `adaptive_roa/adaptive_v2/trainers/final_state_trainer.py`. Copy
`bayesian_mlp_trainer.py`'s structure — `_predictor_cfg`, `_run_lightning`,
`_load_best`, the ensemble-member loop, the post-hoc Laplace fit, the
`best*.ckpt` conventions — and substitute the pieces below.

**The Lightning module** (replaces `_OutcomeModule`):

```python
class _FinalStateModule(pl.LightningModule):
    """Head NLL over endpoint pairs, plus KL/N for variational posteriors."""

    def __init__(self, posterior, head, system, lr, weight_decay, kl_weight, n_train):
        super().__init__()
        self.posterior = posterior
        self.head = head
        self.system = system  # plain attr; methods are device-agnostic
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.kl_weight = float(kl_weight)
        self.n_train = max(int(n_train), 1)

    def forward(self, raw_states):
        embedded = self.system.embed_state_for_model(self.system.normalize_state(raw_states))
        return self.posterior.forward_sample(embedded)

    def _step(self, batch, stage: str):
        params = self(batch["start_state"])
        nll = self.head.nll(params, batch["end_state"]).mean()
        # KL is a per-DATASET term while nll is a per-batch mean, so scale by 1/N.
        kl = self.posterior.kl_divergence() / self.n_train
        loss = nll + self.kl_weight * kl
        # Callbacks monitor val_nll, NOT val_loss: the KL term is data-independent
        # and monotone, so monitoring the total makes EarlyStopping inert and
        # always keeps the last epoch.
        self.log(f"{stage}_nll", nll, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_nll"}}
```

**Datamodule selection** (mirrors `flow_matching_trainer.py:25-31`):

```python
from adaptive_roa.data.pendulum_endpoint_data import PendulumEndpointDataModule
from adaptive_roa.data.cartpole_endpoint_data import CartPoleEndpointDataModule
from adaptive_roa.data.quadrotor2d_endpoint_data import Quadrotor2DEndpointDataModule
from adaptive_roa.data.quadrotor3d_endpoint_data import Quadrotor3DEndpointDataModule

_DATAMODULES = {
    "pendulum": PendulumEndpointDataModule,
    "cartpole_pybullet": CartPoleEndpointDataModule,
    "quadrotor2d": Quadrotor2DEndpointDataModule,
    "quadrotor3d": Quadrotor3DEndpointDataModule,
}

def _create_datamodule(self, dataset_files):
    dm_cls = _DATAMODULES.get(self.system_name)
    if dm_cls is None:
        raise ValueError(
            f"no endpoint datamodule for system {self.system_name!r}; "
            f"expected one of {sorted(_DATAMODULES)}"
        )
    dm = dm_cls(
        data_file=dataset_files["train"],
        validation_file=dataset_files["val"],
        test_file=dataset_files["val"],  # matches FlowMatchingTrainer:87-89
        batch_size=self._predictor_cfg.get("batch_size", 1024),
        val_batch_size=self._predictor_cfg.get("val_batch_size", 2048),
        num_workers=0,  # in-memory tensors; workers break on NFS
    )
    dm.setup()
    return dm
```

**The K >= 2M guard**, in `__init__`:

```python
if self.posterior_kind == "ensemble":
    n_members = int(fs_cfg.get("n_members", 5))
    k = self._resolve_num_mc_samples()
    if k < 2 * n_members:
        raise ValueError(
            f"num_mc_samples={k} cannot resolve an ensemble of {n_members} members; "
            f"forward_sample draws ONE member per call, so an M-atom empirical "
            f"posterior needs num_mc_samples >= 2*M = {2 * n_members}. Raise "
            f"num_mc_samples or lower n_members."
        )
```

where `_resolve_num_mc_samples` reads `predictor.num_mc_samples` if present, else
`probability.num_mc_samples`, else the `endpoint_mc.yaml` default of 10.

**The Laplace sigma**, after `_load_best` and before saving the covariance —
`task="final_state"` requires it explicitly, and a silent default of 1.0 would
rescale the whole posterior covariance:

```python
if self.posterior_kind == "laplace":
    posterior.eval()
    with torch.no_grad():
        starts, ends = _full_training_tensors(data_module)
        embedded = self.system.embed_state_for_model(self.system.normalize_state(starts))
        params = posterior.forward_sample(embedded)
        # Observation noise = RMS geodesic residual of the fitted mean.
        resid = head.distance_per_component(head.mean(params), ends)
        sigma = float(resid.pow(2).mean().sqrt().clamp_min(1e-3))
        posterior.fit(posterior.body(embedded), ends, task="final_state", sigma=sigma)
    torch.save(posterior.posterior_covariance, ckpt_dir / "laplace_cov.pt")
```

with this module-level helper:

```python
def _full_training_tensors(data_module):
    """Stack the whole endpoint training set as ``(starts, ends)`` raw tensors.

    The endpoint datamodules differ from the classification one in two ways that
    matter here. The dataset lives on the PUBLIC ``train_dataset`` attribute (the
    classification module uses a private ``_train``), and it stores
    ``self.data`` as a list of ``(start, end)`` numpy tuples rather than stacked
    tensors — angle wrapping is applied in ``__getitem__``, not at load time
    (``data/cartpole_endpoint_data.py:91-103``). Collating through ``__getitem__``
    is therefore the only way to get correctly wrapped angles.
    """
    from torch.utils.data import default_collate

    ds = data_module.train_dataset
    batch = default_collate([ds[i] for i in range(len(ds))])
    return batch["start_state"].float(), batch["end_state"].float()
```

Everything else — `_run_lightning`, `_load_best`, the per-member ensemble loop with
`torch.manual_seed(seed + m)`, the top-level `best-ensemble.ckpt` write, and
returning `FinalStateModelHandle(posterior, head, self.system).eval().to(device)` —
follows `bayesian_mlp_trainer.py` unchanged apart from the module class and the
monitor string.

- [ ] **Step 4: Write the four configs**

`configs/adaptive_v2/predictor/bnn_mfvi_reg.yaml`:

```yaml
# @package _global_
defaults:
  - /probability: endpoint_mc
predictor:
  type: generative          # family tag: endpoint data, endpoint-MC backend, fast eval branch
  name: bnn_mfvi_reg        # arm name: keys the export registry
  trainer_target: adaptive_roa.adaptive_v2.trainers.final_state_trainer.FinalStateTrainer
  batch_size: 1024
  val_batch_size: 2048
  final_state:
    posterior: mfvi
    hidden_dims: [256, 512, 256]
    lr: 1.0e-3
    weight_decay: 1.0e-5
    max_epochs: 200
    patience: 20
    prior_sigma: 1.0        # isotropic Gaussian prior, identical across arms
    kl_weight: 1.0          # beta = 1: untempered. Any other value is a TEMPERED result.
    beta_nll: 0.5           # Seitzer et al. 2022; 0.0 is plain Gaussian NLL
  lightning_trainer:
    gradient_clip_val: 1.0
    log_every_n_steps: 10
```

`bnn_ensemble_reg.yaml` is identical with `name: bnn_ensemble_reg`,
`posterior: ensemble`, `n_members: 5`, and no `kl_weight`.
`bnn_laplace_reg.yaml` is identical with `name: bnn_laplace_reg`,
`posterior: laplace`, and no `kl_weight`.
`mlp_det.yaml` is identical with `name: mlp_det`, `posterior: deterministic`,
no `kl_weight`, and one extra top-level block marking it a fixed-dataset baseline:

```yaml
acquisition:
  d2_ratio: 0.0             # deterministic arm is a baseline; no conformal acquisition
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/predictors/test_final_state_trainer.py -v`
Expected: PASS (11 tests)

- [ ] **Step 6: Verify the configs compose under Hydra**

Run:
```bash
./env/bin/python -c "
from hydra import compose, initialize_config_dir
import os
d = os.path.abspath('configs/adaptive_v2')
for arm in ['mlp_det','bnn_mfvi_reg','bnn_ensemble_reg','bnn_laplace_reg']:
    with initialize_config_dir(config_dir=d, version_base=None):
        cfg = compose(config_name='default', overrides=[f'predictor={arm}'])
    assert cfg.predictor.name == arm and cfg.predictor.type == 'generative', arm
    print(arm, 'OK', cfg.probability._target_)
"
```
Expected: four `OK` lines naming `EndpointMCProbabilityBackend`.

- [ ] **Step 7: Commit**

```bash
git add adaptive_roa/adaptive_v2/trainers/final_state_trainer.py adaptive_roa/predictors/bayesian_mlp.py configs/adaptive_v2/predictor/ tests/predictors/test_final_state_trainer.py
git commit -m "feat(predictors): final-state trainer and four arm configs"
```

---

### Task 7: Export wrappers and end-to-end smoke

**Files:**
- Create: `adaptive_roa/probabilistic_classifier/bayesian_final_state.py`
- Modify: `adaptive_roa/probabilistic_classifier/__init__.py`
- Test: `tests/predictors/test_final_state_e2e.py`

**Interfaces:**
- Consumes: `FinalStateTrainer` (Task 6), the registry from `adaptive_roa.probabilistic_classifier.registry`
- Produces: `FinalStateProbabilisticClassifier` subclasses registered as `mlp_det`, `bnn_mfvi_reg`, `bnn_ensemble_reg`, `bnn_laplace_reg`

**Read first:** `adaptive_roa/probabilistic_classifier/bayesian.py` — the outcome-arm
export wrappers. These mirror it, with three differences: `predictor_type` is
`"generative"` not `"classifier"`; `native_probs` is `("p_success", "p_failure",
"p_invalid")` because MC-classified endpoints DO produce an unresolved outcome; and
`predict` must MC-sample endpoints and count `system.classify_attractor` labels
rather than applying a sigmoid.

The loader must raise on missing or unexpected parameter keys (the `strict=False`
trap fixed in Plan 1) and must read back `laplace_cov.pt` for the Laplace arm,
raising if it is absent.

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/test_final_state_e2e.py`:

```python
import pytest

ARMS = ["mlp_det", "bnn_mfvi_reg", "bnn_ensemble_reg", "bnn_laplace_reg"]


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_is_registered_for_export(arm):
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class(arm)
    assert cls.predictor_name == arm
    assert cls.predictor_type == "generative"
    assert cls.native_probs == ("p_success", "p_failure", "p_invalid")


def test_arms_do_not_steal_the_legacy_generative_alias():
    """Runs written before arm names existed resolve via the family alias."""
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.flow_matching import FMProbabilisticClassifier

    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/test_final_state_e2e.py -v`
Expected: FAIL — `KeyError: 'mlp_det'`

- [ ] **Step 3: Write the export wrappers**

Create `adaptive_roa/probabilistic_classifier/bayesian_final_state.py` mirroring
`bayesian.py`, then add
`from . import bayesian_final_state as _bayesian_final_state  # noqa: F401,E402`
to `adaptive_roa/probabilistic_classifier/__init__.py` **after** the existing
`_flow_matching` import, so the `"generative"` family alias stays owned by
`FMProbabilisticClassifier` (registration is first-wins).

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/predictors/test_final_state_e2e.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the pipeline smoke test**

Run:
```bash
./env/bin/python scripts/run_adaptive.py \
  --config-name=default system=pendulum predictor=bnn_mfvi_reg \
  sampling_mode=ranked +adaptive_v2.smoke_mode=true n_epochs=2 \
  predictor.final_state.max_epochs=3 predictor.final_state.patience=3
```
Expected: two epochs complete; `epoch_000/artifacts_v2.json` and
`epoch_001/artifacts_v2.json` exist. Note `n_epochs` is the correct top-level key
(`adaptive_v2.max_epochs` does not exist). Repeat for `predictor=mlp_det`. If a run
fails on dataset paths for environmental reasons rather than code, record the exact
command and output and proceed.

- [ ] **Step 6: Run the full suite for regressions**

Run: `./env/bin/python -m pytest tests -q`
Expected: PASS with no failures. The 64 pre-existing `tests/predictors/` tests must
still pass.

- [ ] **Step 7: Commit**

```bash
git add adaptive_roa/probabilistic_classifier/ tests/predictors/test_final_state_e2e.py
git commit -m "feat(export): probabilistic-classifier wrappers for the final-state arms"
```

---

## Self-Review

**Spec coverage for this plan's scope:**

| Spec requirement | Task |
|---|---|
| `Real` component: `mu`, `log sigma`, Gaussian | 1 |
| beta-NLL at `beta = 0.5` | 1 (threaded through 4, configured in 6) |
| `SO2`: `(sin, cos)` → `atan2`, wrapped sampling | 2 |
| `SO3`: unit quaternion + tangent covariance, `q = q_bar (*) exp(xi/2)` | 3 |
| `FinalStateHead` built from `system.manifold_components` | 4 |
| `FinalStateModelHandle`, fresh draw per call | 5 |
| `get_manifold_component_names()` on final-state handles (spec fix #4) | 5 |
| Degeneracy guard test | 5 |
| `mlp_det` as a non-adaptive baseline (`d2_ratio=0`) | 6 |
| BNN final-state arms (mfvi/ensemble/laplace) | 6 |
| `K >= 2M` guard | 6 |
| Export wrappers | 7 |

**Deferred deliberately, not gaps:**
- **MDN likelihood.** The spec marks it optional and motivates it by separatrix
  bimodality. YAGNI: build it once the Gaussian head demonstrably fails there,
  which the separatrix-conditioned reporting (a later plan) will show.
- **`gp_reg` multi-output SVGP.** Its own plan — it carries independent risks
  (minibatched ELBO at the 12k cap, a distinct state-dict schema needing
  `num_tasks`, and the pre-existing bug that `partx/trainer.py:57` writes `gp.pt`
  while `engine.py:140` globs `best*.ckpt`, so the GP arm has never warm-started).
  That plan should also add the **GP export wrapper** deferred from Plan 1 and
  restore `predictor.name` to `gp.yaml`/`gp_optdelta.yaml`.
- HMC reference tier, benchmark orchestration, separatrix-conditioned reporting.

**Type consistency check:** `ComponentLikelihood`'s six methods keep identical
signatures across Tasks 1-3 and are consumed identically in Task 4.
`FinalStateHead(system, beta)` exposing `.n_params`, `.nll`, `.sample`, `.mean`,
`.distance_per_component`, `.component_names` is constructed the same way in Tasks
5, 6 and 7. `FinalStateModelHandle(posterior, head, system, device)` is constructed
identically in Tasks 5-7.

**One placeholder fixed inline:** Task 4's `distance_per_component` as first drafted
contained a leftover loop; the corrected body is given immediately below it in the
same step.

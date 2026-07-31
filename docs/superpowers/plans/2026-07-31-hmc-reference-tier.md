# HMC Reference Tier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Hamiltonian Monte Carlo reference arm for both predictor heads, so the branch can say how good its approximate posteriors actually are rather than only how they rank against each other.

**Architecture:** HMC produces a *set* of weight vectors, which is structurally what `EnsemblePosterior` already holds — so an `HMCPosterior` implementing the existing `Posterior` interface slots into both shipped model handles with no new plumbing. The sampler is hand-rolled leapfrog with dual-averaging step-size adaptation over a flat parameter vector, run at a reduced `[50,50]` backbone where full-batch HMC is routine.

**Tech Stack:** PyTorch 2.5.1, Lightning, Hydra, pytest. No new pip dependencies — the sampler is ~200 lines of torch.

**Design spec:** `docs/superpowers/specs/2026-07-27-probabilistic-predictors-design.md`
**Prior plans (all merged):** the outcome family, the final-state family, and the GP arm. This plan builds on `adaptive_roa/predictors/`.

## Global Constraints

- Branch: create `predictors-hmc` off `main`. Repo root: `/common/home/st1122/Projects/adaptive_roa`.
- Python interpreter is the repo-local env: **`./env/bin/python`**. Never plain `python`.
- Run tests as `./env/bin/python -m pytest <path> -v` from the repo root.
- Import root is `adaptive_roa` (editable install).
- **No new pip dependencies.**
- **Commit messages must contain zero AI/tool attribution** — no `Co-Authored-By`, no `Claude-Session`, no tool or model names. Hard user preference.
- **Another session commits to this repo concurrently.** Stage explicit paths only; never `git add -A` or `git commit -a`. **Destructive git operations are forbidden**: no `git stash`, `git reset --hard`, `git checkout`, `git clean`. Restore any experiment by editing the file back.
- Reference-tier backbone is `hidden_dims: [50, 50]` with **`activation: tanh`**.
- The suite is currently 502 collected with 0 failures (493 passed / 25 skipped, including untracked tests from a concurrent session).

## The design decision this plan rests on

**A reference posterior is only a reference if every arm targets the same posterior — and if that target is actually a posterior.** Two things in the production configuration break that, and the reference tier must pin both:

1. **`pos_weight`.** The outcome arms train with `binary_cross_entropy_with_logits(..., pos_weight=...)` (`adaptive_roa/adaptive_v2/trainers/bayesian_mlp_trainer.py:54`), a class-imbalance reweighting derived from the data. That reweights the likelihood, so the arms target a tempered-per-class object rather than `p(w) · Π p(yᵢ|xᵢ,w)`.

2. **β-NLL.** The final-state arms train with `beta_nll: 0.5` (`heads.py:97`), which multiplies each dimension's loss by a **detached** `σ^(2β)`. That is not a log-likelihood — no posterior corresponds to it. Under β ≠ 0 the MFVI objective is not an ELBO for the stated model.

So the reference tier sets **`pos_weight = 1` and `beta_nll = 0`**. Every arm in that tier then targets one genuine, shared posterior that HMC can be the reference for. The production tier keeps `pos_weight` from the data and β = 0.5 as deliberate optimization aids, and **its fidelity is not directly measured** — the writeup must say so rather than transferring the reference-tier numbers to it.

Recording this because getting it wrong is invisible: HMC would run fine, produce clean R-hat, and report agreement numbers against a target nobody was approximating.

## Verified facts this plan depends on

Measured directly against this environment — do not re-derive:

- **Reference-tier parameter counts** at `[50,50]`: cartpole outcome 2 901, cartpole final-state 3 309, quad3d outcome 3 301, quad3d final-state 4 525. All sit inside the 2 651–20 501 range that published benchmarks run full-batch HMC over on ordinary hardware, so **both** heads are feasible, not just the outcome one.
- **The final-state NLL is differentiable end to end** through the `SO2` and `SO3` branches, and gradients stay finite even for a degenerate all-zero quaternion slice (the `clamp_min` guard in `canonicalize_quaternion` handles it). HMC explores aggressively, so this was worth confirming before committing to `hmc_reg`.
- **`Posterior.predictive_logit_samples`** already exists precisely for finite-support posteriors that enumerate exactly rather than sampling (`adaptive_roa/predictors/posteriors.py`, docstring: *"Posteriors with a FINITE support override this to enumerate it exactly"*). `HMCPosterior` is such a posterior.
- **ReLU degrades leapfrog.** Dinh et al. (NeurIPS 2024) show ReLU's non-differentiability makes the leapfrog local error `Ω(ε)` instead of `O(ε³)`. Hence `activation: tanh` for this whole tier. `build_bayesian_mlp` already accepts `activation` with `tanh` among its options.

## File Structure

**Create:**
- `adaptive_roa/predictors/hmc/__init__.py`
- `adaptive_roa/predictors/hmc/log_posterior.py` — `FlatLogPosterior`: flat-vector ↔ net parameters, plus the differentiable log-posterior for each head
- `adaptive_roa/predictors/hmc/sampler.py` — `leapfrog`, `hmc_chain` with dual-averaging step-size adaptation
- `adaptive_roa/predictors/hmc/diagnostics.py` — function-space R-hat and the HMC-vs-HMC agreement ceiling
- `adaptive_roa/predictors/hmc/posterior.py` — `HMCPosterior` implementing the `Posterior` interface
- `adaptive_roa/adaptive_v2/trainers/hmc_trainer.py` — `HMCTrainer` for both heads
- `configs/adaptive_v2/predictor/{hmc,hmc_reg}.yaml`
- `configs/adaptive_v2/experiment/reference_tier.yaml`
- `tests/predictors/hmc/{__init__.py,test_log_posterior.py,test_sampler.py,test_diagnostics.py,test_hmc_posterior.py,test_hmc_trainer.py}`

**Modify:**
- `adaptive_roa/probabilistic_classifier/` — export wrappers for the two HMC arms

---

### Task 1: `FlatLogPosterior`

**Files:**
- Create: `adaptive_roa/predictors/hmc/__init__.py` (empty), `adaptive_roa/predictors/hmc/log_posterior.py`
- Test: `tests/predictors/hmc/__init__.py` (empty), `tests/predictors/hmc/test_log_posterior.py`

**Interfaces:**
- Consumes: `build_bayesian_mlp` from `adaptive_roa.predictors.bayesian_mlp`; `FinalStateHead` from `adaptive_roa.predictors.heads`
- Produces: `FlatLogPosterior(net, head=None, prior_sigma=1.0)` with `.dim -> int`, `.get_flat() -> Tensor[D]`, `.set_flat(theta)`, `.log_prob(theta, x, y) -> Tensor[]` (scalar), `.grad_log_prob(theta, x, y) -> Tensor[D]`

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/hmc/__init__.py` (empty) and `tests/predictors/hmc/test_log_posterior.py`:

```python
import math

import pytest
import torch

from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.heads import FinalStateHead
from adaptive_roa.predictors.hmc.log_posterior import FlatLogPosterior
from adaptive_roa.systems.cartpole import CartPoleSystem


def _outcome(prior_sigma=1.0):
    net = build_bayesian_mlp(input_dim=5, hidden_dims=[8, 8], output_dim=1,
                             posterior="deterministic", activation="tanh")
    return FlatLogPosterior(net, head=None, prior_sigma=prior_sigma)


def _final_state(prior_sigma=1.0):
    system = CartPoleSystem()
    head = FinalStateHead(system, beta=0.0)
    net = build_bayesian_mlp(input_dim=5, hidden_dims=[8, 8], output_dim=head.n_params,
                             posterior="deterministic", activation="tanh")
    return FlatLogPosterior(net, head=head, prior_sigma=prior_sigma), system


def test_flat_round_trip_is_exact():
    lp = _outcome()
    theta = torch.randn(lp.dim)
    lp.set_flat(theta)
    torch.testing.assert_close(lp.get_flat(), theta)


def test_dim_matches_the_parameter_count():
    lp = _outcome()
    assert lp.dim == sum(p.numel() for p in lp.net.parameters())


def test_outcome_log_prob_equals_the_closed_form():
    """log p = -BCE_sum(logits, y) + log N(theta; 0, prior_sigma^2), summed.
    pos_weight is deliberately absent: the reference tier targets the TRUE
    likelihood so HMC references the same posterior the arms approximate."""
    lp = _outcome(prior_sigma=2.0)
    theta = torch.randn(lp.dim) * 0.1
    x = torch.randn(16, 5)
    y = (torch.rand(16) > 0.5).float()

    lp.set_flat(theta)
    with torch.no_grad():
        logits = lp.net.forward_sample(x).view(-1)
    ll = -torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="sum"
    )
    lprior = (-0.5 * (theta / 2.0) ** 2 - math.log(2.0) - 0.5 * math.log(2 * math.pi)).sum()

    got = lp.log_prob(theta, x, y)
    assert got.item() == pytest.approx((ll + lprior).item(), rel=1e-5)


def test_grad_log_prob_matches_autograd_and_is_finite():
    lp = _outcome()
    theta = torch.randn(lp.dim) * 0.1
    x = torch.randn(16, 5)
    y = (torch.rand(16) > 0.5).float()

    g = lp.grad_log_prob(theta, x, y)
    assert g.shape == (lp.dim,)
    assert torch.isfinite(g).all()

    t = theta.clone().requires_grad_(True)
    lp.log_prob(t, x, y).backward()
    torch.testing.assert_close(g, t.grad, rtol=1e-4, atol=1e-6)


def test_final_state_log_prob_uses_the_head_nll_at_beta_zero():
    """beta-NLL is NOT a likelihood -- it reweights by a detached sigma^(2beta),
    so no posterior corresponds to it. The reference tier must run at beta=0."""
    lp, system = _final_state()
    assert lp.head.beta == 0.0

    theta = torch.randn(lp.dim) * 0.1
    x = torch.randn(12, 5)
    y = torch.randn(12, int(system.state_dim)) * 0.1

    lp.set_flat(theta)
    with torch.no_grad():
        params = lp.net.forward_sample(x)
    expected_ll = -lp.head.nll(params, y).sum()
    lprior = (-0.5 * theta ** 2 - 0.5 * math.log(2 * math.pi)).sum()

    got = lp.log_prob(theta, x, y)
    assert got.item() == pytest.approx((expected_ll + lprior).item(), rel=1e-4)


def test_final_state_gradients_are_finite_including_a_degenerate_quaternion():
    """HMC explores aggressively and will reach parameter regions the optimizer
    never visits. A NaN gradient there silently kills a chain."""
    from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem

    system = Quadrotor3DSystem()
    head = FinalStateHead(system, beta=0.0)
    net = build_bayesian_mlp(input_dim=13, hidden_dims=[8, 8], output_dim=head.n_params,
                             posterior="deterministic", activation="tanh")
    lp = FlatLogPosterior(net, head=head)

    theta = torch.zeros(lp.dim)  # drives the whole output, incl. the quaternion, to 0
    x = torch.randn(8, 13)
    y = torch.randn(8, int(system.state_dim)) * 0.1
    g = lp.grad_log_prob(theta, x, y)
    assert torch.isfinite(g).all()


def test_prior_sigma_scales_the_prior_term():
    theta = torch.randn(_outcome().dim) * 0.5
    x = torch.randn(8, 5)
    y = (torch.rand(8) > 0.5).float()

    tight = _outcome(prior_sigma=0.5)
    loose = _outcome(prior_sigma=5.0)
    torch.manual_seed(0)
    # Same weights in both, so only the prior term differs.
    tight.set_flat(theta)
    loose.set_flat(theta)
    assert tight.log_prob(theta, x, y).item() < loose.log_prob(theta, x, y).item()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_log_posterior.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_roa.predictors.hmc'`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/hmc/__init__.py` (empty file), then `adaptive_roa/predictors/hmc/log_posterior.py`:

```python
"""Flat-vector view of a network plus its differentiable log-posterior.

HMC operates on a single flat parameter vector, so this maps between that view
and the network's parameters and supplies log p(theta | D) with gradients.

Two deliberate omissions, both load-bearing for the reference tier:

* No ``pos_weight``. The production outcome arms reweight the BCE by class
  frequency, which tempers the likelihood per class. A reference posterior must
  be the posterior the arms approximate, and it must actually BE a posterior, so
  this tier targets the unweighted likelihood and the arms are configured to
  match.
* ``beta = 0`` on the final-state head. beta-NLL multiplies each dimension by a
  DETACHED sigma^(2*beta); it is an optimization aid, not a log-likelihood, and
  no posterior corresponds to it.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class FlatLogPosterior:
    """log p(theta | D) for one predictor head, as a function of a flat vector."""

    def __init__(self, net, head=None, prior_sigma: float = 1.0):
        self.net = net
        self.head = head
        self.prior_sigma = float(prior_sigma)
        self._params = [p for p in net.parameters() if p.requires_grad]
        self.dim = int(sum(p.numel() for p in self._params))

    def get_flat(self) -> torch.Tensor:
        return parameters_to_vector(self._params).detach().clone()

    def set_flat(self, theta: torch.Tensor) -> None:
        vector_to_parameters(theta.detach().to(self._params[0].dtype), self._params)

    def _log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        s = self.prior_sigma
        return (-0.5 * (theta / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)).sum()

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Scalar log p(theta | D), differentiable w.r.t. ``theta``.

        Uses functional parameter injection rather than writing into the module,
        so autograd reaches ``theta`` instead of the module's own leaf tensors --
        and so the module is never mutated, which matters because HMC evaluates
        thousands of candidate vectors against one shared net.
        """
        replacements = {}
        offset = 0
        for name, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            replacements[name] = theta[offset:offset + n].view_as(p)
            offset += n
        out = torch.func.functional_call(self.net, replacements, (x,))
        if self.head is None:
            # Outcome head: Bernoulli. No pos_weight -- see the module docstring.
            ll = -F.binary_cross_entropy_with_logits(
                out.view(-1), y.view(-1).to(out.dtype), reduction="sum"
            )
        else:
            # Final-state head: the manifold likelihood at beta = 0.
            ll = -self.head.nll(out, y).sum()
        return ll + self._log_prior(theta)

    def grad_log_prob(self, theta: torch.Tensor, x: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
        t = theta.detach().clone().requires_grad_(True)
        self.log_prob(t, x, y).backward()
        return t.grad.detach().clone()
```

Note `torch.func.functional_call` invokes the module's `forward`, which for a
`Posterior` delegates to `forward_sample` — that delegation exists for exactly
this kind of reuse. If it does not reach the intended path for the posterior kind
in use, read `adaptive_roa/predictors/posteriors.py` and call the correct entry
point; do not silently accept a wrong forward.

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_log_posterior.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/hmc/ tests/predictors/hmc/
git commit -m "feat(hmc): flat-vector log posterior for both predictor heads"
```

---

### Task 2: Leapfrog and the HMC chain

**Files:**
- Create: `adaptive_roa/predictors/hmc/sampler.py`
- Test: `tests/predictors/hmc/test_sampler.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly — the sampler takes callables, so it is testable against analytic targets
- Produces: `leapfrog(theta, momentum, grad_log_prob, step_size, n_steps) -> (theta, momentum)`; `hmc_chain(log_prob, grad_log_prob, theta_init, n_samples, n_warmup, n_leapfrog, target_accept=0.8, seed=0) -> HMCResult` where `HMCResult` has `.samples -> Tensor[n_samples, D]`, `.step_size -> float`, `.accept_rate -> float`, `.divergences -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/hmc/test_sampler.py`:

```python
import math

import pytest
import torch

from adaptive_roa.predictors.hmc.sampler import hmc_chain, leapfrog


def _gaussian_target(sigma=1.0):
    """Standard target with a known answer: N(0, sigma^2 I)."""
    def log_prob(theta):
        return (-0.5 * (theta / sigma) ** 2).sum()

    def grad_log_prob(theta):
        return -theta / (sigma ** 2)

    return log_prob, grad_log_prob


def test_leapfrog_is_reversible():
    """Flip the momentum, run the same number of steps, and you return to the
    start. Reversibility is what makes the MH correction valid; without it the
    chain silently samples the wrong distribution."""
    _lp, glp = _gaussian_target()
    torch.manual_seed(0)
    theta0, mom0 = torch.randn(5), torch.randn(5)

    theta1, mom1 = leapfrog(theta0, mom0, glp, step_size=0.1, n_steps=12)
    theta2, mom2 = leapfrog(theta1, -mom1, glp, step_size=0.1, n_steps=12)

    torch.testing.assert_close(theta2, theta0, atol=1e-5, rtol=0)
    torch.testing.assert_close(-mom2, mom0, atol=1e-5, rtol=0)


def test_leapfrog_conserves_energy_to_second_order():
    """Energy error must shrink like step_size^2 for a smooth target. A first-order
    scheme (a wrong half-step) would show a linear trend instead."""
    lp, glp = _gaussian_target()
    torch.manual_seed(0)
    theta0, mom0 = torch.randn(8), torch.randn(8)

    def energy(th, mo):
        return (-lp(th) + 0.5 * (mo ** 2).sum()).item()

    e0 = energy(theta0, mom0)
    errs = []
    for eps in (0.2, 0.1, 0.05):
        th, mo = leapfrog(theta0, mom0, glp, step_size=eps, n_steps=int(1.0 / eps))
        errs.append(abs(energy(th, mo) - e0))
    # Halving the step size should cut the error by roughly 4x.
    assert errs[1] < errs[0] / 3.0
    assert errs[2] < errs[1] / 3.0


def test_chain_recovers_a_gaussian_target():
    lp, glp = _gaussian_target(sigma=2.0)
    res = hmc_chain(lp, glp, torch.zeros(3), n_samples=800, n_warmup=400,
                    n_leapfrog=20, seed=0)
    assert res.samples.shape == (800, 3)
    assert res.samples.mean(0).abs().max().item() < 0.35
    assert abs(res.samples.std(0).mean().item() - 2.0) < 0.35


def test_step_size_adapts_toward_the_target_acceptance():
    lp, glp = _gaussian_target()
    res = hmc_chain(lp, glp, torch.zeros(4), n_samples=400, n_warmup=400,
                    n_leapfrog=15, target_accept=0.8, seed=0)
    assert 0.6 < res.accept_rate < 0.95
    assert res.step_size > 0.0
    assert res.divergences == 0


def test_different_seeds_give_different_chains_and_same_seed_reproduces():
    lp, glp = _gaussian_target()
    kw = dict(theta_init=torch.zeros(3), n_samples=100, n_warmup=100, n_leapfrog=10)
    a = hmc_chain(lp, glp, seed=0, **kw).samples
    b = hmc_chain(lp, glp, seed=0, **kw).samples
    c = hmc_chain(lp, glp, seed=1, **kw).samples
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


def test_divergences_are_counted_not_silently_accepted():
    """A target with a hard cliff should register divergences rather than
    quietly producing garbage samples."""
    def log_prob(theta):
        return torch.where(theta.abs().max() < 1.0,
                           -0.5 * (theta ** 2).sum(),
                           torch.tensor(-1e6))

    def grad_log_prob(theta):
        return -theta * 1e3

    res = hmc_chain(log_prob, grad_log_prob, torch.zeros(2), n_samples=50,
                    n_warmup=50, n_leapfrog=20, seed=0)
    assert res.divergences > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_sampler.py -v`
Expected: FAIL — `ModuleNotFoundError` for `sampler`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/hmc/sampler.py`:

```python
"""Hand-rolled HMC with dual-averaging step-size adaptation.

Fixed-length leapfrog rather than NUTS: this matches what published BNN
reference benchmarks actually run at this scale, and it is ~200 lines with no
new dependency. The Metropolis correction makes the chain exact for any step
size; adaptation only tunes efficiency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class HMCResult:
    samples: torch.Tensor      # [n_samples, D]
    step_size: float
    accept_rate: float
    divergences: int


def leapfrog(theta: torch.Tensor, momentum: torch.Tensor,
             grad_log_prob: Callable[[torch.Tensor], torch.Tensor],
             step_size: float, n_steps: int):
    """Standard leapfrog: half-kick, then (drift, kick) x n, then half-kick.

    Reversible and volume-preserving, which is what makes the Metropolis
    correction valid.
    """
    theta = theta.clone()
    momentum = momentum + 0.5 * step_size * grad_log_prob(theta)
    for i in range(int(n_steps)):
        theta = theta + step_size * momentum
        g = grad_log_prob(theta)
        if i < n_steps - 1:
            momentum = momentum + step_size * g
        else:
            momentum = momentum + 0.5 * step_size * g
    return theta, momentum


def hmc_chain(log_prob: Callable[[torch.Tensor], torch.Tensor],
              grad_log_prob: Callable[[torch.Tensor], torch.Tensor],
              theta_init: torch.Tensor, n_samples: int, n_warmup: int,
              n_leapfrog: int = 20, target_accept: float = 0.8,
              seed: int = 0, divergence_threshold: float = 1000.0) -> HMCResult:
    """Run one chain. Warmup adapts the step size; only post-warmup draws are kept."""
    gen = torch.Generator(device=theta_init.device).manual_seed(int(seed))
    theta = theta_init.clone()
    dim = theta.numel()

    # Dual averaging state (Hoffman & Gelman).
    log_eps = math.log(0.1)
    log_eps_bar = 0.0
    h_bar = 0.0
    mu = math.log(10.0 * math.exp(log_eps))
    gamma, t0, kappa = 0.05, 10.0, 0.75

    kept = []
    accepts = 0
    divergences = 0
    total = int(n_warmup) + int(n_samples)

    for t in range(1, total + 1):
        eps = math.exp(log_eps if t <= n_warmup else log_eps_bar)
        momentum = torch.randn(dim, generator=gen, device=theta.device, dtype=theta.dtype)

        current_h = -log_prob(theta) + 0.5 * (momentum ** 2).sum()
        new_theta, new_mom = leapfrog(theta, momentum, grad_log_prob, eps, n_leapfrog)
        new_h = -log_prob(new_theta) + 0.5 * (new_mom ** 2).sum()

        delta = (current_h - new_h).item()
        if not math.isfinite(delta) or delta < -divergence_threshold:
            divergences += 1
            accept_prob = 0.0
        else:
            accept_prob = min(1.0, math.exp(min(delta, 0.0)))

        if torch.rand((), generator=gen, device=theta.device).item() < accept_prob:
            theta = new_theta
            if t > n_warmup:
                accepts += 1

        if t <= n_warmup:
            eta = 1.0 / (t + t0)
            h_bar = (1.0 - eta) * h_bar + eta * (target_accept - accept_prob)
            log_eps = mu - math.sqrt(t) / gamma * h_bar
            w = t ** (-kappa)
            log_eps_bar = w * log_eps + (1.0 - w) * log_eps_bar
        else:
            kept.append(theta.detach().clone())

    return HMCResult(
        samples=torch.stack(kept, dim=0) if kept else torch.empty(0, dim),
        step_size=math.exp(log_eps_bar),
        accept_rate=accepts / max(int(n_samples), 1),
        divergences=divergences,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_sampler.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/hmc/sampler.py tests/predictors/hmc/test_sampler.py
git commit -m "feat(hmc): leapfrog integrator and dual-averaging HMC chain"
```

---

### Task 3: Function-space R-hat and the agreement ceiling

**Files:**
- Create: `adaptive_roa/predictors/hmc/diagnostics.py`
- Test: `tests/predictors/hmc/test_diagnostics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks — operates on plain tensors
- Produces: `function_space_rhat(predictions) -> Tensor[N]` taking `[n_chains, n_draws, N]`; `agreement(p, q) -> float`; `total_variation(p, q) -> float`; `hmc_vs_hmc_ceiling(predictions) -> dict`

**Why the ceiling matters:** an approximation's agreement with HMC is only interpretable against how well HMC agrees with *itself*. Reporting "MFVI agrees with HMC 71% of the time" without the ceiling invites the reader to compare it against 100%, when the achievable maximum may be 85%.

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/hmc/test_diagnostics.py`:

```python
import pytest
import torch

from adaptive_roa.predictors.hmc.diagnostics import (
    agreement,
    function_space_rhat,
    hmc_vs_hmc_ceiling,
    total_variation,
)


def test_rhat_is_near_one_for_chains_from_one_distribution():
    torch.manual_seed(0)
    preds = torch.rand(4, 200, 30)  # [chains, draws, points]
    r = function_space_rhat(preds)
    assert r.shape == (30,)
    assert r.max().item() < 1.1


def test_rhat_flags_chains_that_disagree():
    torch.manual_seed(0)
    preds = torch.rand(4, 200, 30) * 0.1
    preds[0] += 5.0  # one chain sampling somewhere else entirely
    assert function_space_rhat(preds).max().item() > 1.2


def test_rhat_needs_at_least_two_chains():
    with pytest.raises(ValueError, match="chains"):
        function_space_rhat(torch.rand(1, 100, 10))


def test_agreement_is_one_for_identical_predictions():
    p = torch.tensor([0.9, 0.2, 0.6])
    assert agreement(p, p.clone()) == pytest.approx(1.0)


def test_agreement_counts_matching_hard_labels():
    p = torch.tensor([0.9, 0.2, 0.6, 0.4])
    q = torch.tensor([0.8, 0.3, 0.4, 0.6])  # last two cross 0.5
    assert agreement(p, q) == pytest.approx(0.5)


def test_total_variation_is_zero_for_identical_and_one_for_opposite():
    p = torch.tensor([0.7, 0.3])
    assert total_variation(p, p.clone()) == pytest.approx(0.0)
    assert total_variation(torch.tensor([1.0, 0.0]),
                           torch.tensor([0.0, 1.0])) == pytest.approx(1.0)


def test_ceiling_reports_below_one_for_finite_chains():
    """The ceiling is what an approximation is actually competing against; it is
    below 1.0 because HMC's own chains differ at finite sample size."""
    torch.manual_seed(0)
    preds = torch.sigmoid(torch.randn(3, 150, 40))
    out = hmc_vs_hmc_ceiling(preds)
    assert set(out) >= {"agreement", "total_variation", "n_chains"}
    assert 0.0 < out["agreement"] <= 1.0
    assert out["n_chains"] == 3


def test_ceiling_needs_at_least_two_chains():
    with pytest.raises(ValueError, match="chains"):
        hmc_vs_hmc_ceiling(torch.rand(1, 50, 10))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_diagnostics.py -v`
Expected: FAIL — `ModuleNotFoundError` for `diagnostics`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/hmc/diagnostics.py`:

```python
"""Convergence and fidelity diagnostics for the HMC reference tier.

R-hat is computed in FUNCTION space (per query point's predictive probability),
not weight space. Weight-space R-hat for a neural network is hopeless — the
posterior is massively multimodal under permutation symmetry, so chains that
agree perfectly on predictions still look divergent in weights. Published BNN
reference work reports the function-space statistic for exactly this reason.
"""
from __future__ import annotations

import torch


def function_space_rhat(predictions: torch.Tensor) -> torch.Tensor:
    """Split-free Gelman-Rubin per point. ``predictions`` is [chains, draws, N]."""
    if predictions.shape[0] < 2:
        raise ValueError(
            f"R-hat needs at least 2 chains, got {predictions.shape[0]}"
        )
    m, n = predictions.shape[0], predictions.shape[1]
    chain_means = predictions.mean(dim=1)                       # [m, N]
    chain_vars = predictions.var(dim=1, unbiased=True)          # [m, N]
    w = chain_vars.mean(dim=0)                                  # within
    b = chain_means.var(dim=0, unbiased=True) * n               # between
    var_hat = (n - 1) / n * w + b / n
    return torch.sqrt((var_hat / w.clamp_min(1e-12)).clamp_min(0.0))


def agreement(p: torch.Tensor, q: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of points where two predictives give the same hard label."""
    return float(((p >= threshold) == (q >= threshold)).float().mean().item())


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """Mean total-variation distance between two Bernoulli predictives."""
    return float((p - q).abs().mean().item())


def hmc_vs_hmc_ceiling(predictions: torch.Tensor) -> dict:
    """How well HMC agrees with ITSELF, leave-one-chain-out.

    This is the ceiling an approximation competes against. Without it, an
    agreement of 0.71 invites comparison against 1.0 when the achievable maximum
    at this sample size may be 0.85.
    """
    if predictions.shape[0] < 2:
        raise ValueError(
            f"the HMC-vs-HMC ceiling needs at least 2 chains, got {predictions.shape[0]}"
        )
    agreements, tvs = [], []
    for i in range(predictions.shape[0]):
        held = predictions[i].mean(dim=0)
        rest = torch.cat([predictions[:i], predictions[i + 1:]], dim=0)
        rest_mean = rest.reshape(-1, rest.shape[-1]).mean(dim=0)
        agreements.append(agreement(held, rest_mean))
        tvs.append(total_variation(held, rest_mean))
    return {
        "agreement": float(sum(agreements) / len(agreements)),
        "total_variation": float(sum(tvs) / len(tvs)),
        "n_chains": int(predictions.shape[0]),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_diagnostics.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/hmc/diagnostics.py tests/predictors/hmc/test_diagnostics.py
git commit -m "feat(hmc): function-space R-hat and the HMC-vs-HMC agreement ceiling"
```

---

### Task 4: `HMCPosterior`

**Files:**
- Create: `adaptive_roa/predictors/hmc/posterior.py`
- Test: `tests/predictors/hmc/test_hmc_posterior.py`

**Interfaces:**
- Consumes: `Posterior` from `adaptive_roa.predictors.posteriors`
- Produces: `HMCPosterior(net, samples)` with `.n_draws -> int`, `.forward_sample(x, generator=None)`, `.predictive_logit_samples(x, S, generator=None)`, `.kl_divergence()`

**The key structural point:** a set of HMC draws is a finite-support posterior, exactly like an ensemble's members. `EnsemblePosterior` overrides `predictive_logit_samples` to enumerate its support rather than sample it, because sampling a finite support with replacement introduces systematic weight error. `HMCPosterior` must do the same — with hundreds of atoms the enumeration is both exact and affordable, and it means the arm's marginal is not a function of the caller's `S`.

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/hmc/test_hmc_posterior.py`:

```python
import pytest
import torch

from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.hmc.posterior import HMCPosterior


def _posterior(n_draws=32, out=1):
    net = build_bayesian_mlp(input_dim=4, hidden_dims=[8, 8], output_dim=out,
                             posterior="deterministic", activation="tanh")
    dim = sum(p.numel() for p in net.parameters())
    torch.manual_seed(0)
    return HMCPosterior(net, torch.randn(n_draws, dim) * 0.3)


def test_reports_its_support_size():
    assert _posterior(n_draws=40).n_draws == 40


def test_rejects_an_empty_sample_set():
    net = build_bayesian_mlp(input_dim=4, hidden_dims=[8], output_dim=1,
                             posterior="deterministic", activation="tanh")
    dim = sum(p.numel() for p in net.parameters())
    with pytest.raises(ValueError, match="at least"):
        HMCPosterior(net, torch.zeros(0, dim))


def test_forward_sample_draws_one_atom_and_varies():
    post = _posterior()
    x = torch.randn(6, 4)
    outs = torch.stack([post.forward_sample(x) for _ in range(40)])
    assert outs.shape == (40, 6, 1)
    assert outs.std(dim=0).mean().item() > 0.0


def test_forward_sample_is_reproducible_under_a_seeded_generator():
    post = _posterior()
    x = torch.randn(6, 4)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    torch.testing.assert_close(post.forward_sample(x, generator=g1),
                               post.forward_sample(x, generator=g2))


def test_predictive_enumerates_the_support_exactly_and_ignores_S():
    """Finite support, same as EnsemblePosterior: sampling it with replacement
    would make the marginal a function of the caller's S. Enumeration is exact."""
    post = _posterior(n_draws=32)
    x = torch.randn(5, 4)
    a = post.predictive_logit_samples(x, S=8)
    b = post.predictive_logit_samples(x, S=1000)
    assert a.shape == (32, 5, 1)
    torch.testing.assert_close(a, b)


def test_each_enumerated_atom_uses_a_distinct_weight_vector():
    post = _posterior(n_draws=16)
    out = post.predictive_logit_samples(torch.randn(3, 4), S=16)
    flat = out.reshape(16, -1)
    assert len({tuple(r.tolist()) for r in flat}) == 16


def test_kl_divergence_is_zero():
    assert _posterior().kl_divergence().item() == 0.0


def test_works_for_a_multi_output_head():
    post = _posterior(n_draws=12, out=9)
    assert post.predictive_logit_samples(torch.randn(4, 4), S=12).shape == (12, 4, 9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_hmc_posterior.py -v`
Expected: FAIL — `ModuleNotFoundError` for `posterior`

- [ ] **Step 3: Write the implementation**

Create `adaptive_roa/predictors/hmc/posterior.py`:

```python
"""A set of HMC draws, exposed through the shared Posterior interface.

Structurally this is the same object as EnsemblePosterior -- a finite support of
weight vectors -- so it plugs into both shipped model handles unchanged. It
follows the ensemble's precedent on the marginal too: enumerate the support
exactly rather than sampling it with replacement, so the predictive is not a
function of the caller's S.
"""
from __future__ import annotations

import torch
from torch.nn.utils import vector_to_parameters

from adaptive_roa.predictors.posteriors import Posterior


class HMCPosterior(Posterior):
    def __init__(self, net, samples: torch.Tensor):
        super().__init__()
        if samples.ndim != 2 or samples.shape[0] < 1:
            raise ValueError(
                f"HMCPosterior needs at least 1 draw shaped [n_draws, dim], "
                f"got {tuple(samples.shape)}"
            )
        self.net = net
        self.register_buffer("samples", samples.detach().clone())
        self._params = [p for p in net.parameters() if p.requires_grad]

    @property
    def n_draws(self) -> int:
        return int(self.samples.shape[0])

    def _forward_with(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        replacements = {}
        offset = 0
        for name, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            replacements[name] = theta[offset:offset + n].view_as(p)
            offset += n
        return torch.func.functional_call(self.net, replacements, (x,))

    def forward_sample(self, x, generator=None):
        idx = int(torch.randint(self.n_draws, (1,), generator=generator,
                                device="cpu").item())
        with torch.no_grad():
            return self._forward_with(self.samples[idx], x)

    def predictive_logit_samples(self, x, S, generator=None):
        """Enumerate the full support: [n_draws, B, out]. ``S`` is ignored."""
        with torch.no_grad():
            return torch.stack([self._forward_with(t, x) for t in self.samples], dim=0)
```

Note `forward_sample` draws its index on CPU deliberately, matching
`EnsemblePosterior`'s convention so a CPU generator works on a CUDA model — the
device-mismatch trap that cost a fix round in an earlier plan.

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_hmc_posterior.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add adaptive_roa/predictors/hmc/posterior.py tests/predictors/hmc/test_hmc_posterior.py
git commit -m "feat(hmc): HMC posterior enumerating its finite support"
```

---

### Task 5: `HMCTrainer`, the two arm configs, and the reference-tier experiment

**Files:**
- Create: `adaptive_roa/adaptive_v2/trainers/hmc_trainer.py`
- Create: `configs/adaptive_v2/predictor/hmc.yaml`, `configs/adaptive_v2/predictor/hmc_reg.yaml`
- Create: `configs/adaptive_v2/experiment/reference_tier.yaml`
- Test: `tests/predictors/hmc/test_hmc_trainer.py`

**Interfaces:**
- Consumes: `FlatLogPosterior` (Task 1), `hmc_chain` (Task 2), `function_space_rhat`/`hmc_vs_hmc_ceiling` (Task 3), `HMCPosterior` (Task 4)
- Produces: `HMCTrainer(cfg, system, system_name)` with `fit(dataset_files, output_dir, resume_checkpoint=None) -> model_handle`

**Read first:** `adaptive_roa/adaptive_v2/trainers/bayesian_mlp_trainer.py` for the outcome-head datamodule and `final_state_trainer.py` for the endpoint one. This trainer selects between them on `predictor.head`, runs `n_chains` HMC chains, assembles an `HMCPosterior`, and returns `OutcomeModelHandle` or `FinalStateModelHandle` accordingly.

**Four requirements specific to this trainer:**

1. **`pos_weight = 1` and `beta_nll = 0`.** See the design section. The trainer must not read a data-derived `pos_weight`, and must build its `FinalStateHead` with `beta=0.0`. Assert both rather than relying on config discipline.
2. **Write the diagnostics artifact.** After sampling, evaluate each chain's predictive on the validation split and write `checkpoints/hmc_diagnostics.json` containing per-chain `accept_rate`, `step_size`, `divergences`, the max and mean function-space R-hat, and the HMC-vs-HMC ceiling. A reference arm whose own convergence is unauditable is not a reference.
3. **Write `checkpoints/best-hmc.ckpt`** so the engine's `checkpoints/best*.ckpt` glob (`adaptive_roa/adaptive_v2/engine.py:140`) matches. Store the draws and enough metadata to rebuild.
4. **No warm start.** HMC restarts from scratch each epoch; accepting a `resume_checkpoint` and ignoring it is the exact trap that cost two fix rounds earlier in this workstream. Raise if one is passed, with a message saying HMC is a reference arm that does not resume.

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/hmc/test_hmc_trainer.py`:

```python
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.trainers.hmc_trainer import HMCTrainer
from adaptive_roa.systems.cartpole import CartPoleSystem


def _write_classification(path, n=160, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 4))
    np.savetxt(path, np.column_stack([X, (np.abs(X[:, 1]) < 0.5).astype(int)]))
    return str(path)


def _write_endpoints(path, n=160, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.5, 0.5, size=(n, 4))
    np.savetxt(path, np.column_stack([X, X * 0.5]))
    return str(path)


def _cfg(head, **overrides):
    hmc = {"hidden_dims": [8, 8], "activation": "tanh", "prior_sigma": 1.0,
           "n_chains": 2, "n_samples": 20, "n_warmup": 20, "n_leapfrog": 8, "seed": 0}
    hmc.update(overrides)
    return OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "classifier" if head == "outcome" else "generative",
                      "name": "hmc" if head == "outcome" else "hmc_reg",
                      "head": head, "batch_size": 64, "hmc": hmc},
    })


@pytest.fixture
def outcome_files(tmp_path):
    return {"train": _write_classification(tmp_path / "tr.txt"),
            "val": _write_classification(tmp_path / "va.txt", n=80, seed=1)}


@pytest.fixture
def endpoint_files(tmp_path):
    return {"train": _write_endpoints(tmp_path / "tr.txt"),
            "val": _write_endpoints(tmp_path / "va.txt", n=80, seed=1)}


def test_outcome_arm_returns_a_working_handle(outcome_files, tmp_path):
    handle = HMCTrainer(_cfg("outcome"), CartPoleSystem(), "cartpole_pybullet").fit(
        outcome_files, str(tmp_path / "out")
    )
    logits = handle(torch.randn(12, 4))
    assert logits.shape in {(12,), (12, 1)}
    assert torch.isfinite(logits).all()
    # The outcome handle marginalizes internally and must be deterministic.
    torch.testing.assert_close(handle(torch.zeros(4, 4)), handle(torch.zeros(4, 4)))


def test_final_state_arm_returns_a_working_handle(endpoint_files, tmp_path):
    handle = HMCTrainer(_cfg("final_state"), CartPoleSystem(), "cartpole_pybullet").fit(
        endpoint_files, str(tmp_path / "out")
    )
    x = torch.randn(12, 4) * 0.3
    out = handle.predict_endpoint(x)
    assert out.shape == (12, 4)
    assert torch.isfinite(out).all()
    # The final-state handle must draw fresh every call.
    assert not torch.allclose(handle.predict_endpoint(x), handle.predict_endpoint(x))


@pytest.mark.parametrize("head", ["outcome", "final_state"])
def test_writes_a_checkpoint_matching_the_engine_glob(head, tmp_path, outcome_files,
                                                      endpoint_files):
    files = outcome_files if head == "outcome" else endpoint_files
    out = tmp_path / f"out_{head}"
    HMCTrainer(_cfg(head), CartPoleSystem(), "cartpole_pybullet").fit(files, str(out))
    assert list((out / "checkpoints").glob("best*.ckpt"))


def test_writes_an_auditable_diagnostics_artifact(outcome_files, tmp_path):
    """A reference arm whose own convergence cannot be audited is not a reference."""
    out = tmp_path / "out"
    HMCTrainer(_cfg("outcome"), CartPoleSystem(), "cartpole_pybullet").fit(
        outcome_files, str(out)
    )
    d = json.loads((out / "checkpoints" / "hmc_diagnostics.json").read_text())
    assert len(d["chains"]) == 2
    for c in d["chains"]:
        assert 0.0 <= c["accept_rate"] <= 1.0
        assert c["step_size"] > 0.0
        assert "divergences" in c
    assert d["rhat_max"] >= 1.0
    assert 0.0 < d["ceiling"]["agreement"] <= 1.0


def test_reference_tier_pins_pos_weight_and_beta(outcome_files, endpoint_files, tmp_path):
    """beta-NLL is not a likelihood and pos_weight tempers one, so neither may
    reach the reference target -- otherwise HMC references a posterior no arm
    is approximating."""
    t = HMCTrainer(_cfg("outcome"), CartPoleSystem(), "cartpole_pybullet")
    t.fit(outcome_files, str(tmp_path / "a"))
    assert t.pos_weight == 1.0

    t2 = HMCTrainer(_cfg("final_state"), CartPoleSystem(), "cartpole_pybullet")
    t2.fit(endpoint_files, str(tmp_path / "b"))
    assert t2.head.beta == 0.0


def test_resume_checkpoint_raises_rather_than_being_ignored(outcome_files, tmp_path):
    """Silently accepting and discarding a resume path is the exact trap that
    made two sibling arms appear to warm-start when they did not."""
    with pytest.raises(ValueError, match="does not resume"):
        HMCTrainer(_cfg("outcome"), CartPoleSystem(), "cartpole_pybullet").fit(
            outcome_files, str(tmp_path / "out"), resume_checkpoint="/some/path.ckpt"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_hmc_trainer.py -v`
Expected: FAIL — `ModuleNotFoundError` for `hmc_trainer`

- [ ] **Step 3: Write the trainer**

Create `adaptive_roa/adaptive_v2/trainers/hmc_trainer.py`, following
`bayesian_mlp_trainer.py`'s `__init__(cfg, system, system_name)` /
`fit(dataset_files, output_dir, resume_checkpoint=None)` contract. Structure:

```python
    def fit(self, dataset_files, output_dir, resume_checkpoint=None):
        if resume_checkpoint:
            raise ValueError(
                "HMC is a reference arm and does not resume: each epoch re-runs "
                "its chains from scratch. Accepting and ignoring a "
                "resume_checkpoint would make warm start appear to work."
            )
```

The body, after the guard above:

```python
        hmc_cfg = self._predictor_cfg.get("hmc", {})
        head_kind = str(self._predictor_cfg.get("head", "outcome"))
        device = "cpu"  # chains are tiny; the gradient is full-batch over ~3k params

        x_train, y_train, x_val, y_val = self._load_split(dataset_files, head_kind)

        # Reference-tier invariants, asserted rather than trusted to config
        # discipline: pos_weight tempers the likelihood and beta-NLL is not a
        # likelihood at all, so with either active HMC would reference a target
        # no arm is approximating.
        self.pos_weight = 1.0
        self.head = None
        if head_kind == "final_state":
            self.head = FinalStateHead(self.system, beta=0.0)
        out_dim = 1 if self.head is None else self.head.n_params

        dummy = torch.zeros(1, int(self.system.state_dim))
        input_dim = int(
            self.system.embed_state_for_model(self.system.normalize_state(dummy)).shape[-1]
        )
        net = build_bayesian_mlp(
            input_dim=input_dim,
            hidden_dims=list(hmc_cfg.get("hidden_dims", [50, 50])),
            output_dim=out_dim,
            posterior="deterministic",
            activation=str(hmc_cfg.get("activation", "tanh")),
        ).to(device)

        prior_sigma = float(hmc_cfg.get("prior_sigma", 1.0))
        lp = FlatLogPosterior(net, head=self.head, prior_sigma=prior_sigma)

        # The net owns normalize+embed for its inputs, so feed it raw states.
        def log_prob(theta):
            return lp.log_prob(theta, self._embed(x_train), y_train)

        def grad_log_prob(theta):
            return lp.grad_log_prob(theta, self._embed(x_train), y_train)

        base_seed = int(hmc_cfg.get("seed", 0))
        n_chains = int(hmc_cfg.get("n_chains", 3))
        results, per_chain_preds = [], []
        for c in range(n_chains):
            gen = torch.Generator().manual_seed(base_seed + c)
            theta0 = torch.randn(lp.dim, generator=gen) * prior_sigma
            res = hmc_chain(
                log_prob, grad_log_prob, theta0,
                n_samples=int(hmc_cfg.get("n_samples", 200)),
                n_warmup=int(hmc_cfg.get("n_warmup", 200)),
                n_leapfrog=int(hmc_cfg.get("n_leapfrog", 20)),
                seed=base_seed + c,
            )
            results.append(res)
            per_chain_preds.append(self._chain_predictions(net, res.samples, x_val))

        posterior = HMCPosterior(net, torch.cat([r.samples for r in results], dim=0))

        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # MUST match the engine's glob, checkpoints/best*.ckpt (engine.py:140).
        torch.save(
            {"samples": posterior.samples, "hidden_dims": list(hmc_cfg.get("hidden_dims", [50, 50])),
             "activation": str(hmc_cfg.get("activation", "tanh")),
             "input_dim": input_dim, "output_dim": out_dim, "head": head_kind},
            ckpt_dir / "best-hmc.ckpt",
        )

        preds = torch.stack(per_chain_preds, dim=0)   # [chains, draws, N]
        rhat = function_space_rhat(preds)
        (ckpt_dir / "hmc_diagnostics.json").write_text(json.dumps({
            "chains": [
                {"accept_rate": r.accept_rate, "step_size": r.step_size,
                 "divergences": r.divergences} for r in results
            ],
            "rhat_max": float(rhat.max()),
            "rhat_mean": float(rhat.mean()),
            "ceiling": hmc_vs_hmc_ceiling(preds),
        }, indent=2))

        if self.head is None:
            return OutcomeModelHandle(posterior, self.system).eval().to(device)
        return FinalStateModelHandle(posterior, self.head, self.system).eval().to(device)
```

Three helpers this needs, which you write to match the existing datamodules:

- `_load_split(dataset_files, head_kind)` — returns `(x_train, y_train, x_val, y_val)` as raw tensors. For `outcome`, reuse `AdaptiveClassificationDataModule` as `bayesian_mlp_trainer` does and read `._train.states` / `._train.labels`. For `final_state`, reuse the endpoint datamodule map and `_full_training_tensors` from `final_state_trainer` — both are module-level there (`_DATAMODULES` at line 36, `_full_training_tensors` at line 416) — and collate the val dataset the same way.
- `_embed(x)` — `self.system.embed_state_for_model(self.system.normalize_state(x))`.
- `_chain_predictions(net, samples, x_val)` — for the outcome head, `sigmoid` of the logits under each draw, shape `[draws, N]`; for the final-state head, the same shape from a scalar summary so R-hat is well defined — use the predicted mean's distance to the success attractor under `system.classify_attractor`'s metric, or simply the first output dimension. Whichever you choose, say so in a comment: R-hat needs a per-point scalar, and the choice determines what "converged" means here.

- [ ] **Step 4: Write the configs**

`configs/adaptive_v2/predictor/hmc.yaml`:

```yaml
# @package _global_
defaults:
  - /probability: classifier_prob
threshold:
  decision_rule: one_sided
calibration:
  decision_rule: one_sided
predictor:
  type: classifier          # family tag
  name: hmc                 # arm name
  head: outcome
  trainer_target: adaptive_roa.adaptive_v2.trainers.hmc_trainer.HMCTrainer
  batch_size: 1024
  hmc:
    hidden_dims: [50, 50]   # reference tier: full-batch HMC is routine at ~3k params
    activation: tanh        # NOT relu: its non-differentiability degrades the
                            # leapfrog local error to O(eps) (Dinh et al. 2024)
    prior_sigma: 1.0        # isotropic Gaussian, identical across arms
    n_chains: 3
    n_samples: 200
    n_warmup: 200
    n_leapfrog: 20
    seed: 0
```

`hmc_reg.yaml` is the same with `type: generative`, `name: hmc_reg`,
`head: final_state`, and `defaults: - /probability: endpoint_mc`; drop the
threshold/calibration overrides, matching the other generative arms.

`configs/adaptive_v2/experiment/reference_tier.yaml` — this is what makes the
tier comparable, and it composes last so its overrides survive:

```yaml
# @package _global_
# Reference tier: every arm at the [50,50] tanh backbone where full-batch HMC is
# feasible, targeting ONE shared posterior so HMC is a valid reference for the
# approximations.
#
# pos_weight and beta-NLL are pinned OFF here. pos_weight tempers the likelihood
# per class; beta-NLL multiplies each dimension by a detached sigma^(2*beta) and
# is not a likelihood at all, so no posterior corresponds to it. With either
# active, HMC would reference a target no arm is approximating -- and it would
# look fine.
#
# The production tier keeps both as deliberate optimization aids. Its fidelity is
# therefore NOT measured by this tier; do not transfer these numbers to it.
#
# Usage:
#   python scripts/run_adaptive.py +experiment=reference_tier predictor=<arm>

predictor:
  classifier:
    hidden_dims: [50, 50]
  bnn:
    hidden_dims: [50, 50]
    activation: tanh
    pos_weight: 1.0
  final_state:
    hidden_dims: [50, 50]
    activation: tanh
    beta_nll: 0.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_hmc_trainer.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Verify the configs compose**

Run:
```bash
./env/bin/python -c "
from hydra import compose, initialize_config_dir
import os
d = os.path.abspath('configs/adaptive_v2')
for arm, fam in [('hmc','classifier'), ('hmc_reg','generative')]:
    with initialize_config_dir(config_dir=d, version_base=None):
        cfg = compose(config_name='default', overrides=[f'predictor={arm}'])
    assert cfg.predictor.name == arm and cfg.predictor.type == fam, arm
    assert list(cfg.predictor.hmc.hidden_dims) == [50, 50]
    assert cfg.predictor.hmc.activation == 'tanh'
    print(arm, 'OK')
with initialize_config_dir(config_dir=d, version_base=None):
    ref = compose(config_name='default', overrides=['+experiment=reference_tier',
                                                    'predictor=bnn_mfvi_reg'])
assert list(ref.predictor.final_state.hidden_dims) == [50, 50]
assert float(ref.predictor.final_state.beta_nll) == 0.0
print('reference_tier OK')
"
```
Expected: `hmc OK`, `hmc_reg OK`, `reference_tier OK`.

- [ ] **Step 7: Commit**

```bash
git add adaptive_roa/adaptive_v2/trainers/hmc_trainer.py configs/adaptive_v2/predictor/hmc.yaml configs/adaptive_v2/predictor/hmc_reg.yaml configs/adaptive_v2/experiment/reference_tier.yaml tests/predictors/hmc/test_hmc_trainer.py
git commit -m "feat(hmc): HMC trainer, both arm configs, and the reference-tier experiment"
```

---

### Task 6: Export wrappers and end-to-end smoke

**Files:**
- Create: `adaptive_roa/probabilistic_classifier/hmc.py`
- Modify: `adaptive_roa/probabilistic_classifier/__init__.py`
- Test: `tests/predictors/hmc/test_hmc_export.py`

**Interfaces:**
- Consumes: `HMCPosterior` (Task 4), `HMCTrainer` (Task 5), the registry
- Produces: `HMCProbabilisticClassifier` (name `hmc`), `HMCRegProbabilisticClassifier` (name `hmc_reg`)

**Read first:** `adaptive_roa/probabilistic_classifier/bayesian.py` (outcome arms) and `bayesian_final_state.py` (final-state arms). These mirror them: `hmc` follows the former, `hmc_reg` the latter, including the shared `endpoint_mc_probabilities` helper the GP plan factored out. Reuse that helper rather than re-implementing the MC loop — duplicating it lost the batching and the empty-input guard once already.

**Two load-bearing points, both previously bitten:** the side-effect import goes **after** the existing ones in `__init__.py` so `ClassifierProbabilisticClassifier` and `FMProbabilisticClassifier` keep the `"classifier"` and `"generative"` family aliases; and the loader must **raise** on a state-dict key mismatch rather than loading an untrained network via `strict=False`.

- [ ] **Step 1: Write the failing test**

Create `tests/predictors/hmc/test_hmc_export.py`:

```python
import pytest


@pytest.mark.parametrize("arm,family", [("hmc", "classifier"), ("hmc_reg", "generative")])
def test_each_arm_is_registered(arm, family):
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class(arm)
    assert cls.predictor_name == arm
    assert cls.predictor_type == family


def test_hmc_arms_do_not_steal_the_legacy_family_aliases():
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.classifier import (
        ClassifierProbabilisticClassifier,
    )
    from adaptive_roa.probabilistic_classifier.flow_matching import FMProbabilisticClassifier

    assert get_probabilistic_classifier_class("classifier") is ClassifierProbabilisticClassifier
    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier


def test_hmc_reg_declares_all_three_probabilities():
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    assert get_probabilistic_classifier_class("hmc_reg").native_probs == (
        "p_success", "p_failure", "p_invalid"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_hmc_export.py -v`
Expected: FAIL — `KeyError: 'hmc'`

- [ ] **Step 3: Write the export wrappers**

Create `adaptive_roa/probabilistic_classifier/hmc.py`:

```python
"""Export wrappers for the two HMC reference arms.

The checkpoint carries its own architecture metadata (hidden_dims, activation,
input/output dims), so the net is rebuilt from the checkpoint rather than from
the run config -- there is no path where a config edit silently produces a
wrong-shaped model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.final_state_handle import FinalStateModelHandle
from adaptive_roa.predictors.handles import OutcomeModelHandle
from adaptive_roa.predictors.heads import FinalStateHead
from adaptive_roa.predictors.hmc.posterior import HMCPosterior
from .base import ProbabilisticClassifier
from .endpoint_mc import endpoint_mc_probabilities
from .registry import register_probabilistic_classifier


def _load_hmc(run_dir, epoch, device):
    path = Path(run_dir) / f"epoch_{epoch:03d}" / "checkpoints" / "best-hmc.ckpt"
    if not path.exists():
        raise FileNotFoundError(f"no HMC checkpoint at {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_bayesian_mlp(
        input_dim=int(ck["input_dim"]), hidden_dims=list(ck["hidden_dims"]),
        output_dim=int(ck["output_dim"]), posterior="deterministic",
        activation=str(ck["activation"]),
    )
    return HMCPosterior(net, ck["samples"]).eval().to(device), ck


@register_probabilistic_classifier
class HMCProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "classifier"
    predictor_name = "hmc"
    native_probs = ("p_success",)

    def __init__(self, handle, system, device):
        self.handle = handle
        self.system = system
        self.device = device

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        out = []
        with torch.no_grad():
            for i in range(0, len(states), 8192):
                x = torch.as_tensor(states[i:i + 8192], dtype=torch.float32,
                                    device=self.device)
                out.append(torch.sigmoid(self.handle(x).view(-1)).double().cpu().numpy())
        p = np.concatenate(out) if out else np.zeros(0)
        return OutcomeProbabilities(p_success=p, p_failure=1.0 - p,
                                    p_invalid=np.zeros_like(p))

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        posterior, ck = _load_hmc(run_dir, epoch, device)
        if ck["head"] != "outcome":
            raise ValueError(f"checkpoint head is {ck['head']!r}, expected 'outcome'")
        handle = OutcomeModelHandle(posterior, system).eval().to(device)
        return cls(handle, system, device)


@register_probabilistic_classifier
class HMCRegProbabilisticClassifier(ProbabilisticClassifier):
    predictor_type = "generative"
    predictor_name = "hmc_reg"
    native_probs = ("p_success", "p_failure", "p_invalid")

    def __init__(self, handle, system, device, attractor_radius, num_mc_samples):
        self.handle = handle
        self.system = system
        self.device = device
        self.attractor_radius = attractor_radius
        self.num_mc_samples = num_mc_samples

    def predict(self, states: np.ndarray) -> OutcomeProbabilities:
        return endpoint_mc_probabilities(
            self.handle, self.system, states,
            attractor_radius=self.attractor_radius,
            num_mc_samples=self.num_mc_samples,
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        prob = cfg.get("probability", {})
        if "attractor_radius" not in prob:
            raise KeyError(
                f"run config at {run_dir} has no probability.attractor_radius; "
                "refusing to guess a radius, which would relabel every endpoint."
            )
        posterior, ck = _load_hmc(run_dir, epoch, device)
        if ck["head"] != "final_state":
            raise ValueError(f"checkpoint head is {ck['head']!r}, expected 'final_state'")
        head = FinalStateHead(system, beta=0.0)
        handle = FinalStateModelHandle(posterior, head, system).eval().to(device)
        return cls(handle, system, device,
                   attractor_radius=float(prob["attractor_radius"]),
                   num_mc_samples=int(prob.get("num_mc_samples", 10)))
```

`endpoint_mc_probabilities` is verified to live at
`adaptive_roa/probabilistic_classifier/endpoint_mc.py:23` with the signature
`(handle, system, states, attractor_radius, num_mc_samples, batch_size=BATCH_SIZE)`.
Use it; do not re-implement the loop. Its docstring records why the batching is
load-bearing rather than merely a memory concern.

Add `from . import hmc as _hmc  # noqa: F401,E402` to
`adaptive_roa/probabilistic_classifier/__init__.py` **after** the existing
side-effect imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./env/bin/python -m pytest tests/predictors/hmc/test_hmc_export.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the pipeline smoke tests**

Run, for each of the two arms:
```bash
./env/bin/python scripts/run_adaptive.py \
  --config-name=default system=pendulum predictor=hmc \
  sampling_mode=ranked +adaptive_v2.smoke_mode=true n_epochs=2 \
  predictor.hmc.n_samples=20 predictor.hmc.n_warmup=20 predictor.hmc.n_chains=2
```
and the same with `predictor=hmc_reg`. Expected: two epochs complete and both
`epoch_00{0,1}/artifacts_v2.json` exist. `n_epochs` is the correct top-level key.
Also confirm `epoch_000/checkpoints/hmc_diagnostics.json` exists and its
`rhat_max` is finite. If a run fails on dataset paths for environmental reasons
rather than code, record the exact command and output and proceed.

- [ ] **Step 6: Run the full suite for regressions**

Run: `./env/bin/python -m pytest tests -q`
Expected: no failures.

- [ ] **Step 7: Commit**

```bash
git add adaptive_roa/probabilistic_classifier/ tests/predictors/hmc/test_hmc_export.py
git commit -m "feat(export): probabilistic-classifier wrappers for the HMC reference arms"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| Hand-rolled leapfrog with dual-averaging | 2 |
| tanh/GELU backbone (not ReLU) | 1 (tests), 5 (configs) |
| `[50,50]` reference configs, all four systems | 5 |
| Both heads — `hmc` and `hmc_reg` | 1, 4, 5, 6 |
| Function-space R-hat | 3, 5 (artifact) |
| HMC-vs-HMC agreement ceiling | 3, 5 (artifact) |
| One shared target: `pos_weight=1`, `beta_nll=0` | 1, 5 |
| Export wrappers | 6 |

**Deferred deliberately:** the MDN likelihood (still optional until the Gaussian head demonstrably fails at separatrices) and benchmark orchestration (Plan 4 — paired random-acquisition controls, matched budgets, prior-scale sweep, separatrix-conditioned reporting).

**Type consistency:** `FlatLogPosterior(net, head, prior_sigma)` with `.dim`, `.get_flat`, `.set_flat`, `.log_prob`, `.grad_log_prob` is used identically in Tasks 1 and 5. `hmc_chain(...) -> HMCResult` with `.samples`, `.step_size`, `.accept_rate`, `.divergences` is used in Tasks 2 and 5. `HMCPosterior(net, samples)` with `.n_draws` is constructed the same way in Tasks 4, 5 and 6. The diagnostics functions are consumed only in Task 5.

**Known risk to watch:** Tasks 1 and 4 both use `torch.func.functional_call` to evaluate a net at a supplied parameter vector. That relies on `Posterior.forward` delegating to `forward_sample`. It does today, and Task 1's brief says to verify rather than assume — but the two call sites should stay in agreement, and a reviewer should check they do.

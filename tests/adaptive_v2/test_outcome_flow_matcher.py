"""Tests for the scalar-outcome flow matcher.

The load-bearing test is `test_exact_readout_matches_analytic_flow`: a constant
velocity field has flow map Psi(x0) = x0 + c, so p = Phi(c) in closed form. If
the bracket/bisect readout is wrong -- wrong tail, off-by-one on the bracket
index, inverted sign convention -- that test fails with an exact expected value
rather than a plausible-looking number.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from adaptive_roa.model.outcome_flow_matcher import (
    OutcomeFlowMatcher,
    OutcomeVelocityMLP,
)


class _IdentitySystem:
    """Minimal system stub: no normalization, no embedding."""

    state_dim = 1

    @staticmethod
    def normalize_state(x):
        return x

    @staticmethod
    def embed_state_for_model(x):
        return x


class _ConstantVelocity(nn.Module):
    """Velocity field v == c, independent of state/x/t. Flow map: x0 -> x0 + c."""

    def __init__(self, c: float):
        super().__init__()
        self.c = float(c)

    def forward(self, condition, x_t, t):
        return torch.full_like(x_t, self.c)


def _matcher(velocity_net) -> OutcomeFlowMatcher:
    return OutcomeFlowMatcher(velocity_net=velocity_net, system=_IdentitySystem(), num_ode_steps=50)


@pytest.mark.parametrize("c", [-1.5, -0.5, 0.0, 0.5, 1.5])
def test_exact_readout_matches_analytic_flow(c):
    """Psi(x0) = x0 + c  =>  p = P(x0 > -c) = Phi(c)."""
    model = _matcher(_ConstantVelocity(c))
    states = torch.zeros(8, 1)

    p, monotone = model.p_success_exact(states)

    expected = 0.5 * (1.0 + math.erf(c / math.sqrt(2.0)))
    assert bool(monotone.all()), "a constant field is monotone; the check said otherwise"
    assert np.allclose(p.numpy(), expected, atol=1e-4), f"got {p[0].item():.6f} want {expected:.6f}"


def test_readout_precision_is_below_the_effect_size():
    """The ablation measures debiased-Brier gaps of 1e-4..1e-3. If the readout's
    own discretisation error reaches that scale, a 'result' can be an artefact.
    Pin both error sources against the analytic Phi(c).
    """
    c = 0.3
    expected = 0.5 * (1.0 + math.erf(c / math.sqrt(2.0)))
    model = OutcomeFlowMatcher(
        velocity_net=_ConstantVelocity(c), system=_IdentitySystem(), num_ode_steps=50
    )
    p, _ = model.p_success_exact(torch.zeros(4, 1), grid_size=33, bisect_iters=20)
    err = float(np.abs(p.numpy() - expected).max())
    assert err < 1e-5, f"readout error {err:.2e} is not safely below the 1e-4 effect scale"


def test_flow_map_integrates_constant_field():
    model = _matcher(_ConstantVelocity(0.75))
    x0 = torch.tensor([-1.0, 0.0, 2.0])
    cond = torch.zeros(3, 1)
    assert torch.allclose(model.flow_map(cond, x0), x0 + 0.75, atol=1e-5)


def test_mc_and_exact_agree_on_analytic_flow():
    """The two readouts must estimate the same quantity; MC only noisier."""
    model = _matcher(_ConstantVelocity(0.4))
    states = torch.zeros(4, 1)

    torch.manual_seed(0)
    p_mc = model.p_success_mc(states, num_samples=20000).numpy()
    p_exact, _ = model.p_success_exact(states)

    # MC standard error at K=20000 is ~0.0035; 4 sigma is a stable bound.
    assert np.allclose(p_mc, p_exact.numpy(), atol=0.014)


def test_forward_returns_logits_matching_p_success():
    """`forward(raw) -> logit(p)` is the contract every discriminative code path
    (threshold optimisation, conformal calibration, ClassifierProbabilityEstimator)
    depends on. If it drifts from `predict_p_success`, calibration silently
    optimises against a different quantity than evaluation reports."""
    model = _matcher(_ConstantVelocity(0.6))
    states = torch.zeros(5, 1)

    logits = model(states)
    p_from_logits = torch.sigmoid(logits).view(-1)
    p_direct = model.predict_p_success(states)

    assert logits.shape == (5, 1), f"expected [B,1] logits, got {tuple(logits.shape)}"
    assert torch.allclose(p_from_logits.double(), p_direct.double(), atol=1e-6)

    expected = 0.5 * (1.0 + math.erf(0.6 / math.sqrt(2.0)))
    assert np.allclose(p_from_logits.numpy(), expected, atol=1e-4)


def test_forward_is_finite_at_saturated_probabilities():
    """A strongly-confident field drives p to within 1e-7 of the tails; without
    clamping the logit is +-inf and every downstream loss becomes NaN."""
    model = _matcher(_ConstantVelocity(20.0))
    logits = model(torch.zeros(3, 1))
    assert torch.isfinite(logits).all(), "logit saturated to inf; clamp is not holding"


def test_anchor_mapping():
    """label 1 -> +1 (success), label 0 -> -1 (failure)."""
    anchors = OutcomeFlowMatcher._anchor(torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert torch.equal(anchors, torch.tensor([1.0, -1.0, 1.0, -1.0]))


def test_nonmonotone_map_is_detected_not_silently_accepted():
    """A field engineered to fold the line must be flagged, so the quadrature
    fallback can never be mistaken for a clean bisection."""

    class _Folding(nn.Module):
        # Pushes both tails positive and the middle negative => 2 sign changes.
        def forward(self, condition, x_t, t):
            return 6.0 * x_t * torch.abs(x_t) - 4.0 * x_t

    model = _matcher(_Folding())
    p, monotone = model.p_success_exact(torch.zeros(4, 1))

    assert not bool(monotone.all()), "folding field should not be reported monotone"
    assert np.all((p.numpy() >= 0.0) & (p.numpy() <= 1.0))


def test_recovers_known_probability_field():
    """End-to-end: train on Bernoulli labels with a known p(x), check recovery.

    This is the test that says the *idea* works, not just the plumbing: velocity
    regression against hard 0/1 anchors has to produce a calibrated interior
    probability, which is the entire premise of the ablation.
    """
    torch.manual_seed(0)
    n = 20000
    x = torch.rand(n, 1)
    p_true = x.view(-1)                      # p(x) = x, exactly
    y = (torch.rand(n) < p_true).float()

    model = OutcomeFlowMatcher(
        velocity_net=OutcomeVelocityMLP(condition_dim=1, hidden_dims=[128, 128]),
        system=_IdentitySystem(),
        lr=3e-3,
        num_ode_steps=50,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(600):
        idx = torch.randint(0, n, (1024,))
        opt.zero_grad()
        loss = model._step({"inputs": x[idx], "label": y[idx]}, "train")
        loss.backward()
        opt.step()

    model.eval()
    probe = torch.linspace(0.05, 0.95, 19).view(-1, 1)
    p_hat, monotone = model.p_success_exact(probe)

    err = (p_hat.numpy() - probe.view(-1).numpy())
    assert bool(monotone.all()), "learned field folded the line on a well-posed problem"
    assert np.abs(err).mean() < 0.06, f"mean |error| {np.abs(err).mean():.4f} too high"
    assert np.abs(err).max() < 0.15, f"max |error| {np.abs(err).max():.4f} too high"

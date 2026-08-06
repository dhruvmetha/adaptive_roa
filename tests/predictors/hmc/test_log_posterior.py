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
    """Assert the DIFFERENCE against the closed form rather than a direction.

    At small theta the -log(sigma) term dominates the -theta^2/(2 sigma^2) one,
    so a TIGHTER prior gives a HIGHER log-density -- an inequality here is easy
    to get backwards. The likelihood term cancels exactly, because log_prob
    injects the same theta into both nets regardless of their own weights.
    """
    theta = torch.randn(_outcome().dim) * 0.5
    x = torch.randn(8, 5)
    y = (torch.rand(8) > 0.5).float()

    def prior_term(sigma):
        return (-0.5 * (theta / sigma) ** 2 - math.log(sigma)
                - 0.5 * math.log(2 * math.pi)).sum().item()

    got = (_outcome(prior_sigma=0.5).log_prob(theta, x, y).item()
           - _outcome(prior_sigma=5.0).log_prob(theta, x, y).item())
    assert got == pytest.approx(prior_term(0.5) - prior_term(5.0), rel=1e-5)

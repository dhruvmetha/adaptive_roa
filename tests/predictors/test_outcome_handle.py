import math

import numpy as np
import pytest
import torch

from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.handles import OutcomeModelHandle
from adaptive_roa.predictors.posteriors import EnsemblePosterior, Posterior
from adaptive_roa.systems.cartpole import CartPoleSystem

_ALL_KINDS = ["deterministic", "mfvi", "ensemble", "laplace"]


def _handle(kind="mfvi", **kw):
    system = CartPoleSystem()
    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
    posterior = build_bayesian_mlp(
        input_dim=input_dim, hidden_dims=[16, 16], output_dim=1, posterior=kind
    )
    return OutcomeModelHandle(posterior, system, **kw), system


def _fit_if_laplace(handle, system, kind, x):
    """Fit the last-layer Laplace posterior so it actually samples.

    Pre-fit, ``LastLayerLaplacePosterior.forward_sample`` falls back to the MAP
    point estimate, which is trivially deterministic and would not exercise the
    sampling path this test is meant to cover.
    """
    if kind != "laplace":
        return
    embedded = system.embed_state_for_model(system.normalize_state(x))
    features = handle.posterior.body(embedded)
    targets = torch.randint(0, 2, (x.shape[0],), dtype=torch.float32, device=x.device)
    handle.posterior.fit(features, targets, task="outcome")


def test_handle_is_deterministic_across_repeated_calls():
    """Threshold optimization, calibration, and eval each call this separately
    and must agree. A stochastic handle silently desynchronizes them."""
    handle, system = _handle()
    x = torch.randn(16, int(system.state_dim))
    assert torch.allclose(handle(x), handle(x))
    assert torch.allclose(handle(x), handle(x))  # and again


def test_handle_returns_logits_the_classifier_estimator_can_consume():
    handle, system = _handle()
    out = handle(torch.randn(16, int(system.state_dim)))
    assert out.shape in {(16,), (16, 1)}
    assert torch.isfinite(out).all()


def test_handle_marginalizes_rather_than_taking_one_draw():
    """More marginal samples => a tighter estimate of the same quantity.

    The seed must be set BEFORE _handle() builds the network, so all eight
    handles share identical weights and the only thing varying is the
    marginalization draw. Seeding afterwards would leave the weights different
    and the measured spread would be weight noise, not MC error.
    """
    x = torch.randn(32, 4)
    spreads = []
    for n in (2, 128):
        outs = []
        for seed in range(8):
            torch.manual_seed(0)
            handle, _ = _handle(n_marginal_samples=n, seed=seed)
            outs.append(handle(x))
        spreads.append(torch.stack(outs).std(dim=0).mean().item())
    assert spreads[1] < spreads[0]


def test_handle_accepts_numpy_and_moves_to_device():
    handle, system = _handle()
    handle.eval().to("cpu")
    out = handle(np.random.randn(8, int(system.state_dim)).astype(np.float32))
    assert torch.is_tensor(out)


def test_deterministic_posterior_handle_matches_a_plain_forward():
    handle, system = _handle(kind="deterministic")
    x = torch.randn(8, int(system.state_dim))
    normalized = system.embed_state_for_model(system.normalize_state(x))
    expected = handle.posterior.forward_sample(normalized).view(-1)
    assert torch.allclose(handle(x), expected, atol=1e-5)


@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_handle_is_deterministic_for_every_posterior_kind(kind):
    """Every posterior kind must agree with itself across repeated calls --
    including ensemble (member index drawn from the generator) and a FITTED
    laplace posterior (the only state in which it actually samples)."""
    handle, system = _handle(kind=kind)
    x = torch.randn(16, int(system.state_dim))
    _fit_if_laplace(handle, system, kind, x)
    assert torch.allclose(handle(x), handle(x))
    assert torch.allclose(handle(x), handle(x))


class _ConstantPosterior(Posterior):
    """Posterior whose every draw is a fixed logit. Lets a test pin exact numbers."""

    def __init__(self, value):
        super().__init__()
        self._unused = torch.nn.Linear(1, 1)  # gives .parameters() a device
        self.value = float(value)

    def forward_sample(self, x, generator=None):
        return torch.full((x.shape[0], 1), self.value, device=x.device, dtype=x.dtype)


class _IdentitySystem:
    state_dim = 1

    def normalize_state(self, x):
        return x

    def embed_state_for_model(self, x):
        return x


@pytest.mark.parametrize("logit", [25.0, -25.0, 100.0, -100.0])
def test_handle_logits_stay_finite_when_the_posterior_saturates(logit):
    """sigmoid saturates to exactly 1.0 in float32 for logits above ~17.

    Averaging probabilities and inverting with log(p/(1-p)) then returns +inf,
    and the obvious guard -- p.clamp(1e-12, 1 - 1e-12) -- does NOT help, because
    float32(1 - 1e-12) rounds to exactly 1.0, making the upper clamp a no-op
    while the lower one works. The handle must marginalize in log space instead.
    A +inf logit propagates silently into thresholds and calibration.
    """
    handle = OutcomeModelHandle(_ConstantPosterior(logit), _IdentitySystem(),
                                n_marginal_samples=8)
    out = handle(torch.zeros(4, 1))
    assert torch.isfinite(out).all()
    # All draws share one logit, so the marginal must be exactly that logit.
    torch.testing.assert_close(out, torch.full((4,), logit), rtol=0, atol=1e-3)


def test_ensemble_marginal_is_exact_enumeration_not_sampling():
    """The ensemble's M atoms carry weight 1/M each; that is computable exactly.

    Drawing atoms with replacement instead gives a multinomial weight vector,
    and because the handle seeds its generator that vector is FIXED for the whole
    run -- a systematic bias, not noise that averages out. With the members below
    the seeded-MC estimate was 0.5125 against an exact 0.5800, which flips a
    decision at lambda* = 0.5.
    """
    probs = [0.9, 0.9, 0.9, 0.1, 0.1]
    logits = [math.log(p / (1.0 - p)) for p in probs]
    posterior = EnsemblePosterior([_ConstantPosterior(v) for v in logits])

    handle = OutcomeModelHandle(posterior, _IdentitySystem(), n_marginal_samples=64)
    p_bar = torch.sigmoid(handle(torch.zeros(3, 1)))

    expected = sum(probs) / len(probs)
    torch.testing.assert_close(p_bar, torch.full((3,), expected), rtol=0, atol=1e-6)

    # And S must not matter: enumeration ignores it.
    coarse = OutcomeModelHandle(posterior, _IdentitySystem(), n_marginal_samples=2)
    torch.testing.assert_close(torch.sigmoid(coarse(torch.zeros(3, 1))), p_bar)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_handle_works_on_cuda(kind):
    """Regression test for a device-mismatch crash: the handle's internal
    generator must live on the same device as the posterior's parameters, and
    every posterior's own sampling (VILinear, EnsemblePosterior's member-index
    draw, LastLayerLaplacePosterior's eps draw) must honor that same device."""
    handle, system = _handle(kind=kind)
    x_cpu = torch.randn(16, int(system.state_dim))
    _fit_if_laplace(handle, system, kind, x_cpu)

    handle.eval().to("cuda:0")
    x = torch.randn(16, int(system.state_dim), device="cuda:0")

    out1 = handle(x)
    out2 = handle(x)
    assert torch.isfinite(out1).all()
    assert torch.allclose(out1, out2)

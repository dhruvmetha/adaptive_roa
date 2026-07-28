import numpy as np
import pytest
import torch

from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.handles import OutcomeModelHandle
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

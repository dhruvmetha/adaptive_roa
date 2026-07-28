import numpy as np
import torch

from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.handles import OutcomeModelHandle
from adaptive_roa.systems.cartpole import CartPoleSystem


def _handle(kind="mfvi", **kw):
    system = CartPoleSystem()
    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
    posterior = build_bayesian_mlp(
        input_dim=input_dim, hidden_dims=[16, 16], output_dim=1, posterior=kind
    )
    return OutcomeModelHandle(posterior, system, **kw), system


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

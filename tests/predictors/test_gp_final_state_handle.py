import numpy as np
import pytest
import torch

from adaptive_roa.predictors.embedding import EmbeddedStateDecoder
from adaptive_roa.predictors.gp_final_state_handle import GPFinalStateHandle
from adaptive_roa.predictors.gp_regressor import GPRegressor
from adaptive_roa.systems.cartpole import CartPoleSystem
from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem


def _handle(cls=CartPoleSystem, n_iters=40):
    system = cls()
    dec = EmbeddedStateDecoder(system)
    rng = np.random.default_rng(0)
    raw = torch.as_tensor(rng.uniform(-0.5, 0.5, size=(200, dec.state_dim)), dtype=torch.float32)
    feats = system.embed_state_for_model(system.normalize_state(raw))
    targets = system.embed_state_for_model(system.normalize_state(raw * 0.5))
    gp = GPRegressor(
        num_tasks=dec.embed_dim, input_dim=dec.embed_dim, n_inducing=16, n_iters=n_iters
    ).fit(feats, targets)
    return GPFinalStateHandle(gp, system), system


def test_predict_endpoint_draws_a_fresh_sample_every_call():
    """The estimator calls this K times on the SAME batch and the spread across
    calls IS the outcome probability. A deterministic handle collapses the arm
    to p in {0, 1}."""
    handle, system = _handle()
    x = torch.randn(16, int(system.state_dim)) * 0.3
    assert not torch.allclose(handle.predict_endpoint(x), handle.predict_endpoint(x))


def test_predict_endpoint_returns_finite_raw_states():
    handle, system = _handle()
    out = handle.predict_endpoint(torch.randn(16, int(system.state_dim)) * 0.3)
    assert out.shape == (16, int(system.state_dim))
    assert torch.isfinite(out).all()


def test_predict_endpoint_takes_a_single_positional_arg():
    import inspect

    sig = inspect.signature(GPFinalStateHandle.predict_endpoint)
    required = [p for n, p in list(sig.parameters.items())[1:]
                if p.default is inspect.Parameter.empty]
    assert len(required) == 1


def test_component_names_and_distance_shapes_agree():
    handle, system = _handle()
    a = handle.predict_endpoint(torch.randn(8, int(system.state_dim)) * 0.3)
    b = handle.predict_endpoint(torch.randn(8, int(system.state_dim)) * 0.3)
    assert handle.compute_manifold_distance_per_component(a, b).shape == (
        8, len(handle.get_manifold_component_names())
    )


def test_endpoint_error_uses_the_normalized_convention():
    """Must match flow_matcher.py:1156-1161, which normalizes both arguments.
    Both land in artifacts_v2.json's endpoint_error, read positionally across
    predictor families."""
    handle, system = _handle()
    a = torch.zeros(1, int(system.state_dim))
    b = torch.zeros(1, int(system.state_dim))
    b[0, 0] = 1.0  # one raw unit on a Euclidean component
    got = handle.compute_manifold_distance_per_component(a, b)
    expected = handle.head.distance_per_component(
        system.normalize_state(a), system.normalize_state(b)
    )
    torch.testing.assert_close(got, expected)


def test_distance_manifold_stays_raw():
    """full_roa.py passes RAW endpoints for the flow-matching family too, so the
    shim must not normalize or endpoint_errors would diverge instead."""
    handle, system = _handle()
    a = torch.zeros(1, int(system.state_dim))
    b = torch.zeros(1, int(system.state_dim))
    b[0, 0] = 1.0
    torch.testing.assert_close(
        handle.distance_manifold.dist(a, b), handle.head.distance_per_component(a, b)
    )


def test_quadrotor3d_endpoints_are_canonicalized():
    handle, system = _handle(cls=Quadrotor3DSystem, n_iters=20)
    out = handle.predict_endpoint(torch.randn(32, int(system.state_dim)) * 0.2)
    q = out[:, 3:7]
    torch.testing.assert_close(torch.linalg.norm(q, dim=-1), torch.ones(32), atol=1e-5, rtol=0)
    assert (q[:, 0] >= 0).all()
    assert system.classify_attractor(out, radius=0.3).shape == (32,)


def test_handle_is_module_shaped_for_the_engine():
    handle, _ = _handle()
    assert handle.eval() is not None
    assert handle.to("cpu") is not None
    assert isinstance(handle.training, bool)

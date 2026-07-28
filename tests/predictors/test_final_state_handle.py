import math

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


def test_distance_manifold_is_present_and_is_the_raw_convention():
    """full_roa.py:642 guards get_manifold_component_names() on hasattr(
    'distance_manifold'), so a handle with one but not the other crashes.

    ``distance_manifold.dist`` must stay RAW: its only caller (full_roa.py:621,
    649) hands the FM family's ``distance_manifold.dist`` raw states too, so
    normalizing here would make ``full_roa['endpoint_errors']`` incomparable
    across the two families -- the same defect C1 fixed in the other field.
    """
    handle, system = _handle()
    assert hasattr(handle, "distance_manifold")
    a = handle.predict_endpoint(torch.randn(8, int(system.state_dim)))
    b = handle.predict_endpoint(torch.randn(8, int(system.state_dim)))
    assert torch.allclose(handle.distance_manifold.dist(a, b),
                          handle.head.distance_per_component(a, b))


def test_endpoint_error_uses_the_flow_matchers_normalized_convention():
    """C1. ``compute_manifold_distance_per_component`` lands in artifacts_v2.json
    under ``endpoint_error`` alongside the FM arms' identically-named,
    identically-shaped output, and compile_adaptive_metrics.py reads it
    POSITIONALLY. The FM family normalizes both arguments first
    (flow_matcher.py:1156-1161), so this must too or the two families' rows are
    tabulated together in different units.

    Pendulum makes the mutation visible: its velocity bound is 2*pi, so a raw
    computation inflates component 1 by exactly 6.28x while leaving the angle
    (which normalize_state does not touch) alone -- which is precisely why the
    discrepancy survived every shape and name check.
    """
    from adaptive_roa.systems.pendulum import PendulumSystem

    handle, system = _handle(cls=PendulumSystem)
    a = handle.predict_endpoint(torch.randn(64, int(system.state_dim)))
    b = handle.predict_endpoint(torch.randn(64, int(system.state_dim)))

    got = handle.compute_manifold_distance_per_component(a, b)
    fm_convention = handle.head.distance_per_component(
        system.normalize_state(a), system.normalize_state(b)
    )
    torch.testing.assert_close(got, fm_convention)

    assert not torch.allclose(got, raw_of(handle, a, b))

    # The exact 6.28x factor, on endpoints inside the system's own bounds.
    # PendulumSystem.normalize_state CLAMPS theta_dot to +/-1, so the scaling is
    # only exact where |theta_dot| <= 2*pi; an untrained net's outputs are not,
    # and the saturation there is genuine (the FM family clamps identically, so
    # the two still agree -- which the assertion above already covers).
    bound = float(system.state_bounds["angular_velocity"][1])
    torch.manual_seed(0)
    a_in = torch.stack([torch.rand(64) * 2 * math.pi - math.pi,
                        torch.rand(64) * 2 * bound - bound], dim=1)
    b_in = torch.stack([torch.rand(64) * 2 * math.pi - math.pi,
                        torch.rand(64) * 2 * bound - bound], dim=1)
    got_in = handle.compute_manifold_distance_per_component(a_in, b_in)
    raw_in = raw_of(handle, a_in, b_in)
    torch.testing.assert_close(raw_in[:, 1], got_in[:, 1] * bound, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(raw_in[:, 0], got_in[:, 0])  # angle is unaffected


def raw_of(handle, a, b):
    """The raw-coordinate convention, i.e. what C1 found here before the fix."""
    return handle.head.distance_per_component(a, b)


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
    """Degeneracy guard: a seeded or deterministic handle yields identical draws,
    so p_success is 0 or 1 at EVERY radius. A stochastic handle must produce a
    genuinely intermediate probability at some radius. Sweeping avoids depending
    on the untrained network's output scale, which a fixed radius cannot."""
    handle, system = _handle()
    x = torch.randn(32, int(system.state_dim))
    draws = torch.stack([handle.predict_endpoint(x) for _ in range(20)], dim=0)

    # The draws themselves must vary -- this is the property under test.
    assert draws.std(dim=0).mean().item() > 0.0

    for radius in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
        labels = torch.stack(
            [system.classify_attractor(d, radius=radius) for d in draws], dim=0
        )
        p_success = (labels == 1).float().mean(dim=0)
        if bool(((p_success > 0.0) & (p_success < 1.0)).any()):
            return
    pytest.fail(
        "no radius in the sweep produced an intermediate p_success; the handle "
        "may be deterministic or seeded"
    )

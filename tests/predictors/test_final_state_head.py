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

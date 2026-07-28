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


@pytest.mark.parametrize("cls,state_dim,n_names,expected_names", SYSTEMS)
def test_parameter_slices_map_to_correct_state_positions(cls, state_dim, n_names, expected_names):
    """Parameter slices must map to the correct state positions. A transposition
    of same-width components (both Real) would pass shape tests but fail this."""
    system = cls()
    head = FinalStateHead(system)

    # Build a params tensor with distinct known values per component
    params = torch.zeros(1, head.n_params)
    expected_state = {}  # component_idx -> expected state value

    # Fill in each component's parameters with distinct recognizable values
    param_offset = 0
    component_idx = 0

    for comp in system.manifold_components:
        lik, _, n_p, _, _ = head._parts[component_idx]

        if comp.manifold_type == "Real":
            # Real(dim): mu and log_sigma structure, n_params = 2*dim
            val = float(10.0 + component_idx)  # 10.0, 11.0, 12.0, etc.
            params[0, param_offset:param_offset + comp.dim] = val
            params[0, param_offset + comp.dim:param_offset + 2 * comp.dim] = 0.0  # log_sigma
            expected_state[component_idx] = torch.full((comp.dim,), val)
        elif comp.manifold_type == "SO2":
            # SO2(1): (sin, cos, log_sigma), n_params = 3
            # Set to a known angle: (component_idx + 1) * pi/4
            angle = (component_idx + 1) * math.pi / 4
            params[0, param_offset] = math.sin(angle)
            params[0, param_offset + 1] = math.cos(angle)
            params[0, param_offset + 2] = 0.0  # log_sigma
            expected_state[component_idx] = torch.tensor([angle])
        elif comp.manifold_type == "SO3":
            # SO3(1): (qw, qx, qy, qz, log_sig_x, log_sig_y, log_sig_z), n_params = 7
            # Use identity quaternion (1, 0, 0, 0)
            q = torch.tensor([1.0, 0.0, 0.0, 0.0])
            params[0, param_offset:param_offset + 4] = q
            params[0, param_offset + 4:param_offset + 7] = 0.0  # log_sigmas
            expected_state[component_idx] = q

        param_offset += n_p
        component_idx += 1

    mean_output = head.mean(params)

    # Verify that each component's mean appears at the correct state position
    state_offset = 0
    component_idx = 0
    for comp in system.manifold_components:
        expected = expected_state[component_idx]
        actual = mean_output[0, state_offset:state_offset + comp.dim]

        # For Real and SO3, check exact match; for SO2 check angle wrapping
        if comp.manifold_type in ("Real", "SO3"):
            assert torch.allclose(actual, expected, atol=1e-5), \
                f"Component {component_idx} ({comp.manifold_type}) mismatch at state[{state_offset}:{state_offset + comp.dim}]"
        elif comp.manifold_type == "SO2":
            # SO2 angle may be wrapped; check it's close to expected angle
            assert torch.allclose(actual, expected, atol=1e-5) or \
                   torch.allclose(actual + 2*math.pi, expected, atol=1e-5) or \
                   torch.allclose(actual - 2*math.pi, expected, atol=1e-5), \
                f"Component {component_idx} (SO2) angle mismatch"

        state_offset += comp.dim
        component_idx += 1


def test_generator_threads_through_all_component_types():
    """Generator reproducibility must work across Real, SO2, and SO3 simultaneously.
    No real system has all three; use a stub."""
    from adaptive_roa.systems.base import ManifoldComponent

    class AllComponentsStub:
        manifold_components = [
            ManifoldComponent("Real", 2, "position"),
            ManifoldComponent("SO2", 1, "angle"),
            ManifoldComponent("SO3", 4, "rotation"),
            ManifoldComponent("Real", 1, "velocity"),
        ]

    head = FinalStateHead(AllComponentsStub())
    params = torch.randn(8, head.n_params)

    # Two identically-seeded generators should produce identical samples
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)

    s1 = head.sample(params, generator=g1)
    s2 = head.sample(params, generator=g2)
    assert torch.allclose(s1, s2), "Generator seeding failed for mixed manifolds"

    # A differently-seeded generator should produce different samples
    g3 = torch.Generator().manual_seed(99)
    s3 = head.sample(params, generator=g3)
    assert not torch.allclose(s1, s3), "Different seeds should produce different samples"

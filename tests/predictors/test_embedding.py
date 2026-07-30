import math

import pytest
import torch

from adaptive_roa.predictors.embedding import EmbeddedStateDecoder
from adaptive_roa.systems.cartpole import CartPoleSystem
from adaptive_roa.systems.quadrotor2d import Quadrotor2DSystem
from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem

SYSTEMS = [(CartPoleSystem, 4, 5), (Quadrotor2DSystem, 6, 7), (Quadrotor3DSystem, 13, 13)]


@pytest.mark.parametrize("cls,state_dim,embed_dim", SYSTEMS)
def test_dims_match_the_system(cls, state_dim, embed_dim):
    dec = EmbeddedStateDecoder(cls())
    assert dec.state_dim == state_dim
    assert dec.embed_dim == embed_dim


@pytest.mark.parametrize("cls,state_dim,embed_dim", SYSTEMS)
def test_decode_inverts_embed_for_in_range_states(cls, state_dim, embed_dim):
    """decode(embed(normalize(x))) == x for states inside the system's bounds.
    This is the round trip the GP relies on: it learns in embedded space and the
    handle must recover the raw state exactly."""
    system = cls()
    dec = EmbeddedStateDecoder(system)

    torch.manual_seed(0)
    raw = system.denormalize_state(torch.rand(64, state_dim) * 2 - 1)
    if cls is Quadrotor3DSystem:  # keep the quaternion on the sphere, qw >= 0
        q = torch.nn.functional.normalize(raw[:, 3:7], dim=-1)
        raw[:, 3:7] = torch.where(q[:, 0:1] < 0, -q, q)

    embedded = system.embed_state_for_model(system.normalize_state(raw))
    assert embedded.shape == (64, embed_dim)
    torch.testing.assert_close(dec.decode(embedded), raw, atol=1e-4, rtol=1e-4)


def test_decode_recovers_angles_across_the_seam():
    """The whole reason the GP works in embedded space: an angle near +/-pi has
    no discontinuity in (sin, cos), so decode must return it correctly."""
    system = CartPoleSystem()
    dec = EmbeddedStateDecoder(system)
    for theta in (math.pi - 1e-3, -math.pi + 1e-3, 3.0, -3.0, 0.0):
        raw = torch.tensor([[0.0, theta, 0.0, 0.0]])
        out = dec.decode(system.embed_state_for_model(system.normalize_state(raw)))
        assert out[0, 1].item() == pytest.approx(theta, abs=1e-4)


def test_decode_normalizes_and_canonicalizes_quaternions():
    """A GP sample is an arbitrary 4-vector, not a unit quaternion. Quadrotor3D's
    classify_attractor compares raw 13-vectors by L2 against an identity-quaternion
    goal, so an un-canonicalized -q sits 2.0 away and is misclassified."""
    system = Quadrotor3DSystem()
    dec = EmbeddedStateDecoder(system)
    embedded = torch.randn(32, dec.embed_dim)
    embedded[:16, 3] = -abs(embedded[:16, 3])  # force qw < 0 on half the rows
    out = dec.decode(embedded)
    q = out[:, 3:7]
    torch.testing.assert_close(torch.linalg.norm(q, dim=-1), torch.ones(32), atol=1e-5, rtol=0)
    assert (q[:, 0] >= 0).all()


def test_decode_output_is_consumable_by_classify_attractor():
    for cls, state_dim, _e in SYSTEMS:
        system = cls()
        dec = EmbeddedStateDecoder(system)
        out = dec.decode(torch.randn(16, dec.embed_dim))
        assert out.shape == (16, state_dim)
        assert system.classify_attractor(out, radius=0.3).shape == (16,)


def test_unknown_component_type_is_rejected():
    from adaptive_roa.systems.base import ManifoldComponent

    class StubSystem:
        manifold_components = [ManifoldComponent("Hyperbolic", 2, "weird")]

    with pytest.raises(ValueError, match="Hyperbolic"):
        EmbeddedStateDecoder(StubSystem())

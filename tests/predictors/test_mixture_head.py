"""Mixture-density endpoint head (tracker C19)."""
import math

import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.predictors.heads import FinalStateHead, MixtureFinalStateHead, build_head
from adaptive_roa.systems.cartpole import CartPoleSystem
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.systems.quadrotor2d import Quadrotor2DSystem
from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem

# One system per manifold mix: SO2 only via pendulum/cartpole, SO2 + Real on
# quad2d, SO3 + Real on quad3d.
SYSTEMS = [PendulumSystem, CartPoleSystem, Quadrotor2DSystem, Quadrotor3DSystem]


def _targets(system, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return system.denormalize_state(torch.rand(n, system.state_dim, generator=g) * 2 - 1)


@pytest.mark.parametrize("cls", SYSTEMS)
def test_one_component_mixture_equals_the_plain_gaussian_nll(cls):
    """With M = 1 the mixture weight is 1, so the NLL must equal the single
    head's at beta = 0 on every manifold."""
    system = cls()
    mix = MixtureFinalStateHead(system, n_components=1)
    single = FinalStateHead(system, beta=0.0)
    comp = torch.randn(16, single.n_params)
    params = torch.cat([torch.randn(16, 1), comp], dim=1)
    y = _targets(system, 16)
    torch.testing.assert_close(mix.nll(params, y), single.nll(comp, y))


@pytest.mark.parametrize("cls", SYSTEMS)
def test_nll_is_the_logsumexp_of_weighted_components(cls):
    system = cls()
    mix = MixtureFinalStateHead(system, n_components=3)
    single = FinalStateHead(system, beta=0.0)
    params = torch.randn(10, mix.n_params)
    y = _targets(system, 10, seed=1)
    logits, comp = params[:, :3], params[:, 3:].reshape(10, 3, single.n_params)
    manual = -torch.logsumexp(
        torch.log_softmax(logits, -1)
        + torch.stack([-single.nll(comp[:, m], y) for m in range(3)], dim=1), dim=1)
    torch.testing.assert_close(mix.nll(params, y), manual)


@pytest.mark.parametrize("cls", SYSTEMS)
def test_nll_backpropagates_finite_gradients(cls):
    system = cls()
    mix = MixtureFinalStateHead(system, n_components=4)
    params = torch.randn(12, mix.n_params, requires_grad=True)
    mix.nll(params, _targets(system, 12)).mean().backward()
    assert torch.isfinite(params.grad).all()


@pytest.mark.parametrize("cls", SYSTEMS)
def test_samples_are_valid_states(cls):
    """Samples feed classify_attractor directly: wrapped angles, unit
    quaternions with qw >= 0, the system's state width."""
    system = cls()
    mix = MixtureFinalStateHead(system, n_components=5)
    params = torch.randn(64, mix.n_params)
    x = mix.sample(params, generator=torch.Generator().manual_seed(0))
    assert x.shape == (64, system.state_dim)
    assert torch.isfinite(x).all()
    off = 0
    for comp in system.manifold_components:
        part = x[:, off:off + comp.dim]
        if comp.manifold_type == "SO2":
            assert (part.abs() <= math.pi + 1e-6).all()
        if comp.manifold_type == "SO3":
            torch.testing.assert_close(part.norm(dim=1), torch.ones(64), atol=1e-5, rtol=0)
            assert (part[:, 0] >= 0).all()
        off += comp.dim


def test_sampling_follows_the_mixture_weights():
    """Two far-apart components at weights 0.25 / 0.75: the share of draws from
    each matches. Real dims only (cart position) decide which one a draw came
    from."""
    system = CartPoleSystem()
    mix = MixtureFinalStateHead(system, n_components=2)
    single = FinalStateHead(system, beta=0.0)
    comp = torch.zeros(2, single.n_params)
    # cart position mu: component 0 at -0.8, component 1 at +0.8 (normalized),
    # tiny sigma on every dim.
    for c, mu in [(0, -0.8), (1, 0.8)]:
        comp[c, 0] = mu
        comp[c, 1] = -6.0           # log sigma, cart position
        comp[c, 2:4] = torch.tensor([0.0, 1.0])
        comp[c, 4] = -6.0           # log sigma, pole angle
        comp[c, 5:9:2] = 0.0
        comp[c, 6:9:2] = -6.0
    logits = torch.log(torch.tensor([0.25, 0.75]))
    params = torch.cat([logits, comp.flatten()]).expand(4000, -1).contiguous()
    x = mix.sample(params, generator=torch.Generator().manual_seed(1))
    share_right = float((x[:, 0] > 0).float().mean())
    assert abs(share_right - 0.75) < 0.03


def test_mean_is_the_heaviest_components_mean():
    system = CartPoleSystem()
    mix = MixtureFinalStateHead(system, n_components=3)
    params = torch.randn(5, mix.n_params)
    params[:, :3] = torch.tensor([0.0, 5.0, 0.0])
    torch.testing.assert_close(mix.mean(params), mix.component_means(params)[:, 1])


def test_responsibilities_sum_to_one():
    system = Quadrotor3DSystem()
    mix = MixtureFinalStateHead(system, n_components=4)
    params = torch.randn(9, mix.n_params)
    r = mix.responsibilities(params, _targets(system, 9))
    torch.testing.assert_close(r.sum(dim=1), torch.ones(9))


@pytest.mark.parametrize("cls", SYSTEMS)
def test_param_layout_partitions_every_output_row(cls):
    mix = MixtureFinalStateHead(cls(), n_components=3)
    lay = mix.param_layout()
    rows = lay["logits"] + sum(lay["locations"], []) + sum(lay["scales"], [])
    assert sorted(rows) == list(range(mix.n_params))


def test_laplace_chunk_stats_shapes_and_ranges():
    system = Quadrotor2DSystem()
    mix = MixtureFinalStateHead(system, n_components=3)
    params = torch.randn(20, mix.n_params)
    st = mix.laplace_chunk_stats(params, _targets(system, 20))
    assert st["logit_lam"].shape == (20,) and (st["logit_lam"] >= 0).all()
    assert (st["logit_lam"] <= 0.25 + 1e-9).all()
    assert st["resp"].shape == (20, 3) and st["sq_resid"].shape == (20, 3)
    assert (st["sq_resid"] >= 0).all()


def test_build_head_defaults_to_the_single_gaussian():
    system = CartPoleSystem()
    head = build_head(system, OmegaConf.create({"beta_nll": 0.3}))
    assert type(head) is FinalStateHead and head.beta == 0.3
    head = build_head(system, None)
    assert type(head) is FinalStateHead and head.beta == 0.5


def test_build_head_builds_the_mixture_and_refuses_beta_nll():
    system = CartPoleSystem()
    head = build_head(system, OmegaConf.create(
        {"beta_nll": 0.0, "head": {"kind": "mixture", "n_components": 4}}))
    assert isinstance(head, MixtureFinalStateHead) and head.n_components == 4
    with pytest.raises(ValueError, match="beta-NLL"):
        build_head(system, OmegaConf.create({"head": {"kind": "mixture"}}))
    with pytest.raises(ValueError, match="unknown head kind"):
        build_head(system, OmegaConf.create({"head": {"kind": "flow"}}))

import pytest
import torch

from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.hmc.posterior import HMCPosterior


def _posterior(n_draws=32, out=1):
    net = build_bayesian_mlp(input_dim=4, hidden_dims=[8, 8], output_dim=out,
                             posterior="deterministic", activation="tanh")
    dim = sum(p.numel() for p in net.parameters())
    torch.manual_seed(0)
    return HMCPosterior(net, torch.randn(n_draws, dim) * 0.3)


def test_reports_its_support_size():
    assert _posterior(n_draws=40).n_draws == 40


def test_rejects_an_empty_sample_set():
    net = build_bayesian_mlp(input_dim=4, hidden_dims=[8], output_dim=1,
                             posterior="deterministic", activation="tanh")
    dim = sum(p.numel() for p in net.parameters())
    with pytest.raises(ValueError, match="at least"):
        HMCPosterior(net, torch.zeros(0, dim))


def test_rejects_a_1d_sample_tensor():
    net = build_bayesian_mlp(input_dim=4, hidden_dims=[8], output_dim=1,
                             posterior="deterministic", activation="tanh")
    dim = sum(p.numel() for p in net.parameters())
    with pytest.raises(ValueError, match="at least"):
        HMCPosterior(net, torch.zeros(dim))


def test_rejects_a_3d_sample_tensor():
    net = build_bayesian_mlp(input_dim=4, hidden_dims=[8], output_dim=1,
                             posterior="deterministic", activation="tanh")
    dim = sum(p.numel() for p in net.parameters())
    with pytest.raises(ValueError, match="at least"):
        HMCPosterior(net, torch.zeros(4, 1, dim))


def test_forward_sample_draws_one_atom_and_varies():
    post = _posterior()
    x = torch.randn(6, 4)
    outs = torch.stack([post.forward_sample(x) for _ in range(40)])
    assert outs.shape == (40, 6, 1)
    assert outs.std(dim=0).mean().item() > 0.0


def test_forward_sample_is_reproducible_under_a_seeded_generator():
    post = _posterior()
    x = torch.randn(6, 4)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    torch.testing.assert_close(post.forward_sample(x, generator=g1),
                               post.forward_sample(x, generator=g2))


def test_forward_sample_is_reproducible_under_a_same_device_generator():
    """forward_sample must draw its index on the SAME device as the
    posterior's own parameters (mirroring EnsemblePosterior and the generator
    OutcomeModelHandle constructs via torch.Generator(device=param_device)),
    not an unconditional CPU draw. This CPU test only shows the CPU/CPU
    pairing is accepted and reproducible -- on this machine `param_device`
    IS `cpu`, so it cannot distinguish the fix from a hardcoded `device="cpu"`
    bug. The cross-device case (a CUDA generator against a hardcoded-CPU
    draw, which raises RuntimeError) cannot be exercised without CUDA; see
    test_forward_sample_accepts_a_device_matched_generator_on_cuda below."""
    post = _posterior()
    x = torch.randn(6, 4)
    param_device = next(post.parameters()).device
    g1 = torch.Generator(device=param_device).manual_seed(11)
    g2 = torch.Generator(device=param_device).manual_seed(11)
    torch.testing.assert_close(post.forward_sample(x, generator=g1),
                               post.forward_sample(x, generator=g2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forward_sample_accepts_a_device_matched_generator_on_cuda():
    """The bug this guards: OutcomeModelHandle builds its generator with
    torch.Generator(device=param_device) (handles.py:62), so on CUDA the index
    draw receives a CUDA generator. A CPU-pinned randint raises there, and no
    CPU-only test can see it -- which is how the sibling arms shipped the mirror
    image of this bug once already.
    """
    post = _posterior().to("cuda:0")
    gen = torch.Generator(device="cuda:0").manual_seed(0)
    out = post.forward_sample(torch.randn(6, 4, device="cuda:0"), generator=gen)
    assert torch.isfinite(out).all()


def test_predictive_enumerates_the_support_exactly_and_ignores_S():
    """Finite support, same as EnsemblePosterior: sampling it with replacement
    would make the marginal a function of the caller's S. Enumeration is exact."""
    post = _posterior(n_draws=32)
    x = torch.randn(5, 4)
    a = post.predictive_logit_samples(x, S=8)
    b = post.predictive_logit_samples(x, S=1000)
    assert a.shape == (32, 5, 1)
    torch.testing.assert_close(a, b)


def test_each_enumerated_atom_uses_a_distinct_weight_vector():
    post = _posterior(n_draws=16)
    out = post.predictive_logit_samples(torch.randn(3, 4), S=16)
    flat = out.reshape(16, -1)
    assert len({tuple(r.tolist()) for r in flat}) == 16


def test_kl_divergence_is_zero():
    assert _posterior().kl_divergence().item() == 0.0


def test_works_for_a_multi_output_head():
    post = _posterior(n_draws=12, out=9)
    assert post.predictive_logit_samples(torch.randn(4, 4), S=12).shape == (12, 4, 9)

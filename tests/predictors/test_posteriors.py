import math

import pytest
import torch

from adaptive_roa.predictors.posteriors import (
    DeterministicPosterior,
    MFVIPosterior,
    VILinear,
)


def test_vilinear_kl_is_finite_and_positive():
    layer = VILinear(4, 3, prior_sigma=1.0)
    kl = layer.kl_divergence()
    assert kl.ndim == 0
    assert torch.isfinite(kl)
    assert kl.item() > 0.0


def test_vilinear_kl_is_zero_when_q_equals_prior():
    """KL(q||p) == 0 exactly when q has the prior's mean and scale."""
    layer = VILinear(4, 3, prior_sigma=1.0)
    with torch.no_grad():
        layer.weight_mu.zero_()
        layer.bias_mu.zero_()
        # softplus(rho) == 1.0  =>  rho == log(e - 1)
        rho = torch.log(torch.expm1(torch.tensor(1.0)))
        layer.weight_rho.fill_(rho)
        layer.bias_rho.fill_(rho)
    assert layer.kl_divergence().abs().item() < 1e-5


def test_vilinear_forward_is_stochastic():
    layer = VILinear(4, 3)
    x = torch.randn(8, 4)
    assert not torch.allclose(layer(x), layer(x))


def test_vilinear_forward_is_reproducible_under_a_seeded_generator():
    layer = VILinear(4, 3)
    x = torch.randn(8, 4)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    assert torch.allclose(layer(x, generator=g1), layer(x, generator=g2))


def test_deterministic_posterior_has_zero_kl_and_no_spread():
    net = torch.nn.Linear(4, 2)
    post = DeterministicPosterior(net)
    x = torch.randn(8, 4)
    assert post.kl_divergence().item() == 0.0
    assert torch.allclose(post.forward_sample(x), post.forward_sample(x))


def test_mfvi_posterior_spreads_and_accumulates_kl():
    net = torch.nn.Sequential(VILinear(4, 6), torch.nn.ReLU(), VILinear(6, 2))
    post = MFVIPosterior(net)
    x = torch.randn(8, 4)
    assert not torch.allclose(post.forward_sample(x), post.forward_sample(x))
    assert post.kl_divergence().item() > 0.0
    samples = post.forward_samples(x, S=5)
    assert samples.shape == (5, 8, 2)
    assert samples.std(dim=0).mean().item() > 0.0


def test_vilinear_kl_matches_closed_form_with_nonzero_mean():
    """Pins the mu**2 term. A KL that drops it still passes the q==prior test,
    because that test zeroes mu."""
    layer = VILinear(4, 3, prior_sigma=1.0)
    with torch.no_grad():
        layer.weight_mu.fill_(0.5)
        layer.bias_mu.fill_(0.5)
        # softplus(rho) == 2.0  =>  rho == log(e**2 - 1)
        rho = torch.log(torch.expm1(torch.tensor(2.0)))
        layer.weight_rho.fill_(rho)
        layer.bias_rho.fill_(rho)

    # per element: log(prior_sigma/sigma) + (sigma**2 + mu**2)/(2*prior_sigma**2) - 0.5
    per_element = math.log(1.0 / 2.0) + (4.0 + 0.25) / 2.0 - 0.5
    n_elements = 4 * 3 + 3  # weights + biases
    expected = per_element * n_elements

    assert layer.kl_divergence().item() == pytest.approx(expected, rel=1e-5)


def test_ensemble_posterior_draws_a_member_per_call():
    from adaptive_roa.predictors.posteriors import EnsemblePosterior

    torch.manual_seed(0)
    nets = [torch.nn.Linear(4, 2) for _ in range(5)]
    # Force members apart so "which member" is observable in the output.
    with torch.no_grad():
        for i, net in enumerate(nets):
            net.bias.fill_(float(i))
    post = EnsemblePosterior(nets)
    x = torch.zeros(1, 4)

    assert post.n_members == 5
    assert post.kl_divergence().item() == 0.0
    draws = {round(post.forward_sample(x)[0, 0].item()) for _ in range(200)}
    assert len(draws) == 5, f"expected all 5 members to be drawn, saw {draws}"


def test_ensemble_forward_samples_has_spread():
    from adaptive_roa.predictors.posteriors import EnsemblePosterior

    torch.manual_seed(0)
    nets = [torch.nn.Linear(4, 2) for _ in range(5)]
    with torch.no_grad():
        for i, net in enumerate(nets):
            net.bias.fill_(float(i))
    post = EnsemblePosterior(nets)
    samples = post.forward_samples(torch.zeros(3, 4), S=40)
    assert samples.shape == (40, 3, 2)
    assert samples.std(dim=0).mean().item() > 0.0


def test_laplace_covariance_is_psd_and_shrinks_with_data():
    from adaptive_roa.predictors.posteriors import LastLayerLaplacePosterior

    torch.manual_seed(0)
    body = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU())
    head = torch.nn.Linear(8, 1)
    post = LastLayerLaplacePosterior(body, head, prior_precision=1.0)

    x_small, x_large = torch.randn(16, 4), torch.randn(512, 4)
    post.fit(body(x_small), torch.randint(0, 2, (16,)).float(), task="outcome")
    cov_small = post.posterior_covariance.clone()
    post.fit(body(x_large), torch.randint(0, 2, (512,)).float(), task="outcome")
    cov_large = post.posterior_covariance.clone()

    # Symmetric positive definite.
    assert torch.allclose(cov_small, cov_small.T, atol=1e-6)
    assert torch.linalg.eigvalsh(cov_small).min().item() > 0.0
    # More data => tighter posterior.
    assert cov_large.diagonal().mean().item() < cov_small.diagonal().mean().item()


def test_laplace_forward_is_stochastic_only_after_fit():
    from adaptive_roa.predictors.posteriors import LastLayerLaplacePosterior

    torch.manual_seed(0)
    body = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU())
    head = torch.nn.Linear(8, 1)
    post = LastLayerLaplacePosterior(body, head, prior_precision=1.0)
    x = torch.randn(8, 4)

    # Before fit: falls back to the MAP point estimate.
    assert torch.allclose(post.forward_sample(x), post.forward_sample(x))
    post.fit(body(x), torch.randint(0, 2, (8,)).float(), task="outcome")
    assert not torch.allclose(post.forward_sample(x), post.forward_sample(x))


def test_laplace_regression_requires_an_explicit_sigma():
    """sigma is the observation noise; silently defaulting it to 1.0 is the
    documented laplace-torch trap and would scale the whole covariance."""
    from adaptive_roa.predictors.posteriors import LastLayerLaplacePosterior

    body = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU())
    head = torch.nn.Linear(8, 3)
    post = LastLayerLaplacePosterior(body, head, prior_precision=1.0)
    with pytest.raises(ValueError, match="sigma"):
        post.fit(body(torch.randn(8, 4)), torch.randn(8, 3), task="final_state")

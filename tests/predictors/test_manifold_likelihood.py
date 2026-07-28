import math

import pytest
import torch

from adaptive_roa.predictors.manifold_likelihood import RealLikelihood


def test_real_needs_two_params_per_dim():
    assert RealLikelihood().n_params(3) == 6


def test_real_nll_matches_the_closed_form_gaussian():
    """Closed form: 0.5*log(2*pi*sigma^2) + (y-mu)^2/(2*sigma^2), summed over dims."""
    lik = RealLikelihood()
    mu = torch.tensor([[1.0, -2.0]])
    log_sigma = torch.tensor([[0.0, math.log(2.0)]])  # sigma = 1.0, 2.0
    params = torch.cat([mu, log_sigma], dim=-1)
    y = torch.tensor([[1.5, -1.0]])

    expected = 0.0
    for m, s, t in ((1.0, 1.0, 1.5), (-2.0, 2.0, -1.0)):
        expected += 0.5 * math.log(2 * math.pi * s ** 2) + (t - m) ** 2 / (2 * s ** 2)

    assert lik.nll(params, y, beta=0.0).item() == pytest.approx(expected, rel=1e-5)


def test_beta_nll_reweights_by_sigma_but_beta_zero_is_plain_nll():
    """beta-NLL multiplies each dim's loss by sigma^(2*beta), detached (Seitzer 2022).
    At beta=0 the weight is 1 and it must reduce exactly to plain Gaussian NLL."""
    lik = RealLikelihood()
    params = torch.cat([torch.zeros(1, 2), torch.tensor([[0.0, math.log(3.0)]])], dim=-1)
    y = torch.ones(1, 2)

    plain = lik.nll(params, y, beta=0.0)
    weighted = lik.nll(params, y, beta=0.5)
    assert not torch.allclose(plain, weighted)

    # Reconstruct the beta=0.5 value from per-dim plain terms times sigma^(2*beta).
    per_dim = []
    for s, t in ((1.0, 1.0), (3.0, 1.0)):
        per_dim.append(0.5 * math.log(2 * math.pi * s ** 2) + t ** 2 / (2 * s ** 2))
    expected = per_dim[0] * (1.0 ** 1.0) + per_dim[1] * (3.0 ** 1.0)
    assert weighted.item() == pytest.approx(expected, rel=1e-5)


def test_real_sample_is_stochastic_and_reproducible_under_a_generator():
    lik = RealLikelihood()
    params = torch.cat([torch.zeros(4, 2), torch.zeros(4, 2)], dim=-1)
    assert not torch.allclose(lik.sample(params), lik.sample(params))
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    assert torch.allclose(lik.sample(params, generator=g1), lik.sample(params, generator=g2))


def test_real_sample_recovers_the_parameters_in_expectation():
    lik = RealLikelihood()
    mu = torch.tensor([[2.0, -1.0]])
    params = torch.cat([mu, torch.log(torch.tensor([[0.5, 1.5]]))], dim=-1)
    draws = torch.stack([lik.sample(params.expand(2000, -1)) for _ in range(1)], dim=0)[0]
    assert torch.allclose(draws.mean(0), mu[0], atol=0.1)
    assert torch.allclose(draws.std(0), torch.tensor([0.5, 1.5]), rtol=0.15)


def test_real_distance_is_absolute_difference_per_dim():
    lik = RealLikelihood()
    a = torch.tensor([[1.0, 5.0]])
    b = torch.tensor([[3.0, 1.0]])
    assert lik.n_dist(2) == 2
    assert torch.allclose(lik.distance(a, b), torch.tensor([[2.0, 4.0]]))


def test_real_names_are_one_per_dim():
    assert RealLikelihood().names(3, "velocity") == ["velocity_0", "velocity_1", "velocity_2"]
    assert RealLikelihood().names(1, "cart_position") == ["cart_position"]

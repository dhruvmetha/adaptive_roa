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


def test_so2_params_are_sin_cos_and_log_sigma():
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood
    assert SO2Likelihood().n_params(1) == 3


def test_so2_mean_recovers_the_angle_via_atan2():
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood

    lik = SO2Likelihood()
    for theta in (0.0, 1.0, -1.0, 3.0, -3.0, math.pi - 1e-3):
        params = torch.tensor([[math.sin(theta), math.cos(theta), 0.0]])
        assert lik.mean(params).item() == pytest.approx(theta, abs=1e-5)


def test_so2_mean_is_unaffected_by_the_magnitude_of_sin_cos():
    """Only the direction matters; the head is not required to emit a unit vector."""
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood

    lik = SO2Likelihood()
    small = torch.tensor([[0.1 * math.sin(2.0), 0.1 * math.cos(2.0), 0.0]])
    large = torch.tensor([[9.0 * math.sin(2.0), 9.0 * math.cos(2.0), 0.0]])
    assert lik.mean(small).item() == pytest.approx(2.0, abs=1e-5)
    assert lik.mean(large).item() == pytest.approx(2.0, abs=1e-5)


def test_so2_samples_stay_wrapped_and_straddle_the_seam():
    """A mean near +pi with real spread must produce samples on BOTH sides of the
    seam, all within [-pi, pi]. This is the property a Euclidean Gaussian lacks."""
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood

    lik = SO2Likelihood()
    theta = math.pi - 0.05
    params = torch.tensor([[math.sin(theta), math.cos(theta), math.log(0.5)]]).expand(4000, -1)
    draws = lik.sample(params)
    assert draws.min() >= -math.pi - 1e-5 and draws.max() <= math.pi + 1e-5
    assert (draws > 0).any() and (draws < 0).any(), "samples never crossed the seam"


def test_so2_distance_is_the_short_way_round():
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood

    lik = SO2Likelihood()
    a = torch.tensor([[math.pi - 0.1]])
    b = torch.tensor([[-math.pi + 0.1]])
    assert lik.n_dist(1) == 1
    assert lik.distance(a, b).item() == pytest.approx(0.2, abs=1e-5)


def test_so2_nll_is_lower_for_a_target_at_the_predicted_angle():
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood

    lik = SO2Likelihood()
    theta = math.pi - 0.05
    params = torch.tensor([[math.sin(theta), math.cos(theta), math.log(0.3)]])
    on = lik.nll(params, torch.tensor([[theta]]))
    # Just across the seam, geodesically 0.1 away.
    near = lik.nll(params, torch.tensor([[-math.pi + 0.05]]))
    far = lik.nll(params, torch.tensor([[0.0]]))
    assert on.item() < near.item() < far.item()


def test_so2_names_mark_the_geodesic():
    from adaptive_roa.predictors.manifold_likelihood import SO2Likelihood
    assert SO2Likelihood().names(1, "pole_angle") == ["pole_angle_geodesic"]


def test_so3_params_are_a_quaternion_plus_three_log_sigmas():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood
    assert SO3Likelihood().n_params(4) == 7


def test_so3_mean_is_unit_norm_and_canonical():
    """qw >= 0 is REQUIRED: Quadrotor3DSystem.classify_attractor uses plain L2
    against an identity-quaternion goal, so -q (the same rotation) sits at
    distance 2 and every near-goal endpoint would be misclassified."""
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    params = torch.tensor([[-0.9, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0]])  # qw < 0, unnormalized
    q = lik.mean(params)
    assert torch.linalg.norm(q, dim=-1).item() == pytest.approx(1.0, abs=1e-6)
    assert q[0, 0].item() >= 0.0


def test_so3_samples_are_unit_norm_and_canonical():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    params = torch.tensor([[1.0, 0.0, 0.0, 0.0, math.log(0.3), math.log(0.3), math.log(0.3)]])
    draws = lik.sample(params.expand(500, -1))
    assert torch.allclose(torch.linalg.norm(draws, dim=-1), torch.ones(500), atol=1e-5)
    assert (draws[:, 0] >= 0).all(), "samples must be canonicalized to qw >= 0"


def test_so3_sample_concentrates_as_sigma_shrinks():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    spreads = []
    for log_sigma in (math.log(0.5), math.log(0.01)):
        params = torch.cat([q, torch.full((1, 3), log_sigma)], dim=-1).expand(800, -1)
        spreads.append(lik.distance(lik.sample(params), q.expand(800, -1)).mean().item())
    assert spreads[1] < spreads[0]


def test_so3_distance_is_the_rotation_angle_and_ignores_double_cover():
    """q and -q are the same rotation, so their geodesic distance must be 0."""
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert lik.n_dist(4) == 1
    assert lik.distance(q, q).item() == pytest.approx(0.0, abs=1e-6)
    assert lik.distance(q, -q).item() == pytest.approx(0.0, abs=1e-5)
    # 180 degrees about x.
    half_turn = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    assert lik.distance(q, half_turn).item() == pytest.approx(math.pi, abs=1e-4)


def test_so3_nll_is_lowest_at_the_predicted_rotation():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    params = torch.tensor([[1.0, 0.0, 0.0, 0.0, math.log(0.3), math.log(0.3), math.log(0.3)]])
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    tilted = torch.tensor([[math.cos(0.4), math.sin(0.4), 0.0, 0.0]])
    assert lik.nll(params, q).item() < lik.nll(params, tilted).item()


def test_so3_nll_is_invariant_to_target_sign():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood

    lik = SO3Likelihood()
    params = torch.tensor([[1.0, 0.0, 0.0, 0.0, math.log(0.3), math.log(0.3), math.log(0.3)]])
    t = torch.tensor([[math.cos(0.4), math.sin(0.4), 0.0, 0.0]])
    assert lik.nll(params, t).item() == pytest.approx(lik.nll(params, -t).item(), rel=1e-5)


def test_so3_names_mark_the_geodesic():
    from adaptive_roa.predictors.manifold_likelihood import SO3Likelihood
    assert SO3Likelihood().names(4, "orientation") == ["orientation_geodesic"]

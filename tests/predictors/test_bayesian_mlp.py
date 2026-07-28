import pytest
import torch

from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.posteriors import (
    DeterministicPosterior,
    EnsemblePosterior,
    LastLayerLaplacePosterior,
    MFVIPosterior,
)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("deterministic", DeterministicPosterior),
        ("mfvi", MFVIPosterior),
        ("ensemble", EnsemblePosterior),
        ("laplace", LastLayerLaplacePosterior),
    ],
)
def test_build_returns_the_right_posterior_and_output_shape(kind, expected):
    model = build_bayesian_mlp(
        input_dim=3, hidden_dims=[16, 16], output_dim=1, posterior=kind, n_members=3
    )
    assert isinstance(model, expected)
    assert model.forward_sample(torch.randn(8, 3)).shape == (8, 1)


def test_unknown_posterior_is_rejected():
    with pytest.raises(ValueError, match="unknown posterior"):
        build_bayesian_mlp(input_dim=3, hidden_dims=[8], output_dim=1, posterior="mcdropout")


def test_mfvi_net_is_built_from_vilinear_layers():
    from adaptive_roa.predictors.posteriors import VILinear

    model = build_bayesian_mlp(input_dim=3, hidden_dims=[8, 8], output_dim=1, posterior="mfvi")
    assert sum(isinstance(m, VILinear) for m in model.modules()) == 3  # 2 hidden + 1 output
    assert model.kl_divergence().item() > 0.0


def test_ensemble_members_are_independently_initialized():
    model = build_bayesian_mlp(
        input_dim=3, hidden_dims=[8], output_dim=1, posterior="ensemble", n_members=4
    )
    assert model.n_members == 4
    first = model.members[0].net[0].weight
    assert not any(torch.allclose(first, m.net[0].weight) for m in list(model.members)[1:])


def test_laplace_exposes_a_body_for_feature_extraction():
    model = build_bayesian_mlp(input_dim=3, hidden_dims=[8, 8], output_dim=1, posterior="laplace")
    assert model.body(torch.randn(5, 3)).shape == (5, 8)


def test_mfvi_net_is_flat_so_generator_reaches_every_layer():
    """MFVIPosterior.forward_sample passes `generator` only to TOP-LEVEL VILinear
    children. A nested Sequential still contains 3 VILinear layers under
    .modules(), so the count test cannot catch it -- but seeded reproducibility
    breaks silently. Assert flatness directly AND end-to-end reproducibility."""
    from adaptive_roa.predictors.posteriors import VILinear

    model = build_bayesian_mlp(input_dim=3, hidden_dims=[8, 8], output_dim=1, posterior="mfvi")

    # Every top-level child is a leaf: no nested containers.
    assert all(
        not isinstance(child, torch.nn.Sequential) for child in model.net
    ), "MFVI net must be flat; a nested Sequential drops `generator` for inner layers"
    assert sum(isinstance(child, VILinear) for child in model.net) == 3

    x = torch.randn(4, 3)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    g3 = torch.Generator().manual_seed(7)
    assert torch.allclose(model.forward_sample(x, generator=g1),
                          model.forward_sample(x, generator=g2))
    assert not torch.allclose(model.forward_sample(x, generator=g1),
                              model.forward_sample(x, generator=g3))


@pytest.mark.parametrize("prior_sigma,expected_precision", [(2.0, 0.25), (0.5, 4.0)])
def test_laplace_converts_prior_sigma_to_precision(prior_sigma, expected_precision):
    """precision = 1/sigma**2. Values other than 1.0 are required: at sigma=1.0
    a passthrough bug is indistinguishable from the correct conversion."""
    model = build_bayesian_mlp(
        input_dim=3, hidden_dims=[8], output_dim=1,
        posterior="laplace", prior_sigma=prior_sigma,
    )
    assert model.prior_precision == pytest.approx(expected_precision)

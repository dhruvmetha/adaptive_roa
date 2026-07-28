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

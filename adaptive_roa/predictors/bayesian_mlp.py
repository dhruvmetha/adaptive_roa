"""Backbone construction for Bayesian MLP arms.

The MLP is split into a ``body`` (everything up to the last hidden activation)
and a single output ``Linear``. Two things need that split: last-layer Laplace
fits its GGN over the body's penultimate features, and the final-state head
attaches a different output layer to the same body.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from adaptive_roa.predictors.posteriors import (
    DeterministicPosterior,
    EnsemblePosterior,
    LastLayerLaplacePosterior,
    MFVIPosterior,
    Posterior,
    VILinear,
)

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class _Body(nn.Module):
    """Hidden stack, exposed as a module so Laplace can call it directly."""

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _hidden_layers(input_dim, hidden_dims, activation, dropout, linear_cls, **linear_kwargs):
    act_cls = _ACTIVATIONS.get(activation)
    if act_cls is None:
        raise ValueError(f"unknown activation {activation!r}; expected one of {sorted(_ACTIVATIONS)}")
    dims = [int(input_dim)] + [int(h) for h in hidden_dims]
    layers: List[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.append(linear_cls(a, b, **linear_kwargs))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return layers, dims[-1]


def build_bayesian_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    posterior: str,
    *,
    prior_sigma: float = 1.0,
    n_members: int = 5,
    dropout: float = 0.0,
    activation: str = "relu",
) -> Posterior:
    """Build an MLP wrapped in the requested approximate weight posterior."""
    kind = str(posterior)

    if kind == "mfvi":
        layers, last_hidden = _hidden_layers(
            input_dim, hidden_dims, activation, dropout, VILinear, prior_sigma=prior_sigma
        )
        layers.append(VILinear(last_hidden, int(output_dim), prior_sigma=prior_sigma))
        return MFVIPosterior(nn.Sequential(*layers))

    if kind == "deterministic":
        layers, last_hidden = _hidden_layers(
            input_dim, hidden_dims, activation, dropout, nn.Linear
        )
        layers.append(nn.Linear(last_hidden, int(output_dim)))
        return DeterministicPosterior(nn.Sequential(*layers))

    if kind == "ensemble":
        members = [
            build_bayesian_mlp(
                input_dim, hidden_dims, output_dim, "deterministic",
                dropout=dropout, activation=activation,
            )
            for _ in range(int(n_members))
        ]
        return EnsemblePosterior(members)

    if kind == "laplace":
        layers, last_hidden = _hidden_layers(
            input_dim, hidden_dims, activation, dropout, nn.Linear
        )
        return LastLayerLaplacePosterior(
            body=_Body(layers),
            head_layer=nn.Linear(last_hidden, int(output_dim)),
            prior_precision=1.0 / float(prior_sigma) ** 2,
        )

    raise ValueError(
        f"unknown posterior {posterior!r}; expected one of "
        "'deterministic', 'mfvi', 'ensemble', 'laplace'"
    )

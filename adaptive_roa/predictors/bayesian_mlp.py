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
    prior_precision=None,
) -> Posterior:
    """Build an MLP wrapped in the requested approximate weight posterior.

    ``prior_precision`` applies to Laplace only. ``None`` (the default, and
    every existing arm) keeps ``1 / prior_sigma^2``. A number fixes it.
    ``"auto"`` selects it per fit by maximizing the Laplace evidence.
    """
    kind = str(posterior)
    if prior_precision is not None and kind != "laplace":
        # Only Laplace reads it; on any other posterior it would be inert and the
        # arm would be described with a prior it never used.
        raise ValueError(
            f"prior_precision={prior_precision!r} is set on a {kind!r} posterior; "
            "only 'laplace' uses it (mfvi takes prior_sigma)")

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
        tune = isinstance(prior_precision, str) and prior_precision == "auto"
        if prior_precision is None or tune:
            delta = 1.0 / float(prior_sigma) ** 2
        elif isinstance(prior_precision, str):
            raise ValueError(
                f"prior_precision must be a number or 'auto', got {prior_precision!r}")
        else:
            delta = float(prior_precision)
        return LastLayerLaplacePosterior(
            body=_Body(layers),
            head_layer=nn.Linear(last_hidden, int(output_dim)),
            prior_precision=delta,
            tune_prior_precision=tune,
        )

    raise ValueError(
        f"unknown posterior {posterior!r}; expected one of "
        "'deterministic', 'mfvi', 'ensemble', 'laplace'"
    )


def embedded_dim(system) -> int:
    """Width the system's embedding presents to the network."""
    dummy = torch.zeros(1, int(system.state_dim))
    return int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])


def build_from_cfg(bnn_cfg, system, posterior_kind: str, output_dim: int = 1) -> Posterior:
    """Build the arm's network straight from a ``predictor.bnn`` config block.

    SINGLE source of the architecture defaults. The trainer and the export
    wrapper (``probabilistic_classifier/bayesian.py``) must agree exactly: they
    build the same net from the same config at different times, and a
    disagreement in even one default yields a checkpoint/skeleton mismatch --
    which, before the strict key check in the export loader, was silent. Keeping
    both callers on this one function makes drift impossible rather than merely
    unlikely.
    """
    cfg = bnn_cfg if bnn_cfg is not None else {}
    return build_bayesian_mlp(
        input_dim=embedded_dim(system),
        hidden_dims=list(cfg.get("hidden_dims", [256, 512, 256])),
        output_dim=int(output_dim),
        posterior=str(posterior_kind),
        prior_sigma=float(cfg.get("prior_sigma", 1.0)),
        n_members=int(cfg.get("n_members", 5)),
        dropout=float(cfg.get("dropout", 0.0)),
        activation=str(cfg.get("activation", "relu")),
        prior_precision=cfg.get("prior_precision", None),
    )


def outcome_handle_from_cfg(posterior: Posterior, system, bnn_cfg):
    """Wrap a posterior in an ``OutcomeModelHandle`` using the config's settings.

    Same rationale as ``build_from_cfg``: ``n_marginal_samples`` and ``seed``
    define the marginalization, so a default that drifts between the trainer and
    the export wrapper would make the exported probabilities differ from the
    ones the run itself recorded.
    """
    from adaptive_roa.predictors.handles import OutcomeModelHandle

    cfg = bnn_cfg if bnn_cfg is not None else {}
    return OutcomeModelHandle(
        posterior, system,
        n_marginal_samples=int(cfg.get("n_marginal_samples", 64)),
        seed=int(cfg.get("seed", 0)),
    )

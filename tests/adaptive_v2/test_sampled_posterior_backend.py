"""BALD on a Gaussian-posterior BNN.

MFVI/Laplace have no finite member set, so the exact-enumeration backend
rejects them. These cover the sampling backend that averages the two BALD
entropy terms over draws from the Gaussian instead.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.ensemble_prob import (
    EnsembleClassifierProbabilityBackend,
    SampledPosteriorClassifierProbabilityBackend,
)


class _GaussianPosterior(torch.nn.Module):
    """Stands in for MFVI/Laplace: continuous q(w), so no n_members."""

    def __init__(self, mu=0.0, sigma=1.0):
        super().__init__()
        self.mu, self.sigma = float(mu), float(sigma)

    def predictive_logit_samples(self, x, S, generator=None):
        return self.mu + self.sigma * torch.randn(int(S), x.shape[0], 1)


class _IdentitySystem:
    def normalize_state(self, x):
        return x

    def embed_state_for_model(self, x):
        return x


def _cfg(**kw):
    return OmegaConf.create({"attractor_radius": 0.3, **kw})


def test_sampling_backend_accepts_a_posterior_with_no_members():
    b = SampledPosteriorClassifierProbabilityBackend(_cfg(), _IdentitySystem(), "cpu")
    b.bind_model(_GaussianPosterior())
    p = b.estimate_members(np.zeros((5, 2), dtype=np.float32))
    assert p.shape == (64, 5), "one row per posterior draw"
    assert ((p >= 0) & (p <= 1)).all()


def test_exact_backend_still_rejects_that_posterior():
    # The reason the sampling backend has to exist at all.
    b = EnsembleClassifierProbabilityBackend(_cfg(), _IdentitySystem(), "cpu")
    with pytest.raises(ValueError, match="at least 2 members"):
        b.bind_model(_GaussianPosterior())


def test_member_sample_size_is_none():
    # That field debiases BINOMIAL noise inside a member. Each atom here is an
    # exact probability for its weight draw, so claiming a sample size would
    # make the epistemic score subtract noise that is not there.
    assert SampledPosteriorClassifierProbabilityBackend.member_sample_size is None


def test_degenerate_posterior_gives_zero_epistemic_spread():
    # sigma=0 collapses q(w) to a point: every atom identical, so BALD must be 0.
    b = SampledPosteriorClassifierProbabilityBackend(_cfg(), _IdentitySystem(), "cpu")
    b.bind_model(_GaussianPosterior(mu=0.7, sigma=0.0))
    p = b.estimate_members(np.zeros((4, 2), dtype=np.float32))
    assert np.allclose(p.std(axis=0), 0.0)


def test_rejects_too_few_samples():
    with pytest.raises(ValueError, match="no epistemic signal"):
        SampledPosteriorClassifierProbabilityBackend(
            _cfg(n_posterior_samples=1), _IdentitySystem(), "cpu")


def test_marginal_is_the_atom_mean():
    b = SampledPosteriorClassifierProbabilityBackend(_cfg(), _IdentitySystem(), "cpu")
    b.bind_model(_GaussianPosterior(mu=0.0, sigma=1e-9))
    out = b.estimate(np.zeros((3, 2), dtype=np.float32))
    assert np.allclose(out.p_success, 0.5, atol=1e-3)
    assert np.allclose(out.p_success + out.p_failure, 1.0)
    assert np.allclose(out.p_invalid, 0.0)

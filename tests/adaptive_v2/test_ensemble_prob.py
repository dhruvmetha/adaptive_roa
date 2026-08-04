import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.ensemble_prob import (
    EnsembleClassifierProbabilityBackend, EnsembleEndpointMCProbabilityBackend,
)


class _FakePosterior:
    """Two members with fixed, different logits."""
    n_members = 2

    def forward_all_members(self, x):
        n = x.shape[0]
        m0 = torch.full((n, 1), 2.0)       # sigmoid ~0.881
        m1 = torch.full((n, 1), -2.0)      # sigmoid ~0.119
        return torch.stack([m0, m1], dim=0)


class _FakeHandle:
    def __init__(self):
        self.posterior = _FakePosterior()
    def eval(self):
        return self


def test_classifier_backend_members_shape_and_values():
    b = EnsembleClassifierProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.1}), system=None, device="cpu")
    b.bind_model(_FakeHandle())
    pm = b.estimate_members(np.zeros((4, 2), dtype=np.float32))
    assert pm.shape == (2, 4)
    assert pm[0].mean() == pytest.approx(0.8808, abs=1e-3)
    assert pm[1].mean() == pytest.approx(0.1192, abs=1e-3)


def test_classifier_backend_marginal_is_mean_of_members():
    b = EnsembleClassifierProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.1}), system=None, device="cpu")
    b.bind_model(_FakeHandle())
    x = np.zeros((4, 2), dtype=np.float32)
    pm = b.estimate_members(x)
    out = b.estimate(x)
    np.testing.assert_allclose(out.p_success, pm.mean(axis=0), atol=1e-9)
    np.testing.assert_allclose(out.p_success + out.p_failure + out.p_invalid, 1.0, atol=1e-9)


def test_classifier_member_sample_size_is_none_because_no_sampling():
    b = EnsembleClassifierProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.1}), system=None, device="cpu")
    assert b.member_sample_size is None


def test_fm_backend_member_sample_size_is_k():
    cfg = OmegaConf.create({"attractor_radius": 0.1, "num_mc_samples": 20})
    b = EnsembleEndpointMCProbabilityBackend(cfg, system=None, device="cpu")
    assert b.member_sample_size == 20


def test_backend_rejects_a_single_member_ensemble():
    class _One:
        n_members = 1
        def forward_all_members(self, x):
            return torch.zeros((1, x.shape[0], 1))
    class _H:
        posterior = _One()
        def eval(self): return self
    b = EnsembleClassifierProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.1}), system=None, device="cpu")
    with pytest.raises(ValueError, match="at least 2 members"):
        b.bind_model(_H())


def test_estimate_before_bind_model_raises():
    b = EnsembleClassifierProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.1}), system=None, device="cpu")
    with pytest.raises(RuntimeError, match="before bind_model"):
        b.estimate_members(np.zeros((2, 2), dtype=np.float32))

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.endpoint_mc import EndpointMCProbabilityBackend
from adaptive_roa.adaptive_v2.probability.classifier_prob import ClassifierProbabilityBackend
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.conformal.config import ConformalConfig
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator


def _mc_cfg():
    return OmegaConf.create({
        "attractor_radius": 0.2,
        "num_mc_samples": 5,
        "refine_invalids": False,
        "refine_t_min": 0.7,
        "refine_t_max": 0.9,
        "refine_num_steps": 100,
        "refine_max_attempts": 5,
        "trajectory_checking": False,
    })


def _clf_cfg():
    return OmegaConf.create({"attractor_radius": 0.2})


class _DummyClassifier:
    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)


def test_mc_backend_stores_attractor_radius():
    backend = EndpointMCProbabilityBackend(_mc_cfg(), system=None, device="cpu")
    assert backend.attractor_radius == 0.2
    assert backend.num_mc_samples == 5


def test_clf_backend_returns_outcome_probabilities():
    backend = ClassifierProbabilityBackend(_clf_cfg(), system=None, device="cpu")
    backend.bind_model(_DummyClassifier())
    out = backend.estimate(np.zeros((7, 4), dtype=np.float32))
    assert isinstance(out, OutcomeProbabilities)
    assert out.p_success.shape == (7,)
    np.testing.assert_allclose(out.p_success, 0.5, rtol=1e-5)
    np.testing.assert_allclose(out.p_failure, 0.5, rtol=1e-5)
    assert np.all(out.p_invalid == 0.0)


class _CountingFlowMatcher:
    """Returns a distinct constant endpoint per call, so passes are traceable."""

    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def predict_endpoint(self, states):
        self.calls += 1
        return torch.full((states.shape[0], 2), float(self.calls))


def test_sample_endpoints_shape_and_contents():
    fm = _CountingFlowMatcher()
    cfg = ConformalConfig(num_mc_samples=99, attractor_radius=0.2)
    estimator = ProbabilityEstimator(fm, system=None, config=cfg, device="cpu")

    out = estimator.sample_endpoints(np.zeros((7, 2), dtype=np.float32), 3, verbose=False)

    assert out.shape == (7, 3, 2)
    # One batch (mc_batch_size=1024 > 7), so the three passes are calls 1, 2, 3.
    np.testing.assert_allclose(out[:, 0, :], 1.0)
    np.testing.assert_allclose(out[:, 1, :], 2.0)
    np.testing.assert_allclose(out[:, 2, :], 3.0)
    # num_samples overrides config.num_mc_samples entirely.
    assert fm.calls == 3


def test_sample_endpoints_respects_mc_batch_size():
    fm = _CountingFlowMatcher()
    cfg = ConformalConfig(num_mc_samples=1, attractor_radius=0.2, mc_batch_size=4)
    estimator = ProbabilityEstimator(fm, system=None, config=cfg, device="cpu")

    out = estimator.sample_endpoints(np.zeros((10, 2), dtype=np.float32), 2, verbose=False)

    assert out.shape == (10, 2, 2)
    # 10 states / batch 4 -> 3 batches, 2 passes each.
    assert fm.calls == 6


def test_sample_endpoints_accepts_torch_input():
    fm = _CountingFlowMatcher()
    cfg = ConformalConfig(attractor_radius=0.2)
    estimator = ProbabilityEstimator(fm, system=None, config=cfg, device="cpu")

    out = estimator.sample_endpoints(torch.zeros((3, 2)), 2, verbose=False)

    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 2, 2)


def test_mc_backend_sample_endpoints_delegates():
    backend = EndpointMCProbabilityBackend(_mc_cfg(), system=None, device="cpu")
    backend.bind_model(_CountingFlowMatcher())

    out = backend.sample_endpoints(np.zeros((5, 2), dtype=np.float32), 4, verbose=False)

    assert out.shape == (5, 4, 2)


def test_mc_backend_sample_endpoints_requires_bind_model():
    backend = EndpointMCProbabilityBackend(_mc_cfg(), system=None, device="cpu")
    with pytest.raises(RuntimeError, match="before bind_model"):
        backend.sample_endpoints(np.zeros((2, 2), dtype=np.float32), 2, verbose=False)


def test_sample_endpoints_handles_empty_input():
    fm = _CountingFlowMatcher()
    cfg = ConformalConfig(attractor_radius=0.2)
    estimator = ProbabilityEstimator(fm, system=None, config=cfg, device="cpu")

    out = estimator.sample_endpoints(np.zeros((0, 2), dtype=np.float32), 3, verbose=False)

    assert out.shape[0] == 0
    assert fm.calls == 0

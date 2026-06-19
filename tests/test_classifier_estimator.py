"""Task 1 tests: ClassifierProbabilityEstimator + factory routing."""
import numpy as np
import torch

from adaptive_roa.conformal.classifier_probability_estimator import ClassifierProbabilityEstimator
from adaptive_roa.conformal.estimator_factory import build_probability_estimator


class _DummyModel:
    """Callable returning a constant logit; mimics ClassifierModule(raw)->logits."""

    def __init__(self, logit: float):
        self.logit = logit

    def eval(self):
        return self

    def __call__(self, x):
        return torch.full((x.shape[0], 1), self.logit, dtype=torch.float32)


def test_estimate_returns_binary_probabilities():
    est = ClassifierProbabilityEstimator(_DummyModel(2.0), system=None, config=None, device="cpu")
    states = np.zeros((5, 6), dtype=np.float32)

    p_success, p_failure, p_invalid = est.estimate(states, verbose=False)

    expected_ps = 1.0 / (1.0 + np.exp(-2.0))
    assert p_success.shape == (5,)
    assert p_success.dtype == np.float64
    np.testing.assert_allclose(p_success, expected_ps, rtol=1e-5)
    np.testing.assert_allclose(p_failure, 1.0 - p_success, rtol=1e-6)
    assert np.all(p_invalid == 0.0)


def test_estimate_batches_large_input():
    est = ClassifierProbabilityEstimator(_DummyModel(0.0), system=None, config=None, device="cpu")
    states = np.zeros((20000, 6), dtype=np.float32)  # > default batch of 8192
    p_success, _, _ = est.estimate(states, verbose=False)
    assert p_success.shape == (20000,)
    np.testing.assert_allclose(p_success, 0.5, rtol=1e-6)


def test_factory_routes_classifier():
    est = build_probability_estimator("classifier", _DummyModel(0.0), None, None, "cpu")
    assert isinstance(est, ClassifierProbabilityEstimator)

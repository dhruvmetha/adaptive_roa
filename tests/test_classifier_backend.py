import numpy as np
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.classifier_prob import ClassifierProbabilityBackend
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class _DummyClassifier:
    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)


def test_backend_returns_outcome_probabilities():
    backend = ClassifierProbabilityBackend(OmegaConf.create({}), system=None, device="cpu")
    backend.bind_model(_DummyClassifier())
    out = backend.estimate(np.zeros((7, 6), dtype=np.float32))
    assert isinstance(out, OutcomeProbabilities)
    assert out.p_success.shape == (7,)
    np.testing.assert_allclose(out.p_success, 0.5, rtol=1e-6)
    np.testing.assert_allclose(out.p_failure, 0.5, rtol=1e-6)
    assert np.all(out.p_invalid == 0.0)

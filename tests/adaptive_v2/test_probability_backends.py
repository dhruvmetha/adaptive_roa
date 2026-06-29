import numpy as np
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.endpoint_mc import EndpointMCProbabilityBackend
from adaptive_roa.adaptive_v2.probability.classifier_prob import ClassifierProbabilityBackend
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


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

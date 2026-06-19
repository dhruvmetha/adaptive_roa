"""Task 7 test: ClassifierProbabilityBackend returns OutcomeProbabilities."""
import numpy as np
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.classifier_prob import ClassifierProbabilityBackend
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


def _cfg():
    return OmegaConf.create({
        "val_batch_size": 2048,
        "conformal": {
            "delta": 0.05, "w": 0.9, "alpha_sampling": 0.1, "num_mc_samples": 10,
            "attractor_radius": 0.2, "optimize_mode": "joint", "decision_rule": "one_sided",
            "lambda_grid_size": 50, "delta_grid_size": 50, "delta_min": 0.05, "delta_max": 0.45,
            "use_p_invalid_veto": True, "optimize_objective": "loss", "target_f1": 0.9,
            "fixed_lambda_star": 0.5, "fixed_delta_star": 0.1,
            "trajectory_checking": False, "refine_invalids": False, "verbose": False,
        },
    })


class _DummyClassifier:
    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)  # logit 0 -> p=0.5


def test_backend_returns_outcome_probabilities():
    backend = ClassifierProbabilityBackend(system=None, cfg=_cfg(), device="cpu")
    backend.bind_model(_DummyClassifier())

    out = backend.estimate(np.zeros((7, 6), dtype=np.float32))
    assert isinstance(out, OutcomeProbabilities)
    assert out.p_success.shape == (7,)
    np.testing.assert_allclose(out.p_success, 0.5, rtol=1e-6)
    np.testing.assert_allclose(out.p_failure, 0.5, rtol=1e-6)
    assert np.all(out.p_invalid == 0.0)

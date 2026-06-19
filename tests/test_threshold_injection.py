"""Task 2 tests: estimator injection into ConformalPredictor + threshold backend."""
import torch
from omegaconf import OmegaConf

from adaptive_roa.conformal import ConformalConfig, ConformalPredictor
from adaptive_roa.conformal.classifier_probability_estimator import ClassifierProbabilityEstimator
from adaptive_roa.adaptive_v2.threshold.conformal_threshold import ConformalThresholdBackend


def _cfg(predictor: str):
    return OmegaConf.create({
        "predictor": predictor,
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
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)


def test_predictor_uses_injected_estimator():
    conf = ConformalConfig.from_hydra(_cfg("classifier"))
    sentinel = object()
    pred = ConformalPredictor(
        flow_matcher=None, system=None, config=conf, device="cpu",
        probability_estimator=sentinel,
    )
    assert pred.prob_estimator is sentinel


def test_threshold_backend_builds_classifier_estimator():
    backend = ConformalThresholdBackend(system=None, cfg=_cfg("classifier"), device="cpu")
    backend.bind_model(_DummyClassifier())
    assert isinstance(backend.predictor.prob_estimator, ClassifierProbabilityEstimator)

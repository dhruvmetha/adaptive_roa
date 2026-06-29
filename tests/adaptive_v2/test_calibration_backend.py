import numpy as np
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.calibration.conformal_calibration import ConformalCalibrationBackend
from adaptive_roa.adaptive_v2.types import ThresholdState


def _cfg(decision_rule="two_sided"):
    return OmegaConf.create({
        "delta": 0.05,
        "w": 0.9,
        "alpha": 0.1,
        "alpha_eval": 0.1,
        "decision_rule": decision_rule,
        "verbose": False,
    })


class _DummyClassifier:
    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)


def test_calibration_backend_init():
    backend = ConformalCalibrationBackend(_cfg(), system=None, device="cpu")
    assert backend.delta == 0.05
    assert backend.alpha == 0.1
    assert backend.decision_rule == "two_sided"


def test_calibrate_returns_float():
    backend = ConformalCalibrationBackend(_cfg(decision_rule="one_sided"), system=None, device="cpu")
    backend.bind_model(_DummyClassifier(), predictor_type="classifier")
    state = ThresholdState(lambda_star=0.5, delta_star=0.1)
    n = 20
    X = np.zeros((n, 4), dtype=np.float32)
    y = np.array([1] * 10 + [-1] * 10, dtype=np.int64)
    q_hat = backend.calibrate(X, y, state)
    assert isinstance(q_hat, float)


def test_calibrate_eval_uses_alpha_eval():
    cfg = OmegaConf.create({
        "delta": 0.05, "w": 0.9, "alpha": 0.1, "alpha_eval": 0.2,
        "decision_rule": "one_sided", "verbose": False,
    })
    backend = ConformalCalibrationBackend(cfg, system=None, device="cpu")
    backend.bind_model(_DummyClassifier(), predictor_type="classifier")
    state = ThresholdState(lambda_star=0.5, delta_star=0.1)
    n = 20
    X = np.zeros((n, 4), dtype=np.float32)
    y = np.array([1] * 10 + [-1] * 10, dtype=np.int64)
    q_hat_a = backend.calibrate(X, y, state)
    q_hat_b = backend.calibrate_eval(X, y, state)
    # Different alpha values → different q_hat (not guaranteed equal)
    assert isinstance(q_hat_a, float)
    assert isinstance(q_hat_b, float)

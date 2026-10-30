"""Prediction store (det paper tracker section 9): counts, endpoints, threshold fit."""
from __future__ import annotations

import numpy as np
import torch

from adaptive_roa.adaptive_v2.eval import prediction_store as ps
from adaptive_roa.adaptive_v2.eval.full_roa import evaluate_full_roa_fast


class _NoisyModel:
    """Endpoint = start + noise, so labels vary across draws."""

    def __init__(self):
        self.g = torch.Generator().manual_seed(0)

    def eval(self):
        return self

    def predict_endpoint(self, x, **kw):
        return x + 0.5 * torch.randn(x.shape, generator=self.g)


class _SignSystem:
    """Success if x0 > 0.3, failure if x0 < -0.3, invalid in between."""

    def classify_attractor(self, pred, radius):
        lab = torch.zeros(pred.shape[0], dtype=torch.long)
        lab[pred[:, 0] > 0.3] = 1
        lab[pred[:, 0] < -0.3] = -1
        return lab


def _states(n=40):
    return np.random.default_rng(1).uniform(-1, 1, size=(n, 2)).astype(np.float32)


def test_counts_sum_to_k_and_match_labels():
    labels, endpoints = ps.mc_predict(_NoisyModel(), _SignSystem(), _states(), 50, 0.1, batch_size=16)
    c = ps.counts_from_labels(labels)
    np.testing.assert_array_equal(c["n_success"] + c["n_failure"] + c["n_invalid"], 50)
    np.testing.assert_array_equal(c["n_success"], (labels == 1).sum(1))
    assert endpoints is None


def test_kept_endpoints_reclassify_to_the_stored_labels():
    X = _states()
    keep = np.array([0, 7, 19, 39])
    labels, endpoints = ps.mc_predict(_NoisyModel(), _SignSystem(), X, 30, 0.1, batch_size=8, keep_rows=keep)
    assert endpoints.shape == (4, 30, 2)
    re = _SignSystem().classify_attractor(torch.from_numpy(endpoints.reshape(-1, 2)), 0.1).numpy().reshape(4, 30)
    np.testing.assert_array_equal(re, labels[keep])


def test_write_split_endpoint_model(tmp_path):
    X, y = _states(), np.where(_states()[:, 0] > 0, 1, -1)
    ps.write_split(tmp_path, "val", X, y, predictor_type="generative", model=_NoisyModel(),
                   system=_SignSystem(), num_mc_samples=20, attractor_radius=0.1,
                   batch_size=16, keep_endpoints=True)
    z = np.load(tmp_path / "predictions_val.npz")
    assert int(z["num_mc_samples"]) == 20 and z["n_success"].dtype == np.int16
    e = np.load(tmp_path / "endpoints_val.npz")
    assert e["endpoints"].shape == (40, 20, 2)


class _Logit:
    def eval(self):
        return self

    def to(self, d):
        return self

    def __call__(self, x):
        return x[:, :1] * 3.0


def test_write_split_classifier_stores_probabilities(tmp_path):
    X = _states()
    ps.write_split(tmp_path, "cal", X, np.ones(len(X)), predictor_type="classifier", model=_Logit(),
                   system=None, num_mc_samples=100, attractor_radius=0.1)
    z = np.load(tmp_path / "predictions_cal.npz")
    np.testing.assert_allclose(z["p_success"], 1 / (1 + np.exp(-3 * X[:, 0])), rtol=1e-5)
    assert not (tmp_path / "endpoints_cal.npz").exists()


def test_test_split_endpoints_from_the_evaluator_match_its_counts(tmp_path):
    X = _states(60)
    rows = np.hstack([X, np.zeros_like(X), np.where(X[:, :1] > 0, 1, 0)])
    f = tmp_path / "eval.txt"
    np.savetxt(f, rows, delimiter=",", fmt="%.6f")
    out = tmp_path / "ep"
    evaluate_full_roa_fast(_NoisyModel(), _SignSystem(), str(f), num_mc_samples=25,
                           lambda_star=0.5, delta=0.1, decision_rule="two_sided", device="cpu",
                           output_dir=str(out), verbose=False, store_endpoints=True,
                           endpoint_rows_cap=10)
    e = np.load(out / "endpoints_test.npz")
    pp = np.load(out / "full_roa_per_point.npz")
    assert e["endpoints"].shape == (10, 25, 2)
    np.testing.assert_array_equal(e["rows"], ps.endpoint_test_rows(60, 10))
    lab = _SignSystem().classify_attractor(torch.from_numpy(e["endpoints"].reshape(-1, 2)), 0.1)
    n_succ = (lab.numpy().reshape(10, 25) == 1).sum(1)
    np.testing.assert_allclose(n_succ / 25, pp["p_success"][e["rows"]])


def test_endpoint_rows_are_fixed_and_capped():
    a, b = ps.endpoint_test_rows(1000, 100), ps.endpoint_test_rows(1000, 100)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 100 and np.all(np.diff(a) > 0)
    np.testing.assert_array_equal(ps.endpoint_test_rows(50, 100), np.arange(50))


def test_threshold_fit_is_captured_and_written(tmp_path):
    from omegaconf import OmegaConf
    from adaptive_roa.adaptive_v2.threshold.conformal_threshold import ConformalThresholdBackend
    cfg = OmegaConf.create({
        "predictor_type": "classifier", "decision_rule": "one_sided", "delta": 0.05,
        "attractor_radius": 0.2, "num_mc_samples": 10, "optimize_mode": "joint",
        "optimize_objective": "loss", "target_f1": 0.9, "lambda_grid_size": 20,
        "delta_grid_size": 10, "delta_min": 0.05, "delta_max": 0.45, "fixed_lambda_star": 0.5,
        "fixed_delta_star": 0.1, "use_p_invalid_veto": True, "verbose": False,
    })
    b = ConformalThresholdBackend(cfg, system=None, device="cpu")
    b.bind_model(_Logit())
    X = _states(80)
    y = np.where(X[:, 0] > 0, 1, -1)
    b.optimize(X, y)
    fit = b.last_fit
    np.testing.assert_array_equal(fit["labels"], y)
    ps.write_threshold_fit(tmp_path, fit)
    z = np.load(tmp_path / "predictions_thrfit.npz")
    assert z["states"].shape == (80, 2) and z["p_success"].shape == (80,)

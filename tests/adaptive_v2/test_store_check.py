"""The store check passes a consistent run and catches a tampered one (C08)."""
from __future__ import annotations

import json

import numpy as np
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval import prediction_store as ps
from adaptive_roa.adaptive_v2.eval.full_roa import evaluate_full_roa_fast
from adaptive_roa.adaptive_v2.eval.store_check import check_run


class _M:
    def __init__(self):
        self.g = torch.Generator().manual_seed(0)

    def eval(self):
        return self

    def predict_endpoint(self, x, **kw):
        return x + 0.5 * torch.randn(x.shape, generator=self.g)


class _S:
    def classify_attractor(self, pred, radius):
        lab = torch.zeros(pred.shape[0], dtype=torch.long)
        lab[pred[:, 0] > 0.3] = 1
        lab[pred[:, 0] < -0.3] = -1
        return lab


def _run(tmp_path):
    run = tmp_path / "run"
    (run / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"decision_rule": "two_sided", "prediction_store": {
        "enabled": True, "endpoint_epochs": [0], "endpoint_test_rows": 10}}), run / ".hydra" / "config.yaml")
    X = np.random.default_rng(2).uniform(-1, 1, size=(50, 2)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 1, -1)
    f = tmp_path / "test.txt"
    np.savetxt(f, np.hstack([X, np.zeros_like(X), (y == 1)[:, None]]), delimiter=",", fmt="%.6f")
    ep = run / "epoch_000"
    m = evaluate_full_roa_fast(_M(), _S(), str(f), num_mc_samples=20, lambda_star=0.5, delta=0.1,
                               decision_rule="two_sided", device="cpu", output_dir=str(ep),
                               verbose=False, store_endpoints=True, endpoint_rows_cap=10)
    for split in ("val", "cal"):
        ps.write_split_states_once(run, split, X[:20], y[:20])
        ps.write_split(ep, split, X[:20], y[:20], predictor_type="generative", model=_M(), system=_S(),
                       num_mc_samples=20, attractor_radius=0.1, keep_endpoints=True)
    ps.write_threshold_fit(ep, {"states": X, "labels": y, "p_success": np.zeros(50),
                                "p_failure": np.zeros(50), "p_invalid": np.zeros(50), "num_mc_samples": 10})
    art = {"threshold_state": {"lambda_star": 0.5}, "eval_metrics": json.loads(json.dumps(m, default=float))}
    (ep / "artifacts_v2.json").write_text(json.dumps(art))
    return run


def test_consistent_run_passes(tmp_path):
    assert check_run(_run(tmp_path)) == []


def test_tampered_counts_fail(tmp_path):
    run = _run(tmp_path)
    p = run / "epoch_000" / "artifacts_v2.json"
    art = json.loads(p.read_text())
    art["eval_metrics"]["lambda_delta"]["tp"] += 1
    p.write_text(json.dumps(art))
    assert any("replayed counts differ" in x for x in check_run(run))


def test_missing_reporting_endpoints_fail(tmp_path):
    run = _run(tmp_path)
    (run / "epoch_000" / "endpoints_cal.npz").unlink()
    assert any("endpoints_cal" in x for x in check_run(run))

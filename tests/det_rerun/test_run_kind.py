"""The lambda*/delta*-only post-hoc tools refuse FPR-capped (tau-rule) runs.

Idea 6 arms store the K=20 val fit states in predictions_thrfit.npz and record
tau_s/tau_f/tau_inv, so a lambda/delta refit or replay on them would misread both
(better-thresh-opt, 2026-11-02).
"""
import json

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.eval import fpr_cap
from adaptive_roa.adaptive_v2.eval.run_kind import require_lambda_delta, uses_tau_rule


def _run(tmp_path, name, tau):
    run = tmp_path / name
    (run / "epoch_000").mkdir(parents=True)
    ts = {"lambda_star": 0.5, "delta_star": 0.1}
    if tau:
        ts.update(tau_s=0.6, tau_f=0.7, tau_inv=1.01, alpha_s=0.01)
    (run / "epoch_000" / "artifacts_v2.json").write_text(json.dumps({"threshold_state": ts}))
    return run


def test_tau_runs_are_recognised_by_name_or_recorded_thresholds(tmp_path):
    assert uses_tau_rule(_run(tmp_path, "dr_q3d30_fm_ranked_fprcap1pct_s99512", tau=False))
    assert uses_tau_rule(_run(tmp_path, "dr_q3d30_fm_ranked_other", tau=True))
    assert not uses_tau_rule(_run(tmp_path, "dr_q3d30_fm_ranked_s99512", tau=False))
    assert not uses_tau_rule(tmp_path / "empty")


@pytest.mark.parametrize("call", [
    lambda r: __import__("adaptive_roa.adaptive_v2.eval.store_threshold", fromlist=["x"]).refit_run(r),
    lambda r: __import__("adaptive_roa.adaptive_v2.eval.store_k", fromlist=["x"]).k_check_run(r),
    lambda r: __import__("adaptive_roa.adaptive_v2.eval.store_diag", fromlist=["x"]).diagnose_run(r),
    lambda r: __import__("adaptive_roa.adaptive_v2.eval.store_boundary", fromlist=["x"]).decision_maps(r),
    lambda r: __import__("adaptive_roa.adaptive_v2.eval.store_score", fromlist=["x"]).score_run(r, "run"),
    lambda r: __import__("adaptive_roa.adaptive_v2.eval.store_score", fromlist=["x"]).score_run(r, "val"),
])
def test_lambda_delta_tools_refuse_tau_runs(tmp_path, call):
    run = _run(tmp_path, "dr_pen_fm_ranked_fprcap1pct_s99512", tau=True)
    with pytest.raises(ValueError, match="tau rule"):
        call(run)


def test_require_lambda_delta_passes_lambda_delta_runs(tmp_path):
    require_lambda_delta(_run(tmp_path, "dr_pen_fm_ranked_s99512", tau=False), "test")


def test_a_both_off_fit_reports_no_commits_not_all_invalid(tmp_path, monkeypatch):
    ep = tmp_path / "dr_pen_fm_ranked_fprcap1pct_s99512" / "epoch_000"
    ep.mkdir(parents=True)
    y = np.array([1, -1, 1, -1])
    np.savez(ep / "predictions_val.npz", true_labels=y, n_success=np.array([5, 1, 4, 0]),
             n_failure=np.array([5, 9, 6, 10]), n_invalid=np.zeros(4, int), num_mc_samples=np.int32(10))
    np.savez(ep / "full_roa_per_point.npz", p_success=np.array([0.5, 0.1, 0.4, 0.0]),
             p_failure=np.array([0.5, 0.9, 0.6, 1.0]), p_invalid=np.zeros(4), true_labels=y)
    off = {a: {"tau_s": 1.01, "tau_f": 1.01, "tau_inv": 1.01, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
               "n_success": 2, "n_failure": 2} for a in (0.01,)}
    monkeypatch.setattr(fpr_cap, "fit", lambda *a, **k: off)
    (row,) = fpr_cap.score_run(ep.parent, alphas=[0.01])
    assert row["no_commits"] and row["n_success_claims"] == 0
    assert np.isnan(row["abstain_invalid"]) and np.isnan(row["abstain_uncertain"])


def test_reported_rates_are_nan_without_the_class():
    m = fpr_cap.split_metrics(np.array([-1, 0, 0]), np.array([-1, -1, -1]))   # no true successes
    assert np.isnan(m["success_recovery"]) and np.isnan(m["ffr_all_successes"])

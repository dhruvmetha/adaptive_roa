"""Conformal alpha sweep from the prediction store (tracker H02; REVIEW_TODO R01-R03, R15).

Two protocols, both computed so the paper can state the one it uses:

- "all": what the code does today. calibrate_qhat scores every cal point, and
  coverage is the share of all test points whose true label is in the set.
- "filtered": what the manuscript states. Cal and test points with
  p_invalid >= the invalid threshold are excluded from both calibration and
  coverage, and the retained fractions are reported.

Both reuse the run's frozen lambda*, delta*, decision rule and the stored cal
predictions at the eval K, so no model is loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.full_roa import _predict_lambda_delta, _predict_qhat_prediction_sets
from adaptive_roa.conformal.calibrator import (
    conformal_quantile,
    nonconformity_scores_batch_one_sided,
    nonconformity_scores_batch_two_sided,
)

ALPHAS = (0.10, 0.05, 0.02, 0.01)


def _probs(z):
    if "p_success" in z.files and "n_success" not in z.files:
        p = np.asarray(z["p_success"], float)
        return p, 1.0 - p, np.zeros_like(p)
    k = float(z["num_mc_samples"])
    return z["n_success"] / k, z["n_failure"] / k, z["n_invalid"] / k


def _scores(p_s, p_f, y, lam, delta, rule):
    if rule == "two_sided":
        return nonconformity_scores_batch_two_sided(p_s, p_f, y, lam, delta)
    return nonconformity_scores_batch_one_sided(p_s, y, lam, delta)


def sweep(cal, test, lam, delta, rule, inv_thr, alphas=ALPHAS) -> list[dict]:
    """cal and test are (p_success, p_failure, p_invalid, y) tuples, y in {-1, 1}."""
    thr = float((lam - delta) if inv_thr is None else inv_thr)
    pc_s, pc_f, pc_i, yc = cal
    pt_s, pt_f, pt_i, yt = test
    keep_c, keep_t = pc_i < thr, pt_i < thr
    sc = _scores(pc_s, pc_f, yc, lam, delta, rule)
    st = _scores(pt_s, pt_f, yt, lam, delta, rule)
    base, _ = _predict_lambda_delta(pt_s, pt_f, pt_i, lam, delta, rule, inv_thr)
    rows = []
    for protocol, cal_mask, test_mask in (("all", np.ones_like(keep_c), np.ones_like(keep_t)),
                                          ("filtered", keep_c, keep_t)):
        for a in alphas:
            q = conformal_quantile(sc[cal_mask], a)
            pred, ex = _predict_qhat_prediction_sets(pt_s, pt_f, pt_i, yt, lam, delta, q, rule, inv_thr)
            covered = st <= q
            rows.append({
                "protocol": protocol, "alpha": a, "q_hat": q,
                "n_cal_used": int(cal_mask.sum()), "cal_retained": float(cal_mask.mean()),
                "test_retained": float(test_mask.mean()),
                "coverage": float(covered[test_mask].mean()) if test_mask.any() else float("nan"),
                "avg_set_size": ex["avg_set_size"],
                "committed_fraction": float(np.mean((pred == 0) | (pred == 1))),
                "decisions_changed": float(np.mean(pred != base)),
                "success_precision_committed": (float(np.mean(yt[pred == 1] == 1))
                                                if np.any(pred == 1) else float("nan")),
            })
    return rows


def sweep_run(run_dir) -> list[dict]:
    run = Path(run_dir)
    cfg_path = run / "resolved_config.yaml"
    cfg = OmegaConf.load(cfg_path if cfg_path.exists() else run / ".hydra" / "config.yaml")
    rule = str(cfg.get("decision_rule"))
    out = []
    for art_path in sorted(run.glob("epoch_*/artifacts_v2.json")):
        ep_dir = art_path.parent
        art = json.loads(art_path.read_text())
        ld = (art.get("eval_metrics") or {}).get("lambda_delta")
        cal_f = ep_dir / "predictions_cal.npz"
        if ld is None or not cal_f.exists():
            continue
        zc, pp = np.load(cal_f), np.load(ep_dir / "full_roa_per_point.npz")
        cal = (*_probs(zc), zc["true_labels"].astype(int))
        test = (pp["p_success"], pp["p_failure"], pp["p_invalid"], pp["true_labels"].astype(int))
        for r in sweep(cal, test, float(ld["lambda_star"]), float(ld["delta"]), rule, ld.get("invalid_threshold")):
            r.update(run=run.name, epoch=int(art["epoch"]), train_trajectories=int(art["train_trajectories"]))
            out.append(r)
    return out

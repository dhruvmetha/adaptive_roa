"""Abstention and error diagnostics from the store (tracker H07; REVIEW_TODO R04, R07, E5).

Per run and epoch on test, under the run's own lambda*/delta* rule:
- invalid and uncertain shares, and the true success share inside each, so the
  abstained set's class composition is visible;
- valid-but-wrong: the committed states whose label is wrong, as a share of all
  committed states (the error p_invalid cannot see);
- whether p_invalid, or the disagreement 1 - max(p_s, p_f, p_inv), ranks the
  states the rule gets wrong or abstains on above the ones it gets right (AUROC).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.full_roa import _predict_lambda_delta
from adaptive_roa.adaptive_v2.eval.store_score import _auroc


def diagnose(p_s, p_f, p_i, y, lam, delta, rule, inv_thr) -> dict:
    y01 = (np.asarray(y) == 1).astype(int)
    pred, _ = _predict_lambda_delta(p_s, p_f, p_i, lam, delta, rule, inv_thr)
    inv, unc = pred == -2, pred == -1
    committed = (pred == 0) | (pred == 1)
    wrong = committed & (pred != y01)
    bad = (wrong | inv | unc).astype(int)
    disagreement = 1.0 - np.max(np.stack([p_s, p_f, p_i]), axis=0)
    return {
        "invalid_share": float(inv.mean()), "uncertain_share": float(unc.mean()),
        "success_share_in_invalid": float(y01[inv].mean()) if inv.any() else float("nan"),
        "success_share_in_uncertain": float(y01[unc].mean()) if unc.any() else float("nan"),
        "success_share_overall": float(y01.mean()),
        "valid_but_wrong_share_of_committed": float(wrong.sum() / committed.sum()) if committed.any() else float("nan"),
        "auroc_p_invalid_ranks_bad": _auroc(np.asarray(p_i, float), bad),
        "auroc_disagreement_ranks_bad": _auroc(disagreement, bad),
        "auroc_p_invalid_ranks_wrong_among_committed": (_auroc(np.asarray(p_i, float)[committed], wrong[committed].astype(int))
                                                        if committed.any() and wrong[committed].any() else float("nan")),
    }


def diagnose_run(run_dir) -> list[dict]:
    run = Path(run_dir)
    cfg = OmegaConf.load(run / "resolved_config.yaml")
    rows = []
    for art_path in sorted(run.glob("epoch_*/artifacts_v2.json")):
        art = json.loads(art_path.read_text())
        ld = (art.get("eval_metrics") or {}).get("lambda_delta")
        if ld is None:
            continue
        pp = np.load(art_path.parent / "full_roa_per_point.npz")
        r = diagnose(pp["p_success"], pp["p_failure"], pp["p_invalid"], pp["true_labels"],
                     float(ld["lambda_star"]), float(ld["delta"]), str(cfg.decision_rule), ld.get("invalid_threshold"))
        r.update(run=run.name, epoch=int(art["epoch"]), train_trajectories=int(art["train_trajectories"]))
        rows.append(r)
    return rows

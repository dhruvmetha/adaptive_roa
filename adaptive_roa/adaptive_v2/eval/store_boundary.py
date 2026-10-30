"""Decision maps over the test states at chosen budgets (tracker H06; REVIEW_TODO E5, R11).

Per run and epoch: each test state's decision under the run's own lambda*/delta*
rule (success, failure, or abstain, which joins invalid and uncertain) beside its
rollout label, and the shares a caption quotes. The figure is drawn by
scripts/det_rerun/boundary_figure.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.full_roa import _predict_lambda_delta

SUCCESS, FAILURE, ABSTAIN = 1, 0, -1


def decisions(p_s, p_f, p_i, lam, delta, rule, inv_thr) -> np.ndarray:
    pred, _ = _predict_lambda_delta(p_s, p_f, p_i, lam, delta, rule, inv_thr)
    return np.where(pred < 0, ABSTAIN, pred).astype(int)


def shares(dec, y01) -> dict:
    committed = dec != ABSTAIN
    wrong = committed & (dec != y01)
    return {
        "committed_share": float(committed.mean()),
        "abstain_share": float((~committed).mean()),
        "wrong_share_of_committed": float(wrong.sum() / committed.sum()) if committed.any() else float("nan"),
        "success_recovery": float(np.mean(dec[y01 == 1] == SUCCESS)) if np.any(y01 == 1) else float("nan"),
        "success_precision": float(np.mean(y01[dec == SUCCESS] == 1)) if np.any(dec == SUCCESS) else float("nan"),
    }


def decision_maps(run_dir, epochs=None, dims=(0, 1)) -> list[dict]:
    """One entry per stored epoch (or per epoch in `epochs`), oldest first."""
    run = Path(run_dir)
    cfg = OmegaConf.load(run / "resolved_config.yaml")
    rule = str(cfg.decision_rule)
    out = []
    for art_path in sorted(run.glob("epoch_*/artifacts_v2.json")):
        art = json.loads(art_path.read_text())
        ld = (art.get("eval_metrics") or {}).get("lambda_delta")
        if ld is None or (epochs is not None and int(art["epoch"]) not in epochs):
            continue
        pp = np.load(art_path.parent / "full_roa_per_point.npz")
        y01 = (pp["true_labels"] == 1).astype(int)
        dec = decisions(pp["p_success"], pp["p_failure"], pp["p_invalid"], float(ld["lambda_star"]),
                        float(ld["delta"]), rule, ld.get("invalid_threshold"))
        out.append({"run": run.name, "epoch": int(art["epoch"]), "train_trajectories": int(art["train_trajectories"]),
                    "xy": np.asarray(pp["start_states"])[:, list(dims)], "decision": dec, "label": y01,
                    **shares(dec, y01)})
    return out

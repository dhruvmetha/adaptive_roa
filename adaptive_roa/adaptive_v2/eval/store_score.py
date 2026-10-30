"""Score det-rerun runs from the prediction store alone (tracker H01; REVIEW_TODO E1, R14).

Per run and evaluated epoch, on the held-out test split:

- the run's own lambda*/delta* rule, with every denominator explicit:
  success precision TP/(TP+FP); success recovery TP/N_success, abstained
  successes included; FPR FP/N_failure over all true failures; abstention
  split into invalid and uncertain; selective F1 kept as a secondary number;
- threshold-free scores of p_success over all states: AUROC, AUPRC, Brier and
  log loss with p clipped to [eps, 1 - eps];
- a fixed-precision operating point: the smallest threshold whose val-split
  success precision reaches the target, applied unchanged to test. The val
  split never touches test, so an unmet target on test is a result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.full_roa import _predict_lambda_delta

LOG_EPS = 1e-3


def _p_from_store(npz) -> np.ndarray:
    if "p_success" in npz.files:
        return np.asarray(npz["p_success"], dtype=float)
    return npz["n_success"].astype(float) / float(npz["num_mc_samples"])


def _auroc(p, y01):
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    # average ranks for ties
    sp = p[order]
    i = 0
    r = np.arange(1, len(p) + 1, dtype=float)
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks[order] = r
    n_pos, n_neg = int(y01.sum()), int(len(y01) - y01.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y01 == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _auprc(p, y01):
    """Average precision (step-wise, as sklearn's average_precision_score)."""
    if y01.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y01[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    # keep the last index of each distinct threshold
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    prec = tp[last] / (tp[last] + fp[last])
    rec = tp[last] / ys.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def threshold_for_precision(p_val, y_val01, target: float) -> float | None:
    """Smallest threshold t with val precision of {p >= t} at least target, or None."""
    order = np.argsort(-p_val, kind="mergesort")
    ps, ys = p_val[order], y_val01[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    ok = last[prec[last] >= target]
    if ok.size == 0:
        return None
    return float(ps[ok.max()])


def score_epoch(p_test, pf_test, pinv_test, y_test, lam, delta, rule, inv_thr,
                p_val=None, y_val=None, precision_target=0.99) -> dict:
    y01 = (np.asarray(y_test) == 1).astype(int)
    n_s, n_f = int(y01.sum()), int(len(y01) - y01.sum())
    pred, _ = _predict_lambda_delta(p_test, pf_test, pinv_test, lam, delta, rule, inv_thr)
    tp = int(np.sum((pred == 1) & (y01 == 1)))
    fp = int(np.sum((pred == 1) & (y01 == 0)))
    tn = int(np.sum((pred == 0) & (y01 == 0)))
    fn = int(np.sum((pred == 0) & (y01 == 1)))
    n = len(y01)
    row = {
        "n_test": n, "n_success": n_s, "n_failure": n_f,
        "success_precision": tp / (tp + fp) if tp + fp else float("nan"),
        "success_recovery": tp / n_s if n_s else float("nan"),
        "fpr_all_failures": fp / n_f if n_f else float("nan"),
        "abstain_invalid": float(np.mean(pred == -2)),
        "abstain_uncertain": float(np.mean(pred == -1)),
        "commit_rate": float(np.mean((pred == 0) | (pred == 1))),
        "selective_f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
        "auroc": _auroc(np.asarray(p_test, float), y01),
        "auprc": _auprc(np.asarray(p_test, float), y01),
        "brier": float(np.mean((np.asarray(p_test, float) - y01) ** 2)),
        "log_loss": float(-np.mean(y01 * np.log(np.clip(p_test, LOG_EPS, 1 - LOG_EPS))
                                   + (1 - y01) * np.log(np.clip(1 - np.asarray(p_test), LOG_EPS, 1 - LOG_EPS)))),
    }
    if p_val is not None:
        t = threshold_for_precision(np.asarray(p_val, float), (np.asarray(y_val) == 1).astype(int),
                                    precision_target)
        row["p99_threshold"] = t
        if t is None:
            row.update(p99_test_precision=float("nan"), p99_test_recovery=0.0)
        else:
            adm = np.asarray(p_test) >= t
            tpa = int(np.sum(adm & (y01 == 1)))
            row["p99_test_precision"] = tpa / adm.sum() if adm.sum() else float("nan")
            row["p99_test_recovery"] = tpa / n_s if n_s else float("nan")
    return row


def _refit_on_val(cfg, z) -> tuple[float, float]:
    import contextlib
    import io

    from adaptive_roa.adaptive_v2.eval.store_threshold import refit    # imports this module
    tcfg = OmegaConf.to_container(cfg.threshold, resolve=True)
    if "n_success" in z.files:
        k = float(z["num_mc_samples"])
        p_s, p_f, p_i = z["n_success"] / k, z["n_failure"] / k, z["n_invalid"] / k
    else:                                  # classifier: one probability per state
        p_s = np.asarray(z["p_success"], float)
        p_f, p_i = 1.0 - p_s, np.zeros_like(p_s)
    with contextlib.redirect_stdout(io.StringIO()):                     # the optimizer prints its grid
        return refit(tcfg, float(tcfg.get("w", 0.9)), p_s, p_f, p_i, z["true_labels"])


def score_run(run_dir: str | Path, thresholds: str = "run") -> list[dict]:
    """thresholds="run" scores each epoch with the run's own lambda*/delta*;
    "val" refits them on the stored val split at its stored K (tracker D27b)
    with the run's threshold settings, and scores test with the refit rule."""
    if thresholds not in ("run", "val"):
        raise ValueError(f"thresholds must be 'run' or 'val', not {thresholds!r}")
    run = Path(run_dir)
    cfg_path = run / "resolved_config.yaml"
    cfg = OmegaConf.load(cfg_path if cfg_path.exists() else run / ".hydra" / "config.yaml")
    rule = str(cfg.get("decision_rule"))
    rows = []
    for art_path in sorted(run.glob("epoch_*/artifacts_v2.json")):
        ep_dir = art_path.parent
        art = json.loads(art_path.read_text())
        ev = art.get("eval_metrics") or {}
        if ev.get("skipped") or "lambda_delta" not in ev:
            continue
        ld = ev["lambda_delta"]
        pp = np.load(ep_dir / "full_roa_per_point.npz")
        val = ep_dir / "predictions_val.npz"
        p_val = y_val = None
        if val.exists():
            z = np.load(val)
            p_val, y_val = _p_from_store(z), z["true_labels"]
        lam, delta, inv_thr = float(ld["lambda_star"]), float(ld["delta"]), ld.get("invalid_threshold")
        if thresholds == "val":
            if not val.exists():
                continue
            lam, delta = _refit_on_val(cfg, z)
            inv_thr = None                # lambda* - delta*, as the rule defines it
        row = score_epoch(pp["p_success"], pp["p_failure"], pp["p_invalid"], pp["true_labels"],
                          lam, delta, rule, inv_thr, p_val, y_val)
        row.update(run=run.name, epoch=int(art["epoch"]), train_trajectories=int(art["train_trajectories"]),
                   thresholds=thresholds, lambda_star=lam, delta_star=delta)
        rows.append(row)
    return rows

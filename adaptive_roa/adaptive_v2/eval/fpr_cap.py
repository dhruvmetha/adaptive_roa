"""FPR-capped decision thresholds (det rerun, Idea 6).

One rule labels a state from its outcome fractions (p_s, p_f, p_inv); classifiers
have p_f = 1 - p_s and p_inv = 0. Both sides of every comparison are rounded with
``conformal.decision.tie`` first.

- invalid (-2):   p_inv >= tau_inv, or the tied veto p_inv > 1 - min(tau_s, tau_f);
- success (1):    otherwise, p_s >= tau_s and p_f < tau_f;
- failure (0):    otherwise, p_f >= tau_f and p_s < tau_s;
- uncertain (-1): everything else, i.e. both labels fire or neither does.

A threshold of 1.01 turns its label off. ``fit`` picks (tau_s, tau_f, tau_inv) on a
fit split: for a success cap alpha_s it maximizes (TP + TN) / N subject to
FP / N_fail <= alpha_s and FN / N_succ <= alpha_f, with tau_s >= tau_s_min. A true
success is label == 1; every other label is a true failure, and abstentions stay
in both denominators. Ties go to fewer errors (FP + FN), then to the earliest
(tau_inv, tau_s, tau_f) in grid order. (1.01, 1.01) commits nothing, so a fit
always has a feasible point; a cap nothing else meets gives tau_s = 1.01.

With tau_s + tau_f >= 1 the rule is the two-sided lambda/delta rule at
lambda = (1 + tau_s - tau_f) / 2, delta = (tau_s + tau_f - 1) / 2 (``to_lambda_delta``);
below 1 delta is negative and both labels can fire. The converted values are
logged only.

Reference implementation: the better-thresh-opt session's fpr_sweep_floor.py and
scripts/det_rerun/analysis/tau_rule.py. ``fit`` reproduces the former's picks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from adaptive_roa.conformal.decision import tie

GRID = np.sort(tie(np.r_[np.linspace(0.01, 1.0, 100), 0.995, 0.999, 0.9999, 1.01]))
TAU_INV_GRID = tie(np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]))
OFF = 1.01                 # above every probability: that label never fires
TAU_S_MIN = 0.5            # a success claim needs at least half of all samples to succeed
ALPHA_F = 0.05             # failure cap: FN / N_succ
POSTHOC_ALPHAS = (0.0025, 0.005, 0.01)

# A classifier's continuous p is binned before the fit so the cell aggregation stays
# small. Only the fit sees the binned p; test labels use the raw probability.
_CLASSIFIER_MAX_DISTINCT = 2000
_CLASSIFIER_DECIMALS = 4
_EPS_CAP = 1e-12           # rates are compared with this slack, as the reference does
_EPS_FLOOR = 1e-9


def _probs(p_s, p_f=None, p_inv=None):
    """Arrays (p_s, p_f, p_inv); p_f defaults to 1 - p_s and p_inv to 0 (a classifier)."""
    ps = np.asarray(p_s, dtype=float)
    pf = 1.0 - ps if p_f is None else np.asarray(p_f, dtype=float)
    pi = np.zeros_like(ps) if p_inv is None else np.asarray(p_inv, dtype=float)
    return ps, pf, pi


def to_lambda_delta(tau_s: float, tau_f: float) -> tuple[float, float]:
    """The (lambda, delta) whose two-sided rule has these thresholds; delta can be negative."""
    return (1.0 + tau_s - tau_f) / 2.0, (tau_s + tau_f - 1.0) / 2.0


def predict(p_s, p_f, p_inv, tau_s: float, tau_f: float, tau_inv: float) -> np.ndarray:
    """Labels as int8: 1 success, 0 failure, -1 uncertain, -2 invalid."""
    ps, pf, pi = (tie(a) for a in _probs(p_s, p_f, p_inv))
    ts, tf, ti = float(tie(tau_s)), float(tie(tau_f)), float(tie(tau_inv))
    s_fire, f_fire = ps >= ts, pf >= tf
    inv = (pi >= ti) | (pi > float(tie(1.0 - min(ts, tf))))
    pred = np.full(ps.shape, -1, dtype=np.int8)
    pred[inv] = -2
    pred[s_fire & ~f_fire & ~inv] = 1
    pred[f_fire & ~s_fire & ~inv] = 0
    return pred


def both_fire(p_s, p_f, p_inv, tau_s: float, tau_f: float, tau_inv: float) -> np.ndarray:
    """States where both labels fire and the invalid test does not (labelled uncertain)."""
    ps, pf, _ = (tie(a) for a in _probs(p_s, p_f, p_inv))
    pred = predict(p_s, p_f, p_inv, tau_s, tau_f, tau_inv)
    return (ps >= float(tie(tau_s))) & (pf >= float(tie(tau_f))) & (pred != -2)


def acquisition_score(p_s, p_f, p_inv, tau_s: float, tau_f: float, tau_inv: float) -> np.ndarray:
    """Ranked-acquisition score; the lowest scores are acquired.

    0 for a state the rule labels uncertain or invalid. A committed state scores its
    distance to the nearest boundary that would uncommit it:
    success min(p_s - tau_s, tau_f - p_f, tau_inv - p_inv),
    failure min(p_f - tau_f, tau_s - p_s, tau_inv - p_inv).
    """
    ps, pf, pi = (tie(a) for a in _probs(p_s, p_f, p_inv))
    ts, tf, ti = float(tie(tau_s)), float(tie(tau_f)), float(tie(tau_inv))
    pred = predict(ps, pf, pi, ts, tf, ti)
    score = np.zeros(ps.shape, dtype=float)
    m = pred == 1
    score[m] = np.minimum.reduce([ps[m] - ts, tf - pf[m], ti - pi[m]])
    m = pred == 0
    score[m] = np.minimum.reduce([pf[m] - tf, ts - ps[m], ti - pi[m]])
    return score


def _aggregate(p_s, p_f, p_inv, y):
    """Distinct (p_s, p_f, p_inv) cells and their true-success / true-failure counts."""
    ps, pf, pi = _probs(p_s, p_f, p_inv)
    if (np.all(pi == 0) and np.allclose(ps + pf, 1.0, atol=1e-6)
            and len(np.unique(tie(ps))) > _CLASSIFIER_MAX_DISTINCT):
        ps = np.round(ps, _CLASSIFIER_DECIMALS)
        pf = 1.0 - ps
    key = np.stack([tie(ps), tie(pf), tie(pi)], 1)
    cells, ix = np.unique(key, axis=0, return_inverse=True)
    ix = ix.ravel()
    succ = np.asarray(y) == 1
    c1 = np.bincount(ix, weights=succ.astype(float), minlength=len(cells))
    c0 = np.bincount(ix, weights=(~succ).astype(float), minlength=len(cells))
    return cells, c1, c0


def grid_counts(cells, c1, c0, tau_inv: float, grid=GRID):
    """TP, FP, TN, FN on the whole (tau_s, tau_f) grid at one tau_inv; arrays [T_s, T_f]."""
    s, f, i = cells.T
    s_fire = s[None, :] >= grid[:, None]                          # [T, C]
    f_fire = f[None, :] >= grid[:, None]
    veto = tie(1.0 - np.minimum(grid[:, None], grid[None, :]))    # [T_s, T_f]
    inv_ti = i >= float(tie(tau_inv))
    T = len(grid)
    tp, fp, tn, fn = (np.zeros((T, T)) for _ in range(4))
    for a in range(T):
        inv = (i[None, :] > veto[a][:, None]) | inv_ti[None, :]  # [T_f, C]
        succ = s_fire[a][None, :] & ~f_fire & ~inv
        fail = f_fire & ~s_fire[a][None, :] & ~inv
        tp[a], fp[a], tn[a], fn[a] = succ @ c1, succ @ c0, fail @ c0, fail @ c1
    return tp, fp, tn, fn


def _rate(num, den):
    return num / den if den > 0 else np.zeros_like(num)


def _pick(counts, P, N, alpha_s, alpha_f, tau_s_min, grid=GRID):
    """Best feasible (tau_s, tau_f) index at one tau_inv, and its objective and errors."""
    tp, fp, tn, fn = counts
    ok = ((_rate(fp, N) <= alpha_s + _EPS_CAP) & (_rate(fn, P) <= alpha_f + _EPS_CAP)
          & (grid[:, None] >= tau_s_min - _EPS_FLOOR))
    obj = np.where(ok, (tp + tn) / (P + N), -1.0)
    flat = int(np.lexsort(((fp + fn).ravel(), -obj.ravel()))[0])
    a, b = np.unravel_index(flat, obj.shape)
    return int(a), int(b), float(obj[a, b]), float(fp[a, b] + fn[a, b])


def tau_inv_grid(p_inv) -> np.ndarray:
    """TAU_INV_GRID, or only OFF when the fit split carries no invalid mass."""
    return TAU_INV_GRID if np.any(np.asarray(p_inv) > 0) else np.array([OFF])


def fit(p_s, p_f, p_inv, y, alphas: Iterable[float], alpha_f: float = ALPHA_F,
        tau_s_min: float = TAU_S_MIN) -> dict[float, dict[str, Any]]:
    """Fit (tau_s, tau_f, tau_inv) per success cap; {alpha_s: thresholds and fit-split counts}."""
    ps, pf, pi = _probs(p_s, p_f, p_inv)
    y = np.asarray(y)
    P, N = int(np.sum(y == 1)), int(np.sum(y != 1))
    alphas = [float(a) for a in alphas]
    if P + N == 0:
        return {a: {"tau_s": OFF, "tau_f": OFF, "tau_inv": OFF, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                    "n_success": 0, "n_failure": 0} for a in alphas}
    cells, c1, c0 = _aggregate(ps, pf, pi, y)
    counts = {float(ti): grid_counts(cells, c1, c0, ti) for ti in tau_inv_grid(pi)}
    out = {}
    for a_s in alphas:
        best = None
        for ti, cnt in counts.items():           # earliest tau_inv wins a tie
            a, b, obj, err = _pick(cnt, P, N, a_s, alpha_f, tau_s_min)
            cand = (obj, -err, ti, a, b)
            if best is None or cand[:2] > best[:2]:
                best = cand
        _, _, ti, a, b = best
        tp, fp, tn, fn = (int(round(c[a, b])) for c in counts[ti])
        out[a_s] = {"tau_s": float(GRID[a]), "tau_f": float(GRID[b]), "tau_inv": float(ti),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_success": P, "n_failure": N}
    return out


def split_metrics(pred, y) -> dict[str, Any]:
    """Every rate over all states of a split: abstentions stay in the denominators."""
    from adaptive_roa.adaptive_v2.eval.full_roa import _conservative_metrics

    pred, y = np.asarray(pred), np.asarray(y)
    s, f = y == 1, y != 1
    n_s, n_f, n = int(s.sum()), int(f.sum()), len(y)
    tp = int(np.sum((pred == 1) & s))
    fp = int(np.sum((pred == 1) & f))
    tn = int(np.sum((pred == 0) & f))
    fn = int(np.sum((pred == 0) & s))
    nan = float("nan")
    return {
        "n_test": n, "n_success": n_s, "n_failure": n_f,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "success_recovery": tp / n_s if n_s else nan,
        "fpr_all_failures": fp / n_f if n_f else nan,
        "ffr_all_successes": fn / n_s if n_s else nan,
        "tnr_all_failures": tn / n_f if n_f else nan,
        "abstain_uncertain": float(np.mean(pred == -1)) if n else nan,
        "abstain_invalid": float(np.mean(pred == -2)) if n else nan,
        "success_precision": tp / (tp + fp) if tp + fp else nan,
        "selective_f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else nan,
        "conservative_f1": float(_conservative_metrics(np.asarray(pred), np.where(s, 1, -1))["f1"]),
        "n_success_claims": int(np.sum(pred == 1)),
    }


def load_val(npz, fit_k: int | None = None):
    """(p_s, p_f, p_inv, y, K) from predictions_val.npz.

    Endpoint models give their outcome fractions over all K draws, or over the first
    ``fit_k`` (the n_*_k<fit_k> keys) when asked; K is None for a classifier, whose
    stored p_success is used as is.
    """
    z = np.load(npz) if not hasattr(npz, "files") else npz
    y = np.asarray(z["true_labels"])
    if "n_success" not in z.files:
        ps = np.asarray(z["p_success"], dtype=float)
        return ps, 1.0 - ps, np.zeros_like(ps), y, None
    if fit_k is None:
        k = int(z["num_mc_samples"])
        n_s, n_f = z["n_success"].astype(int), z["n_failure"].astype(int)
    else:
        k = int(fit_k)
        if f"n_success_k{k}" not in z.files:
            raise KeyError(f"no n_success_k{k} in the val store")
        n_s, n_f = z[f"n_success_k{k}"].astype(int), z[f"n_failure_k{k}"].astype(int)
    return n_s / k, n_f / k, (k - n_s - n_f) / k, y, k


def score_run(run_dir, alphas: Iterable[float] = POSTHOC_ALPHAS, epochs: Iterable[int] | None = None,
              fit_k: int | None = None, alpha_f: float = ALPHA_F,
              tau_s_min: float = TAU_S_MIN) -> list[dict[str, Any]]:
    """One row per (epoch, alpha_s): fit on the stored val split, score the test split.

    Reads every epoch that has both predictions_val.npz and full_roa_per_point.npz.
    The fit uses all stored val draws unless ``fit_k`` names a stored subsample. When
    the epoch's artifacts record thresholds (an fpr_cap arm), the row carries them as
    run_tau_* for comparison.
    """
    run = Path(run_dir)
    keep = None if epochs is None else {int(e) for e in epochs}
    alphas = [float(a) for a in alphas]
    rows = []
    for ep_dir in sorted(run.glob("epoch_*")):
        m = re.fullmatch(r"epoch_(\d+)", ep_dir.name)
        if m is None or not ep_dir.is_dir():
            continue
        epoch = int(m.group(1))
        if keep is not None and epoch not in keep:
            continue
        val, test = ep_dir / "predictions_val.npz", ep_dir / "full_roa_per_point.npz"
        if not (val.exists() and test.exists()):
            continue
        vs, vf, vi, vy, k = load_val(val, fit_k)
        tz = np.load(test)
        ts_, tf_, ti_ = (np.asarray(tz[n], dtype=float) for n in ("p_success", "p_failure", "p_invalid"))
        ty = np.asarray(tz["true_labels"])
        art_path = ep_dir / "artifacts_v2.json"
        art = json.loads(art_path.read_text()) if art_path.exists() else {}
        thr = art.get("threshold_state") or {}
        fits = fit(vs, vf, vi, vy, alphas, alpha_f=alpha_f, tau_s_min=tau_s_min)
        for a_s in alphas:
            r = fits[a_s]
            pred = predict(ts_, tf_, ti_, r["tau_s"], r["tau_f"], r["tau_inv"])
            lam, dl = to_lambda_delta(r["tau_s"], r["tau_f"])
            row = {
                "run": run.name, "epoch": epoch, "train_trajectories": art.get("train_trajectories"),
                "alpha_s": a_s, "alpha_f": alpha_f, "val_k": k if k is not None else "classifier",
                "tau_s": r["tau_s"], "tau_f": r["tau_f"], "tau_inv": r["tau_inv"],
                "lambda_conv": lam, "delta_conv": dl,
                "val_n_success": r["n_success"], "val_n_failure": r["n_failure"],
                "val_tp": r["tp"], "val_fp": r["fp"], "val_tn": r["tn"], "val_fn": r["fn"],
                "tau_s_on_floor": bool(abs(r["tau_s"] - tau_s_min) < _EPS_FLOOR),
                **split_metrics(pred, ty),
                "n_both_fire": int(np.sum(both_fire(ts_, tf_, ti_, r["tau_s"], r["tau_f"], r["tau_inv"]))),
            }
            if thr.get("tau_s") is not None:
                row.update(run_alpha_s=thr.get("alpha_s"), run_tau_s=thr.get("tau_s"),
                           run_tau_f=thr.get("tau_f"), run_tau_inv=thr.get("tau_inv"))
            rows.append(row)
    return rows

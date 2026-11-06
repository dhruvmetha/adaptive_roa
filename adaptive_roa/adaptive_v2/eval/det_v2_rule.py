"""det_v2 decision rule and threshold opt (method sections 1 and 4).

One rule labels a state from (p_s, p_f); classifiers have p_f = 1 - p_s:

- S: p_s > tau_s and p_f <= tau_f
- F: p_f > tau_f and p_s <= tau_s
- A: otherwise. A_out when both pass, A_inv when neither does.

There is no tau_inv. Labels use the evaluator's codes: 1 = S, 0 = F, -1 = A_out
(outcome-uncertain), -2 = A_inv (invalidity abstention). Both sides of every
comparison are rounded with ``conformal.decision.tie`` first, so a final-state
p = n / K that equals a grid threshold compares equal.

``fit`` picks (tau_s, tau_f) on the grid {0.01, ..., 0.99}^2 by minimizing

- J_sel = (1 - w) MR + w AR, with MR = 0 when nothing is answered, or
- J_ver = (1 - w) FCR + w (1 - REC), ties broken by J_sel (ablation A1),

on the threshold-fit rows (held-out rows of the acquired data). A true success
is label == 1; every other label is a true failure. Classifiers search only
tau_s + tau_f >= 1, which loses nothing (section 1). Remaining ties go to the
smallest tau_s, then the smallest tau_f.

Each state reduces to the cell (i_s, i_f): the number of grid thresholds its p_s
and p_f exceed. The pair at grid indices (a, b) then passes S exactly when
i_s > a, and F when i_f > b, so one integer bincount over the cells and 2-D
prefix sums give every pair's counts; there is no loop over pairs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.conformal.decision import tie

GRID = tie(np.arange(1, 100) / 100.0)      # tau values 0.01 .. 0.99
N_GRID = len(GRID)
OBJECTIVES = ("j_sel", "j_ver")
REGIONS = ("inv", "out")
S, F, A_OUT, A_INV = 1, 0, -1, -2


def _arrays(p_s, p_f=None):
    ps = np.asarray(p_s, dtype=float)
    pf = 1.0 - ps if p_f is None else np.asarray(p_f, dtype=float)
    return tie(ps), tie(pf)


def predict(p_s, p_f, tau_s: float, tau_f: float) -> np.ndarray:
    """Labels as int8: 1 S, 0 F, -1 A_out (both pass), -2 A_inv (neither passes)."""
    ps, pf = _arrays(p_s, p_f)
    ts, tf = float(tie(tau_s)), float(tie(tau_f))
    s_pass, f_pass = ps > ts, pf > tf
    out = np.full(ps.shape, A_INV, dtype=np.int8)
    out[s_pass & f_pass] = A_OUT
    out[s_pass & ~f_pass] = S
    out[f_pass & ~s_pass] = F
    return out


def cell_index(p) -> np.ndarray:
    """Number of grid thresholds strictly below each tie-rounded p (0 .. 99)."""
    return np.searchsorted(GRID, tie(np.asarray(p, dtype=float)), side="left")


def cell_keys(p_s, p_f=None) -> np.ndarray:
    """Each state's cell (i_s, i_f) as one integer, i_s * 100 + i_f."""
    ps, pf = _arrays(p_s, p_f)
    return cell_index(ps) * (N_GRID + 1) + cell_index(pf)


def grid_counts(p_s, p_f, y, weights=None) -> dict[str, np.ndarray]:
    """TP, FP, FN (successes called F), TN on the whole grid; arrays [99, 99] indexed (tau_s, tau_f).

    ``weights`` (one per row) gives weighted counts as floats, e.g. a bootstrap
    resample's multiplicities; without it the counts are exact integers.
    """
    return counts_from_keys(cell_keys(p_s, p_f), np.asarray(y) == 1, weights)


def counts_from_keys(key: np.ndarray, succ: np.ndarray, weights=None) -> dict[str, np.ndarray]:
    """grid_counts from precomputed cell_keys and success flags."""
    size = N_GRID + 1
    if weights is None:
        c1 = np.bincount(key[succ], minlength=size * size).reshape(size, size).astype(np.int64)
        c0 = np.bincount(key[~succ], minlength=size * size).reshape(size, size).astype(np.int64)
        n, n_succ = np.int64(len(succ)), np.int64(succ.sum())
    else:
        wt = np.asarray(weights, dtype=float)
        c1 = np.bincount(key[succ], weights=wt[succ], minlength=size * size).reshape(size, size)
        c0 = np.bincount(key[~succ], weights=wt[~succ], minlength=size * size).reshape(size, size)
        n, n_succ = wt.sum(), wt[succ].sum()

    def s_and_f(c):
        # cum[i, j] = #{i_s <= i, i_f <= j}
        cum = c.cumsum(0).cumsum(1)
        a = np.arange(N_GRID)
        low = cum[np.ix_(a, a)]                       # i_s <= a, i_f <= b
        in_s = cum[-1, a][None, :] - low              # i_s > a,  i_f <= b
        in_f = cum[a, -1][:, None] - low              # i_s <= a, i_f > b
        return in_s, in_f

    tp, fn = s_and_f(c1)
    fp, tn = s_and_f(c0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n, "n_success": n_succ, "n_failure": n - n_succ}


def objectives(counts: dict[str, np.ndarray], w: float) -> dict[str, np.ndarray]:
    """MR, AR, FCR, REC, J_sel and J_ver on the grid."""
    tp, fp, fn, tn = (counts[k].astype(float) for k in ("tp", "fp", "fn", "tn"))
    n, n_succ = float(counts["n"]), float(counts["n_success"])
    answered = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        mr = np.where(answered > 0, (fp + fn) / np.maximum(answered, 1), 0.0)
        fcr = np.where(tp + fp > 0, fp / np.maximum(tp + fp, 1), 0.0)
    ar = (n - answered) / n if n > 0 else np.zeros_like(answered)
    rec = tp / n_succ if n_succ > 0 else np.zeros_like(tp)
    j_sel = (1.0 - w) * mr + w * ar
    j_ver = (1.0 - w) * fcr + w * (1.0 - rec)
    return {"mr": mr, "ar": ar, "fcr": fcr, "rec": rec, "j_sel": j_sel, "j_ver": j_ver}


def region_mask(region: str | None) -> np.ndarray:
    """[99, 99] grid pairs in a regime: "inv" = tau_s + tau_f >= 1 (no A_out), "out" = tau_s + tau_f < 1."""
    a, b = np.meshgrid(np.arange(N_GRID), np.arange(N_GRID), indexing="ij")
    inv = (a + 1) + (b + 1) >= 100             # in hundredths, so exact
    if region is None:
        return np.ones_like(inv)
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS} or None, got {region!r}")
    return inv if region == "inv" else ~inv


def region_of(tau_s: float, tau_f: float) -> str:
    return "inv" if round(tau_s * 100) + round(tau_f * 100) >= 100 else "out"


def fit(p_s, p_f, y, *, w: float = 0.1, objective: str = "j_sel",
        classifier: bool = False, region: str | None = None) -> dict[str, Any]:
    """tau* = (tau_s, tau_f) minimizing the objective; returns the pair and its fit-split diagnostics.

    ``region`` restricts the search to one regime of region_mask (ablation A9).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    y = np.asarray(y)
    if len(y) == 0:
        raise ValueError("threshold opt needs at least one fit row")
    counts = grid_counts(p_s, p_f, y)
    obj = objectives(counts, w)
    primary = obj[objective]
    secondary = obj["j_sel"] if objective == "j_ver" else np.zeros_like(primary)
    a, b = np.meshgrid(np.arange(N_GRID), np.arange(N_GRID), indexing="ij")
    ok = region_mask(region)
    if classifier:
        ok = ok & region_mask("inv")             # tau_s + tau_f >= 1
        if not ok.any():
            raise ValueError("a classifier's search is tau_s + tau_f >= 1; region='out' leaves no pair")
    # Round so values equal in exact arithmetic tie here too; then smallest tau_s, tau_f.
    keys = (b[ok].ravel(), a[ok].ravel(), np.round(secondary[ok], 12).ravel(), np.round(primary[ok], 12).ravel())
    best = int(np.lexsort(keys)[0])
    ia, ib = int(a[ok].ravel()[best]), int(b[ok].ravel()[best])
    pick = {k: int(counts[k][ia, ib]) for k in ("tp", "fp", "fn", "tn")}
    labels = predict(p_s, p_f, GRID[ia], GRID[ib])
    return {
        "tau_s": float(GRID[ia]), "tau_f": float(GRID[ib]),
        "objective": objective, "w": float(w),
        "j": float(primary[ia, ib]), "j_sel": float(obj["j_sel"][ia, ib]), "j_ver": float(obj["j_ver"][ia, ib]),
        "mr": float(obj["mr"][ia, ib]), "ar": float(obj["ar"][ia, ib]),
        "fcr": float(obj["fcr"][ia, ib]), "rec": float(obj["rec"][ia, ib]),
        "ar_inv": float(np.mean(labels == A_INV)), "ar_out": float(np.mean(labels == A_OUT)),
        **pick,
        "n": int(counts["n"]), "n_success": int(counts["n_success"]), "n_failure": int(counts["n_failure"]),
    }

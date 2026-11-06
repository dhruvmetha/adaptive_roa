"""Section 6 certificates on stored predictions (method tab, sections 6a and 6c).

All of it runs after training, on the per-epoch prediction store: D_val fixes a testing
order, D_cal certifies, D_test reports. Nothing here touches a running job.

Verdicts use the det_v2 rule's strict edges:
    S(t_s, t_f) = {p_s > t_s and p_f <= t_f}      F(u_s, u_f) = {p_f > u_f and p_s <= u_s}
A classifier has one real edge per verdict (p_f = 1 - p_s): S_h = {p_s > h}, F_l = {p_s <= l}.

6a (verification, main): P(FCR(S_hat) <= alpha) >= 1 - delta. Candidates are tested in a
   D_val order on D_cal with the Clopper-Pearson bound and the walk stops at the first
   failure. The final verdicts are S = S_hat, F = F_0 minus S_hat (F_0 is tau*'s F, no claim), A the rest.
6c (two-sided): the 6a walk at delta/2 for S, then a walk at delta/2 for F on the ROA loss
   L(F) = P(F | success) <= beta. The final verdicts are S = S_hat, F = F_hat minus S_hat.

Testing orders (fixed on D_val):
  classifier S: h walks down the classifier grid from the largest h whose S holds
                >= start_min D_val states (nested, each set contains the one before).
  classifier F: l walks up the grid from the smallest l whose F is nonempty on D_val.
  final-state S: Pareto testing. Among pairs whose S holds >= start_min D_val states, keep the front of
                lower D_val FCR and higher D_val REC, sorted by D_val REC ascending; pairs
                giving the same D_val set count once, as the tightest (largest t_s, then smallest t_f).
  final-state F: front of lower D_val L and higher D_val failure recall, sorted by failure
                recall ascending, starting at the first nonempty set; duplicates count once
                as the tightest (largest u_f, then smallest u_s).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist

from adaptive_roa.adaptive_v2.eval import det_v2_rule

GRID = det_v2_rule.GRID                                   # 0.01 .. 0.99
N_GRID = det_v2_rule.N_GRID


def classifier_grid() -> np.ndarray:
    """The plain grid plus log-spaced ends, 10 per decade: t in [1e-6, 0.01) and 1 - t."""
    low = 10.0 ** (-6 + np.arange(40) / 10)               # 1e-6 .. 10^-2.1
    high = 1.0 - low[::-1]                                # 1 - 10^-2.1 .. 1 - 1e-6
    return det_v2_rule.tie(np.concatenate([low, GRID, high]))


CLF_GRID = classifier_grid()


def m_min(alpha: float, delta: float) -> int:
    """Smallest set that can pass the Clopper-Pearson FCR test with zero failures."""
    return int(math.ceil(math.log(delta) / math.log(1.0 - alpha)))


def cp_upper(k, n, delta: float) -> np.ndarray:
    """One-sided (1 - delta) Clopper-Pearson upper bound on a rate with k events out of n (1 when n = 0 or k = n)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.ones(np.broadcast(k, n).shape)
    ok = (n > 0) & (k < n)
    out[ok] = beta_dist.ppf(1.0 - delta, k[ok] + 1.0, n[ok] - k[ok])
    return out


# --- loading -------------------------------------------------------------------------------------

def split_probs(z) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """(p_s, p_f or None for a classifier, success flags) from a predictions_{val,cal}.npz or full_roa_per_point.npz.

    Final-state and FM splits store integer counts (exact k/K); a classifier or Part-X stores p_success only.
    """
    y = np.asarray(z["true_labels"]) == 1
    if "n_success" in z.files:
        n = np.stack([z["n_success"], z["n_failure"], z["n_invalid"]]).astype(float)
        k = float(z["num_mc_samples"]) if "num_mc_samples" in z.files else n.sum(0).max()
        return det_v2_rule.tie(n[0] / k), det_v2_rule.tie(n[1] / k), y
    ps = np.asarray(z["p_success"], dtype=float)
    pf = np.asarray(z["p_failure"], dtype=float) if "p_failure" in z.files else None
    return ps, pf, y


# --- membership ----------------------------------------------------------------------------------

def in_S(ps, pf, t_s, t_f):
    ps = det_v2_rule.tie(ps)
    if pf is None:
        return ps > det_v2_rule.tie(t_s)
    return (ps > det_v2_rule.tie(t_s)) & (det_v2_rule.tie(pf) <= det_v2_rule.tie(t_f))


def in_F(ps, pf, u_s, u_f):
    ps = det_v2_rule.tie(ps)
    if pf is None:
        return ps <= det_v2_rule.tie(u_s)
    return (det_v2_rule.tie(pf) > det_v2_rule.tie(u_f)) & (ps <= det_v2_rule.tie(u_s))


# --- grid tables ---------------------------------------------------------------------------------

def _pair_tables(ps, pf, y, weights=None):
    """Per (t_s, t_f) grid pair: S size and successes, F size and successes (det_v2_rule grid, [99, 99])."""
    c = det_v2_rule.grid_counts(ps, pf, y, weights)
    return {"S_n": c["tp"] + c["fp"], "S_succ": c["tp"], "F_n": c["fn"] + c["tn"], "F_succ": c["fn"],
            "n": c["n"], "n_succ": c["n_success"]}


def _set_keys(ps, pf, n_rows, seed=0):
    """Two random integer weightings per state; equal sums on a pair mean the same D_val set (exact in float64)."""
    rng = np.random.default_rng(seed)
    keys = []
    for _ in range(2):
        w = rng.integers(1, 2 ** 30, n_rows).astype(float)
        keys.append(_pair_tables(ps, pf, np.ones(n_rows, bool), weights=w))
    return keys


def _pareto(cost: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Indices of points no other point beats on both (lower cost, higher gain)."""
    order = np.lexsort((-gain, cost))                      # cost ascending, then gain descending
    keep, best = [], -np.inf
    for i in order:
        if gain[i] > best:
            keep.append(i)
            best = gain[i]
    return np.array(keep, dtype=int)


# --- testing orders ------------------------------------------------------------------------------

def order_S_final_state(val, start_min: int) -> list[tuple[float, float]]:
    """6a order for a final-state model: Pareto front on D_val, sorted by D_val REC ascending."""
    ps, pf, y = val
    T = _pair_tables(ps, pf, y)
    a, b = np.meshgrid(np.arange(N_GRID), np.arange(N_GRID), indexing="ij")
    n, s = T["S_n"].ravel(), T["S_succ"].ravel()
    ok = n >= start_min
    if not ok.any():
        return []
    k1, k2 = (K["S_n"].ravel() for K in _set_keys(ps, pf, len(y)))
    cand = np.flatnonzero(ok)
    # one pair per distinct D_val set: the tightest, largest t_s then smallest t_f
    tight = cand[np.lexsort((b.ravel()[cand], -a.ravel()[cand]))]
    seen, uniq = set(), []
    for i in tight:
        sig = (k1[i], k2[i])
        if sig not in seen:
            seen.add(sig)
            uniq.append(i)
    uniq = np.array(uniq)
    fcr = (n[uniq] - s[uniq]) / n[uniq]
    rec = s[uniq] / max(T["n_succ"], 1)
    front = uniq[_pareto(fcr, rec)]
    front = front[np.argsort(s[front] / max(T["n_succ"], 1), kind="stable")]
    return [(float(GRID[a.ravel()[i]]), float(GRID[b.ravel()[i]])) for i in front]


def order_S_classifier(val, start_min: int) -> list[tuple[float, None]]:
    ps, _, _ = val
    ps = np.sort(det_v2_rule.tie(ps))
    sizes = len(ps) - np.searchsorted(ps, CLF_GRID, side="right")     # |{p_s > h}|
    ok = np.flatnonzero(sizes >= start_min)
    if not len(ok):
        return []
    start = ok.max()                                       # largest h with enough states
    return [(float(CLF_GRID[i]), None) for i in range(start, -1, -1)]


def order_F_final_state(val) -> list[tuple[float, float]]:
    """6c F order: Pareto front of lower D_val ROA loss and higher D_val failure recall, by failure recall ascending."""
    ps, pf, y = val
    T = _pair_tables(ps, pf, y)
    a, b = np.meshgrid(np.arange(N_GRID), np.arange(N_GRID), indexing="ij")
    n, s = T["F_n"].ravel(), T["F_succ"].ravel()
    ok = n > 0
    if not ok.any():
        return []
    k1, k2 = (K["F_n"].ravel() for K in _set_keys(ps, pf, len(y)))
    cand = np.flatnonzero(ok)
    tight = cand[np.lexsort((a.ravel()[cand], -b.ravel()[cand]))]      # largest u_f, then smallest u_s
    seen, uniq = set(), []
    for i in tight:
        sig = (k1[i], k2[i])
        if sig not in seen:
            seen.add(sig)
            uniq.append(i)
    uniq = np.array(uniq)
    n_fail = max(T["n"] - T["n_succ"], 1)
    loss = s[uniq] / max(T["n_succ"], 1)
    frec = (n[uniq] - s[uniq]) / n_fail
    front = uniq[_pareto(loss, frec)]
    front = front[np.argsort((n[front] - s[front]) / n_fail, kind="stable")]
    return [(float(GRID[a.ravel()[i]]), float(GRID[b.ravel()[i]])) for i in front]


def order_F_classifier(val) -> list[tuple[float, None]]:
    ps, _, _ = val
    ps = np.sort(det_v2_rule.tie(ps))
    sizes = np.searchsorted(ps, CLF_GRID, side="right")                 # |{p_s <= l}|
    ok = np.flatnonzero(sizes > 0)
    if not len(ok):
        return []
    return [(float(CLF_GRID[i]), None) for i in range(ok.min(), len(CLF_GRID))]


# --- walks ---------------------------------------------------------------------------------------

def walk_S(order, cal, alpha: float, delta: float) -> dict[str, Any]:
    """Test H_S: FCR(S) > alpha along `order` on D_cal; stop at the first failure; return the last pass."""
    ps, pf, y = cal
    last, steps = None, 0
    for t_s, t_f in order:
        m = in_S(ps, pf, t_s, t_f)
        n, fp = int(m.sum()), int((m & ~y).sum())
        if cp_upper(fp, n, delta) > alpha:
            break
        last, steps = {"t_s": t_s, "t_f": t_f, "cal_n": n, "cal_fp": fp, "cal_bound": float(cp_upper(fp, n, delta))}, steps + 1
    return {"passed": last is not None, "steps": steps, "n_order": len(order), **(last or {})}


def walk_F(order, cal, beta_: float, delta: float) -> dict[str, Any]:
    """Test H_F: L(F) > beta along `order` on D_cal (k successes in F of n_succ); stop at the first failure."""
    ps, pf, y = cal
    n_succ = int(y.sum())
    last, steps = None, 0
    for u_s, u_f in order:
        m = in_F(ps, pf, u_s, u_f)
        k = int((m & y).sum())
        if cp_upper(k, n_succ, delta) > beta_:
            break
        last, steps = {"u_s": u_s, "u_f": u_f, "cal_n": int(m.sum()), "cal_succ": k,
                       "cal_bound": float(cp_upper(k, n_succ, delta))}, steps + 1
    return {"passed": last is not None, "steps": steps, "n_order": len(order), **(last or {})}


# --- reporting -----------------------------------------------------------------------------------

def report(test, S_thr=None, F_thr=None, tau=None) -> dict[str, float | int | None]:
    """Test-set view of the final verdicts: S = S_hat; F = (F_hat, else tau*'s F_0) minus S_hat; A the rest."""
    ps, pf, y = test
    n, n_succ = len(y), int(y.sum())
    S = in_S(ps, pf, *S_thr) if S_thr else np.zeros(n, bool)
    if F_thr:
        F = in_F(ps, pf, *F_thr)
    elif tau is not None and tau[0] is not None:          # F_0 under the det_v2 rule at tau*
        F = det_v2_rule.predict(ps, pf, tau[0], tau[1]) == det_v2_rule.F
    else:
        F = np.zeros(n, bool)
    F = F & ~S
    nS, fp = int(S.sum()), int((S & ~y).sum())
    nF, fs = int(F.sum()), int((F & y).sum())
    return {
        "test_S": nS, "test_fcr": fp / nS if nS else None, "test_rec": (nS - fp) / n_succ if n_succ else None,
        "test_F": nF, "test_roa_loss": fs / n_succ if n_succ else None,
        "test_fail_rec": (nF - fs) / max(n - n_succ, 1), "test_abstain": 1.0 - (nS + nF) / n,
    }


def certify(val, cal, test, *, classifier: bool, tau=None, alpha=0.05, beta_=0.05, delta=0.05) -> dict[str, dict]:
    """6a and 6c for one epoch. val/cal/test are split_probs tuples; tau = (tau_s*, tau_f*) or None (Part-X)."""
    out = {}
    if classifier:
        val = (val[0], None, val[2]); cal = (cal[0], None, cal[2]); test = (test[0], None, test[2])
    order_S = order_S_classifier if classifier else order_S_final_state

    # 6a: S walk at delta, F = F_0 minus S_hat (no claim)
    mm = m_min(alpha, delta)
    s6a = walk_S(order_S(val, 2 * mm), cal, alpha, delta)
    S6a = (s6a["t_s"], s6a["t_f"]) if s6a["passed"] else None
    out["6a"] = {**s6a, "alpha": alpha, "delta": delta, "m_min": mm, **report(test, S6a, None, tau)}

    # 6c: S walk at delta/2, F walk at delta/2
    mm2 = m_min(alpha, delta / 2)
    s6c = walk_S(order_S(val, 2 * mm2), cal, alpha, delta / 2)
    f6c = walk_F(order_F_classifier(val) if classifier else order_F_final_state(val), cal, beta_, delta / 2)
    S6c = (s6c["t_s"], s6c["t_f"]) if s6c["passed"] else None
    F6c = (f6c["u_s"], f6c["u_f"]) if f6c["passed"] else None
    out["6c"] = {"S": s6c, "F": f6c, "alpha": alpha, "beta": beta_, "delta": delta, "m_min": mm2,
                 **report(test, S6c, F6c, None)}
    return out

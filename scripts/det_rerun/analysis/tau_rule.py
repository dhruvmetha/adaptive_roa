"""Separate success / failure thresholds in place of (lambda, delta).

Rule (user proposal, 2026-10-31):
  success fires if p_s >= tau_s, failure fires if p_f >= tau_f;
  exactly one fires -> that label; both or neither -> outcome-uncertain;
  invalid if p_inv > 1 - min(tau_s, tau_f).
The veto generalises D28: with tau_s = lambda + delta and tau_f = 1 - lambda + delta it is
max(lambda, 1 - lambda) - delta, and it can only flag a state where neither fires
(p_s + p_f < min(tau_s, tau_f)). The (lambda, delta) rule covers tau_s + tau_f > 1;
tau_s + tau_f <= 1 is new, the only region where both can fire.

Fitted by grid search over [0.05, 0.95]^2 (step 0.01, no sum constraint) with the
optimizer's objective w * misclass_rate + (1 - w) * unknown_rate
(adaptive_roa/conformal/lambda_optimizer.py), unknown = uncertain + invalid.
"""
from __future__ import annotations

import numpy as np

from adaptive_roa.conformal.decision import tie

GRID = tie(np.linspace(0.05, 0.95, 91))


def predict(ps, pf, pi, tau_s: float, tau_f: float) -> tuple[np.ndarray, int]:
    """Labels 1 success, 0 failure, -1 uncertain, -2 invalid; and the both-fire count."""
    ps, pf, pi = tie(ps), tie(pf), tie(pi)
    ts, tf = float(tie(tau_s)), float(tie(tau_f))
    s_fire, f_fire = ps >= ts, pf >= tf
    inv = pi > float(tie(1.0 - min(ts, tf)))
    pred = np.full(len(ps), -1, dtype=np.int8)
    pred[inv] = -2
    pred[s_fire & ~f_fire & ~inv] = 1
    pred[f_fire & ~s_fire & ~inv] = 0
    return pred, int(np.count_nonzero(s_fire & f_fire & ~inv))


def _loss(mis, conf, n, w):
    err = np.where(conf > 0, mis / np.maximum(conf, 1), 0.0)
    unk = (n - conf) / n
    return w * err + (1 - w) * unk, err, unk


def _fit_general(ps, pf, pi, y, w, grid):
    """Count-valued probabilities: aggregate identical (p_s, p_f, p_inv) cells first."""
    key = np.stack([tie(ps), tie(pf), tie(pi)], 1)
    cells, inv_idx = np.unique(key, axis=0, return_inverse=True)
    inv_idx = inv_idx.ravel()
    c1 = np.bincount(inv_idx, weights=(y == 1).astype(float), minlength=len(cells))
    c0 = np.bincount(inv_idx, weights=(y != 1).astype(float), minlength=len(cells))
    s, f, i = cells[:, 0], cells[:, 1], cells[:, 2]
    T = len(grid)
    s_fire = s[None, :] >= grid[:, None]                         # (T, C) over tau_s
    f_fire = f[None, :] >= grid[:, None]                         # (T, C) over tau_f
    mins = tie(1.0 - np.minimum(grid[:, None], grid[None, :]))   # (T, T)
    best = None
    for a in range(T):
        inv = i[None, :] > mins[a][:, None]                      # (T_f, C)
        succ = s_fire[a][None, :] & ~f_fire & ~inv
        fail = f_fire & ~s_fire[a][None, :] & ~inv
        mis = succ @ c0 + fail @ c1
        conf = succ @ (c0 + c1) + fail @ (c0 + c1)
        loss, err, unk = _loss(mis, conf, len(y), w)
        b = int(np.lexsort((unk, loss))[0])
        cand = (loss[b], unk[b], float(grid[a]), float(grid[b]), err[b])
        if best is None or cand[:2] < best[:2]:
            best = cand
    return best


def fit(ps, pf, pi, y, w: float, grid=GRID) -> dict:
    """Grid-search (tau_s, tau_f) on a fit set; returns the optimum and its fit-set terms."""
    ps, pf, pi, y = (np.asarray(a) for a in (ps, pf, pi, y))
    if np.all(pi == 0) and np.allclose(ps + pf, 1.0, atol=1e-6) and len(np.unique(tie(ps))) > 2000:
        # A continuous classifier: bin p_s to 1e-4 so the cell aggregation stays small.
        ps = np.round(ps, 4)
        pf = 1.0 - ps
    loss, unk, ts, tf, err = _fit_general(ps, pf, pi, y, w, grid)
    return {"tau_s": ts, "tau_f": tf, "loss": float(loss), "fit_err": float(err), "fit_unk": float(unk),
            "sum_le_1": bool(ts + tf <= 1.0 + 1e-12)}

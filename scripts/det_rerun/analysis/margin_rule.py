"""Margin rule: commit on the gap between the two outcome masses.

  success if p_s - p_f >= t_s, failure if p_f - p_s >= t_f, otherwise outcome-uncertain.
Invalid mass is not vetoed separately; it shrinks both masses and so the margin.
This is the ordering behind the risk-coverage curves (confidence = |p_s - p_f|).
For a classifier (p_f = 1 - p_s) it is a band rule on p_s with edges (1 + t_s) / 2 and
(1 - t_f) / 2. Fitted on a grid t_s, t_f in [0, 0.95] (step 0.01, t_s + t_f > 0) with the
optimizer's objective w * misclass_rate + (1 - w) * unknown_rate.
"""
from __future__ import annotations

import numpy as np

from adaptive_roa.conformal.decision import tie

GRID = tie(np.linspace(0.0, 0.95, 96))


def predict(ps, pf, t_s: float, t_f: float) -> np.ndarray:
    m = tie(np.asarray(ps, float) - np.asarray(pf, float))
    pred = np.full(len(m), -1, dtype=np.int8)
    pred[m >= tie(t_s)] = 1
    pred[(-m >= tie(t_f)) & (pred != 1)] = 0
    return pred


def fit(ps, pf, y, w: float, grid=GRID) -> dict:
    m = tie(np.round(np.asarray(ps, float) - np.asarray(pf, float), 4))
    cells, idx = np.unique(m, return_inverse=True)
    c1 = np.bincount(idx, weights=(np.asarray(y) == 1).astype(float), minlength=len(cells))
    c0 = np.bincount(idx, weights=(np.asarray(y) != 1).astype(float), minlength=len(cells))
    n = float(len(m))
    succ = cells[None, :] >= grid[:, None]                  # (T_s, C)
    fail = -cells[None, :] >= grid[:, None]                 # (T_f, C)
    best = None
    for a in range(len(grid)):
        f = fail & ~succ[a][None, :]
        mis = (succ[a] @ c0) + f @ c1
        conf = (succ[a] @ (c0 + c1)) + f @ (c0 + c1)
        err = np.where(conf > 0, mis / np.maximum(conf, 1), 0.0)
        unk = (n - conf) / n
        loss = w * err + (1 - w) * unk
        ok = (grid[a] + grid) > 0
        loss = np.where(ok, loss, np.inf)
        b = int(np.lexsort((unk, loss))[0])
        cand = (loss[b], unk[b], float(grid[a]), float(grid[b]))
        if best is None or cand[:2] < best[:2]:
            best = cand
    return {"t_s": best[2], "t_f": best[3], "loss": float(best[0]), "fit_unk": float(best[1])}

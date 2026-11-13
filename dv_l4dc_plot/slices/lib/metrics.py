"""Ground-truth clarity, TERA-FM match and the rank score of one slice.

Labels are int (ny, nx), 1 = success, 0 = fail. Decisions are int8 (ny, nx), 1 = Success,
0 = Fail, -1 = Abstain.

The in-box rule: a slice npz may carry ``box_x`` / ``box_y`` (float[2], NaN = unbounded side),
the dataset's start-sampling range along each axis. A cell is in the box when its centre lies
in both ranges (inclusive). Because the box is axis-aligned and the grid is rectangular, the
in-box cells form one sub-rectangle, and the primary metrics are computed on that crop.
``slice_metrics`` returns those primary numbers plus full-slice copies with a ``_full`` suffix,
and ``rank_score`` uses the in-box numbers. Without box keys the whole slice is in the box.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import ndimage

CC_MIN_FRAC = 0.001           # components smaller than 0.1% of cells are speckle
_EIGHT = np.ones((3, 3), dtype=int)

GT_KEYS = ["n_cells", "succ_frac", "balance", "boundary_len", "boundary_frac", "n_cc_succ", "n_cc_fail",
           "speckle_frac", "gt_clarity"]
TERA_KEYS = ["R_s", "R_f", "abstain_frac", "FDR", "FOR", "n_wrong_s", "n_wrong_f", "wrong_frac", "tera_match"]


def _div(a, b) -> float:
    return float(a) / float(b) if b else float("nan")


def gt_metrics(labels) -> dict:
    """Clarity of a ground-truth label image (see the README for each definition)."""
    L = np.asarray(labels).astype(np.int8)
    if L.ndim != 2:
        raise ValueError(f"labels must be 2-D, got {L.shape}")
    ny, nx = L.shape
    n = L.size
    if n == 0:
        return {k: float("nan") for k in GT_KEYS} | {"n_cells": 0}
    succ = L == 1
    sf = float(succ.mean())
    balance = 1.0 - abs(2.0 * sf - 1.0)
    dh = L[:, 1:] != L[:, :-1]
    dv = L[1:, :] != L[:-1, :]
    boundary_len = (int(dh.sum()) + int(dv.sum())) / nx
    edge = np.zeros_like(succ)
    edge[:, 1:] |= dh
    edge[:, :-1] |= dh
    edge[1:, :] |= dv
    edge[:-1, :] |= dv
    min_size = CC_MIN_FRAC * n
    n_cc, speckle = {}, 0
    for name, m in (("succ", succ), ("fail", ~succ)):
        lab, k = ndimage.label(m, structure=_EIGHT)
        sizes = np.bincount(lab.ravel())[1:] if k else np.zeros(0, int)
        big = sizes >= min_size
        n_cc[name] = int(big.sum())
        speckle += int(sizes[~big].sum())
    speckle_frac = speckle / n
    clarity = math.sqrt(balance) * (1.0 - speckle_frac) / max(1, n_cc["succ"] + n_cc["fail"] - 1)
    return {"n_cells": int(n), "succ_frac": sf, "balance": balance, "boundary_len": float(boundary_len),
            "boundary_frac": float(edge.mean()), "n_cc_succ": n_cc["succ"], "n_cc_fail": n_cc["fail"],
            "speckle_frac": float(speckle_frac), "gt_clarity": float(clarity)}


def tera_metrics(labels, decision) -> dict:
    """How TERA-FM's certified decisions match the ground truth. NaN where a denominator is 0."""
    L = np.asarray(labels)
    d = np.asarray(decision)
    if L.shape != d.shape:
        raise ValueError(f"labels {L.shape} and decision {d.shape} differ in shape")
    if not np.isin(d, (-1, 0, 1)).all():
        raise ValueError("decision values must be in {-1, 0, 1}")
    gs, gf = L == 1, L != 1
    S, F = d == 1, d == 0
    n_wrong_s = int((S & gf).sum())
    n_wrong_f = int((F & gs).sum())
    n = L.size
    R_s = _div((S & gs).sum(), gs.sum())
    R_f = _div((F & gf).sum(), gf.sum())
    wrong_frac = _div(n_wrong_s + n_wrong_f, n)
    rr = [r for r in (R_s, R_f) if not math.isnan(r)]
    match = float(np.clip(np.mean(rr) - 5.0 * wrong_frac, 0.0, 1.0)) if rr and n else float("nan")
    return {"R_s": R_s, "R_f": R_f, "abstain_frac": _div((d == -1).sum(), n),
            "FDR": _div(n_wrong_s, S.sum()), "FOR": _div(n_wrong_f, F.sum()),
            "n_wrong_s": n_wrong_s, "n_wrong_f": n_wrong_f, "wrong_frac": wrong_frac, "tera_match": match}


def rank_score(gt_m: dict, tera_m: dict) -> float:
    """gt_clarity * tera_match (the in-box numbers when given slice_metrics output)."""
    return float(gt_m["gt_clarity"]) * float(tera_m["tera_match"])


# --- in-box handling -----------------------------------------------------------------------------

def _range_mask(c, rng) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    if rng is None:
        return np.ones(c.shape, bool)
    lo, hi = (float(v) for v in rng)
    m = np.ones(c.shape, bool)
    if not math.isnan(lo):
        m &= c >= lo
    if not math.isnan(hi):
        m &= c <= hi
    return m


def box_crop(sl: dict) -> tuple[slice, slice]:
    """(row slice, column slice) of the in-box cells of slice dict ``sl`` (whole slice without box keys).

    Empty slices when no cell centre lies in the box.
    """
    mx = _range_mask(sl["xs"], sl.get("box_x"))
    my = _range_mask(sl["ys"], sl.get("box_y"))

    def span(m):
        idx = np.flatnonzero(m)
        if idx.size == 0:
            return slice(0, 0)
        if idx[-1] - idx[0] + 1 != idx.size:
            raise ValueError("in-box cells are not contiguous; are xs/ys ascending?")
        return slice(int(idx[0]), int(idx[-1]) + 1)

    return span(my), span(mx)


def has_box(sl: dict) -> bool:
    return sl.get("box_x") is not None or sl.get("box_y") is not None


def slice_metrics(sl: dict, decision=None) -> tuple[dict, dict | None]:
    """(gt_m, tera_m) for a loaded slice: in-box primary numbers plus ``<key>_full`` copies.

    gt_m also carries ``in_box_frac``; tera_m (None without a decision) also carries
    ``rank_score`` (in-box) and ``rank_score_full``.
    """
    L = np.asarray(sl["labels"])
    rs, cs = box_crop(sl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g_in, g_full = gt_metrics(L[rs, cs]), gt_metrics(L)
        gt_m = dict(g_in) | {f"{k}_full": v for k, v in g_full.items()}
        gt_m["in_box_frac"] = g_in["n_cells"] / L.size if L.size else float("nan")
        if decision is None:
            return gt_m, None
        d = np.asarray(decision)
        if d.shape != L.shape:
            raise ValueError(f"decision {d.shape} does not match labels {L.shape}")
        t_in, t_full = tera_metrics(L[rs, cs], d[rs, cs]), tera_metrics(L, d)
    tera_m = dict(t_in) | {f"{k}_full": v for k, v in t_full.items()}
    tera_m["rank_score"] = rank_score(g_in, t_in)
    tera_m["rank_score_full"] = rank_score(g_full, t_full)
    return gt_m, tera_m

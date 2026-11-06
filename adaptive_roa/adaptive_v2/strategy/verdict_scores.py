"""det_v2 acquisition scores (method section 5) and batch rules.

q_A(x) is the predictive probability that x's verdict under tau* is A:

- final-state model: exactly, q_A = 1 - a - b + 2 P(B) under Dirichlet(n + 1), with
  a = P(p_s > tau_s), b = P(p_f > tau_f) (Beta tails) and P(B) the both-pass
  integral, computed with fixed composite Gauss-Legendre quadrature over
  [tau_s, 1 - tau_f] (empty when tau_s + tau_f >= 1). Computed once per distinct
  count triple, so K = 100 costs at most 5,151 integrals per round.
- BNN classifier: the fraction of posterior draws of p_s whose verdict is A.
- MLP classifier: 1 when x's verdict is A, else 0.

U_H(x) = p_inv + (1 - p_inv) H2(p_s / (p_s + p_f)) in bits (ablation A7), with the
second term 0 when p_s + p_f = 0; for classifiers H2(p_s).

Batch rules: ``topk`` takes the k highest scores, ties going to the candidate
farthest from the nearest state already rolled out, then to pool order;
``proportional`` samples without replacement with probability proportional to U
among U > 0, and the engine tops a short batch up with uniform starts.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import betainc, betaln, xlogy

from adaptive_roa.adaptive_v2.eval import det_v2_rule

# 64 panels x 16 Gauss-Legendre nodes. The Beta(n + 1, .) density at K = 100 is about
# 0.05 wide, so each panel sees a near-polynomial piece of it.
QUAD_PANELS = 64
QUAD_NODES = 16
_GL_X, _GL_W = np.polynomial.legendre.leggauss(QUAD_NODES)


def _beta_logpdf(x, a, b):
    return xlogy(a - 1.0, x) + xlogy(b - 1.0, 1.0 - x) - betaln(a, b)


def _quad_nodes(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(lo, hi, QUAD_PANELS + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    x = (mid[:, None] + half[:, None] * _GL_X[None, :]).ravel()
    w = (half[:, None] * _GL_W[None, :]).ravel()
    return x, w


def both_pass_probability(n_s, n_f, n_inv, tau_s: float, tau_f: float) -> np.ndarray:
    """P(p_s > tau_s and p_f > tau_f) under Dirichlet(n_s + 1, n_f + 1, n_inv + 1)."""
    n_s, n_f, n_inv = (np.asarray(v, dtype=float) for v in (n_s, n_f, n_inv))
    if tau_s + tau_f >= 1.0:
        return np.zeros(np.broadcast(n_s, n_f, n_inv).shape)
    x, w = _quad_nodes(float(tau_s), 1.0 - float(tau_f))
    n = n_s + n_f + n_inv
    # p_s ~ Beta(n_s + 1, n - n_s + 2); given p_s = x, p_f / (1 - x) ~ Beta(n_f + 1, n_inv + 1).
    a_s, b_s = n_s[..., None] + 1.0, (n - n_s)[..., None] + 2.0
    dens = np.exp(_beta_logpdf(x, a_s, b_s))
    ratio = np.clip(tau_f / (1.0 - x), 0.0, 1.0)
    tail = 1.0 - betainc(n_f[..., None] + 1.0, n_inv[..., None] + 1.0, ratio)
    return (dens * tail) @ w


def verdict_probabilities(n_s, n_f, n_inv, tau_s: float, tau_f: float) -> dict[str, np.ndarray]:
    """q_S, q_F, q_A for final-state counts, computed once per distinct triple."""
    tri = np.stack([np.asarray(n_s), np.asarray(n_f), np.asarray(n_inv)], 1).astype(np.int64)
    uniq, inv = np.unique(tri, axis=0, return_inverse=True)
    inv = inv.ravel()
    us, uf, ui = (uniq[:, j].astype(float) for j in range(3))
    n = us + uf + ui
    a = 1.0 - betainc(us + 1.0, n - us + 2.0, float(tau_s))
    b = 1.0 - betainc(uf + 1.0, n - uf + 2.0, float(tau_f))
    pb = both_pass_probability(us, uf, ui, tau_s, tau_f)
    q_s, q_f = a - pb, b - pb
    q_a = 1.0 - a - b + 2.0 * pb
    # Quadrature round-off leaves values a few 1e-15 outside [0, 1]; later ratios and logs need them inside.
    return {k: np.clip(v, 0.0, 1.0)[inv] for k, v in (("q_s", q_s), ("q_f", q_f), ("q_a", q_a))}


def classifier_verdict_probabilities(p_draws: np.ndarray, tau_s: float, tau_f: float) -> dict[str, np.ndarray]:
    """q_S, q_F, q_A from posterior draws of p_s, shape [S, B] (one row for an MLP)."""
    p_draws = np.atleast_2d(np.asarray(p_draws, dtype=float))
    lab = det_v2_rule.predict(p_draws, None, tau_s, tau_f)
    return {"q_s": (lab == det_v2_rule.S).mean(0), "q_f": (lab == det_v2_rule.F).mean(0),
            "q_a": ((lab == det_v2_rule.A_OUT) | (lab == det_v2_rule.A_INV)).mean(0)}


def binary_entropy_bits(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return -(xlogy(p, p) + xlogy(1.0 - p, 1.0 - p)) / np.log(2.0)


def u_h(p_s, p_f=None, p_inv=None) -> np.ndarray:
    """U_H in bits; a classifier (p_f, p_inv None) gives H2(p_s)."""
    p_s = np.asarray(p_s, dtype=float)
    if p_f is None:
        return binary_entropy_bits(p_s)
    p_f, p_inv = np.asarray(p_f, dtype=float), np.asarray(p_inv, dtype=float)
    valid = p_s + p_f
    ratio = np.divide(p_s, valid, out=np.zeros_like(p_s), where=valid > 0)
    outcome = np.where(valid > 0, binary_entropy_bits(ratio), 0.0)
    return p_inv + (1.0 - p_inv) * outcome


def verdict_entropy_bits(q: dict[str, np.ndarray]) -> np.ndarray:
    """H(q_S, q_F, q_A) in bits: logged as a diagnostic, never used for selection."""
    total = sum(-xlogy(q[k], q[k]) for k in ("q_s", "q_f", "q_a"))
    return total / np.log(2.0)


def nearest_distance(candidates: np.ndarray, rolled_out: np.ndarray) -> np.ndarray:
    """Euclidean distance from each candidate to its nearest rolled-out state (inf when none)."""
    if len(rolled_out) == 0:
        return np.full(len(candidates), np.inf)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(rolled_out, dtype=float)).query(np.asarray(candidates, dtype=float), k=1)
    return np.asarray(d, dtype=float)


def select_topk(scores: np.ndarray, k: int, distance_fn) -> tuple[np.ndarray, dict[str, Any]]:
    """Positions of the k highest scores; ties go to the larger distance, then to pool order.

    ``distance_fn(positions)`` returns the nearest-rolled-out distance of those
    candidates; it is called only for the tie group straddling the k-th place.
    """
    u = np.round(np.asarray(scores, dtype=float), 12)
    n = len(u)
    k = min(int(k), n)
    if k <= 0:
        return np.zeros(0, dtype=np.int64), {"n_tie_group": 0, "u_kth": None}
    order = np.lexsort((np.arange(n), -u))              # U descending, then pool order
    u_kth = u[order[k - 1]]
    above = order[u[order] > u_kth]
    tied = np.flatnonzero(u == u_kth)
    need = k - len(above)
    if need < len(tied):
        dist = np.asarray(distance_fn(tied), dtype=float)
        tied = tied[np.lexsort((tied, -dist))][:need]  # farthest first, then pool order
    picks = np.concatenate([above, tied]).astype(np.int64)
    return picks, {"n_tie_group": int(np.sum(u == u_kth)), "n_tie_taken": int(need), "u_kth": float(u_kth)}


def select_proportional(scores: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Up to k positions sampled without replacement with probability proportional to U among U > 0."""
    u = np.asarray(scores, dtype=float)
    pos = np.flatnonzero(u > 0)
    take = min(int(k), len(pos))
    if take <= 0:
        return np.zeros(0, dtype=np.int64), {"n_positive": int(len(pos))}
    picks = rng.choice(pos, size=take, replace=False, p=u[pos] / u[pos].sum())
    return np.sort(picks).astype(np.int64), {"n_positive": int(len(pos))}

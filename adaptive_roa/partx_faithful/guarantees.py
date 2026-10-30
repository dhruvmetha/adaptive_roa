"""Part-X's probabilistic guarantees on one snapshot (arXiv 2110.10729; tracker C21).

Per classified leaf i at depth j with n_i evaluations, the paper's Lemma 1 gives

    delta_i = [ln(1/p_i) + ln(1/alpha_j) + 2 ln n_i + 1] / n_i,   alpha_j = alpha / B^(j-1)

where p_i is the GP-predicted probability of the event that would make the leaf
wrong: P(f <= 0) for a satisfying ('+') leaf, P(f > 0) for a violating ('-') one.
Lemmas 2 and 4 (eq. 9 across levels) then bound the misclassified volume of each
classified set: with probability at least prod_i (1 - delta_i), it is at most
sum_i delta_i v(sigma_i) / v(S).

What this does NOT reproduce, and says so in its output:
- the reclassification terms of Theorems 1-2, which need the partition history;
- Theorem 3's squared factor, which its own Lemma 6 does not support (Lemma 6 is used);
- any check of the conditioning event C_k (defined through the unknown level set),
  of the assumed-known GP hyperparameters, or of independence across leaves.
A side is "not computable" when a leaf's factor is vacuous (delta >= 1) or p rounds
to 0; 'r', 'r+', 'r-', 'u' and 'i' leaves sit outside every bound.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from adaptive_roa.partx_faithful.sampling import lhs

UNCHECKED = [
    "every bound conditions on C_k, defined through the unknown zero level set",
    "GP hyperparameters are assumed known (Assumption 3); they are fitted here",
    "leaves are assumed independent (Lemma 2)",
    "only the current partition: Theorems 1-2's reclassification terms are not reconstructed",
]


def leaf_delta(p: float, n: int, depth: int, alpha: float, B: int) -> float:
    """Lemma 1's delta for a leaf at tree depth `depth` (root = 0, so level j = depth + 1)."""
    if n <= 0 or not p > 0.0:
        return math.inf
    alpha_j = alpha / B ** depth
    return (math.log(1.0 / p) + math.log(1.0 / alpha_j) + 2.0 * math.log(n) + 1.0) / n


def snapshot_guarantees(leaves, models, n_samples, root_volume: float, alpha: float, B: int,
                        R: int, M: int, rng) -> dict:
    """Satisfying and violating bounds for the classified leaves of one snapshot."""
    out = {"alpha": alpha, "B": B, "unchecked_assumptions": UNCHECKED, "sides": {}}
    for side, rtype in (("satisfying", "+"), ("violating", "-")):
        rows, reasons = [], []
        for leaf in leaves:
            if leaf.rtype != rtype:
                continue
            model = models.get(leaf.rid)
            X = np.vstack([lhs(M, leaf.low, leaf.high, rng) for _ in range(R)])
            m, s = model.latent(X)
            p_fals = float(np.mean(norm.cdf(-np.asarray(m) / np.asarray(s))))
            p = p_fals if rtype == "+" else 1.0 - p_fals
            n = int(n_samples[leaf.rid])
            d = leaf_delta(p, n, leaf.depth, alpha, B)
            rows.append({"rid": int(leaf.rid), "depth": int(leaf.depth), "n": n, "p": p,
                         "delta": d, "volume_frac": float(leaf.volume() / root_volume)})
            if not p > 0.0:
                reasons.append(f"leaf {leaf.rid}: p rounds to 0")
            elif d >= 1.0:
                reasons.append(f"leaf {leaf.rid}: vacuous, delta={d:.3f} with n={n}")
        entry = {"n_leaves": len(rows), "leaves": rows}
        if not rows:
            entry.update(computable=False, reason=f"no '{rtype}' leaf in this snapshot")
        elif reasons:
            entry.update(computable=False, reason="; ".join(reasons[:5])
                         + (f"; and {len(reasons) - 5} more" if len(reasons) > 5 else ""))
        else:
            entry.update(computable=True,
                         probability_at_least=float(np.prod([1.0 - r["delta"] for r in rows])),
                         misclassified_volume_frac_at_most=float(sum(r["delta"] * r["volume_frac"]
                                                                      for r in rows)))
        out["sides"][side] = entry
    return out

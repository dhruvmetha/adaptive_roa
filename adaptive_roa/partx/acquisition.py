from __future__ import annotations

import numpy as np


def straddle_score(m, s2, beta=1.96):
    return beta * np.sqrt(np.maximum(s2, 0.0)) - np.abs(m)


def select_pool_indices(tree, cand_states, cand_indices, m, s2,
                        target_count, beta=1.96, allocation="per_region_volume"):
    cand_states = np.asarray(cand_states)
    scores = straddle_score(np.asarray(m), np.asarray(s2), beta)
    leaf_of = np.asarray(tree.assign(cand_states))
    leaves = tree.leaves()
    unresolved = [i for i, r in enumerate(leaves) if r.region_class in ("r", "min")]
    eligible = np.array([i for i in range(len(cand_states))
                         if leaf_of[i] in set(unresolved)], dtype=int)
    diag = {"n_unresolved_leaves": len(unresolved), "n_eligible": int(eligible.size),
            "n_candidates": int(len(cand_states)),
            "n_outside_tree": int(np.sum(leaf_of < 0)),
            "fallback_used": False}
    if target_count <= 0:
        return [], diag

    if eligible.size == 0:
        # No candidate lies in an unresolved leaf. Returning [] here is what let
        # an arm train on unchanged data for a whole epoch while still writing a
        # normal-looking artifact -- q3d_nd048_partx did that for all 18 epochs
        # and appeared in the standings as an adaptive arm. Degrade to a global
        # straddle top-k instead, and say so, so the run keeps spending its
        # budget and the fallback is visible in the artifact.
        diag["fallback_used"] = True
        diag["fallback_reason"] = (
            "no candidate in an unresolved leaf "
            f"({diag['n_outside_tree']}/{diag['n_candidates']} outside the tree)"
        )
        order = np.argsort(-scores)[:target_count]
        selected = [int(cand_indices[i]) for i in order]
        diag["n_selected"] = len(selected)
        print(f"WARNING partx: {diag['fallback_reason']}; "
              f"falling back to global straddle top-{len(selected)}")
        return selected, diag

    if allocation == "global_top":
        order = eligible[np.argsort(-scores[eligible])]
        chosen = order[:target_count]
    else:  # per_region_volume
        vols = {i: leaves[i].volume() for i in unresolved}
        total = sum(vols.values()) or 1.0
        chosen = []
        remaining = target_count
        # volume-proportional quota per leaf, floor of 1 for non-empty leaves
        for j, i in enumerate(unresolved):
            in_leaf = eligible[leaf_of[eligible] == i]
            if in_leaf.size == 0:
                continue
            quota = max(1, int(round(target_count * vols[i] / total)))
            quota = min(quota, in_leaf.size, remaining)
            top = in_leaf[np.argsort(-scores[in_leaf])][:quota]
            chosen.extend(top.tolist())
            remaining -= len(top)
            if remaining <= 0:
                break
        # top up globally if rounding left us short
        if remaining > 0:
            rest = [i for i in eligible.tolist() if i not in set(chosen)]
            rest = sorted(rest, key=lambda i: -scores[i])[:remaining]
            chosen.extend(rest)
        chosen = np.array(chosen[:target_count], dtype=int)

    # Eligible candidates can run out before the quota is met (quad2D corridor
    # spent 500, 500, ... then 29, 8, 4, 1 as its unresolved leaves emptied).
    # An arm that silently banks the shortfall is not on the same budget as the
    # arms it is being ranked against, so top up with the best remaining
    # candidates and record how many came from outside the unresolved set.
    chosen = list(np.asarray(chosen, dtype=int).tolist())
    shortfall = target_count - len(chosen)
    n_topup = 0
    if shortfall > 0:
        taken = set(chosen)
        rest = [i for i in np.argsort(-scores).tolist() if i not in taken]
        topup = rest[:shortfall]
        chosen.extend(topup)
        n_topup = len(topup)
    diag["n_topup_outside_unresolved"] = n_topup
    if n_topup:
        print(f"NOTE partx: unresolved leaves supplied "
              f"{len(chosen) - n_topup}/{target_count}; topped up {n_topup} "
              f"from the global straddle ranking")

    selected = [int(cand_indices[i]) for i in chosen]
    diag["n_selected"] = len(selected)
    return selected, diag

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
    diag = {"n_unresolved_leaves": len(unresolved), "n_eligible": int(eligible.size)}
    if eligible.size == 0 or target_count <= 0:
        return [], diag

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

    selected = [int(cand_indices[i]) for i in chosen]
    diag["n_selected"] = len(selected)
    return selected, diag

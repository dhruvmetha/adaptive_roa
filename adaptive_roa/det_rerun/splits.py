"""Disjoint pool / cal / val / test partitions for the deterministic-paper rerun.

The old det datasets drew test_set and cal_set from the same trajectories the training
pool reads, so runs were scored on states they trained on. Every partition here is
built by trajectory and checked with `assert_disjoint` before anything is written.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def row_key(values: Iterable[float], decimals: int = 6) -> str:
    """Canonical text key of a state, at the eval files' 6 decimals; -0 prints as 0."""
    out = []
    for v in values:
        s = f"{float(v):.{decimals}f}"
        if s.startswith("-") and float(s) == 0.0:
            s = s[1:]
        out.append(s)
    return ",".join(out)


def partition_ids(
    ids: Sequence,
    sizes: Mapping[str, int],
    rest: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Split ids at random into the named sizes, in order; whatever is left goes to `rest`."""
    ids = np.asarray(ids)
    need = sum(sizes.values())
    if need > len(ids):
        raise ValueError(f"requested {need} ids from {len(ids)}")
    perm = np.random.default_rng(seed).permutation(len(ids))
    out: Dict[str, np.ndarray] = {}
    start = 0
    for name, n in sizes.items():
        out[name] = ids[perm[start:start + n]]
        start += n
    out[rest] = ids[perm[start:]]
    return out


def group_split(
    groups: Sequence,
    sizes: Mapping[str, int],
    rest: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Split row indices so that all rows of a group land in one split.

    Groups are visited in a seeded random order and assigned to each named split until
    it holds at least its target row count, so a split can exceed its target by less
    than one group. Remaining groups go to `rest`.
    """
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members: List[List[int]] = [[] for _ in uniq]
    for row, g in enumerate(inverse):
        members[g].append(row)
    order = np.random.default_rng(seed).permutation(len(uniq))
    out: Dict[str, List[int]] = {name: [] for name in sizes}
    out[rest] = []
    names = list(sizes)
    k = 0
    for g in order:
        while k < len(names) and len(out[names[k]]) >= sizes[names[k]]:
            k += 1
        target = names[k] if k < len(names) else rest
        out[target].extend(members[g])
    if any(len(out[n]) < sizes[n] for n in names):
        raise ValueError(f"not enough rows for {dict(sizes)}: got {{ {', '.join(f'{n}: {len(out[n])}' for n in names)} }}")
    return {name: np.asarray(sorted(rows), dtype=np.int64) for name, rows in out.items()}


def assert_disjoint(named_sets: Mapping[str, set]) -> None:
    """Raise naming every pair of sets that share an element."""
    names = list(named_sets)
    clashes = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = len(named_sets[a] & named_sets[b])
            if n:
                clashes.append(f"{a} & {b}: {n}")
    if clashes:
        raise ValueError("partition overlap: " + "; ".join(clashes))


def pool_order(names: Sequence[str], seed: int) -> List[str]:
    """Seeded permutation of the pool's trajectory names; the run seed picks the order."""
    perm = np.random.default_rng(seed).permutation(len(names))
    return [names[i] for i in perm]

"""Pendulum trial switches: val_split=interleave and acquisition.tie_tol (both off by default)."""
import numpy as np
import pytest

from adaptive_roa.adaptive.dataset_builder import AdaptiveDatasetBuilder, DatasetSplit
from adaptive_roa.adaptive_v2.strategy import verdict_scores as vs


def builder(val_split, ids, val_ratio=0.2):
    b = AdaptiveDatasetBuilder.__new__(AdaptiveDatasetBuilder)
    b.train_split = DatasetSplit(ids)
    b.val_ratio, b.val_split = val_ratio, val_split
    return b


def grow(n0=50, k=35, rounds=10):
    ids = list(range(n0))
    yield list(ids)
    nxt = 10_000
    for _ in range(rounds):
        ids += list(range(nxt, nxt + k))
        nxt += k
        yield list(ids)


def test_prefix_is_the_old_slicing():
    for ids in grow():
        tr, va = builder("prefix", ids)._start_split()
        n_val = max(1, int(len(ids) * 0.2))
        assert va == ids[:n_val] and tr == ids[n_val:]


def test_prefix_strips_the_initial_batch_and_interleave_keeps_it():
    last = list(grow())[-1]
    tr_p, _ = builder("prefix", last)._start_split()
    tr_i, va_i = builder("interleave", last)._start_split()
    assert sum(x < 50 for x in tr_p) == 0                    # round 10: no initial trajectory trains
    assert sum(x < 50 for x in tr_i) == 40 and sum(x < 50 for x in va_i) == 10
    assert set(tr_i).isdisjoint(va_i) and sorted(tr_i + va_i) == sorted(last)
    assert len(va_i) == 80                                   # same fit-set size as prefix at N = 400


def test_interleave_membership_never_changes_side():
    side = {}
    for ids in grow():
        tr, va = builder("interleave", ids)._start_split()
        for x in va:
            assert side.setdefault(x, "val") == "val"
        for x in tr:
            assert side.setdefault(x, "train") == "train"


def test_bad_val_split_is_refused(tmp_path):
    with pytest.raises(ValueError):
        AdaptiveDatasetBuilder(data_source=None, output_dir=str(tmp_path), val_split="shuffle")


def reference_topk(scores, k, distance_fn):
    """select_topk as it was at 0f343ff (no tie_tol)."""
    u = np.round(np.asarray(scores, dtype=float), 12)
    n = len(u)
    order = np.lexsort((np.arange(n), -u))
    u_kth = u[order[k - 1]]
    above = order[u[order] > u_kth]
    tied = np.flatnonzero(u == u_kth)
    need = k - len(above)
    if need < len(tied):
        dist = np.asarray(distance_fn(tied), dtype=float)
        tied = tied[np.lexsort((tied, -dist))][:need]
    return np.concatenate([above, tied]).astype(np.int64)


def test_tie_tol_zero_matches_the_old_rule():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = np.round(rng.random(500), 2)                     # plenty of exact ties
        dist = rng.random(500)
        fn = lambda pos: dist[pos]
        got, _ = vs.select_topk(s, 37, fn)
        assert np.array_equal(got, reference_topk(s, 37, fn))


def two_clusters():
    """Cluster A (x near 0) scores a hair higher than cluster B (x near 10); both saturate near 1."""
    rng = np.random.default_rng(1)
    xa, xb = rng.normal(0, 0.1, 40), rng.normal(10, 0.1, 40)
    X = np.concatenate([xa, xb])[:, None]
    s = np.concatenate([np.full(40, 1 - 1e-7) - rng.random(40) * 1e-8, np.full(40, 1 - 5e-7)])
    return X, s


def test_saturated_scores_pack_one_cluster_without_tie_tol():
    X, s = two_clusters()
    picks, _ = vs.select_topk(s, 10, lambda pos: np.zeros(len(pos)))
    assert (X[picks, 0] < 5).all()                           # all ten from cluster A


def test_tie_tol_spreads_the_batch_over_both_clusters():
    X, s = two_clusters()

    def spread(band, need, above):
        return band[vs.spread_farthest_point(X[band], X[above], need)]

    picks, diag = vs.select_topk(s, 10, lambda pos: np.zeros(len(pos)), tie_tol=1e-3, spread_fn=spread)
    assert len(picks) == 10 and len(set(picks.tolist())) == 10
    assert (X[picks, 0] < 5).any() and (X[picks, 0] > 5).any()
    assert diag["n_tie_group"] == 80 and diag["tie_tol"] == 1e-3


def test_tie_tol_still_takes_clear_winners_outright():
    X, s = two_clusters()
    s = s.copy()
    s[[5, 50]] = 0.5                                         # far below the band: never picked
    s[[0, 45]] = 2.0                                         # far above the band: always picked
    spread = lambda band, need, above: band[vs.spread_farthest_point(X[band], X[above], need)]
    picks, _ = vs.select_topk(s, 6, lambda pos: np.zeros(len(pos)), tie_tol=1e-3, spread_fn=spread)
    assert {0, 45} <= set(picks.tolist()) and not {5, 50} & set(picks.tolist())


def test_farthest_point_maximises_spacing_on_a_line():
    cand = np.arange(11, dtype=float)[:, None]               # 0..10
    rows = vs.spread_farthest_point(cand, np.array([[0.0]]), 2)
    assert rows.tolist() == [10, 5]

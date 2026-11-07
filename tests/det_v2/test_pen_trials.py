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


class _StubSource:
    def get_state_dim(self):
        return 2


def external_builder(tmp_path, ids):
    f = tmp_path / "val_set.txt"
    f.write_text("0.1,0.2,0.0,0.05,1\n-1.0,3.0,2.1,0.0,0\n0.5,-0.5,0.01,-0.06,1\n")
    b = AdaptiveDatasetBuilder.__new__(AdaptiveDatasetBuilder)
    b.train_split = DatasetSplit(ids)
    b.val_ratio, b.val_split, b.external_val_file, b._external_val = 0.2, "external", str(f), None
    b.data_source, b.output_dir, b.candidate_mode = _StubSource(), tmp_path, "start"
    return b


def test_external_trains_on_every_trajectory_and_fits_on_d_val(tmp_path):
    ids = list(grow())[-1]
    b = external_builder(tmp_path, ids)
    tr, va = b._start_split()
    assert tr == ids and va == []
    X, E, y, g = b.get_val_data(return_groups=True)
    assert X.shape == (3, 2) and E.shape == (3, 2)
    assert y.tolist() == [1, -1, 1]                          # internal labels, as the val/cal store reads them
    assert g.tolist() == [0, 1, 2]
    Xl, yl = b.get_val_labels()
    assert np.array_equal(Xl, X) and np.array_equal(yl, y)


def test_external_writes_d_val_as_the_val_files(tmp_path):
    b = external_builder(tmp_path, list(range(50)))
    pairs = np.loadtxt(b.build_val_dataset())
    assert pairs.shape == (3, 4) and np.allclose(pairs[1], [-1.0, 3.0, 2.1, 0.0])
    clf = np.loadtxt(b.build_val_classification_dataset())
    assert clf.shape == (3, 3) and clf[:, 2].tolist() == [1.0, 0.0, 1.0]


def test_external_needs_the_file_and_start_mode(tmp_path):
    with pytest.raises(ValueError):
        AdaptiveDatasetBuilder(data_source=None, output_dir=str(tmp_path), val_split="external")
    with pytest.raises(ValueError):
        AdaptiveDatasetBuilder(data_source=None, output_dir=str(tmp_path), val_split="external",
                               external_val_file="x.txt", candidate_mode="intermediate")


def test_external_val_honours_the_row_cap_with_the_seeded_subset(tmp_path):
    from adaptive_roa.adaptive.data_source import split_subset
    b = external_builder(tmp_path, list(range(50)))
    b.external_val_rows = 2
    X, E, y = b.get_val_data()
    idx = split_subset(3, 2)
    assert len(y) == 2 and y.tolist() == np.array([1, -1, 1])[idx].tolist()


def _sorted_split_file(path, n=2000):
    """A split file in the order cp-ppo's are: failures first, then successes (label sorted)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 2)); E = rng.normal(size=(n, 2))
    y = (np.arange(n) >= int(0.8 * n)).astype(int)          # last 20% are successes
    np.savetxt(path, np.column_stack([X, E, y]), delimiter=",", fmt="%.6f")
    return y


def test_split_caps_take_a_seeded_random_subset_not_a_prefix(tmp_path):
    from omegaconf import OmegaConf
    from adaptive_roa.adaptive_v2.engine import _load_split
    f = tmp_path / "val_set.txt"
    _sorted_split_file(f)
    cfg = OmegaConf.create({"data_source": {"val_set_max_rows": 500}})
    X1, y1 = _load_split(cfg, "val", str(f), None, 2)
    X2, y2 = _load_split(cfg, "val", str(f), None, 2)
    assert len(y1) == 500 and np.array_equal(X1, X2)               # the same rows every time
    assert 0.14 < (y1 == 1).mean() < 0.26                           # a prefix would have 0 successes
    Xa, ya = _load_split(OmegaConf.create({"data_source": {}}), "val", str(f), None, 2)
    assert len(ya) == 2000                                          # no cap: the whole file


def test_builder_and_engine_read_the_same_d_val_rows(tmp_path):
    from omegaconf import OmegaConf
    from adaptive_roa.adaptive_v2.engine import _load_split
    f = tmp_path / "val_set.txt"
    _sorted_split_file(f)
    b = AdaptiveDatasetBuilder.__new__(AdaptiveDatasetBuilder)
    b.train_split = DatasetSplit(list(range(50)))
    b.val_ratio, b.val_split, b.external_val_file, b._external_val = 0.2, "external", str(f), None
    b.external_val_rows = 500
    b.data_source, b.output_dir, b.candidate_mode = _StubSource(), tmp_path, "start"
    Xb, _, yb = b.get_val_data()
    Xe, ye = _load_split(OmegaConf.create({"data_source": {"val_set_max_rows": 500}}), "val", str(f), None, 2)
    assert np.allclose(Xb, Xe) and np.array_equal(yb, ye)

import numpy as np
import pytest

from adaptive_roa.partx_faithful.region import Region, child_masks
from adaptive_roa.partx_faithful.sampling import ContinuousSampler, PoolSampler, SampleStore, lhs


def _region(low, high, rid=0):
    return Region(rid=rid, parent=-1, low=np.asarray(low, float), high=np.asarray(high, float),
                  branch_dir=0, depth=0)


def test_split_bounds_halves_one_dimension():
    (lo0, hi0), (lo1, hi1) = _region([0, 0], [2, 4]).split_bounds(1)
    assert hi0[1] == 2.0 and lo1[1] == 2.0
    assert lo0[0] == 0.0 and hi1[0] == 2.0


def test_child_masks_assign_every_point_exactly_once():
    bounds = _region([0.0], [2.0]).split_bounds(0)
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    masks = child_masks(x, bounds, 0)
    assert np.all(sum(m.astype(int) for m in masks) == 1)
    assert masks[0].tolist() == [True, True, False, False, False]
    assert masks[1].tolist() == [False, False, True, True, True]


def test_lhs_puts_one_point_in_each_stratum():
    X = lhs(8, [0.0, 0.0], [1.0, 2.0], np.random.default_rng(0))
    for d, width in enumerate([1.0, 2.0]):
        assert sorted(np.floor(X[:, d] / width * 8).astype(int).tolist()) == list(range(8))


def test_sample_store_grows_and_round_trips():
    s = SampleStore(2)
    assert s.add(np.ones((100, 2)), np.arange(100), np.arange(100)) == list(range(100))
    assert s.add([[5.0, 6.0]], [7.0], [-1]) == [100]
    np.testing.assert_array_equal(s.X([100]), [[5.0, 6.0]])
    assert s.y([3])[0] == 3.0 and s.pool_idx([100])[0] == -1 and s.n == 101
    assert s.X([]).shape == (0, 2)


def _pool(n=400, seed=0, n_used=10):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, 1, size=(n, 2))
    labels = np.where(starts[:, 0] < 0.5, 1.0, -1.0)
    used = np.zeros(n, dtype=bool)
    used[:n_used] = True
    return PoolSampler(starts, labels, used, scale=np.ones(2)), starts


def _split_root(ps):
    root = _region([0, 0], [1, 1])
    ps.init_root(root)
    kids = [Region(rid=i + 1, parent=0, low=lo, high=hi, branch_dir=1, depth=1)
            for i, (lo, hi) in enumerate(root.split_bounds(0))]
    all_cand = root.pool_candidates.copy()
    ps.split(root, kids, 0)
    return root, kids, all_cand


def test_split_partitions_the_candidates():
    ps, _ = _pool()
    root, kids, all_cand = _split_root(ps)
    assert root.pool_candidates is None
    both = np.concatenate([k.pool_candidates for k in kids])
    assert sorted(both.tolist()) == sorted(all_cand.tolist())


def test_snap_returns_unused_unique_in_box_points():
    ps, starts = _pool()
    _, kids, _ = _split_root(ps)
    h = ps.snap_design(lhs(20, kids[0].low, kids[0].high, np.random.default_rng(1)), kids[0])
    assert len(h) == 20 and len(set(h.tolist())) == 20
    assert not ps.used[h].any()
    assert np.all(kids[0].contains(starts[h]))
    assert np.all(h >= 10)                                     # the initial set is excluded


def test_take_marks_used_and_refuses_reuse():
    ps, _ = _pool()
    _, kids, _ = _split_root(ps)
    h = ps.snap_design(lhs(3, kids[0].low, kids[0].high, np.random.default_rng(1)), kids[0])
    X, y, p = ps.take(h)
    assert ps.used[p].all() and X.shape == (3, 2) and set(y.tolist()) <= {1.0, -1.0}
    with pytest.raises(RuntimeError):
        ps.take(h[:1])


def test_candidates_respect_the_cap_and_empty_boxes():
    ps, _ = _pool()
    root = _region([0, 0], [1, 1])
    ps.init_root(root)
    X, h = ps.candidates(root, 5, np.random.default_rng(0))
    assert len(h) == 5 and X.shape == (5, 2)
    empty = _region([0, 0], [1, 1], rid=9)
    empty.pool_candidates = np.empty(0, dtype=np.int64)
    assert ps.n_available(empty) == 0
    assert len(ps.snap_design(np.array([[0.5, 0.5]]), empty)) == 0


def test_continuous_sampler_evaluates_the_function():
    cs = ContinuousSampler(lambda x: x[0] + x[1])
    X, y, p = cs.take(np.array([[1.0, 2.0]]))
    assert y[0] == 3.0 and p[0] == -1
    U, h = cs.candidates(_region([0, 0], [1, 1]), 7, np.random.default_rng(0))
    assert U.shape == (7, 2) and np.all((U >= 0) & (U <= 1))

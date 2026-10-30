"""Disjoint pool/cal/val/test partitions and the overlap guard."""
import numpy as np
import pytest

from adaptive_roa.det_rerun.splits import (
    assert_disjoint,
    group_split,
    partition_ids,
    pool_order,
    row_key,
)


def test_partition_ids_sizes_and_disjoint():
    ids = np.arange(100)
    parts = partition_ids(ids, {"pool": 50, "cal": 10, "val": 10}, rest="test", seed=7)
    assert [len(parts[k]) for k in ("pool", "cal", "val", "test")] == [50, 10, 10, 30]
    assert sorted(np.concatenate(list(parts.values())).tolist()) == ids.tolist()
    assert_disjoint({k: set(v.tolist()) for k, v in parts.items()})


def test_partition_ids_is_seeded():
    ids = np.arange(100)
    a = partition_ids(ids, {"pool": 50}, rest="test", seed=7)
    b = partition_ids(ids, {"pool": 50}, rest="test", seed=7)
    c = partition_ids(ids, {"pool": 50}, rest="test", seed=8)
    assert np.array_equal(a["pool"], b["pool"])
    assert not np.array_equal(a["pool"], c["pool"])


def test_partition_ids_refuses_oversize():
    with pytest.raises(ValueError):
        partition_ids(np.arange(10), {"pool": 8, "cal": 5}, rest="test", seed=0)


def test_assert_disjoint_raises_on_overlap():
    with pytest.raises(ValueError, match="pool & test"):
        assert_disjoint({"pool": {1, 2}, "cal": {3}, "test": {2, 9}})


def test_group_split_keeps_groups_whole():
    # 30 rows from 12 groups of uneven size; a group never straddles two splits.
    groups = np.array([g for g in range(12) for _ in range(1 + g % 3)])
    out = group_split(groups, {"cal": 6, "val": 6}, rest="test", seed=3)
    owner = {}
    for name, rows in out.items():
        for r in rows:
            owner.setdefault(groups[r], set()).add(name)
    assert all(len(v) == 1 for v in owner.values())
    assert len(out["cal"]) >= 6 and len(out["val"]) >= 6
    assert sum(len(v) for v in out.values()) == len(groups)


def test_pool_order_is_a_seeded_permutation():
    names = [f"sequence_{i}.txt" for i in range(20)]
    o1, o2 = pool_order(names, 99512), pool_order(names, 99512)
    assert o1 == o2 and sorted(o1) == sorted(names) and o1 != names
    assert pool_order(names, 33407) != o1


def test_row_key_formats_six_decimals():
    assert row_key([-1.34159, 0.000325657]) == "-1.341590,0.000326"
    assert row_key([-0.0000001, 2.0]) == "0.000000,2.000000"

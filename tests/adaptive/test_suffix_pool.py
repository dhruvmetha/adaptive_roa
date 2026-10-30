"""Suffix-pool bookkeeping (det paper D22, C20): one suffix per trajectory, picks
cost a rollout each, val fixed at first pick, random candidates, ordered starts."""
from __future__ import annotations

import numpy as np
import pytest

from adaptive_roa.adaptive.suffix_pool import SuffixPoolState

LENGTHS = [10, 10, 10, 1, 10, 10, 10, 10, 10, 10]
TIMEOUTS = {2}


def _pool(val_ratio=0.2, seed=7, n=None):
    lengths = LENGTHS if n is None else [10] * n
    timeouts = TIMEOUTS if n is None else set()
    return SuffixPoolState(len(lengths), val_ratio, seed,
                           length_fn=lambda i: lengths[i],
                           is_timeout_fn=lambda i: i in timeouts)


def test_earlier_row_replaces_the_suffix_and_both_picks_count():
    p = _pool()
    p.record_picks([(0, 6)], "acquisition")
    assert p.available_rows(0) == 6
    p.record_picks([(0, 2)], "acquisition")
    assert p.suffix_start[0] == 2
    assert p.available_rows(0) == 2
    assert p.n_picks == 2
    assert p.traj_order == [0]


def test_rows_at_or_after_the_suffix_start_are_refused():
    p = _pool()
    p.record_picks([(0, 4)], "acquisition")
    with pytest.raises(ValueError, match="not available"):
        p.record_picks([(0, 4)], "acquisition")
    with pytest.raises(ValueError, match="not available"):
        p.record_picks([(0, 7)], "acquisition")


def test_terminal_row_is_never_a_candidate():
    p = _pool()
    assert p.available_rows(0) == 9
    with pytest.raises(ValueError):
        p.record_picks([(0, 9)], "acquisition")


def test_timeout_trajectory_offers_only_its_start():
    p = _pool()
    assert p.available_rows(2) == 1
    with pytest.raises(ValueError):
        p.record_picks([(2, 3)], "acquisition")
    p.record_picks([(2, 0)], "uniform")
    assert p.full[2]


def test_length_one_trajectory_still_offers_its_start():
    p = _pool()
    assert p.available_rows(3) == 1


def test_two_states_of_one_trajectory_in_a_batch_are_refused():
    p = _pool()
    with pytest.raises(ValueError, match="same trajectory"):
        p.record_picks([(0, 3), (0, 5)], "acquisition")


def test_val_is_exactly_floor_n_r_and_fixed_at_first_pick():
    p = _pool(n=50)
    for t in range(20):
        p.record_picks([(t, 5)], "acquisition")
    assert len(p.val_trajectories()) == 4
    before = {t: p.is_val[t] for t in range(20)}
    p.record_picks([(t, 0) for t in range(20)], "uniform")   # replacements
    assert {t: p.is_val[t] for t in range(20)} == before


def test_candidates_are_one_per_trajectory_and_skip_full_and_excluded():
    p = _pool(n=200)
    p.record_picks([(t, 0) for t in range(50)], "uniform")
    c = p.draw_candidates(100, exclude_trajs=range(50, 60))
    trajs = [t for t, _ in c]
    assert len(trajs) == len(set(trajs)) == 100
    assert not set(trajs) & set(range(60))
    assert all(0 <= r < p.available_rows(t) for t, r in c)


def test_candidates_are_not_the_first_n_in_order():
    p = _pool(n=1000)
    trajs = sorted(t for t, _ in p.draw_candidates(50))
    assert trajs != list(range(50))
    assert max(trajs) > 500


def test_uniform_starts_follow_pool_order():
    p = _pool(n=30)
    p.record_picks([(1, 4)], "acquisition")      # partial: its start is still free
    p.record_picks([(3, 0)], "uniform")          # full: skipped
    assert p.next_uniform_starts(4) == [(0, 0), (1, 0), (2, 0), (4, 0)]


def test_uniform_data_at_a_budget_is_independent_of_step_size():
    a, b = _pool(n=100), _pool(n=100)
    a.record_picks(a.next_uniform_starts(10), "initial")
    b.record_picks(b.next_uniform_starts(10), "initial")
    for _ in range(4):
        a.record_picks(a.next_uniform_starts(5), "uniform")
    b.record_picks(b.next_uniform_starts(10), "uniform")
    b.record_picks(b.next_uniform_starts(10), "uniform")
    assert a.suffix_start == b.suffix_start
    assert a.is_val == b.is_val


def test_resume_replays_to_the_same_state_and_the_same_next_draw():
    p = _pool(n=500)
    p.record_picks(p.next_uniform_starts(20), "initial")
    p.record_picks(p.draw_candidates(15), "acquisition")
    saved = p.to_dict()
    expected = p.draw_candidates(30)
    q = _pool(n=500)
    q.restore(saved)
    assert q.suffix_start == p.suffix_start and q.is_val == p.is_val
    assert q.n_picks == p.n_picks
    assert q.draw_candidates(30) == expected


def test_restore_refuses_a_different_pool():
    p = _pool(n=500)
    with pytest.raises(ValueError, match="does not match"):
        _pool(n=400).restore(p.to_dict())

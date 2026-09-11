import numpy as np
import pytest

from adaptive_roa.partx_faithful.algorithm import PartX, PartXConfig, multinomial_counts
from adaptive_roa.partx_faithful.sampling import PoolSampler
from adaptive_roa.partx_faithful.state import load_state, save_state


class _Stub:
    """g(x) = x0 - 0.5, sd 0.1/sqrt(n). Deterministic and instant."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        self.n = max(len(X), 1)
        return self

    def latent(self, X):
        X = np.atleast_2d(X)
        return X[:, 0] - 0.5, np.full(len(X), 0.1 / np.sqrt(self.n))


def _px(budget, n=4000, n0=4, n_bo=2, n_c=6, seed=0, initial=40, x0_max=1.0):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, 1, size=(n, 2))
    starts[:, 0] *= x0_max
    labels = np.where(starts[:, 0] < 0.5, 1.0, -1.0)
    used = np.zeros(n, dtype=bool)
    used[:initial] = True
    ps = PoolSampler(starts, labels, used, scale=np.ones(2))
    cfg = PartXConfig(n0=n0, n_bo=n_bo, n_c=n_c, R=3, M=40, seed=seed)
    px = PartX(cfg, np.zeros(2), np.ones(2), ps, _Stub, budget,
               init_X=starts[:initial], init_y=labels[:initial], init_pool_idx=np.arange(initial))
    return px, ps


def test_the_budget_is_spent_exactly_and_logged_once():
    px, _ = _px(budget=120)
    px.run()
    assert px.done and px.used == 120 == len(px.log) == len(set(px.log))


def test_a_straight_boundary_gets_classified_on_the_right_sides():
    px, _ = _px(budget=400)
    px.run()
    leaves = px.leaves()
    plus = [r for r in leaves if r.rtype == "+"]
    minus = [r for r in leaves if r.rtype == "-"]
    assert plus and minus
    assert all(r.low[0] >= 0.5 - 1e-12 for r in plus)       # g > 0 only where x0 > 0.5
    assert all(r.high[0] <= 0.5 + 1e-12 for r in minus)


def test_branch_directions_follow_the_permutation():
    px, _ = _px(budget=400)
    px.run_root()
    px.step()
    root = px.regions[0]
    d0 = int(px.perm[0])
    kids = [px.regions[k] for k in root.children]
    assert kids[0].high[d0] == pytest.approx(0.5) and kids[1].low[d0] == pytest.approx(0.5)
    px.step()
    grandkids = [px.regions[k] for k in kids[0].children]
    assert grandkids
    d1 = int(px.perm[1 % 2])
    assert grandkids[0].high[d1] == pytest.approx((kids[0].low[d1] + kids[0].high[d1]) / 2)


def test_children_inherit_the_parent_samples_exactly():
    px, _ = _px(budget=400)
    px.run_root()
    before = set(px.regions[0].samples)
    px.step()
    kids = [px.regions[k] for k in px.regions[0].children]
    inherited = [s for k in kids for s in k.samples if s in before]
    assert sorted(inherited) == sorted(before)


def test_the_gate_falls_through_to_the_final_phase():
    px, _ = _px(budget=5, n0=4, n_bo=2)
    px.run_root()
    assert px.used == 2
    px.step()
    assert px.phase == "final" and px.used == 5
    px.step()
    assert px.done


def test_on_sample_fires_after_every_batch():
    px, _ = _px(budget=60)
    seen = []
    px.on_sample = lambda p: seen.append(p.used)
    px.run()
    assert seen[-1] == 60 and seen == sorted(seen)


def test_a_box_without_pool_points_becomes_infeasible():
    px, _ = _px(budget=200, x0_max=0.5)          # the pool covers only x0 < 0.5
    px.run()
    assert any(r.rtype == "i" and r.low[0] >= 0.5 for r in px.leaves())


def test_the_same_seed_gives_the_same_run():
    a, _ = _px(budget=150)
    a.run()
    b, _ = _px(budget=150)
    b.run()
    assert a.log == b.log
    assert [r.rtype for r in a.leaves()] == [r.rtype for r in b.leaves()]


def test_a_pickled_run_resumes_exactly(tmp_path):
    full, _ = _px(budget=150)
    full.run()
    part, _ = _px(budget=150)
    part.run_root()
    part.step()
    part.step()
    save_state(part, tmp_path / "s.pkl", "h")
    resumed = load_state(tmp_path / "s.pkl", "h")
    _, fresh = _px(budget=150)                   # same pool, only the initial set used
    resumed.attach(fresh, _Stub)
    resumed.run()
    assert resumed.log == full.log
    with pytest.raises(RuntimeError):
        load_state(tmp_path / "s.pkl", "other")


def test_multinomial_counts():
    rng = np.random.default_rng(0)
    c = multinomial_counts([0.0, 1.0, 3.0], 400, rng)
    assert c[0] == 0 and c.sum() == 400 and c[2] > c[1]
    assert multinomial_counts([0.0, 0.0], 10, rng).tolist() == [0, 0]
    assert multinomial_counts([1.0], 0, rng).tolist() == [0]


def test_the_budget_must_cover_the_root():
    with pytest.raises(ValueError):
        _px(budget=1, n0=4, n_bo=2)


def test_the_final_phase_ends_the_run_even_when_the_pool_falls_short():
    px, ps = _px(budget=400)
    px.run_root()                                     # used = n_bo = 2
    free = np.flatnonzero(~ps.used)[:2]
    ps.used[:] = True
    ps.used[free] = False                             # only 2 unused pool points remain
    px.budget = px.used + 3                           # the gate needs 4, so the final phase runs with 3
    px.step()
    assert px.phase == "final" and px.done            # spec §3: the final phase ends the loop
    assert px.budget - px.used == 1 == px.shortfall   # D8: the unmet evaluation stays unspent

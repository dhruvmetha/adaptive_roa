import numpy as np
import pytest
from scipy.stats import norm

from adaptive_roa.partx_faithful.bo import expected_improvement, sample_bo
from adaptive_roa.partx_faithful.region import Region
from adaptive_roa.partx_faithful.sampling import PoolSampler, SampleStore


def test_ei_at_the_incumbent_is_phi0_times_sd():
    assert expected_improvement(np.array([1.0]), np.array([2.0]), 1.0)[0] == pytest.approx(2.0 * norm.pdf(0.0))


def test_ei_prefers_a_lower_mean_and_a_larger_sd():
    e = expected_improvement(np.array([0.0, 1.0]), np.array([1.0, 1.0]), 0.5)
    assert e[0] > e[1]
    e = expected_improvement(np.array([1.0, 1.0]), np.array([0.1, 1.0]), 0.5)
    assert e[1] > e[0]


class _Linear:
    """g(x) = x0 with a constant sd. At fixed sd, EI falls as the mean rises, so
    each EI step takes the lowest available x0. The sd is 0.05, not tiny, so EI
    stays well above underflow for every candidate and argmax never breaks ties
    by index."""

    def fit(self, X, y):
        return self

    def latent(self, X):
        X = np.atleast_2d(X)
        return X[:, 0].copy(), np.full(len(X), 0.05)


def _setup(n=300, all_used=False):
    rng = np.random.default_rng(0)
    starts = rng.uniform(0, 1, size=(n, 2))
    used = np.full(n, all_used)
    ps = PoolSampler(starts, np.ones(n), used, scale=np.ones(2))
    region = Region(rid=0, parent=-1, low=np.zeros(2), high=np.ones(2), branch_dir=0, depth=0)
    ps.init_root(region)
    store = SampleStore(2)

    def take(handles):
        if len(handles) == 0:
            return []
        ids = store.add(*ps.take(handles))
        region.samples.extend(ids)
        return ids

    return ps, region, store, take, starts


def test_tops_up_by_lhs_then_takes_the_ei_argmax():
    ps, region, store, take, starts = _setup()
    res = sample_bo(region, store, ps, lambda r: _Linear(), take, n0=6, n_bo=3,
                    cand_cap=10_000, rng=np.random.default_rng(0))
    assert not res["infeasible"] and res["shortfall"] == 0 and len(res["new_ids"]) == 9
    topped_up = store.pool_idx(res["new_ids"][:6])
    rest = np.delete(starts[:, 0], topped_up)
    np.testing.assert_allclose(np.sort(store.X(res["new_ids"][6:])[:, 0]), np.sort(rest)[:3])


def test_skips_the_top_up_when_the_region_already_has_n0():
    ps, region, store, take, _ = _setup()
    take(np.arange(6))
    res = sample_bo(region, store, ps, lambda r: _Linear(), take, n0=6, n_bo=2,
                    cand_cap=100, rng=np.random.default_rng(0))
    assert len(res["new_ids"]) == 2 and len(region.samples) == 8


def test_marks_an_empty_box_infeasible():
    ps, region, store, take, _ = _setup(all_used=True)
    res = sample_bo(region, store, ps, lambda r: _Linear(), take, n0=6, n_bo=3,
                    cand_cap=100, rng=np.random.default_rng(0))
    assert res == {"new_ids": [], "shortfall": 9, "infeasible": True}

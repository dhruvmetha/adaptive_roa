"""Section 6 certificates (adaptive_roa/adaptive_v2/eval/certificates.py)."""
import numpy as np
import pytest

from adaptive_roa.adaptive_v2.eval import certificates as C
from adaptive_roa.adaptive_v2.eval import det_v2_rule as R


def counts_split(n, K=100, seed=0, p_succ=0.3):
    """Final-state style split: k/K probabilities on a simplex and labels that follow p_s."""
    rng = np.random.default_rng(seed)
    a = rng.dirichlet([0.6, 0.6, 0.2], size=n)
    k = np.stack([rng.multinomial(K, row) for row in a])
    ps, pf = R.tie(k[:, 0] / K), R.tie(k[:, 1] / K)
    y = rng.random(n) < np.clip(a[:, 0] / (a[:, 0] + a[:, 1]), 0, 1)
    return ps, pf, y


def clf_split(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.beta(0.5, 0.5, n)                       # true success probability per state
    y = rng.random(n) < q
    ps = np.clip(q + rng.normal(0, 0.05, n), 1e-7, 1 - 1e-7)
    return ps, None, y


def test_m_min_and_cp_bound():
    assert C.m_min(0.05, 0.05) == 59 and C.m_min(0.05, 0.025) == 72
    assert C.cp_upper(0, 59, 0.05) <= 0.05 < C.cp_upper(0, 58, 0.05)
    assert C.cp_upper(3, 3, 0.05) == 1.0 and C.cp_upper(0, 0, 0.05) == 1.0


def test_pair_tables_match_direct_membership():
    ps, pf, y = counts_split(3000, seed=1)
    T = C._pair_tables(ps, pf, y)
    rng = np.random.default_rng(2)
    for a, b in rng.integers(0, R.N_GRID, size=(60, 2)):
        ts, tf = R.GRID[a], R.GRID[b]
        S, F = C.in_S(ps, pf, ts, tf), C.in_F(ps, pf, ts, tf)
        assert T["S_n"][a, b] == S.sum() and T["S_succ"][a, b] == (S & y).sum()
        assert T["F_n"][a, b] == F.sum() and T["F_succ"][a, b] == (F & y).sum()


def test_final_state_order_is_a_deduplicated_front_by_recall():
    ps, pf, y = counts_split(4000, seed=3)
    order = C.order_S_final_state((ps, pf, y), start_min=118)
    assert order
    sets = [C.in_S(ps, pf, ts, tf) for ts, tf in order]
    n = np.array([s.sum() for s in sets]); tp = np.array([(s & y).sum() for s in sets])
    fcr, rec = (n - tp) / n, tp / y.sum()
    assert (n >= 118).all()
    assert (np.diff(rec) > 0).all() and (np.diff(fcr) >= -1e-12).all()       # a front, sorted by recall
    assert len({s.tobytes() for s in sets}) == len(sets)                        # one pair per D_val set
    for (ts, tf), s in zip(order, sets):                                         # each is the tightest pair for its set
        for dts, dtf in ((0.01, 0.0), (0.0, -0.01)):
            t2 = (round(ts + dts, 2), round(tf + dtf, 2))
            if 0.01 <= t2[0] <= 0.99 and 0.01 <= t2[1] <= 0.99:
                assert not np.array_equal(C.in_S(ps, pf, *t2), s)


def test_classifier_order_is_nested_from_the_start_size():
    ps, pf, y = clf_split(5000, seed=4)
    order = C.order_S_classifier((ps, pf, y), start_min=118)
    sets = [C.in_S(ps, None, h, None) for h, _ in order]
    assert sets[0].sum() >= 118
    assert all((a <= b).all() for a, b in zip(sets, sets[1:]))                 # each set contains the one before
    h0 = order[0][0]
    bigger = C.CLF_GRID[C.CLF_GRID > h0]
    assert not len(bigger) or C.in_S(ps, None, bigger.min(), None).sum() < 118


def test_walk_stops_at_the_first_failure():
    ps, pf, y = clf_split(4000, seed=5)
    order = [(0.999, None), (0.99, None), (0.05, None), (0.95, None)]          # third fails, fourth would pass
    out = C.walk_S(order, (ps, None, y), alpha=0.05, delta=0.05)
    assert out["passed"] and out["t_s"] == 0.99 and out["steps"] == 2


def test_6c_F_walk_bounds_roa_loss_and_final_F_excludes_S():
    ps, pf, y = counts_split(6000, seed=6)
    val, cal, test = (ps[:2000], pf[:2000], y[:2000]), (ps[2000:4000], pf[2000:4000], y[2000:4000]), (ps[4000:], pf[4000:], y[4000:])
    res = C.certify(val, cal, test, classifier=False, tau=(0.5, 0.3))
    f = res["6c"]["F"]
    if f["passed"]:
        assert C.cp_upper(f["cal_succ"], cal[2].sum(), 0.025) <= 0.05
    r = C.report(test, (0.2, 0.9), (0.9, 0.05))
    S = C.in_S(test[0], test[1], 0.2, 0.9); F = C.in_F(test[0], test[1], 0.9, 0.05) & ~S
    assert r["test_S"] == S.sum() and r["test_F"] == F.sum()


@pytest.mark.parametrize("classifier", [True, False])
def test_6a_controls_fcr_over_cal_draws(classifier):
    """P(FCR(S_hat) > alpha) <= delta over fresh D_cal draws, with a large test split as the truth."""
    rng = np.random.default_rng(7)
    gen = clf_split if classifier else counts_split
    val = gen(2000, seed=100)
    test = gen(60000, seed=101)
    bad, trials = 0, 120
    for t in range(trials):
        cal = gen(2000, seed=1000 + t)
        out = C.certify(val, cal, test, classifier=classifier)["6a"]
        if out["passed"] and out["test_fcr"] is not None and out["test_fcr"] > 0.05:
            bad += 1
    # delta = 0.05; allow sampling slack (3 SE over 120 trials) on the realized rate
    assert bad / trials <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / trials)


def test_6a_walks_at_delta_and_6c_at_half_delta():
    ps, pf, y = counts_split(9000, seed=8)
    val, cal, test = (ps[:3000], pf[:3000], y[:3000]), (ps[3000:6000], pf[3000:6000], y[3000:6000]), (ps[6000:], pf[6000:], y[6000:])
    res = C.certify(val, cal, test, classifier=False, tau=(0.5, 0.3), alpha=0.05, delta=0.05)
    a, cS, cF = res["6a"], res["6c"]["S"], res["6c"]["F"]
    assert a["m_min"] == 59 and res["6c"]["m_min"] == 72
    if a["passed"]:
        assert a["cal_bound"] == pytest.approx(float(C.cp_upper(a["cal_fp"], a["cal_n"], 0.05)))
    if cS["passed"]:
        assert cS["cal_bound"] == pytest.approx(float(C.cp_upper(cS["cal_fp"], cS["cal_n"], 0.025)))
    if cF["passed"]:
        assert cF["cal_bound"] == pytest.approx(float(C.cp_upper(cF["cal_succ"], cal[2].sum(), 0.025)))
    # every 6c S candidate starts at >= 2 * 72 D_val states
    order = C.order_S_final_state(val, 2 * 72)
    assert all(C.in_S(val[0], val[1], *t).sum() >= 144 for t in order)

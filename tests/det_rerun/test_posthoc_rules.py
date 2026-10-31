"""The post-hoc analysis rules (scripts/det_rerun/analysis) keep their stated invariants."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "det_rerun" / "analysis"))
import calib  # noqa: E402
import margin_rule  # noqa: E402
import tau_rule  # noqa: E402

from adaptive_roa.adaptive_v2.eval.full_roa import _predict_lambda_delta  # noqa: E402


def _splits(k=10):
    s, f = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
    keep = s + f <= k
    s, f = s[keep] / k, f[keep] / k
    return s, f, 1 - s - f


def test_tau_rule_reduces_to_lambda_delta_when_the_sum_exceeds_one():
    ps, pf, pi = _splits()
    lam, d = 0.437, 0.093            # off the 0.1 grid, so >= and > agree
    new, _ = tau_rule.predict(ps, pf, pi, lam + d, 1 - lam + d)
    old, _ = _predict_lambda_delta(ps, pf, pi, lam, d, "two_sided", None)
    assert np.array_equal(new, old)


def test_tau_veto_only_flags_states_where_neither_side_fires():
    ps, pf, pi = _splits()
    for ts, tf in [(0.15, 0.25), (0.35, 0.3), (0.6, 0.7)]:
        pred, _ = tau_rule.predict(ps, pf, pi, ts, tf)
        v = pred == -2
        assert not np.any(v & ((ps >= ts) | (pf >= tf)))


def test_tau_fit_returns_a_grid_point_and_both_fire_counts():
    rng = np.random.default_rng(0)
    ps, pf, pi = _splits()
    idx = rng.integers(0, len(ps), 4000)
    y = np.where(ps[idx] + rng.normal(0, 0.1, idx.size) > pf[idx], 1, -1)
    f = tau_rule.fit(ps[idx], pf[idx], pi[idx], y, 0.9)
    assert f["tau_s"] in tau_rule.GRID and f["tau_f"] in tau_rule.GRID
    _, n_both = tau_rule.predict(ps, pf, pi, 0.2, 0.2)
    assert n_both == int(np.sum((ps >= 0.2) & (pf >= 0.2)))


def test_margin_rule_is_a_band_for_a_classifier():
    p = np.linspace(0, 1, 101)
    pred = margin_rule.predict(p, 1 - p, 0.4, 0.2)    # success p >= 0.7, failure p <= 0.4
    assert np.all(pred[p >= 0.7 - 1e-12] == 1) and np.all(pred[p <= 0.4 + 1e-12] == 0)
    assert np.all(pred[(p > 0.4 + 1e-12) & (p < 0.7 - 1e-12)] == -1)


def test_renorm_drops_invalid_votes_and_leaves_classifiers_alone():
    assert np.allclose(calib.renorm(np.array([0.3]), np.array([0.0]), 10), (3 + 0.5) / (3 + 0 + 1))
    p = np.array([0.2, 0.9])
    assert np.allclose(calib.renorm(p, 1 - p, None), calib.raw(p, None))

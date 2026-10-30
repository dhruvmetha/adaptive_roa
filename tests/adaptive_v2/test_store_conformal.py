"""Alpha sweep from the store (H02): both protocols, coverage near 1 - alpha on exchangeable data."""
import numpy as np

from adaptive_roa.adaptive_v2.eval.store_conformal import sweep


def _split(rng, n):
    y = np.where(rng.random(n) < 0.4, 1, -1)
    p_s = np.clip(np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.25, n), 0, 1)
    p_i = np.clip(rng.random(n) * 0.3, 0, 1)
    p_s = np.minimum(p_s, 1 - p_i)
    return p_s, 1 - p_s - p_i, p_i, y


def test_coverage_tracks_one_minus_alpha_for_both_protocols():
    rng = np.random.default_rng(0)
    cal, test = _split(rng, 4000), _split(rng, 20000)
    rows = sweep(cal, test, lam=0.5, delta=0.1, rule="two_sided", inv_thr=0.25)
    for r in rows:
        assert r["coverage"] >= 1 - r["alpha"] - 0.02, r
    filt = [r for r in rows if r["protocol"] == "filtered"]
    assert all(r["cal_retained"] < 1 and r["test_retained"] < 1 for r in filt)
    assert all(r["cal_retained"] == 1 for r in rows if r["protocol"] == "all")


def test_smaller_alpha_never_shrinks_q_hat():
    rng = np.random.default_rng(1)
    rows = sweep(_split(rng, 500), _split(rng, 2000), 0.5, 0.1, "one_sided", None)
    for proto in ("all", "filtered"):
        qs = [r["q_hat"] for r in rows if r["protocol"] == proto]
        assert qs == sorted(qs)

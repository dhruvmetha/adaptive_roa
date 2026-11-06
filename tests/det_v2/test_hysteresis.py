"""Ablation A9: tau* regime hysteresis (threshold=j_sel_hysteresis)."""
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.eval import det_v2_rule as R
from adaptive_roa.adaptive_v2.threshold import hysteresis as H

K = 100


def _data(seed, n_traj=60, rows=8):
    """Final-state fractions k/K with p_s + p_f <= 1, rows grouped by trajectory."""
    rng = np.random.default_rng(seed)
    t = np.repeat(np.arange(n_traj), rows)
    groups = t * 7 + 3                                          # arbitrary trajectory ids
    y = np.where(rng.random(n_traj) < 0.5, 1, -1)[t]
    # Successes lean to p_s, failures to p_f, some mass on invalid; then K draws.
    conc = np.where(y[:, None] == 1, [3.0, 1.0, 1.0], [1.0, 3.0, 1.0]) * rng.uniform(0.3, 2.0, (len(y), 1))
    counts = np.array([rng.multinomial(K, rng.dirichlet(c)) for c in conc])
    return counts[:, 0] / K, counts[:, 1] / K, y, groups


def _cfg(**over):
    base = {"predictor_type": "endpoint", "objective": "j_sel", "w": 0.1, "num_mc_samples": K,
            "verbose": False, "n_boot": 50, "se_mult": 2.0, "margin": None, "seed": 7}
    base.update(over)
    return SimpleNamespace(**base, get=lambda k, d=None: base.get(k, d))


def _backend(**over):
    cfg = _cfg(**over)
    return H.HysteresisJSelThresholdBackend(cfg, system=None, device="cpu")


def _j(ps, pf, y):
    return R.objectives(R.grid_counts(ps, pf, y), 0.1)["j_sel"]


def _gap_data(want_better):
    """A dataset whose global best J_sel lies in regime ``want_better``, with the gap to the other."""
    for seed in range(400):
        ps, pf, y, g = _data(seed)
        j_inv, j_out = H.regime_minima(_j(ps, pf, y))
        better = "inv" if j_inv < j_out else "out"
        if better == want_better and abs(j_inv - j_out) > 1e-3:
            return ps, pf, y, g, abs(j_inv - j_out)
    raise AssertionError(f"no seed gives a clear {want_better} optimum")


# --- the rule ------------------------------------------------------------------------------

@pytest.mark.parametrize("region", ["inv", "out"])
def test_region_restricted_fit_is_the_brute_force_best_in_that_region(region):
    ps, pf, y, _ = _data(1)
    res = R.fit(ps, pf, y, w=0.1, region=region)
    j = _j(ps, pf, y)
    mask = R.region_mask(region)
    assert R.region_of(res["tau_s"], res["tau_f"]) == region
    best = np.round(j, 12)[mask].min()
    ties = [(a, b) for a, b in zip(*np.nonzero(mask & (np.round(j, 12) == best)))]
    assert (round(res["tau_s"] * 100) - 1, round(res["tau_f"] * 100) - 1) == min(ties)   # smallest tau_s, then tau_f
    assert np.isclose(res["j"], j[mask].min(), atol=1e-12)


def test_region_masks_split_the_grid_at_tau_s_plus_tau_f_equal_one():
    inv, out = R.region_mask("inv"), R.region_mask("out")
    assert not (inv & out).any() and (inv | out).all()
    assert R.region_of(0.5, 0.5) == "inv" and R.region_of(0.49, 0.5) == "out" and R.region_of(0.01, 0.99) == "inv"


def test_weighted_counts_equal_duplicated_rows():
    ps, pf, y, _ = _data(2)
    mult = np.random.default_rng(0).integers(0, 4, len(y))
    wc = R.grid_counts(ps, pf, y, weights=mult)
    dup = R.grid_counts(np.repeat(ps, mult), np.repeat(pf, mult), np.repeat(y, mult))
    for k in ("tp", "fp", "fn", "tn", "n", "n_success", "n_failure"):
        assert np.array_equal(np.asarray(wc[k]), np.asarray(dup[k]).astype(float)), k
    ones = R.grid_counts(ps, pf, y, weights=np.ones(len(y)))
    plain = R.grid_counts(ps, pf, y)
    assert plain["tp"].dtype == np.int64 and np.array_equal(ones["tp"], plain["tp"])


# --- the backend ---------------------------------------------------------------------------

def test_round_zero_takes_the_global_argmin():
    for want in ("inv", "out"):
        ps, pf, y, g, _ = _gap_data(want)
        st = _backend(margin=0.5).optimize_probs(None, y, ps, pf, 1 - ps - pf, K, groups=g)
        glob = R.fit(ps, pf, y, w=0.1)
        assert (st.tau_s_star, st.tau_f_star) == (glob["tau_s"], glob["tau_f"])
        assert st.threshold_region == want


@pytest.mark.parametrize("better", ["inv", "out"])
def test_holds_the_incumbent_unless_the_other_regime_wins_by_more_than_the_margin(better):
    ps, pf, y, g, gap = _gap_data(better)
    incumbent = H.OTHER[better]
    for margin, want in ((2 * gap, incumbent), (gap / 2, better)):
        b = _backend(margin=margin)
        b.restore({"threshold_region": incumbent})
        st = b.optimize_probs(None, y, ps, pf, 1 - ps - pf, K, groups=g)
        res = R.fit(ps, pf, y, w=0.1, region=want)                 # refit inside the regime
        assert st.threshold_region == want and b.region == want
        assert (st.tau_s_star, st.tau_f_star) == (res["tau_s"], res["tau_f"])
        assert st.threshold_margin == margin
        j_inv, j_out = H.regime_minima(_j(ps, pf, y))
        assert (st.threshold_j_inv, st.threshold_j_out) == (j_inv, j_out)


def test_bootstrap_resamples_trajectories_not_rows():
    ps, pf, y, g = _data(3, n_traj=80, rows=1)
    se = H.bootstrap_gap_se(ps, pf, y, g, 0.1, 100, np.random.default_rng(5))
    m = 6                                                           # every trajectory gets 6 identical rows
    rep = lambda a: np.repeat(a, m)
    se_traj = H.bootstrap_gap_se(rep(ps), rep(pf), rep(y), rep(g), 0.1, 100, np.random.default_rng(5))
    se_rows = H.bootstrap_gap_se(rep(ps), rep(pf), rep(y), np.arange(len(y) * m), 0.1, 100, np.random.default_rng(5))
    assert se > 0 and np.isclose(se_traj, se, rtol=1e-9)           # copies move together: same SE
    assert se_rows < 0.7 * se_traj                                  # resampling rows would understate it


def test_bootstrap_margin_is_two_se_and_deterministic():
    ps, pf, y, g = _data(4)
    a = _backend(n_boot=60).optimize_probs(None, y, ps, pf, 1 - ps - pf, K, groups=g)
    b = _backend(n_boot=60).optimize_probs(None, y, ps, pf, 1 - ps - pf, K, groups=g)
    assert a.threshold_margin == b.threshold_margin > 0
    rng = np.random.default_rng([7, len(y), len(np.unique(g))])
    assert np.isclose(a.threshold_margin, 2.0 * H.bootstrap_gap_se(ps, pf, y, g, 0.1, 60, rng))
    with pytest.raises(ValueError, match="groups"):
        _backend().optimize_probs(None, y, ps, pf, 1 - ps - pf, K)


def test_classifiers_and_other_objectives_are_refused():
    with pytest.raises(ValueError, match="classifier"):
        _backend(predictor_type="classifier")
    with pytest.raises(ValueError, match="j_sel"):
        _backend(objective="j_ver")


# --- state, resume and wiring --------------------------------------------------------------

def test_regime_fields_are_stored_and_dropped_only_when_unset(tmp_path):
    from adaptive_roa.adaptive_v2.engine import _drop_unset_taus, restore_threshold_backend
    from adaptive_roa.adaptive_v2.types import ThresholdState
    ps, pf, y, g, _ = _gap_data("out")
    st = asdict(_backend(margin=0.01).optimize_probs(None, y, ps, pf, 1 - ps - pf, K, groups=g))
    _drop_unset_taus(st)
    assert {"threshold_region", "threshold_j_inv", "threshold_j_out", "threshold_margin"} <= st.keys()
    plain = asdict(ThresholdState(lambda_star=0.5, delta_star=0.0, tau_s_star=0.3, tau_f_star=0.4,
                                  threshold_objective="j_sel", threshold_w=0.1))
    _drop_unset_taus(plain)
    assert "threshold_region" not in plain and "tau_s_star" in plain

    (tmp_path / "epoch_002").mkdir()
    (tmp_path / "epoch_002" / "artifacts_v2.json").write_text(json.dumps({"threshold_state": st}))
    b = _backend()
    restore_threshold_backend(b, tmp_path, 0)                       # a fresh run keeps none
    assert b.region is None
    restore_threshold_backend(b, tmp_path, 3)
    assert b.region == st["threshold_region"] == "out"
    restore_threshold_backend(SimpleNamespace(), tmp_path, 3)       # backends without restore are left alone


class _Source:
    """TrajectoryDataSource with stubbed trajectories: trajectory i has i % 3 + 2 pairs."""
    timeout_intermediates = "keep"

    def get_all_endpoint_pairs_from_trajectory(self, idx, mode):
        n = idx % 3 + 2
        return np.full((n, 2), idx, dtype=np.float32), np.zeros((n, 2), dtype=np.float32)

    def get_label(self, idx):
        return 1 if idx % 2 else -1

    def get_trajectory_length(self, idx):
        return idx % 3 + 2

    def _drops_intermediates(self, idx, n_rows):
        return False


def _source():
    from adaptive_roa.adaptive.data_source import TrajectoryDataSource
    src = TrajectoryDataSource.__new__(TrajectoryDataSource)
    for k in ("get_all_endpoint_pairs_from_trajectory", "get_label", "get_trajectory_length", "_drops_intermediates"):
        setattr(src, k, getattr(_Source(), k))
    src.timeout_intermediates = "keep"
    return src


def test_endpoint_dataset_groups_name_each_pairs_trajectory():
    src = _source()
    starts, ends, labels, groups = src.build_endpoint_dataset([5, 9, 4], mode="train", return_groups=True)
    assert np.array_equal(groups, [5] * 4 + [9] * 2 + [4] * 3)
    assert np.array_equal(starts[:, 0], groups) and np.array_equal(labels, [1] * 6 + [-1] * 3)
    assert len(src.build_endpoint_dataset([5, 9, 4], mode="train")) == 3     # default unchanged


def test_val_data_groups_cover_the_first_n_val_train_trajectories():
    from adaptive_roa.adaptive.dataset_builder import AdaptiveDatasetBuilder
    b = AdaptiveDatasetBuilder.__new__(AdaptiveDatasetBuilder)
    b.candidate_mode, b.train_split, b.val_ratio, b.data_source = "start", [7, 2, 11, 4, 8], 0.4, _source()
    X, _e, y, g = b.get_val_data(return_groups=True)
    assert sorted(set(g.tolist())) == [2, 7] and len(g) == len(X) == len(y)
    b.candidate_mode = "suffix"
    with pytest.raises(NotImplementedError):
        b.get_val_data(return_groups=True)


def test_engine_hands_groups_to_the_backend_and_the_store_keeps_them(tmp_path, monkeypatch):
    from adaptive_roa.adaptive_v2 import engine as E
    from adaptive_roa.adaptive_v2.eval import prediction_store as ps
    ps_, pf_, y_, g_ = _data(5)
    X = np.c_[ps_, pf_].astype(np.float32)
    builder = SimpleNamespace(get_val_data=lambda return_groups=False: (X, X, y_, g_) if return_groups else (X, X, y_))
    monkeypatch.setattr(ps, "predict_split", lambda X, y, **kw: {"states": X, "true_labels": y})
    monkeypatch.setattr(ps, "fit_probabilities", lambda pred, k: (ps_, pf_, 1 - ps_ - pf_, K))
    backend = _backend(margin=0.0)
    fake = SimpleNamespace(threshold_backend=backend, pool=SimpleNamespace(dataset_builder=builder),
                           predictor_type="endpoint", system=None, evaluator=SimpleNamespace(attractor_radius=0.1),
                           cfg={}, device="cpu")
    st = E.AdaptiveEngine._fit_on_acquired_rows(fake, model_handle=None)
    assert st.threshold_region in ("inv", "out")
    assert np.array_equal(backend.last_fit["trajectory"], g_)
    ps.write_threshold_fit(tmp_path, backend.last_fit)
    z = np.load(tmp_path / "predictions_thrfit.npz")
    assert np.array_equal(z["trajectory"], g_)

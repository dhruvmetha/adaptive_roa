"""The FPR-capped rule, its fit and its acquisition score (adaptive_v2/eval/fpr_cap.py, Idea 6)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_roa.adaptive_v2.eval import fpr_cap
from adaptive_roa.adaptive_v2.eval import prediction_store as ps
from adaptive_roa.conformal.calibrator import nonconformity_scores_batch_two_sided
from adaptive_roa.conformal.decision import tie

OFF = fpr_cap.OFF


def _counts(rng, n, k, invalid_scale=0.3):
    """Random outcome fractions over k draws and labels correlated with them."""
    q = rng.beta(0.6, 0.6, n)
    pinv = rng.uniform(0, invalid_scale, n) * rng.integers(0, 2, n)
    probs = np.stack([q * (1 - pinv), (1 - q) * (1 - pinv), pinv], 1)
    c = np.array([rng.multinomial(k, p) for p in probs])
    ps_, pf_, pi_ = c[:, 0] / k, c[:, 1] / k, c[:, 2] / k
    y = np.where(rng.random(n) < np.clip(q + rng.normal(0, 0.15, n), 0, 1), 1, -1)
    return ps_, pf_, pi_, y


def _classifier(rng, n, distinct=None):
    p = rng.beta(0.5, 0.5, n)
    if distinct is not None:
        p = np.round(p * distinct) / distinct
    y = np.where(rng.random(n) < p, 1, -1)
    return p, 1.0 - p, np.zeros(n), y


# ---------------------------------------------------------------- predict


def _one(ps_, pf_, pi_, ts, tf, ti):
    return int(fpr_cap.predict(np.array([ps_]), np.array([pf_]), np.array([pi_]), ts, tf, ti)[0])


def test_predict_truth_table():
    assert _one(0.7, 0.2, 0.1, 0.6, 0.5, OFF) == 1           # success fires alone
    assert _one(0.1, 0.8, 0.1, 0.6, 0.5, OFF) == 0           # failure fires alone
    assert _one(0.6, 0.4, 0.0, 0.5, 0.3, OFF) == -1          # both fire -> uncertain
    assert _one(0.3, 0.3, 0.4, 0.5, 0.5, OFF) == -1          # neither fires -> uncertain
    assert _one(0.6, 0.1, 0.3, 0.5, 0.5, 0.3) == -2          # p_inv >= tau_inv beats a success
    assert _one(0.1, 0.6, 0.3, 0.5, 0.5, 0.3) == -2          # ... and a failure
    assert _one(0.3, 0.2, 0.5, 0.6, 0.7, OFF) == -2          # tied veto: 0.5 > 1 - min(0.6, 0.7)
    assert _one(0.3, 0.3, 0.4, 0.6, 0.7, OFF) == -1          # veto is strict: 0.4 is not > 0.4
    assert _one(1.0, 0.0, 0.0, OFF, 0.5, OFF) == -1          # tau = 1.01 turns success off
    assert _one(0.0, 1.0, 0.0, 0.5, OFF, OFF) == -1          # ... and failure


def test_predict_classifier_inputs():
    p = np.array([0.95, 0.5, 0.05, 0.9])
    want = np.array([1, -1, 0, 1], dtype=np.int8)
    assert np.array_equal(fpr_cap.predict(p, 1 - p, np.zeros(4), 0.9, 0.9, OFF), want)
    assert np.array_equal(fpr_cap.predict(p, None, None, 0.9, 0.9, OFF), want)
    assert fpr_cap.predict(p, None, None, 0.9, 0.9, OFF).dtype == np.int8


def test_predict_ties_compare_equal():
    s = np.float64(0.24) + np.float64(0.21)
    assert s != 0.45 and tie(s) == tie(0.45)                  # the float sum misses 0.45
    assert _one(s, 0.1, 1 - s - 0.1, 0.45, 0.5, OFF) == 1    # success fires at the tie
    assert _one(0.1, s, 1 - s - 0.1, 0.5, 0.45, OFF) == 0    # failure fires at the tie
    assert _one(0.1, 0.1, s, 0.5, 0.5, 0.45) == -2            # invalid fires at the tie
    # The tied veto is strict: 0.1 + 0.2 = 0.30000000000000004 ties 1 - 0.7, so not invalid.
    v = np.float64(0.1) + np.float64(0.2)
    assert v > 0.3 and _one(0.35, 0.35, v, 0.7, 0.8, OFF) == -1


# ---------------------------------------------------------------- fit


@pytest.mark.parametrize("seed", range(6))
def test_fit_meets_both_caps_and_the_floor(seed):
    rng = np.random.default_rng(seed)
    data = _counts(rng, 3000, 20) if seed % 2 == 0 else _classifier(rng, 3000)
    ps_, pf_, pi_, y = data
    caps = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.05)
    fits = fpr_cap.fit(ps_, pf_, pi_, y, caps)
    P, N = int(np.sum(y == 1)), int(np.sum(y != 1))
    for a in caps:
        r = fits[a]
        assert r["tau_s"] >= 0.5
        assert r["tau_s"] in fpr_cap.GRID and r["tau_f"] in fpr_cap.GRID
        assert r["tau_inv"] in fpr_cap.tau_inv_grid(pi_)
        if seed % 2 == 0:       # count data: the fit sees exactly what predict sees
            pred = fpr_cap.predict(ps_, pf_, pi_, r["tau_s"], r["tau_f"], r["tau_inv"])
            fp = int(np.sum((pred == 1) & (y != 1)))
            fn = int(np.sum((pred == 0) & (y == 1)))
            assert (fp, fn) == (r["fp"], r["fn"])
            assert int(np.sum((pred == 1) & (y == 1))) == r["tp"]
            assert int(np.sum((pred == 0) & (y != 1))) == r["tn"]
        assert r["fp"] / N <= a + 1e-12
        assert r["fn"] / P <= fpr_cap.ALPHA_F + 1e-12


def test_classifier_fit_bins_many_distinct_probabilities():
    rng = np.random.default_rng(3)
    ps_, pf_, pi_, y = _classifier(rng, 5000)
    assert len(np.unique(ps_)) > 2000
    cells, _, _ = fpr_cap._aggregate(ps_, pf_, pi_, y)
    assert len(cells) <= 10001
    assert np.array_equal(fpr_cap.tau_inv_grid(pi_), [OFF])


def _brute_force(ps_, pf_, pi_, y, a_s, a_f=fpr_cap.ALPHA_F, floor=fpr_cap.TAU_S_MIN):
    """The fit as a plain grid loop over predict(), in grid order (no aggregation)."""
    P, N = int(np.sum(y == 1)), int(np.sum(y != 1))
    best = None
    for ti in fpr_cap.tau_inv_grid(pi_):
        for ts in fpr_cap.GRID:
            if ts < floor - 1e-9:
                continue
            for tf in fpr_cap.GRID:
                pred = fpr_cap.predict(ps_, pf_, pi_, ts, tf, ti)
                tp = np.sum((pred == 1) & (y == 1)); fp = np.sum((pred == 1) & (y != 1))
                tn = np.sum((pred == 0) & (y != 1)); fn = np.sum((pred == 0) & (y == 1))
                if fp / N > a_s + 1e-12 or fn / P > a_f + 1e-12:
                    continue
                key = (-(tp + tn), fp + fn)           # max correct, then min errors; first wins ties
                if best is None or key < best[0]:
                    best = (key, (float(ts), float(tf), float(ti)))
    return best[1]


@pytest.mark.parametrize("kind,seed,cap", [("counts", 0, 0.01), ("counts", 1, 0.05), ("counts", 2, 0.0),
                                           ("classifier", 0, 0.02), ("classifier", 1, 0.1)])
def test_vectorized_fit_matches_a_brute_force_grid_loop(kind, seed, cap):
    rng = np.random.default_rng(100 + seed)
    ps_, pf_, pi_, y = _counts(rng, 60, 10) if kind == "counts" else _classifier(rng, 60, distinct=20)
    r = fpr_cap.fit(ps_, pf_, pi_, y, [cap])[cap]
    assert (r["tau_s"], r["tau_f"], r["tau_inv"]) == _brute_force(ps_, pf_, pi_, y, cap)


def test_an_infeasible_success_cap_turns_success_off():
    # True failures at p_s = 1 fire success for every tau_s <= 1, so no cap of 0 can hold.
    rng = np.random.default_rng(7)
    ps_, pf_, pi_, y = _counts(rng, 500, 20)
    ps_ = np.r_[ps_, np.ones(5)]; pf_ = np.r_[pf_, np.zeros(5)]; pi_ = np.r_[pi_, np.zeros(5)]
    y = np.r_[y, -np.ones(5, dtype=int)]
    r = fpr_cap.fit(ps_, pf_, pi_, y, [0.0])[0.0]
    assert r["tau_s"] == OFF
    assert r["tp"] == 0 and r["fp"] == 0
    assert not np.any(fpr_cap.predict(ps_, pf_, pi_, r["tau_s"], r["tau_f"], r["tau_inv"]) == 1)


def test_an_empty_or_one_class_split_still_returns_a_point():
    r = fpr_cap.fit([], [], [], [], [0.005])[0.005]
    assert (r["tau_s"], r["tau_f"], r["tau_inv"]) == (OFF, OFF, OFF)
    p = np.linspace(0, 1, 11)
    r = fpr_cap.fit(p, 1 - p, np.zeros(11), -np.ones(11), [0.005])[0.005]   # no true success
    assert r["fp"] == 0 and r["tau_s"] >= 0.5


# ---------------------------------------------------------------- acquisition


@pytest.mark.parametrize("seed", range(5))
def test_acquisition_score_is_the_two_sided_score_when_the_sum_reaches_one(seed):
    rng = np.random.default_rng(seed)
    ps_, pf_, pi_, _ = _counts(rng, 4000, 20) if seed % 2 == 0 else _classifier(rng, 4000)
    n_checked = 0
    for _ in range(40):
        ts, tf = (float(x) for x in rng.choice(fpr_cap.GRID, 2))
        if ts + tf < 1.0:
            continue
        lam, dl = fpr_cap.to_lambda_delta(ts, tf)
        old = nonconformity_scores_batch_two_sided(ps_, pf_, np.zeros(len(ps_), int), lam, dl)
        new = fpr_cap.acquisition_score(ps_, pf_, pi_, ts, tf, OFF)
        assert np.allclose(old, new, atol=1e-9, rtol=0), (ts, tf)
        n_checked += 1
    assert n_checked > 5


def _fake_estimator(ps_, pf_, pi_):
    class E:
        def estimate(self, states, **kw):
            idx = np.asarray(states, dtype=int).ravel()
            return ps_[idx], pf_[idx], pi_[idx]
    return E()


def test_both_fire_and_invalid_states_score_zero_and_are_direct_eligible():
    from adaptive_roa.adaptive.balanced_sampler import UncertainSampler

    ts, tf, ti = 0.4, 0.3, 0.5
    ps_ = np.array([0.6, 0.2, 0.45, 0.9, 0.05, 0.3])
    pf_ = np.array([0.35, 0.2, 0.4, 0.1, 0.9, 0.1])
    pi_ = 1 - ps_ - pf_                                   # [0.05, 0.6, 0.15, 0, 0.05, 0.6]
    pred = fpr_cap.predict(ps_, pf_, pi_, ts, tf, ti)
    assert pred.tolist() == [-1, -2, -1, 1, 0, -2]        # both, inv, both, succ, fail, inv
    score = fpr_cap.acquisition_score(ps_, pf_, pi_, ts, tf, ti)
    assert np.all(score[[0, 1, 2, 5]] == 0)
    assert np.all(score[[3, 4]] > 0)
    assert np.isclose(score[3], min(0.9 - ts, tf - 0.1, ti - 0.0))
    assert np.isclose(score[4], min(0.9 - tf, ts - 0.05, ti - 0.05))
    assert fpr_cap.both_fire(ps_, pf_, pi_, ts, tf, ti).tolist() == [True, False, True, False, False, False]

    sampler = UncertainSampler(dataset_builder=None, target_count=6)
    states = np.arange(6)[:, None]
    unc, cert = sampler._classify_batch_direct(states, list(range(6)), _fake_estimator(ps_, pf_, pi_),
                                               0.0, 0.0, "two_sided", rule="tau", taus=(ts, tf, ti))
    assert unc == [0, 1, 2, 5] and cert == [3, 4]
    # The lambda/delta test with the converted (negative) delta, for contrast: it calls
    # the both-fire states certain.
    lam, dl = fpr_cap.to_lambda_delta(ts, tf)
    unc_old, _ = sampler._classify_batch_direct(states, list(range(6)), _fake_estimator(ps_, pf_, pi_),
                                                lam, dl, "two_sided")
    assert 0 not in unc_old and 2 not in unc_old


def test_sampler_refuses_an_unknown_rule_or_missing_taus():
    from adaptive_roa.adaptive.balanced_sampler import _check_rule
    with pytest.raises(ValueError):
        _check_rule("margin", None)
    with pytest.raises(ValueError):
        _check_rule("tau", None)
    _check_rule("lambda_delta", None)


# ---------------------------------------------------------------- the store's fit-K counts


class _FakeSystem:
    def classify_attractor(self, x, radius):
        import torch
        v = x[:, 0]
        return torch.where(v > 0.3, 1, torch.where(v < -0.3, -1, 0)).to(torch.int8)


class _FakeModel:
    """predict_endpoint draws one noisy endpoint per call (the MC draw)."""
    def __init__(self):
        import torch
        self.g = torch.Generator().manual_seed(0)

    def eval(self):
        return self

    def predict_endpoint(self, xb):
        import torch
        return xb + torch.randn(xb.shape, generator=self.g)


def test_fit_k_counts_equal_the_first_k_stored_endpoints(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.5, (40, 2)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 1, -1)
    system = _FakeSystem()
    pred = ps.predict_split(X, y, predictor_type="generative", model=_FakeModel(), system=system,
                            num_mc_samples=30, attractor_radius=0.1, keep_endpoints=True)
    ps.write_split_prediction(tmp_path, "val", pred, sub_k=20)
    z = np.load(tmp_path / "predictions_val.npz")
    e = np.load(tmp_path / "endpoints_val.npz")
    import torch
    first = np.stack([system.classify_attractor(torch.from_numpy(e["endpoints"][:, s]), 0.1).numpy()
                      for s in range(20)], 1)
    assert np.array_equal(z["n_success_k20"], (first == 1).sum(1))
    assert np.array_equal(z["n_failure_k20"], (first == -1).sum(1))
    assert int(z["num_mc_samples"]) == 30
    assert np.all(z["n_success"] + z["n_failure"] + z["n_invalid"] == 30)
    # fit_probabilities reads the same first-20 counts.
    p_s, p_f, p_i, k = ps.fit_probabilities(pred, 20)
    assert k == 20 and np.array_equal(p_s * 20, z["n_success_k20"])
    vs, vf, vi, vy, kk = fpr_cap.load_val(tmp_path / "predictions_val.npz", fit_k=20)
    assert kk == 20 and np.allclose(vs, p_s) and np.allclose(vi, p_i)


# ---------------------------------------------------------------- score_run


def _write_run(root: Path, taus=None):
    rng = np.random.default_rng(5)
    run = root / "dr_x_fm_ranked_s1"
    ep = run / "epoch_000"
    ep.mkdir(parents=True)
    vs, vf, vi, vy = _counts(rng, 800, 100)
    k20 = {"n_success_k20": np.rint(vs * 20).astype(np.int16), "n_failure_k20": np.rint(vf * 20).astype(np.int16)}
    np.savez(ep / "predictions_val.npz", true_labels=vy.astype(np.int8),
             n_success=np.rint(vs * 100).astype(np.int16), n_failure=np.rint(vf * 100).astype(np.int16),
             n_invalid=np.rint(vi * 100).astype(np.int16), num_mc_samples=np.int32(100), **k20)
    ts_, tf_, ti_, ty = _counts(rng, 3000, 100)
    np.savez(ep / "full_roa_per_point.npz", p_success=ts_, p_failure=tf_, p_invalid=ti_, true_labels=ty)
    thr = {"lambda_star": 0.5, "delta_star": 0.1}
    if taus is not None:
        thr.update(tau_s=taus[0], tau_f=taus[1], tau_inv=taus[2], alpha_s=0.005)
    (ep / "artifacts_v2.json").write_text(json.dumps({"epoch": 0, "train_trajectories": 100,
                                                      "threshold_state": thr}))
    return run, (vs, vf, vi, vy), (ts_, tf_, ti_, ty)


def test_score_run_rows_match_a_direct_fit(tmp_path):
    run, val, test = _write_run(tmp_path, taus=(0.6, 0.5, 0.5))
    rows = fpr_cap.score_run(run, (0.0025, 0.005, 0.01))
    assert [r["alpha_s"] for r in rows] == [0.0025, 0.005, 0.01]
    fits = fpr_cap.fit(*val, (0.0025, 0.005, 0.01))
    for r in rows:
        f = fits[r["alpha_s"]]
        assert (r["tau_s"], r["tau_f"], r["tau_inv"]) == (f["tau_s"], f["tau_f"], f["tau_inv"])
        pred = fpr_cap.predict(*test[:3], f["tau_s"], f["tau_f"], f["tau_inv"])
        y = test[3]
        assert r["n_success_claims"] == int(np.sum(pred == 1))
        assert r["success_recovery"] == np.sum((pred == 1) & (y == 1)) / np.sum(y == 1)
        assert r["fpr_all_failures"] == np.sum((pred == 1) & (y != 1)) / np.sum(y != 1)
        assert r["val_k"] == 100 and r["run_tau_s"] == 0.6
    assert "matches_run" not in rows[0] and "matches_run" in rows[1]   # only at the run's own cap
    rows20 = fpr_cap.score_run(run, (0.005,), fit_k=20)
    assert rows20[0]["val_k"] == 20
    f20 = fpr_cap.fit(np.rint(val[0] * 20) / 20, np.rint(val[1] * 20) / 20,
                      1 - np.rint(val[0] * 20) / 20 - np.rint(val[1] * 20) / 20, val[3], [0.005])[0.005]
    assert rows20[0]["tau_s"] == f20["tau_s"] and rows20[0]["tau_f"] == f20["tau_f"]


def test_score_runs_cli_writes_the_fprcap_csv(tmp_path, monkeypatch):
    import csv
    import importlib.util
    import sys
    _write_run(tmp_path / "pen")
    spec = importlib.util.spec_from_file_location(
        "score_runs", Path(__file__).resolve().parents[2] / "scripts/det_rerun/score_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "det_rerun_scores_fprcap.csv"
    monkeypatch.setattr(sys, "argv", ["score_runs.py", "--root", str(tmp_path), "--thresholds", "fprcap",
                                      "--out", str(out)])
    mod.main()
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 3 and {r["alpha_s"] for r in rows} == {"0.0025", "0.005", "0.01"}
    assert all(r["system"] == "pen" and r["thresholds"] == "fprcap" for r in rows)


# ---------------------------------------------------------------- golden cells


EXP = Path("/common/users/shared/pracsys/adaptive_roa_experiments/det_rerun")
GOLDEN = [  # ranked arm, post-hoc fit at the stored 100-sample val counts (spec table)
    ("q3d30/dr_q3d30_fm_ranked_s99512", 10, 0.005, (0.56, 0.59, 0.5), 0.648253, 0.005095),
    ("q3d30/dr_q3d30_mlpclf_ranked_s99512", 10, 0.005, (0.90, 0.94, 1.01), 0.608741, 0.005262),
    ("pen/dr_pen_fm_ranked_s99512", 10, 0.005, (0.50, 0.16, 0.90), 0.957110, 0.000469),
    ("q3d110/dr_q3d110_bnnreg_ranked_s99512", 5, 0.0025, (1.01, 0.40, 0.70), 0.0, 0.0),
]


@pytest.mark.parametrize("cell,epoch,cap,taus,recovery,fpr", GOLDEN, ids=[g[0].split("/")[1] for g in GOLDEN])
def test_golden_cells(cell, epoch, cap, taus, recovery, fpr):
    ep = EXP / cell / f"epoch_{epoch:03d}"
    if not ((ep / "predictions_val.npz").exists() and (ep / "full_roa_per_point.npz").exists()):
        pytest.skip(f"shared data missing: {ep}")
    (row,) = fpr_cap.score_run(EXP / cell, [cap], epochs=[epoch])
    assert (row["tau_s"], row["tau_f"], row["tau_inv"]) == taus
    assert abs(row["success_recovery"] - recovery) < 5e-7
    assert abs(row["fpr_all_failures"] - fpr) < 5e-7

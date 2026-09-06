"""Level-set metrics for the stochastic campaigns.

At a level beta both the continuous truth p and the prediction p_hat are
thresholded at the same beta and a confusion matrix is read off. The tests pin
the identities that make the curve trustworthy: the counts agree with direct
thresholding, the rates agree with sklearn, a perfect model scores one, and the
integral of FP + FN over beta is exactly N times the MAE (so the raw-count area
adds nothing over MAE, and the value of the view is in the normalised rates).
"""
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# The scripts import each other by bare name after inserting their own
# directory, so tests do the same rather than spec_from_file_location, which
# leaves the module out of sys.modules and breaks @dataclass processing.
_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

spm = importlib.import_module("stoch_prob_metrics")


def _field(n=4000, seed=0):
    """A truth field with mass at both ends and in the interior, and a noisy p_hat."""
    rng = np.random.default_rng(seed)
    p = np.concatenate([np.zeros(n // 4), np.ones(n // 4), rng.uniform(size=n // 2)])
    p_hat = np.clip(p + rng.normal(0, 0.15, size=n), 0, 1)
    return p_hat, p


# ---------------------------------------------------------------- level_set_table
def test_default_levels_are_the_ten_bin_centres():
    assert list(spm.LEVELS) == pytest.approx([0.05, 0.15, 0.25, 0.35, 0.45,
                                              0.55, 0.65, 0.75, 0.85, 0.95])


def test_counts_match_direct_thresholding_at_a_level():
    p = np.array([0.0, 0.2, 0.6, 0.9, 1.0])
    p_hat = np.array([0.1, 0.7, 0.4, 0.95, 0.5])
    (row,) = spm.level_set_table(p_hat, p, betas=(0.5,))
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (2, 1, 1, 1)
    assert row["beta"] == 0.5
    assert row["n_pos_true"] == 3 and row["n_neg_true"] == 2


def test_rates_match_sklearn_at_every_default_level():
    sk = pytest.importorskip("sklearn.metrics")
    p_hat, p = _field()
    for row in spm.level_set_table(p_hat, p):
        b = row["beta"]
        y, yh = (p >= b).astype(int), (p_hat >= b).astype(int)
        assert row["tpr"] == pytest.approx(sk.recall_score(y, yh))
        assert row["tnr"] == pytest.approx(sk.recall_score(y, yh, pos_label=0))
        assert row["fpr"] == pytest.approx(1 - row["tnr"])
        assert row["fnr"] == pytest.approx(1 - row["tpr"])
        assert row["acc"] == pytest.approx(sk.accuracy_score(y, yh))
        assert row["bal_acc"] == pytest.approx(sk.balanced_accuracy_score(y, yh))
        assert row["f1"] == pytest.approx(sk.f1_score(y, yh))
        assert row["f05"] == pytest.approx(sk.fbeta_score(y, yh, beta=0.5))
        assert row["prec"] == pytest.approx(sk.precision_score(y, yh))


def test_perfect_model_scores_one_at_every_level():
    _, p = _field()
    for row in spm.level_set_table(p, p):
        for k in ("tpr", "tnr", "acc", "bal_acc", "f1", "f05", "prec", "vol_ratio"):
            assert row[k] == pytest.approx(1.0), (row["beta"], k)
        assert row["fpr"] == 0.0 and row["fnr"] == 0.0


def test_integral_of_fp_plus_fn_over_beta_is_n_times_mae():
    p_hat, p = _field()
    betas = np.linspace(0, 1, 4001)
    rows = spm.level_set_table(p_hat, p, betas=betas)
    err = np.array([r["fp"] + r["fn"] for r in rows], dtype=float)
    area = np.trapezoid(err, betas) if hasattr(np, "trapezoid") else np.trapz(err, betas)
    assert area == pytest.approx(len(p) * np.mean(np.abs(p_hat - p)), rel=2e-3)


def test_realized_success_is_mean_true_p_over_the_claimed_set():
    p_hat, p = _field()
    (row,) = spm.level_set_table(p_hat, p, betas=(0.7,))
    assert row["realized"] == pytest.approx(p[p_hat >= 0.7].mean())


def test_realized_and_volume_ratio_are_nan_when_a_set_is_empty():
    p = np.array([0.1, 0.2, 0.3])
    p_hat = np.array([0.1, 0.2, 0.3])
    (row,) = spm.level_set_table(p_hat, p, betas=(0.9,))
    assert math.isnan(row["realized"]) and math.isnan(row["vol_ratio"])
    assert math.isnan(row["tpr"]) and row["tnr"] == 1.0


def test_volume_ratio_is_predicted_over_true_set_size():
    p = np.array([0.0, 0.6, 0.7, 0.8])
    p_hat = np.array([0.9, 0.9, 0.1, 0.1])
    (row,) = spm.level_set_table(p_hat, p, betas=(0.5,))
    assert row["vol_ratio"] == pytest.approx(2 / 3)


def test_table_always_has_the_ten_levels_even_when_a_true_set_is_empty():
    """A fixed grid: the practitioner's beta is always there. An empty true set
    gives NaN rates and zero counts, never a missing row."""
    p = np.random.default_rng(1).uniform(0.15, 1.0, 1000)      # nothing below 0.1
    rows = {r["beta"]: r for r in spm.level_set_table(p, p)}
    assert set(rows) == set(spm.LEVELS)
    assert rows[0.05]["n_neg_true"] == 0 and math.isnan(rows[0.05]["tnr"])
    assert rows[0.05]["tpr"] == 1.0 and rows[0.45]["tnr"] == 1.0
    assert "used" not in rows[0.05]


# --------------------------------------------------------------- level_set_oracle
def _interior(n=20000, seed=3):
    return np.random.default_rng(seed).uniform(0.05, 0.95, n)


def test_oracle_rows_align_with_arm_rows():
    p = _interior()
    arm = spm.level_set_table(p, p)
    orc = spm.level_set_oracle(p, k=None, m=100)
    assert [r["beta"] for r in orc] == [r["beta"] for r in arm]
    assert set(orc[0]) == set(arm[0])


def test_oracle_is_near_one_for_huge_m_and_below_one_for_m_100():
    p = _interior()
    big = {r["beta"]: r for r in spm.level_set_oracle(p, k=None, m=1e7)}
    real = {r["beta"]: r for r in spm.level_set_oracle(p, k=None, m=100)}
    for b in (0.25, 0.55, 0.85):
        assert big[b]["tpr"] > 0.999 and big[b]["tnr"] > 0.999
        assert real[b]["tpr"] < 0.995 and real[b]["tnr"] < 0.995
        assert real[b]["bal_acc"] < big[b]["bal_acc"] - 0.01


def test_oracle_k_noise_lowers_the_ceiling_further():
    p = _interior()
    exact = {r["beta"]: r for r in spm.level_set_oracle(p, k=None, m=100)}
    mc = {r["beta"]: r for r in spm.level_set_oracle(p, k=100, m=100)}
    for b in (0.25, 0.55, 0.85):
        assert mc[b]["bal_acc"] < exact[b]["bal_acc"]


def test_oracle_is_deterministic_for_a_seed():
    p = _interior(n=2000)
    a = spm.level_set_oracle(p, k=100, m=100, seed=7)
    b = spm.level_set_oracle(p, k=100, m=100, seed=7)
    c = spm.level_set_oracle(p, k=100, m=100, seed=8)
    assert [r["tpr"] for r in a] == [r["tpr"] for r in b]
    assert [r["tpr"] for r in a] != [r["tpr"] for r in c]


def test_oracle_with_m_of_one_is_the_perfect_model():
    """Deterministic levels ship trials = 1; there is no noise to resample."""
    p = np.random.default_rng(5).integers(0, 2, 3000).astype(float)
    for r in spm.level_set_oracle(p, k=None, m=1):
        assert r["tpr"] == 1.0 and r["tnr"] == 1.0


# -------------------------------------------------------------- level_set_summary
def _rows(**over):
    base = []
    for i, b in enumerate(spm.LEVELS):
        r = {k: 0.5 for k in spm.LEVEL_STATS}
        r.update(beta=b, realized=b + 0.05)
        base.append(r)
    for i, kv in over.items():
        base[int(i)].update(kv)
    return base


def test_summary_averages_all_ten_levels():
    rows = _rows(**{"4": dict(tpr=1.0)})
    s = spm.level_set_summary(rows, _rows())
    assert s["n_levels"] == 10
    assert s["auc_tpr"] == pytest.approx((9 * 0.5 + 1.0) / 10)
    assert s["auc_tpr_oracle"] == pytest.approx(0.5)


def test_summary_skips_nan_levels_and_counts_the_defined_ones():
    rows = _rows(**{"3": dict(tpr=float("nan"), bal_acc=float("nan"))})
    s = spm.level_set_summary(rows, _rows())
    assert s["auc_tpr"] == pytest.approx(0.5)
    assert s["n_levels_defined"] == 9


def test_worst_overclaim_is_the_largest_shortfall_of_realized_below_beta():
    rows = _rows(**{"2": dict(realized=0.25 - 0.08), "6": dict(realized=0.65 - 0.03),
                    "9": dict(realized=0.95 - 0.12)})
    s = spm.level_set_summary(rows, _rows())
    assert s["worst_overclaim"] == pytest.approx(0.12)
    assert spm.level_set_summary(_rows(), _rows())["worst_overclaim"] == 0.0


def test_all_metrics_exposes_area_scalars_and_level_count():
    rng = np.random.default_rng(6)
    p_hat, p = _field(n=3000)
    succ = rng.binomial(100, p).astype(float)
    out = spm.all_metrics(p_hat, p, succ, np.full(len(p), 100.0), k=100)
    for st in spm.LEVEL_STATS:
        assert f"auc_{st}" in out and f"auc_{st}_oracle" in out
    assert out["n_levels"] == 10 and out["n_levels_defined"] == 10
    assert 0.0 < out["auc_bal_acc"] <= out["auc_bal_acc_oracle"] + 0.05
    assert "worst_overclaim" in out
    assert all(not isinstance(v, (list, dict)) for v in out.values()), \
        "all_metrics must stay flat: callers write it straight to CSV"


# ------------------------------------------------------------- epoch scoring / cli
@pytest.fixture
def synthetic_epoch(tmp_path):
    """A ground-truth grid plus one scored epoch, the shapes the pipeline writes."""
    rng = np.random.default_rng(11)
    n = 2500
    starts = rng.uniform(-1, 1, (n, 2))
    p_hat, p = _field(n=n, seed=12)
    trials = np.full(n, 100.0)
    succ = rng.binomial(100, p).astype(float)
    ds = tmp_path / "level"
    ds.mkdir()
    np.savez(ds / "eval_success_prob.npz", starts=starts, p_success=p,
             successes=succ, trials=trials)
    run = tmp_path / "run"
    ep = run / "epoch_003"
    ep.mkdir(parents=True)
    np.savez(ep / "full_roa_per_point.npz", start_states=starts.astype(np.float32),
             p_success=p_hat.astype(np.float32), p_failure=(1 - p_hat).astype(np.float32),
             p_invalid=np.zeros(n, np.float32))
    (ep / "artifacts_v2.json").write_text(
        '{"eval_metrics": {"num_mc_samples": 100}, "train_trajectories": 400}')
    return ds, run, ep


def test_score_epoch_full_returns_flat_scalars_and_one_long_row_per_level(synthetic_epoch):
    ds, run, ep = synthetic_epoch
    gt = spm.load_ground_truth(ds)
    row, levels = spm.score_epoch_full(ep, gt, None, None)
    assert row == spm.score_epoch(ep, gt, None, None)      # unchanged public shape
    assert len(levels) == len(spm.LEVELS)
    first = levels[0]
    for key in ("beta", "n_pos_true", "n_neg_true", "tp", "tn", "fp", "fn"):
        assert key in first
    for st in spm.LEVEL_STATS:
        assert st in first and f"{st}_oracle" in first
    assert "tp_oracle" in first
    # the scalar area must be the mean of the long rows it summarises
    assert row["auc_bal_acc"] == pytest.approx(np.nanmean([r["bal_acc"] for r in levels]))
    assert row["n_levels"] == len(levels)


def test_cli_writes_level_sets_csv_next_to_metrics_csv(synthetic_epoch, tmp_path):
    import csv, json, subprocess, sys
    ds, run, ep = synthetic_epoch
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([dict(predictor="fm", level="level", arm="toy",
                                     run_dir=str(run))]))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(Path(_SCRIPTS) / "stoch_prob_metrics.py"),
                        "--spec", str(spec), "--data-root", str(tmp_path),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    wide = list(csv.DictReader((out / "metrics.csv").open()))
    long = list(csv.DictReader((out / "level_sets.csv").open()))
    assert len(wide) == 1 and "auc_bal_acc" in wide[0]
    assert len(long) == len(spm.LEVELS)
    assert [long[0][k] for k in ("predictor", "level", "arm", "epoch")] == ["fm", "level", "toy", "3"]
    assert {float(r["beta"]) for r in long} == set(spm.LEVELS)
    assert all(r["tp"].isdigit() for r in long)
    assert "n_pos_true" in long[0] and "used" not in long[0]


# ------------------------------------------------------ incremental scorer helpers
inc = importlib.import_module("score_stoch_incremental")


def test_scored_epochs_ignores_rows_missing_the_level_set_column():
    rows = [
        {"level": "low", "arm": "epi_var", "epoch": "0", "KL": "0.1", "auc_bal_acc": "0.9"},
        {"level": "low", "arm": "epi_var", "epoch": "1", "KL": "0.1"},                      # pre-level-set row
        {"level": "low", "arm": "epi_var", "epoch": "2", "KL": "0.1", "auc_bal_acc": ""},   # blank counts as missing
        {"level": "med", "arm": "epi_var", "epoch": "3", "KL": "0.1", "auc_bal_acc": "0.9"},
    ]
    assert inc.scored_epochs(rows, "low") == {"epi_var": {0}}
    assert inc.scored_epochs(rows, "med") == {"epi_var": {3}}


def test_scored_epochs_with_rescore_reports_nothing_as_scored():
    rows = [{"level": "low", "arm": "epi_var", "epoch": "0", "auc_bal_acc": "0.9"}]
    assert inc.scored_epochs(rows, "low", rescore=True) == {}


def test_merge_rows_new_wins_and_appends_new_columns_after_the_old_order():
    lead = ["level", "arm", "epoch"]
    old = [{"level": "low", "arm": "a", "epoch": "0", "KL": "0.5", "sAUROC": "0.9"},
           {"level": "low", "arm": "a", "epoch": "1", "KL": "0.4", "sAUROC": "0.9"}]
    new = [{"level": "low", "arm": "a", "epoch": "1", "KL": "0.3", "sAUROC": "0.9", "auc_bal_acc": "0.8"},
           {"level": "low", "arm": "b", "epoch": "0", "KL": "0.2", "sAUROC": "0.9", "auc_bal_acc": "0.7"}]
    key = lambda r: (r["level"], r["arm"], int(r["epoch"]))
    ordered, fields, clash = inc.merge_rows(old, new, key, lead)
    assert fields == ["level", "arm", "epoch", "KL", "sAUROC", "auc_bal_acc"]
    assert clash == 1
    assert [key(r) for r in ordered] == [("low", "a", 0), ("low", "a", 1), ("low", "b", 0)]
    assert ordered[1]["KL"] == "0.3" and ordered[0].get("auc_bal_acc", "") == ""


def test_merge_rows_with_no_old_rows_leads_with_the_key_columns():
    lead = ["level", "arm", "epoch", "beta"]
    new = [{"tp": "1", "level": "low", "arm": "a", "epoch": "0", "beta": "0.05", "n_pos_true": "9"}]
    _, fields, _ = inc.merge_rows([], new, lambda r: (r["level"], r["arm"], int(r["epoch"]), float(r["beta"])), lead)
    assert fields[:4] == lead and set(fields[4:]) == {"tp", "n_pos_true"}


def test_levelsets_csv_path_sits_next_to_the_campaign_csv():
    p = inc.levelsets_path(Path("/x/docs/pendulum/lqr/gaussian_all_levels.csv"))
    assert p == Path("/x/docs/pendulum/lqr/gaussian_all_levels_levelsets.csv")


# --------------------------------------------------------------------------- alc
alc_mod = importlib.import_module("stoch_alc")


def test_alc_of_a_constant_curve_is_the_constant():
    assert alc_mod.alc([100, 200, 400], [0.3, 0.3, 0.3]) == pytest.approx(0.3)


def test_alc_of_a_linear_curve_is_the_midpoint_regardless_of_spacing():
    assert alc_mod.alc([0, 1, 10], [0.0, 0.1, 1.0]) == pytest.approx(0.5)


def test_alc_needs_two_points_and_increasing_budget():
    assert math.isnan(alc_mod.alc([100], [0.3]))
    with pytest.raises(ValueError):
        alc_mod.alc([200, 100], [0.3, 0.3])


def _campaign_rows():
    rows = []
    for arm, vals in {"dir00_s42": [0.5, 0.4, 0.3], "dir00_s43": [0.5, 0.42, 0.32],
                      "dir00_s44": [0.5, 0.38, 0.28], "epi_var": [0.5, 0.2, 0.1]}.items():
        for e, (v, tt) in enumerate(zip(vals, [100, 200, 300])):
            rows.append({"level": "low", "predictor": "fm", "arm": arm, "epoch": str(e),
                         "train_trajectories": str(tt), "KL": str(v),
                         "auc_bal_acc": str(1 - v)})
    rows.append({"level": "med", "predictor": "fm", "arm": "epi_var", "epoch": "0",
                 "train_trajectories": "100", "KL": "9", "auc_bal_acc": "0"})
    return rows


def test_alc_table_has_one_row_per_arm_plus_control_mean_and_floor():
    t = {r["arm"]: r for r in alc_mod.alc_table(_campaign_rows(), "low",
                                                metrics=("KL", "auc_bal_acc"))}
    assert set(t) == {"dir00_s42", "dir00_s43", "dir00_s44", "epi_var", "dir00_mean", "dir00_2sd"}
    # trapezoid over x = 100,200,300 of 0.5,0.4,0.3 is 0.4 (linear); 0.5,0.2,0.1 -> (0.35+0.15)/2
    assert t["dir00_s42"]["alc_KL"] == pytest.approx(0.4)
    assert t["epi_var"]["alc_KL"] == pytest.approx(0.25)
    assert t["epi_var"]["alc_auc_bal_acc"] == pytest.approx(0.75)
    assert t["epi_var"]["n_epochs"] == 3 and t["epi_var"]["budget_lo"] == 100 \
        and t["epi_var"]["budget_hi"] == 300
    seeds = [t[s]["alc_KL"] for s in ("dir00_s42", "dir00_s43", "dir00_s44")]
    assert t["dir00_mean"]["alc_KL"] == pytest.approx(np.mean(seeds))
    assert t["dir00_2sd"]["alc_KL"] == pytest.approx(2 * np.std(seeds, ddof=1))
    assert all(r["level"] == "low" for r in t.values())


def test_alc_table_skips_a_metric_an_arm_lacks_rather_than_dying():
    rows = _campaign_rows()
    for r in rows:
        if r["arm"] == "epi_var":
            r["auc_bal_acc"] = ""
    t = {r["arm"]: r for r in alc_mod.alc_table(rows, "low", metrics=("KL", "auc_bal_acc"))}
    assert math.isnan(t["epi_var"]["alc_auc_bal_acc"]) and t["epi_var"]["alc_KL"] == pytest.approx(0.25)


# ------------------------------------------------------------ level-set plotter
pls = importlib.import_module("plot_stoch_levelsets")


def _long_rows(arms=("dir00_s42", "dir00_s43", "dir00_s44", "epi_var", "clf_dir00"),
               epochs=(0, 1, 2)):
    rows = []
    for arm in arms:
        for e in epochs:
            for b in spm.LEVELS:
                r = {"level": "low", "predictor": "clf" if arm.startswith("clf") else "fm",
                     "arm": arm, "epoch": str(e), "beta": str(b),
                     "n_pos_true": str(int(1000 * (1 - b))), "n_neg_true": str(int(1000 * b))}
                for st in spm.LEVEL_STATS:
                    v = 0.5 + 0.1 * e + (0.05 if arm == "epi_var" else 0.0)
                    r[st] = str(v)
                    r[f"{st}_oracle"] = str(0.95)
                for c in ("tp", "tn", "fp", "fn"):
                    r[c] = "10"; r[f"{c}_oracle"] = "10.0"
                rows.append(r)
    return rows


def test_final_epoch_rows_picks_the_deepest_epoch_per_arm():
    rows = _long_rows()
    rows = [r for r in rows if not (r["arm"] == "clf_dir00" and r["epoch"] == "2")]
    by_arm, depth = pls.final_epoch_rows(rows)
    assert depth == {"dir00_s42": 2, "dir00_s43": 2, "dir00_s44": 2, "epi_var": 2, "clf_dir00": 1}
    assert set(by_arm["epi_var"]) == set(spm.LEVELS)
    assert by_arm["epi_var"][0.45]["tpr"] == pytest.approx(0.75)
    assert by_arm["clf_dir00"][0.45]["tpr"] == pytest.approx(0.6)


def test_draw_level_writes_a_png(tmp_path):
    by_arm, depth = pls.final_epoch_rows(_long_rows())
    out = pls.draw_level(by_arm, depth, tmp_path / "levelsets_low.png", "toy — low")
    assert out.exists() and out.stat().st_size > 10_000


# ------------------------------------------------------- all-levels plotter guard
pal = importlib.import_module("plot_stoch_all_levels")


def test_all_levels_plotter_carries_the_balanced_accuracy_area():
    assert "auc_bal_acc" in [m[0] for m in pal.METRICS]


def test_missing_columns_names_the_arm_and_epoch_that_lack_a_metric():
    m = {"epi_var": {0: {"KL": "0.1", "auc_bal_acc": "0.9"},
                     1: {"KL": "0.1", "auc_bal_acc": ""}},
         "dir00_s42": {0: {"KL": "0.1"}}}
    miss = pal.missing_columns(m, ["KL", "auc_bal_acc"])
    assert miss == [("dir00_s42", 0, "auc_bal_acc"), ("epi_var", 1, "auc_bal_acc")]
    assert pal.missing_columns(m, ["KL"]) == []


def test_merge_rows_prunes_a_column_no_row_fills():
    """A rescore replaces every row; a column the new scorer no longer writes
    must leave the file rather than survive as an all-blank column."""
    lead = ["level", "arm", "epoch"]
    old = [{"level": "low", "arm": "a", "epoch": "0", "KL": "0.5", "n_levels_used": "8"}]
    new = [{"level": "low", "arm": "a", "epoch": "0", "KL": "0.3", "n_levels": "10"}]
    _, fields, _ = inc.merge_rows(old, new, lambda r: (r["level"], r["arm"], int(r["epoch"])), lead)
    assert fields == ["level", "arm", "epoch", "KL", "n_levels"]


# ------------------------------------------------------------- budget window
def test_within_budget_keeps_rows_at_or_below_the_campaign_budget():
    rows = [{"arm": "a", "epoch": "0", "train_trajectories": "10000"},
            {"arm": "a", "epoch": "4", "train_trajectories": "30000"},
            {"arm": "a", "epoch": "5", "train_trajectories": "35000"}]
    assert [r["epoch"] for r in pal.within_budget(rows, {"budget": 30000})] == ["0", "4"]
    assert pal.within_budget(rows, {}) == rows          # no cap configured: untouched


def test_quad3d_corridor_campaign_reports_to_thirty_five_thousand_trajectories():
    assert pal.CAMPAIGNS["quad3d_cs"]["budget"] == 35000


def test_level_rows_within_budget_joins_the_wide_csv_for_the_trajectory_count():
    wide = [{"level": "L", "arm": "a", "epoch": "0", "train_trajectories": "10000"},
            {"level": "L", "arm": "a", "epoch": "1", "train_trajectories": "40000"},
            {"level": "L", "arm": "b", "epoch": "0", "train_trajectories": "10000"}]
    long = [{"level": "L", "arm": "a", "epoch": "0", "beta": "0.05"},
            {"level": "L", "arm": "a", "epoch": "1", "beta": "0.05"},
            {"level": "L", "arm": "b", "epoch": "0", "beta": "0.05"}]
    kept = pls.level_rows_within_budget(long, wide, {"budget": 30000})
    assert [(r["arm"], r["epoch"]) for r in kept] == [("a", "0"), ("b", "0")]
    assert pls.level_rows_within_budget(long, wide, {}) == long

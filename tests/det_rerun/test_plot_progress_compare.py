"""--compare renders the method figure and the FM-regime figure as a pair."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def pp(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("plot_progress", ROOT / "scripts/det_rerun/plot_progress.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", tmp_path)
    cell = ([10.0, 20.0], [[0.5, 0.4], [0.02, 0.01], [0.7, 0.8], [0.8, 0.85], [20.0, 15.0], [5.0, 4.0],
                           [15.0, 11.0], [0.1, 0.08], [0.2, 0.15]], True)
    data = {s: {("fm_det", m): cell for m in ("uniform", "direct", "ranked", "entropy")} for s in mod.SETTINGS}
    data["pen"][("det_mlp_clf", "ranked")] = cell
    monkeypatch.setattr(mod, "load", lambda: data)
    monkeypatch.setattr(sys, "argv", ["plot_progress.py", "--compare"])
    return mod, data


def test_compare_prints_both_figures_then_nothing(pp, capsys, tmp_path):
    mod, _ = pp
    mod.main()
    printed = capsys.readouterr().out.split()
    assert printed == [str(tmp_path / "compare_ranked.png"), str(tmp_path / "compare_fm_regimes.png")]
    assert all(Path(p).stat().st_size > 0 for p in printed)
    mod.main()
    assert capsys.readouterr().out == ""


def test_change_in_one_figure_reprints_the_pair(pp, capsys, tmp_path):
    mod, data = pp
    mod.main()
    capsys.readouterr()
    xs, ys, _ = data["q2d"][("fm_det", "entropy")]       # FM-regime figure only
    data["q2d"][("fm_det", "entropy")] = (xs, ys, False)
    mod.main()
    assert len(capsys.readouterr().out.split()) == 2


def test_own_uniform_cell_wins_over_the_cut_110k_run():
    spec = importlib.util.spec_from_file_location("pp2", ROOT / "scripts/det_rerun/plot_progress.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    long = ([10.0, 20.0, 30.0, 40.0], [[1.0, 2.0, 3.0, 4.0]], True)
    own = ([10.0, 12.0, 14.0], [[9.0, 8.0, 7.0]], False)
    data = {s: {} for s in mod.SETTINGS}
    data["q3d110"][("fm_det", "uniform")] = long
    data["q3d110"][("det_mlp_clf", "uniform")] = long
    data["q3d30"][("fm_det", "uniform")] = own
    out = mod.fill_shared_uniform(data)
    assert out["q3d30"][("fm_det", "uniform")] == own                       # its own cell
    assert out["q3d30"][("det_mlp_clf", "uniform")][0] == [10.0, 20.0, 30.0]  # cut until it banks


def test_folded_epochs_are_detected(tmp_path):
    import numpy as np
    spec = importlib.util.spec_from_file_location("pp3", ROOT / "scripts/det_rerun/plot_progress.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ps = np.array([0.2, 0.9, 0.0])
    np.savez(tmp_path / "full_roa_per_point.npz", p_success=ps, p_failure=1 - ps, p_invalid=np.zeros(3))
    assert mod.folded(tmp_path)
    np.savez(tmp_path / "full_roa_per_point.npz", p_success=ps, p_failure=1 - ps - 0.1, p_invalid=np.full(3, 0.1))
    assert not mod.folded(tmp_path)


def test_folded_epoch_reads_the_rescored_artifact(tmp_path, monkeypatch):
    import json
    import numpy as np
    spec = importlib.util.spec_from_file_location("pp4", ROOT / "scripts/det_rerun/plot_progress.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = tmp_path / "q3drl40" / "dr_q3drl40_fm_ranked_s99512"
    art = lambda inv: {"train_trajectories": 10000, "eval_metrics": {
        "lambda_delta": {"tp": 1, "tn": 1, "fp": 0, "fn": 0, "f1": 1.0, "invalid_pct": inv, "uncertain_pct": 0.0},
        "threshold_free": {"log_score": 0.1}}}
    for e, folded_ in ((0, True), (1, True), (2, False)):
        d = run / f"epoch_{e:03d}"
        d.mkdir(parents=True)
        (d / "artifacts_v2.json").write_text(json.dumps(art(0.0 if folded_ else 0.2)))
        ps = np.array([0.3, 0.7])
        pi = np.zeros(2) if folded_ else np.full(2, 0.1)
        np.savez(d / "full_roa_per_point.npz", p_success=ps, p_failure=1 - ps - pi, p_invalid=pi)
    resc = run / "rescored_ternary" / "epoch_000"
    resc.mkdir(parents=True)
    (resc / "artifacts_v2.json").write_text(json.dumps(art(0.15)))
    (tmp_path / "manifest.tsv").write_text(
        "name\tsystem\tarm\tmode\tseed\tmem\tpriority\toverrides\tscript\n"
        f"dr_q3drl40_fm_ranked_s99512\tq3drl40\tfm_det\tranked\t99512\t16G\t1\toutput_dir={run}\tscripts/run_adaptive.py\n")
    monkeypatch.setattr(mod, "HERE", tmp_path)
    xs, ys, fin = mod.load()["q3drl40"][("fm_det", "ranked")]
    assert len(xs) == 2                                 # epoch 1 folded, not re-scored: left out
    inv = mod.METRICS.index("Invalid %")
    assert [round(v, 2) for v in ys[inv]] == [15.0, 20.0]  # invalid %: re-scored epoch 0, then epoch 2


def test_invalid_and_outcome_unc_share_a_scale_per_row(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("pp5", ROOT / "scripts/det_rerun/plot_progress.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    figs = []
    monkeypatch.setattr(mod.plt, "close", lambda fig: figs.append(fig))
    lo = ([10.0, 20.0], [[0.5, 0.4], [0.02, 0.01], [0.7, 0.8], [0.8, 0.85], [3.0, 2.0], [1.0, 2.0], [2.0, 0.5],
                         [0.1, 0.08], [0.2, 0.15]], True)
    hi = ([10.0, 20.0], [[0.5, 0.4], [0.02, 0.01], [0.7, 0.8], [0.8, 0.85], [30.0, 20.0], [12.0, 9.0], [18.0, 11.0],
                         [0.1, 0.08], [0.2, 0.15]], True)
    data = {s: {("fm_det", "ranked"): lo} for s in mod.SETTINGS}
    data["q3d110"] = {("fm_det", "ranked"): hi}
    mod.plot_compare(data, tmp_path / "c.png")
    axes = figs[0].axes
    n = len(mod.METRICS)
    inv, unc = mod.METRICS.index("Invalid %"), mod.METRICS.index("Outcome-unc %")
    for i, s in enumerate(mod.SETTINGS):
        a, b = axes[i * n + inv].get_ylim(), axes[i * n + unc].get_ylim()
        assert a == b, (s, a, b)
    i = mod.SETTINGS.index("q3d110")
    assert axes[i * n + inv].get_ylim()[1] > 18.0 > axes[0 * n + inv].get_ylim()[1]   # per row, not global


def test_false_certification_rate_and_success_recall_follow_kl():
    import math
    spec = importlib.util.spec_from_file_location("pp6", ROOT / "scripts/det_rerun/plot_progress.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.METRICS[:4] == ["KL (nats)", "False cert. rate", "Success recall", "F1"]
    # 100 success claims, 10 of them true failures: FCR = 10 / 100.
    # 120 true successes: 90 certified, 5 called failure, 25 abstained. Every one not
    # certified is a false negative, abstentions included: recall = 90 / 120.
    art = {"eval_metrics": {
        "threshold_free": {"log_score": 0.3},
        "lambda_delta": {"tp": 90, "fp": 10, "tn": 50, "fn": 5, "f1": 0.9, "invalid_pct": 0.1, "uncertain_pct": 0.05},
        "conservative_lambda_delta": {"tp": 90, "fp": 10, "tn": 75, "fn": 30}}}
    kl, fcr, rec, f1, unc, inv, ounc, fpr, fnr = mod.metrics(art)
    assert (kl, f1) == (0.3, 0.9)
    assert math.isclose(fcr, 0.1) and math.isclose(rec, 0.75)
    assert math.isclose(fnr, 5 / 95) and math.isclose(fpr, 10 / 60)   # committed states only, unchanged
    assert math.isclose(unc, 15.0) and math.isclose(inv, 10.0) and math.isclose(ounc, 5.0)
    none = {"eval_metrics": {"lambda_delta": {"tp": 0, "fp": 0, "tn": 4, "fn": 0}}}
    _, fcr, rec, *_ = mod.metrics(none)
    assert math.isnan(fcr)                    # no success claims
    assert math.isnan(rec)                    # no conservative block: the true-success total is unknown

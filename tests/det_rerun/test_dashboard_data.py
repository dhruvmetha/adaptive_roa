"""The dashboard's FPR-capped 110k/100k quad3d settings are the 200k runs' first 11 epochs."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location("dashboard_data", ROOT / "scripts/det_rerun/dashboard_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_read_off_takes_epochs_0_to_10_of_the_200k_cells():
    dd = _mod()
    rows = [{"setting": "q3d200", "cell": "c", "epoch": e, "budget": 200000, "n_epochs": 20, "status": "running"}
            for e in range(13)] + [{"setting": "pen", "cell": "p", "epoch": 0, "budget": 600, "n_epochs": 11,
                                     "status": "running"}]
    out = dd.read_off(rows)
    assert [r["epoch"] for r in out] == list(range(11))
    assert all(r["setting"] == "q3d110" and r["budget"] == 110000 and r["n_epochs"] == 11 for r in out)
    assert all(r["status"] == "finished" for r in out)                    # 13 epochs banked >= 11
    assert all(r["setting"] == "q3d200" for r in rows if r["cell"] == "c")   # source rows untouched
    assert dd.read_off([dict(r, epoch=0) for r in rows[:1]])[0]["status"] == "running"


HEADER = "name\tsystem\tarm\tmode\tseed\tmem\tpriority\toverrides\tscript\n"


def _uniform_cell(root, system="pen", folded=False):
    """An FM uniform cell with one epoch whose tau-rule labels are the same at every valid fit.

    Val: pure successes and pure failures at K=100, so every K'=20 subsample is the same and
    the fit carries no invalid mass (tau_inv off). Test, with p = (p_s, p_f, p_inv):
    4 true successes (1, 0, 0) certified, 1 true success (0, 0, 1) invalid, 1 true success
    (0, 1, 0) called failure, 3 true failures (0, 1, 0), 1 true failure (1, 0, 0) certified.
    """
    import json
    import numpy as np
    run = root / system / f"dr_{system}_fm_uniform_s99512"
    d = run / "epoch_000"
    d.mkdir(parents=True)
    yv = np.array([1] * 6 + [-1] * 6, dtype=np.int8)
    n_s = np.where(yv == 1, 100, 0).astype(np.int16)
    np.savez(d / "predictions_val.npz", true_labels=yv, n_success=n_s, n_failure=(100 - n_s).astype(np.int16),
             n_invalid=np.zeros(12, np.int16), num_mc_samples=np.int32(100), attractor_radius=np.float64(0.1))
    p = np.array([[1, 0, 0]] * 4 + [[0, 0, 1], [0, 1, 0]] + [[0, 1, 0]] * 3 + [[1, 0, 0]], dtype=float)
    y = np.array([1] * 6 + [-1] * 4)
    three_way = dict(p_success=p[:, 0], p_failure=p[:, 1], p_invalid=p[:, 2], true_labels=y)
    art = lambda kl: {"train_trajectories": 600, "threshold_state": {"lambda_star": 0.5, "delta": 0.05},
                      "eval_metrics": {"threshold_free": {"log_score": kl},
                                       "lambda_delta": {"tp": 1, "tn": 1, "fp": 0, "fn": 0, "f1": 0.11,
                                                        "invalid_pct": 0.0, "uncertain_pct": 0.0}}}
    if folded:     # the two-way test scoring, and its three-way re-score beside it
        np.savez(d / "full_roa_per_point.npz", p_success=p[:, 0], p_failure=1 - p[:, 0], p_invalid=np.zeros(10),
                 true_labels=y)
        (d / "artifacts_v2.json").write_text(json.dumps(art(9.9)))
        rd = run / "rescored_ternary" / "epoch_000"
        rd.mkdir(parents=True)
        np.savez(rd / "full_roa_per_point.npz", **three_way)
        (rd / "artifacts_v2.json").write_text(json.dumps(art(0.55)))
    else:
        np.savez(d / "full_roa_per_point.npz", **three_way)
        (d / "artifacts_v2.json").write_text(json.dumps(art(0.42)))
    return f"dr_{system}_fm_uniform_s99512\t{system}\tfm_det\tuniform\t99512\t16G\t1\toutput_dir={run} n_epochs=1\tx\n"


def test_fm_uniform_is_rescored_into_the_fprcap_rows(tmp_path):
    dd = _mod()
    (tmp_path / "m.tsv").write_text(HEADER + _uniform_cell(tmp_path / "exp"))
    nominal, fprcap = dd.build(tmp_path / "m.tsv", tmp_path / "cache.json")
    assert [(r["model"], r["regime"], r["f1"]) for r in nominal] == [("FM", "uniform", 0.11)]
    assert "tau_s" not in nominal[0] and not nominal[0].get("rescored")
    (r,) = fprcap
    assert (r["model"], r["regime"], r["setting"], r["rescored"], r["status"]) == ("FM", "uniform", "pen", True, "finished")
    assert r["kl"] == 0.42                         # threshold-free: the nominal epoch's own
    assert r["tau_s"] >= 0.5 and r["tau_inv"] > 1.0
    assert r["fcr"] == 0.2                         # 1 of 5 success claims is a true failure
    assert abs(r["success_recall"] - 4 / 6) < 1e-6  # the invalid and the called-failure success both miss
    assert (r["fpr"], r["fnr"]) == (0.25, 0.2)     # committed states: 1 / (1 + 3), 1 / (1 + 4)
    assert (r["invalid_pct"], r["uncertain_pct"], r["f1"]) == (10.0, 0.0, 0.8)


def test_rescore_cache_is_reused_until_a_store_file_changes(tmp_path, monkeypatch):
    import os
    from adaptive_roa.adaptive_v2.eval import fpr_cap
    dd = _mod()
    (tmp_path / "m.tsv").write_text(HEADER + _uniform_cell(tmp_path / "exp"))
    first = dd.build(tmp_path / "m.tsv", tmp_path / "cache.json")[1]
    calls = []
    real_fit = fpr_cap.fit
    monkeypatch.setattr(fpr_cap, "fit", lambda *a, **k: calls.append(1) or real_fit(*a, **k))
    assert dd.build(tmp_path / "m.tsv", tmp_path / "cache.json")[1] == first and not calls
    test = next((tmp_path / "exp").rglob("full_roa_per_point.npz"))
    st = test.stat()
    os.utime(test, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert dd.build(tmp_path / "m.tsv", tmp_path / "cache.json")[1] == first and len(calls) == 1


def test_folded_uniform_epoch_is_rescored_from_its_three_way_rescore(tmp_path):
    dd = _mod()
    (tmp_path / "m.tsv").write_text(HEADER + _uniform_cell(tmp_path / "exp", system="q3drl40", folded=True))
    (r,) = dd.build(tmp_path / "m.tsv", tmp_path / "cache.json")[1]
    assert (r["kl"], r["invalid_pct"], r["fcr"]) == (0.55, 10.0, 0.2)


def test_rescored_uniform_rows_are_not_read_off(tmp_path):
    dd = _mod()
    (tmp_path / "m.tsv").write_text(HEADER + _uniform_cell(tmp_path / "exp", system="q3d200"))
    _, fprcap = dd.build(tmp_path / "m.tsv", tmp_path / "cache.json")
    assert [r["setting"] for r in fprcap] == ["q3d200"]   # q3d110 has its own uniform cell


def test_script_imports_its_own_tree_without_pythonpath(tmp_path):
    """The cycle runs it with no PYTHONPATH; the env's editable install is another checkout."""
    import os
    import subprocess
    import sys
    code = ("import runpy; runpy.run_path(%r, run_name='dd'); import adaptive_roa; print(adaptive_roa.__file__)"
            % str(ROOT / "scripts/det_rerun/dashboard_data.py"))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert Path(out.stdout.strip().splitlines()[-1]).resolve().is_relative_to(ROOT)

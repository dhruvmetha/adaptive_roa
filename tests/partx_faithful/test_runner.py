import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from adaptive_roa.partx.tree import build_root
from adaptive_roa.partx_faithful.algorithm import PartX, PartXConfig
from adaptive_roa.partx_faithful.runner import resolve_n0, run_partx_faithful, schedule
from adaptive_roa.systems.pendulum import PendulumSystem

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
spm = importlib.import_module("stoch_prob_metrics")


def _p_true(X):
    return 1.0 / (1.0 + np.exp(-4.0 * (1.0 - np.abs(X[:, 0]))))


@pytest.fixture
def toy(tmp_path):
    system = PendulumSystem()
    root = build_root(system)
    rng = np.random.default_rng(0)
    starts = rng.uniform(root.low, root.high, size=(3000, 2))
    labels = np.where(rng.random(3000) < _p_true(starts), 1.0, -1.0)
    g0, g1 = np.meshgrid(np.linspace(-3.0, 3.0, 15), np.linspace(-6.0, 6.0, 15))
    grid = np.round(np.column_stack([g0.ravel(), g1.ravel()]), 6)
    p = _p_true(grid)
    droot = tmp_path / "data"
    droot.mkdir()
    np.savetxt(droot / "test_set.txt", np.column_stack([grid, p]), delimiter=",", fmt="%.6f")
    np.savez(droot / "eval_success_prob.npz", starts=grid, p_success=p,
             successes=np.round(50 * p), trials=np.full(len(p), 50.0))
    return system, starts, labels, droot


def _go(toy, out):
    system, starts, labels, droot = toy
    return run_partx_faithful(
        system=system, starts=starts, labels=labels, eval_states_file=str(droot / "test_set.txt"),
        output_dir=str(out), initial=60, samples_per_epoch=30, n_epochs=4,
        pcfg=PartXConfig(n0=5, n_bo=2, n_c=10, R=4, M=50, seed=0),
        gp_kwargs=dict(n_restarts=1, max_points=500), falsify="failure",
        attractor_radius=0.1, extra={"test": True})


def test_schedule_and_n0():
    assert schedule(100, 100, 20) == (1900, [100 + 100 * e for e in range(20)])
    assert resolve_n0("auto", 2) == 10 and resolve_n0("auto", 13) == 30
    assert resolve_n0(None, 4) == 30 and resolve_n0(30, 2) == 30


def test_every_epoch_is_written_and_scoreable(toy, tmp_path):
    out = tmp_path / "run"
    _go(toy, out)
    tt, d2 = [], []
    for e in range(4):
        d = out / f"epoch_{e:03d}"
        for f in ("full_roa_per_point.npz", "artifacts_v2.json", "partx_leaves.npz"):
            assert (d / f).exists(), f"{d.name}/{f}"
        a = json.loads((d / "artifacts_v2.json").read_text())
        assert a["sampling_mode"] == "partx_faithful"
        tt.append(a["train_trajectories"])
        d2.append(a["acquisition"]["d2_indices"])
    assert tt == [60, 90, 120, 150]
    assert [len(x) for x in d2] == [30, 30, 30, 0]
    flat = [i for x in d2 for i in x]
    assert len(set(flat)) == 90 and min(flat) >= 60          # never the initial set
    with np.load(out / "epoch_003" / "full_roa_per_point.npz") as z:
        assert np.all((z["p_success"] > 0) & (z["p_success"] < 1))
    row, _ = spm.score_epoch_full(out / "epoch_003", spm.load_ground_truth(toy[3]), None, None)
    assert np.isfinite(row["KL"]) and row["train_trajectories"] == 150
    assert json.loads((out / "final_results.json").read_text())["used"] == 90
    assert json.loads((out / "epoch_003" / "artifacts_v2.json").read_text())["acquisition"]["diagnostics"]["done"]


def _epoch(d):
    a = json.loads((d / "artifacts_v2.json").read_text())
    with np.load(d / "full_roa_per_point.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}
    with np.load(d / "partx_leaves.npz") as z:
        leaves = {k: z[k].copy() for k in z.files}
    return a, arrays, leaves


def _assert_same_run(a_dir, b_dir, n_epochs=4):
    for e in range(n_epochs):
        ja, arr_a, lv_a = _epoch(a_dir / f"epoch_{e:03d}")
        jb, arr_b, lv_b = _epoch(b_dir / f"epoch_{e:03d}")
        assert ja == jb, f"epoch {e}: artifacts_v2.json differs"
        for k in arr_a:
            np.testing.assert_array_equal(arr_a[k], arr_b[k])
        for k in lv_a:
            np.testing.assert_array_equal(lv_a[k], lv_b[k])
    fa = json.loads((a_dir / "final_results.json").read_text())
    fb = json.loads((b_dir / "final_results.json").read_text())
    fa.pop("elapsed_s_this_process")
    fb.pop("elapsed_s_this_process")
    assert fa == fb


@pytest.mark.parametrize("crash_at", [3, 4])
def test_a_crash_between_iterations_resumes_identically(toy, tmp_path, monkeypatch, crash_at):
    _go(toy, tmp_path / "a")
    real_step = PartX.step
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == crash_at:
            raise RuntimeError("simulated preemption")
        return real_step(self)

    monkeypatch.setattr(PartX, "step", flaky)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        _go(toy, tmp_path / "b")
    monkeypatch.setattr(PartX, "step", real_step)
    _go(toy, tmp_path / "b")
    _assert_same_run(tmp_path / "a", tmp_path / "b")


def test_a_crash_mid_iteration_after_a_checkpoint_resumes_identically(toy, tmp_path, monkeypatch):
    _go(toy, tmp_path / "a")
    real_take = PartX._take
    marker = tmp_path / "b" / "epoch_002" / "artifacts_v2.json"

    def flaky(self, region, handles):
        ids = real_take(self, region, handles)
        if marker.exists():
            raise RuntimeError("simulated preemption")
        return ids

    monkeypatch.setattr(PartX, "_take", flaky)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        _go(toy, tmp_path / "b")
    monkeypatch.setattr(PartX, "_take", real_take)
    _go(toy, tmp_path / "b")
    _assert_same_run(tmp_path / "a", tmp_path / "b")


def test_a_pool_too_small_for_the_budget_leaves_the_rest_unspent(toy, tmp_path):
    system, starts, labels, droot = toy
    out = tmp_path / "small"
    run_partx_faithful(
        system=system, starts=starts[:110], labels=labels[:110],
        eval_states_file=str(droot / "test_set.txt"), output_dir=str(out), initial=60,
        samples_per_epoch=30, n_epochs=4, pcfg=PartXConfig(n0=5, n_bo=2, n_c=10, R=4, M=50, seed=0),
        gp_kwargs=dict(n_restarts=1, max_points=500), falsify="failure", attractor_radius=0.1,
        extra={"test": True})
    tt = [json.loads((out / f"epoch_{e:03d}" / "artifacts_v2.json").read_text())["train_trajectories"]
          for e in range(4)]
    assert tt[:2] == [60, 90] and tt[2] == tt[3] < 150      # D8: never topped up
    last = json.loads((out / "epoch_003" / "artifacts_v2.json").read_text())["acquisition"]["diagnostics"]
    assert last["done"] and last["unspent"] == 90 - (tt[3] - 60)

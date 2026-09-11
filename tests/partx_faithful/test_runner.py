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


def test_a_crash_and_resume_matches_an_uninterrupted_run(toy, tmp_path, monkeypatch):
    _go(toy, tmp_path / "a")
    real_step = PartX.step
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated preemption")
        return real_step(self)

    monkeypatch.setattr(PartX, "step", flaky)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        _go(toy, tmp_path / "b")
    monkeypatch.setattr(PartX, "step", real_step)
    _go(toy, tmp_path / "b")
    for e in range(4):
        da, db = tmp_path / "a" / f"epoch_{e:03d}", tmp_path / "b" / f"epoch_{e:03d}"
        with np.load(da / "full_roa_per_point.npz") as za, np.load(db / "full_roa_per_point.npz") as zb:
            np.testing.assert_array_equal(za["p_success"], zb["p_success"])
        ja = json.loads((da / "artifacts_v2.json").read_text())
        jb = json.loads((db / "artifacts_v2.json").read_text())
        assert ja["acquisition"]["d2_indices"] == jb["acquisition"]["d2_indices"]
        assert ja["train_trajectories"] == jb["train_trajectories"]

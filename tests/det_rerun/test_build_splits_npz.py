"""End-to-end run of scripts/det_rerun/build_splits_npz.py on a tiny synthetic npz dataset."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/det_rerun/build_splits_npz.py"
DS = "stochastic/quadrotor3D/corridor_sine_ambient/ppo_1500K/f_0.00"
K = 13


def _write(root: Path, n_flights=80, seed=1, corrupt=False):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    lens = rng.integers(2, 7, size=n_flights)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    states = rng.uniform(-1, 1, size=(offsets[-1], K)).astype(np.float32)
    labels = (np.arange(n_flights) % 4 == 0).astype(np.uint8)
    np.savez(root / "train.npz", states=states, offsets=offsets, starts=states[offsets[:-1]].astype(np.float64),
             labels=labels, timeout=np.zeros(n_flights, np.uint8))
    tid, row = [], []
    for t in range(n_flights):              # two picks per flight, one of them the start
        for r in sorted({0, int(rng.integers(0, lens[t]))}):
            tid.append(t)
            row.append(offsets[t] + r)
    tid, row = np.array(tid), np.array(row)
    picked = states[row].copy()
    if corrupt:
        picked[3, 0] += 0.5
    np.savez(root / "eval_fps.npz", tid=tid, row=row, states=picked)
    finals = states[offsets[tid + 1] - 1][:, None, :]
    p = labels[tid].astype(float)
    np.savez(root / "eval_success_prob.npz", finals=finals, p_success=p, trials=np.ones(len(tid), np.int32))
    lines = [",".join(f"{v:.6f}" for v in s) + f",{pi:.4f}" for s, pi in zip(picked, p)]
    (root / "eval_states.txt").write_text("\n".join(lines) + "\n")
    return states, offsets, tid


def _run(tmp_path, sizes="pool=40,cal=6,val=6,test=10", corrupt=False):
    root = tmp_path / DS
    data = _write(root, corrupt=corrupt)
    env = dict(os.environ, DATA_DIR=str(tmp_path), PYTHONPATH=str(REPO))
    res = subprocess.run([sys.executable, str(SCRIPT), "--system", "quad3d_rl", "--sizes", sizes],
                         env=env, capture_output=True, text=True)
    return root, data, res


def test_splits_are_disjoint_and_in_det_layout(tmp_path):
    root, (states, offsets, tid), res = _run(tmp_path)
    assert res.returncode == 0, res.stderr
    out = root / "splits_det_rerun"
    man = json.loads((out / "split_manifest.json").read_text())
    assert man["pool_trajectories"]["n"] == 40
    pool = {int(s.split("_")[1].split(".")[0]) for s in (out / "pool_trajectories.txt").read_text().split()}
    pool_rows = {states[i].tobytes() for t in pool for i in range(offsets[t], offsets[t + 1])}
    seen = set()
    for split in ("cal", "val", "test"):
        lines = (out / f"{split}_set.txt").read_text().split()
        assert man["splits"][split]["n"] == len(lines) > 0
        for ln in lines:
            v = ln.split(",")
            assert len(v) == 2 * K + 1 and v[-1] in ("0", "1")
            start = np.array([float(x) for x in v[:K]], dtype=np.float32)
            assert not any(np.allclose(start, np.frombuffer(b, np.float32), atol=1e-6) for b in pool_rows)
            assert ln not in seen
            seen.add(ln)
        flights = set((out / f"{split}_trajectories.txt").read_text().split())
        assert not flights & {f"sequence_{t}.txt" for t in pool}
    npz = np.load(out / "train.npz")
    assert len(npz["offsets"]) == 41 and int(npz["labels"].sum()) == man["pool_trajectories"]["success"]
    order = [int(x) for x in (out / "pool_order_99512.txt").read_text().split()]
    assert sorted(order) == list(range(40))


def test_refuses_a_source_map_that_does_not_match_the_stored_rows(tmp_path):
    _, _, res = _run(tmp_path, corrupt=True)
    assert res.returncode != 0 and "differ from the stored rows" in res.stderr


def test_refuses_to_overwrite(tmp_path):
    root, _, res = _run(tmp_path)
    assert res.returncode == 0, res.stderr
    env = dict(os.environ, DATA_DIR=str(tmp_path), PYTHONPATH=str(REPO))
    again = subprocess.run([sys.executable, str(SCRIPT), "--system", "quad3d_rl", "--sizes", "pool=40,cal=6,val=6,test=10"],
                           env=env, capture_output=True, text=True)
    assert again.returncode != 0 and "refusing to overwrite" in again.stderr


def _write_grid_dataset(root: Path, refly: Path, n_flights=60, n_eval=50, timed_out=(), bad_label=False):
    rng = np.random.default_rng(2)
    root.mkdir(parents=True)
    lens = rng.integers(2, 6, size=n_flights)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    states = rng.uniform(-1, 1, size=(offsets[-1], 4)).astype(np.float32)
    labels = (np.arange(n_flights) % 3 == 0).astype(np.uint8)
    np.savez(root / "train.npz", states=states, offsets=offsets, starts=states[offsets[:-1]].astype(np.float64),
             labels=labels)
    grid = rng.uniform(-1, 1, size=(n_eval, 4))
    lab = (np.arange(n_eval) % 4 == 0).astype(int)
    (root / "eval_states.txt").write_text("\n".join(",".join(f"{v:.6f}" for v in g) + f",{l:.4f}"
                                                    for g, l in zip(grid, lab)) + "\n")
    refly.mkdir(parents=True)
    term = np.ones(n_eval, np.uint8)
    term[list(timed_out)] = 0
    succ = lab.astype(np.uint8).copy()
    if bad_label:
        succ[0] ^= 1
    np.savez(refly / "shard_000.npz", index=np.arange(n_eval), stored_label=lab.astype(np.uint8), success=succ,
             terminated=term, final=rng.uniform(-1, 1, size=(n_eval, 4)))


def _run_grid(tmp_path, **kw):
    ds = "stochastic/cartpole/gaussian_signal/safe_explorer_ppo/baseline"
    root, refly = tmp_path / ds, tmp_path / "refly" / "cartpole"
    _write_grid_dataset(root, refly, **kw)
    env = dict(os.environ, DATA_DIR=str(tmp_path), DET_REFLY_DIR=str(tmp_path / "refly"), PYTHONPATH=str(REPO))
    res = subprocess.run([sys.executable, str(SCRIPT), "--system", "cartpole_ppo", "--sizes",
                          "pool=60,cal=5,val=5,test=30"], env=env, capture_output=True, text=True)
    return root, res


def test_grid_eval_drops_timeouts_and_uses_refly_finals(tmp_path):
    root, res = _run_grid(tmp_path, timed_out=(3, 7))
    assert res.returncode == 0, res.stderr
    out = root / "splits_det_rerun"
    man = json.loads((out / "split_manifest.json").read_text())
    assert man["scan"]["eval_rows_timed_out_in_refly"] == 2 and man["pool_trajectories"]["n"] == 60
    rows = [ln for split in ("cal", "val", "test") for ln in (out / f"{split}_set.txt").read_text().split()]
    assert len(rows) == len(set(rows)) == 40 and all(len(r.split(",")) == 9 for r in rows)
    starts = {",".join(r.split(",")[:4]) for r in rows}
    lines = (root / "eval_states.txt").read_text().split()
    assert not starts & {",".join(lines[i].split(",")[:4]) for i in (3, 7)}


def test_grid_eval_refuses_a_refly_that_disagrees(tmp_path):
    _, res = _run_grid(tmp_path, bad_label=True)
    assert res.returncode != 0 and "disagrees with 1 stored labels" in res.stderr


def test_grid_eval_can_drop_disagreeing_rows(tmp_path):
    ds = "stochastic/quadrotor2D/corridor_sine_ambient/rl/baseline"
    root, refly = tmp_path / ds, tmp_path / "refly" / "q2d"
    _write_grid_dataset(root, refly, bad_label=True)
    # q2d states are 6-D; widen the synthetic 4-D files to the q2d layout.
    tr = dict(np.load(root / "train.npz"))
    pad = lambda a: np.concatenate([a, np.zeros((len(a), 2), a.dtype)], 1)
    np.savez(root / "train.npz", **{**tr, "states": pad(tr["states"]), "starts": pad(tr["starts"])})
    lines = [ln.split(",") for ln in (root / "eval_states.txt").read_text().split()]
    (root / "eval_states.txt").write_text("\n".join(",".join(v[:4] + ["0.000000", "0.000000", v[4]]) for v in lines) + "\n")
    sh = dict(np.load(refly / "shard_000.npz"))
    np.savez(refly / "shard_000.npz", **{**sh, "final": pad(sh["final"])})
    env = dict(os.environ, DATA_DIR=str(tmp_path), DET_REFLY_DIR=str(tmp_path / "refly"), PYTHONPATH=str(REPO))
    res = subprocess.run([sys.executable, str(SCRIPT), "--system", "quad2d_corridor0", "--sizes",
                          "pool=60,cal=5,val=5,test=30"], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    man = json.loads((root / "splits_det_rerun" / "split_manifest.json").read_text())
    assert man["scan"]["eval_rows_dropped_refly_label_disagreement"] == 1
    kept = {ln.split(",")[0] for s in ("cal", "val", "test")
            for ln in (root / "splits_det_rerun" / f"{s}_set.txt").read_text().split()}
    assert lines[0][0] not in kept

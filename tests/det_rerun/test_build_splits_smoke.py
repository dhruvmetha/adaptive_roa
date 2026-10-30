"""End-to-end run of scripts/det_rerun/build_splits.py on tiny synthetic datasets."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/det_rerun/build_splits.py"


def _write_dataset(root: Path, n_traj: int, k: int, picks: bool, rng):
    (root / "trajectories").mkdir(parents=True)
    (root / "train_test_splits").mkdir()
    names, labels, eval_lines = [], [], []
    for t in range(n_traj):
        rows = rng.uniform(-1, 1, size=(int(rng.integers(2, 6)), k))
        name = f"sequence_{t}.txt"
        (root / "trajectories" / name).write_text("\n".join(",".join(f"{v:.6f}" for v in r) for r in rows) + "\n")
        lab = int(t % 3 == 0)
        names.append(name)
        labels.append(lab)
        fmt = lambda r: ",".join(f"{v:.6f}" for v in r)
        if not picks:
            eval_lines.append(f"{fmt(rows[0])},{fmt(rows[-1])},{lab}")
        else:
            for r in rng.choice(len(rows), size=min(2, len(rows)), replace=False):
                eval_lines.append(f"{fmt(rows[r])},{fmt(rows[-1])},{lab}")
    (root / "train_test_splits/all_shuffled_indices.txt").write_text("\n".join(names) + "\n")
    (root / "train_test_splits/all_shuffled_labels.txt").write_text("\n".join(map(str, labels)) + "\n")
    (root / "eval_states.txt").write_text("\n".join(eval_lines) + "\n")


def _run(tmp_path, system, ds_dir, k, picks, sizes):
    rng = np.random.default_rng(1)
    root = tmp_path / "deterministic" / ds_dir
    _write_dataset(root, 60, k, picks, rng)
    env = dict(os.environ, DATA_DIR=str(tmp_path), PYTHONPATH=str(REPO))
    subprocess.run([sys.executable, str(SCRIPT), "--system", system, "--validate", "20", "--sizes", sizes],
                   check=True, env=env, capture_output=True, text=True)
    out = root / "splits_det_rerun"
    return root, out, json.loads((out / "split_manifest.json").read_text())


def _states(path, k):
    return {",".join(ln.split(",")[:k]) for ln in path.read_text().splitlines() if ln}


def test_starts_branch_disjoint(tmp_path):
    root, out, man = _run(tmp_path, "cartpole", "cartpole_pybullet", 4, False, "pool=30,cal=5,val=5")
    assert man["pool_trajectories"]["n"] == 30
    assert man["splits"]["cal"]["n"] == 5 and man["splits"]["val"]["n"] == 5 and man["splits"]["test"]["n"] == 20
    pool = (out / "pool_trajectories.txt").read_text().split()
    pool_rows = set()
    for n in pool:
        pool_rows |= {ln for ln in (root / "trajectories" / n).read_text().splitlines() if ln}
    for split in ("cal", "val", "test"):
        assert not (_states(out / f"{split}_set.txt", 4) & pool_rows)
    order = (out / "pool_order_99512.txt").read_text().split()
    assert sorted(order) == sorted(pool)


def test_picks_branch_groups_by_trajectory(tmp_path):
    root, out, man = _run(tmp_path, "quad3d", "quadrotor3D_lqr", 13, True, "pool=30,cal=6,val=6")
    assert man["validation"]["missed_by_string_scan"] == 0
    tr = {s: set((out / f"{s}_trajectories.txt").read_text().split()) for s in ("cal", "val", "test")}
    pool = set((out / "pool_trajectories.txt").read_text().split())
    assert not (tr["cal"] & tr["val"]) and not (tr["cal"] & tr["test"]) and not (tr["val"] & tr["test"])
    assert not (pool & (tr["cal"] | tr["val"] | tr["test"]))
    pool_rows = set()
    for n in pool:
        pool_rows |= {ln for ln in (root / "trajectories" / n).read_text().splitlines() if ln}
    for split in ("cal", "val", "test"):
        assert not (_states(out / f"{split}_set.txt", 13) & pool_rows)

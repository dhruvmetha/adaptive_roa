#!/usr/bin/env python
"""Prep splits/metadata for the v2 stochastic pendulum datasets (2026-07-25).

The new layout at noisy/pendulum/lqr/{level} ships:
- train.npz: 300k rollouts from uniform-random starts (states/offsets/starts/labels/seeds)
- eval_success_prob.npz + success_probabilities.txt: p_success per 158x315 grid
  cell from ~90 rollouts/cell (Jeffreys-SD-converged)

This script generates the files the adaptive_v2 pipeline needs (no data copied):
- train_test_splits/shuffled_indices_0.txt / shuffled_labels_0.txt (plain rollout
  shuffle, seed 0 — starts are unique so no block structure)
- cal_set.txt (10k grid cells) / test_set.txt (rest), 3-col theta,theta_dot,p,
  shuffle order (seed 0) shared across levels
- dataset_description.json with achieved_bounds (required by PendulumSystem)
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

ROOT = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr")
LEVELS = ["low", "med", "high", "xhigh"]
SHUFFLE_SEED = 0
N_CAL = 10_000


def prepare_level(level: str, cell_perm: np.ndarray | None) -> np.ndarray:
    d = ROOT / level
    with np.load(d / "train.npz") as z:
        n_traj = len(z["offsets"]) - 1
        labels = z["labels"].astype(int)
        starts = z["starts"]
    assert len(labels) == n_traj

    rng = np.random.default_rng(SHUFFLE_SEED)
    idx = rng.permutation(n_traj)
    (d / "train_test_splits").mkdir(exist_ok=True)
    np.savetxt(d / "train_test_splits" / "shuffled_indices_0.txt", idx, fmt="%d")
    np.savetxt(d / "train_test_splits" / "shuffled_labels_0.txt", labels[idx], fmt="%d")

    ev = np.loadtxt(d / "success_probabilities.txt", delimiter=",")
    n_cells = len(ev)
    if cell_perm is None:
        cell_perm = np.random.default_rng(SHUFFLE_SEED).permutation(n_cells)
    assert len(cell_perm) == n_cells
    rows = ev[cell_perm]
    fmt = "%.6f,%.6f,%.6f"
    np.savetxt(d / "eval_states.txt", rows, fmt=fmt)
    np.savetxt(d / "cal_set.txt", rows[:N_CAL], fmt=fmt)
    np.savetxt(d / "test_set.txt", rows[N_CAL:], fmt=fmt)

    train_desc = json.loads((d / "train_description.json").read_text())
    eval_desc = json.loads((d / "eval_description.json").read_text())
    desc = {
        "dataset_name": f"Stochastic pendulum LQR ({level} noise), v2 splits",
        "prep_date": datetime.date.today().isoformat(),
        "source": {"train": "train_description.json", "eval": "eval_description.json"},
        "shuffle_seed": SHUFFLE_SEED,
        "splits": {
            "train_pool": int(n_traj),
            "cal_cells": N_CAL,
            "test_cells": int(n_cells - N_CAL),
            "cell_shuffle": "shared across noise levels (identical grid)",
        },
        "achieved_bounds": {
            "description": "min/max over sampled train start states; sampling ranges from train_description",
            "theta": {"min": float(starts[:, 0].min()), "max": float(starts[:, 0].max()), "unit": "rad"},
            "theta_dot": {"min": float(starts[:, 1].min()), "max": float(starts[:, 1].max()), "unit": "rad/s"},
        },
        "train_stats": {k: train_desc[k] for k in ("num_trajectories", "success_rate", "mean_length")},
        "eval_stats": {k: eval_desc[k] for k in ("num_cells", "n_batches", "mean_se", "success_rate")},
    }
    (d / "dataset_description.json").write_text(json.dumps(desc, indent=2))
    print(f"[{level}] prep done: {n_traj} train rollouts, {n_cells} eval cells "
          f"(mean p = {ev[:, 2].mean():.4f})", flush=True)
    return cell_perm


def verify_level(level: str) -> None:
    d = ROOT / level
    with np.load(d / "train.npz") as z:
        labels = z["labels"].astype(int)
    idx = np.loadtxt(d / "train_test_splits" / "shuffled_indices_0.txt", dtype=int)
    lab = np.loadtxt(d / "train_test_splits" / "shuffled_labels_0.txt", dtype=int)
    assert sorted(idx.tolist()) == list(range(len(labels)))
    assert np.array_equal(labels[idx], lab)
    es = np.loadtxt(d / "eval_states.txt", delimiter=",")
    cal = np.loadtxt(d / "cal_set.txt", delimiter=",")
    tst = np.loadtxt(d / "test_set.txt", delimiter=",")
    assert np.array_equal(np.vstack([cal, tst]), es) and len(cal) == N_CAL
    src = np.loadtxt(d / "success_probabilities.txt", delimiter=",")
    assert np.allclose(np.sort(es[:, 2]), np.sort(src[:, 2]))
    print(f"[{level}] verification OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=LEVELS)
    args = ap.parse_args()

    # precondition: identical eval grids across levels
    ref = np.loadtxt(ROOT / args.levels[0] / "success_probabilities.txt", delimiter=",")[:, :2]
    for lv in args.levels[1:]:
        g = np.loadtxt(ROOT / lv / "success_probabilities.txt", delimiter=",")[:, :2]
        assert np.allclose(ref, g), f"grid mismatch: {lv}"
    print("precondition OK: eval grids identical across levels", flush=True)

    perm = None
    for lv in args.levels:
        perm = prepare_level(lv, perm)
        verify_level(lv)
    print("all levels done", flush=True)


if __name__ == "__main__":
    main()

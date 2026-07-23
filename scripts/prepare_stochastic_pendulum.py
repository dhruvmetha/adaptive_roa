#!/usr/bin/env python
"""Prepare stochastic pendulum LQR datasets for adaptive FM training.

Reads noisy/pendulum_lqr_stoch-dyn-ctrl-{level}/trajectories.npz (per-grid-cell
layout) and writes noisy/pendulum/lqr/{level}/ with a flat train/eval npz split
at the start-state level plus probabilistic eval/cal/test files.

See docs/superpowers/specs/2026-07-22-stochastic-pendulum-dataset-prep-design.md
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy")
DST_ROOT = SRC_ROOT / "pendulum" / "lqr"
LEVELS = ["low", "med", "high", "xhigh"]

SHUFFLE_SEED = 0
N_STARTS = 49_770
ROLLOUTS_PER_START = 10
N_TRAIN_STARTS = 10_000
N_CAL_STARTS = 10_000


def src_dir(level: str) -> Path:
    return SRC_ROOT / f"pendulum_lqr_stoch-dyn-ctrl-{level}"


def check_grids_match() -> None:
    """Precondition: identical grid cells (hence start states) across levels."""
    ref = None
    for level in LEVELS:
        with np.load(src_dir(level) / "trajectories.npz") as z:
            grid = (z["cells_low"], z["cells_high"], z["cell_lengths"])
        if ref is None:
            ref = grid
        else:
            for a, b in zip(ref, grid):
                if not np.array_equal(a, b):
                    raise AssertionError(f"grid mismatch between {LEVELS[0]} and {level}")
    print("precondition OK: grids identical across levels", flush=True)


def start_split() -> tuple[np.ndarray, np.ndarray]:
    """Shared shuffle of start ids -> (train_start_ids, eval_start_ids), shuffle order."""
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(N_STARTS)
    return perm[:N_TRAIN_STARTS], perm[N_TRAIN_STARTS:]


def prepare_level(level: str, train_ids: np.ndarray, eval_ids: np.ndarray) -> None:
    dst = DST_ROOT / level
    (dst / "train_test_splits").mkdir(parents=True, exist_ok=True)
    z = np.load(src_dir(level) / "trajectories.npz")

    cell_lengths = z["cell_lengths"]
    assert len(cell_lengths) == N_STARTS and (cell_lengths == ROLLOUTS_PER_START).all()

    # Pass 1: per-cell metadata (small arrays)
    lengths = np.empty((N_STARTS, ROLLOUTS_PER_START), dtype=np.int64)
    starts = np.empty((N_STARTS, ROLLOUTS_PER_START, 2))
    seeds = np.empty((N_STARTS, ROLLOUTS_PER_START), dtype=np.int64)
    labels = np.empty((N_STARTS, ROLLOUTS_PER_START), dtype=np.uint8)
    for i in range(N_STARTS):
        off = z[f"offsets_{i}"]
        lengths[i] = np.diff(off)
        starts[i] = z[f"starts_{i}"]
        seeds[i] = z[f"seeds_{i}"]
        term = z[f"terminated_{i}"]
        trunc = z[f"truncated_{i}"]
        assert not (term & trunc).any()
        labels[i] = term.astype(np.uint8)
    print(f"[{level}] pass 1 done", flush=True)

    in_train = np.zeros(N_STARTS, dtype=bool)
    in_train[train_ids] = True

    # Pass 2: fill flat state buffers, splits in ascending start-id order
    out = {}
    for name, mask in (("train", in_train), ("eval", ~in_train)):
        ids = np.flatnonzero(mask)  # ascending start ids
        n_roll = len(ids) * ROLLOUTS_PER_START
        roll_lengths = lengths[ids].reshape(-1)
        offsets = np.zeros(n_roll + 1, dtype=np.int64)
        np.cumsum(roll_lengths, out=offsets[1:])
        out[name] = {
            "ids": ids,
            "states": np.empty((offsets[-1], 2)),
            "offsets": offsets,
            "starts": starts[ids].reshape(-1, 2),
            "labels": labels[ids].reshape(-1),
            "start_ids": np.repeat(ids, ROLLOUTS_PER_START).astype(np.int32),
            "seeds": seeds[ids].reshape(-1),
        }
        # row position of each start's block within this split
        out[name]["block_pos"] = {int(s): k for k, s in enumerate(ids)}

    cursor = {"train": 0, "eval": 0}
    for i in range(N_STARTS):
        name = "train" if in_train[i] else "eval"
        s = z[f"states_{i}"]
        c = cursor[name]
        out[name]["states"][c : c + len(s)] = s
        cursor[name] = c + len(s)
    for name in ("train", "eval"):
        assert cursor[name] == out[name]["offsets"][-1]
        np.savez(
            dst / f"{name}.npz",
            states=out[name]["states"],
            offsets=out[name]["offsets"],
            starts=out[name]["starts"],
            labels=out[name]["labels"],
            start_ids=out[name]["start_ids"],
            seeds=out[name]["seeds"],
        )
    print(f"[{level}] train.npz / eval.npz written", flush=True)

    # shuffled_indices_0: rollout row ids into train.npz, blocks of 10 per start,
    # block order = shuffle order over the training starts
    tr = out["train"]
    block_rows = np.array([tr["block_pos"][int(s)] for s in train_ids])
    idx = (block_rows[:, None] * ROLLOUTS_PER_START + np.arange(ROLLOUTS_PER_START)).reshape(-1)
    np.savetxt(dst / "train_test_splits" / "shuffled_indices_0.txt", idx, fmt="%d")
    np.savetxt(dst / "train_test_splits" / "shuffled_labels_0.txt", tr["labels"][idx], fmt="%d")

    # eval_states / cal / test: shuffle order over eval starts; theta, theta_dot, p_success
    ev = out["eval"]
    p_success = labels.mean(axis=1)
    vertex = starts[:, 0, :]  # all rollouts of a start share the reset state
    assert np.allclose(starts, vertex[:, None, :])
    rows = np.column_stack([vertex[eval_ids], p_success[eval_ids]])
    fmt = "%.6f,%.6f,%.4f"
    np.savetxt(dst / "eval_states.txt", rows, fmt=fmt)
    np.savetxt(dst / "cal_set.txt", rows[:N_CAL_STARTS], fmt=fmt)
    np.savetxt(dst / "test_set.txt", rows[N_CAL_STARTS:], fmt=fmt)

    desc = json.loads((src_dir(level) / "dataset_description.json").read_text())
    desc["prep"] = {
        "date": datetime.date.today().isoformat(),
        "source_dir": str(src_dir(level)),
        "spec": "docs/superpowers/specs/2026-07-22-stochastic-pendulum-dataset-prep-design.md",
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle": "np.random.default_rng(seed).permutation(49770); first 10000 -> train, rest -> eval; shared across noise levels",
        "split": {
            "train_starts": N_TRAIN_STARTS,
            "eval_starts": N_STARTS - N_TRAIN_STARTS,
            "cal_starts": N_CAL_STARTS,
            "test_starts": N_STARTS - N_TRAIN_STARTS - N_CAL_STARTS,
            "rollouts_per_start": ROLLOUTS_PER_START,
            "level": "start-state (a start's 10 rollouts never straddle a split)",
        },
        "files": {
            "train.npz / eval.npz": "flat arrays in ascending start-id order: states (sumT,2), offsets (N+1,), starts (N,2), labels (N,), start_ids (N,), seeds (N,); rollout id = row index",
            "train_test_splits/shuffled_indices_0.txt": "rollout ids into train.npz, contiguous blocks of 10 per start, block order = shuffled start order",
            "train_test_splits/shuffled_labels_0.txt": "binary labels aligned with shuffled_indices_0 (1=success, 0=failure/timeout)",
            "eval_states.txt": "theta_start, theta_dot_start, p_success per eval start, shuffle order; cal_set.txt = first 10000 rows, test_set.txt = rest",
        },
    }
    (dst / "dataset_description.json").write_text(json.dumps(desc, indent=2))
    z.close()
    print(f"[{level}] prep complete", flush=True)


def verify_level(level: str, train_ids: np.ndarray, eval_ids: np.ndarray) -> None:
    dst = DST_ROOT / level
    zsrc = np.load(src_dir(level) / "trajectories.npz")
    # Materialize member arrays once: NpzFile re-reads the full member from
    # disk on EVERY access, which is catastrophic inside the loops below.
    with np.load(dst / "train.npz") as z:
        tr = {k: z[k] for k in z.files}
    with np.load(dst / "eval.npz") as z:
        ev = {k: z[k] for k in z.files}

    assert len(tr["labels"]) == N_TRAIN_STARTS * ROLLOUTS_PER_START
    assert len(ev["labels"]) == (N_STARTS - N_TRAIN_STARTS) * ROLLOUTS_PER_START
    assert not (set(tr["start_ids"]) & set(ev["start_ids"])), "train/eval start overlap"
    assert set(tr["start_ids"]) == set(train_ids.tolist())

    idx = np.loadtxt(dst / "train_test_splits" / "shuffled_indices_0.txt", dtype=int)
    lab = np.loadtxt(dst / "train_test_splits" / "shuffled_labels_0.txt", dtype=int)
    assert len(idx) == len(lab) == N_TRAIN_STARTS * ROLLOUTS_PER_START
    assert sorted(idx.tolist()) == list(range(len(idx)))
    assert np.array_equal(tr["labels"][idx], lab)
    blocks = tr["start_ids"][idx].reshape(-1, ROLLOUTS_PER_START)
    assert (blocks == blocks[:, :1]).all(), "shuffled_indices blocks mix starts"
    assert np.array_equal(blocks[:, 0], train_ids), "block order != recorded shuffle"

    es = np.loadtxt(dst / "eval_states.txt", delimiter=",")
    cal = np.loadtxt(dst / "cal_set.txt", delimiter=",")
    tst = np.loadtxt(dst / "test_set.txt", delimiter=",")
    assert es.shape == (N_STARTS - N_TRAIN_STARTS, 3)
    assert cal.shape[0] == N_CAL_STARTS and tst.shape[0] == es.shape[0] - N_CAL_STARTS
    assert np.array_equal(np.vstack([cal, tst]), es), "cal+test != eval_states"

    # overall success-rate consistency (train+eval vs source description)
    stats = json.loads((src_dir(level) / "dataset_description.json").read_text())["dataset_statistics"]
    total_succ = int(tr["labels"].sum()) + int(ev["labels"].sum())
    assert total_succ == stats["successful_trajectories"]["count"], "label total mismatch"

    # round-trip random rollouts against the source (states + label + p_success rows)
    rng = np.random.default_rng(1)
    eval_pos = {int(s): k for k, s in enumerate(np.flatnonzero(~np.isin(np.arange(N_STARTS), train_ids)))}
    shuffle_pos = {int(s): k for k, s in enumerate(eval_ids)}
    for sid in rng.choice(N_STARTS, size=25, replace=False):
        sid = int(sid)
        src_states = zsrc[f"states_{sid}"]
        src_off = zsrc[f"offsets_{sid}"]
        src_term = zsrc[f"terminated_{sid}"]
        if sid in eval_pos:
            z, row0 = ev, eval_pos[sid] * ROLLOUTS_PER_START
        else:
            tr_pos = {int(s): k for k, s in enumerate(np.unique(tr["start_ids"]))}
            z, row0 = tr, tr_pos[sid] * ROLLOUTS_PER_START
        for j in range(ROLLOUTS_PER_START):
            r = row0 + j
            got = z["states"][z["offsets"][r] : z["offsets"][r + 1]]
            assert np.array_equal(got, src_states[src_off[j] : src_off[j + 1]])
            assert z["labels"][r] == src_term[j]
        if sid in shuffle_pos:
            row = es[shuffle_pos[sid]]
            assert np.allclose(row[:2], src_states[0], atol=1e-6)
            assert abs(row[2] - src_term.mean()) < 1e-3
    zsrc.close()
    print(f"[{level}] verification OK  (mean p_success eval = {es[:, 2].mean():.4f})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=LEVELS)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    check_grids_match()
    train_ids, eval_ids = start_split()
    for level in args.levels:
        if not args.verify_only:
            prepare_level(level, train_ids, eval_ids)
        verify_level(level, train_ids, eval_ids)
    print("all levels done", flush=True)


if __name__ == "__main__":
    main()

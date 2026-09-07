#!/usr/bin/env python3
"""Build a leakage-free eval split for the legacy deterministic pendulum.

The legacy eval_states.txt contains all 49,770 grid trajectories, while
shuffled_indices_0.txt exposes 20,000 of those same trajectories to adaptive
training.  Rows in eval_states.txt are aligned with all_shuffled_indices.txt,
so the exact held-out complement can be selected by filename without parsing or
relabeling trajectory files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_split(source_dir: Path, output_dir: Path, cal_size: int, seed: int) -> dict:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    eval_path = source_dir / "eval_states.txt"
    all_indices_path = source_dir / "train_test_splits" / "all_shuffled_indices.txt"
    train_indices_path = source_dir / "train_test_splits" / "shuffled_indices_0.txt"

    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("output directory must be outside the source dataset tree")

    rows = np.loadtxt(eval_path, delimiter=",", dtype=np.float64)
    all_names = _read_names(all_indices_path)
    train_names = _read_names(train_indices_path)
    if rows.ndim != 2 or rows.shape[1] != 5:
        raise ValueError(f"expected eval rows [N,5], got {rows.shape}")
    if len(rows) != len(all_names):
        raise ValueError(
            f"eval/index alignment length mismatch: {len(rows)} != {len(all_names)}"
        )
    if len(set(all_names)) != len(all_names):
        raise ValueError("all_shuffled_indices.txt contains duplicate filenames")
    if len(set(train_names)) != len(train_names):
        raise ValueError("shuffled_indices_0.txt contains duplicate filenames")

    all_name_set = set(all_names)
    train_name_set = set(train_names)
    missing = train_name_set - all_name_set
    if missing:
        raise ValueError(f"training filenames absent from all-indices list: {len(missing)}")

    heldout_mask = np.fromiter(
        (name not in train_name_set for name in all_names),
        dtype=bool,
        count=len(all_names),
    )
    heldout_rows = rows[heldout_mask]
    heldout_names = [name for name, keep in zip(all_names, heldout_mask) if keep]
    expected_heldout = len(all_names) - len(train_name_set)
    if len(heldout_rows) != expected_heldout:
        raise RuntimeError(f"held-out count mismatch: {len(heldout_rows)} != {expected_heldout}")
    if not 0 < cal_size < len(heldout_rows):
        raise ValueError(f"cal_size must be in (0, {len(heldout_rows)}), got {cal_size}")

    permutation = np.random.default_rng(seed).permutation(len(heldout_rows))
    heldout_rows = heldout_rows[permutation]
    heldout_names = [heldout_names[int(i)] for i in permutation]
    cal_rows = heldout_rows[:cal_size]
    test_rows = heldout_rows[cal_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    heldout_path = output_dir / "eval_states.txt"
    cal_path = output_dir / "cal_set.txt"
    test_path = output_dir / "test_set.txt"
    np.savetxt(heldout_path, heldout_rows, delimiter=",", fmt="%.9g")
    np.savetxt(cal_path, cal_rows, delimiter=",", fmt="%.9g")
    np.savetxt(test_path, test_rows, delimiter=",", fmt="%.9g")
    (output_dir / "heldout_indices.txt").write_text("\n".join(heldout_names) + "\n")

    train_starts = rows[~heldout_mask, :2]
    heldout_starts = heldout_rows[:, :2]
    train_start_keys = {tuple(x) for x in train_starts.tolist()}
    heldout_start_keys = {tuple(x) for x in heldout_starts.tolist()}
    overlap = train_start_keys & heldout_start_keys
    if overlap:
        raise RuntimeError(f"train/held-out start-state overlap remains: {len(overlap)}")

    metadata = {
        "source_dir": str(source_dir),
        "method": "complement shuffled_indices_0 from all_shuffled_indices; seeded heldout shuffle",
        "seed": seed,
        "n_all": len(all_names),
        "n_train_excluded": len(train_name_set),
        "n_heldout": len(heldout_rows),
        "n_cal": len(cal_rows),
        "n_test": len(test_rows),
        "train_heldout_start_overlap": 0,
        "heldout_success": int(np.sum(heldout_rows[:, -1] == 1)),
        "cal_success": int(np.sum(cal_rows[:, -1] == 1)),
        "test_success": int(np.sum(test_rows[:, -1] == 1)),
        "input_sha256": {
            "eval_states.txt": _sha256(eval_path),
            "all_shuffled_indices.txt": _sha256(all_indices_path),
            "shuffled_indices_0.txt": _sha256(train_indices_path),
        },
        "output_sha256": {
            "eval_states.txt": _sha256(heldout_path),
            "cal_set.txt": _sha256(cal_path),
            "test_set.txt": _sha256(test_path),
        },
    }
    metadata_path = output_dir / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cal-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    metadata = build_split(args.source_dir, args.output_dir, args.cal_size, args.seed)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

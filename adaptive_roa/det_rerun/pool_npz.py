"""Flat npz pools for NpzTrajectoryDataSource.

Acquisition over intermediate states reads tens of thousands of random
(trajectory, row) states per round, and the text pool opens one file per
trajectory. A single `train.npz` holds the whole pool in memory instead:
`states` (sumT, d) float32, `offsets` (N+1,) int64, `starts` (N, d),
`labels` (N,) uint8 success, `timeout` (N,) bool. Pool orders list integer
row ids into it, the format `NpzTrajectoryDataSource._parse_rollout_ids` reads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


def parse_rows(text: str) -> np.ndarray:
    """Comma-separated trajectory text -> [T, d] float32, as the text loader casts it."""
    first = text.lstrip().split("\n", 1)[0]
    d = first.count(",") + 1
    flat = np.fromstring(text.strip().replace("\n", ","), sep=",")  # C strtod, same doubles as float()
    if flat.size % d:
        raise ValueError(f"ragged trajectory text: {flat.size} values for {d} columns")
    return flat.reshape(-1, d).astype(np.float32)


def write_pool_npz(
    out_dir: Path,
    names: Sequence[str],
    arrays: Sequence[np.ndarray],
    success: Sequence[int],
    timeout: Sequence[bool],
    seeds: Iterable[int],
    horizon_steps: int,
) -> Dict:
    """Write train.npz, npz_row_names.txt, pool orders and labels, and train_description.json.

    Rows are written in the given order. Pool orders draw only the non-timeout rows, so a
    timeout row is on record (flagged) but never enters training.
    """
    out_dir = Path(out_dir)
    lengths = np.array([len(a) for a in arrays], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    states = np.concatenate(arrays).astype(np.float32)
    success = np.asarray(success, dtype=np.uint8)
    timeout = np.asarray(timeout, dtype=bool)
    if (success.astype(bool) & timeout).any():
        raise ValueError("a row is flagged both success and timeout")
    if lengths.max() > horizon_steps + 1:
        raise ValueError(f"a trajectory has {lengths.max()} rows, more than horizon_steps + 1 = {horizon_steps + 1}")
    np.savez(out_dir / "train.npz", states=states, offsets=offsets, starts=states[offsets[:-1]],
             labels=success, timeout=timeout)
    (out_dir / "npz_row_names.txt").write_text("\n".join(names) + "\n")
    eligible = np.where(~timeout)[0]
    for s in seeds:
        order = eligible[np.random.default_rng(s).permutation(len(eligible))]
        (out_dir / f"pool_order_{s}.txt").write_text("\n".join(map(str, order.tolist())) + "\n")
        (out_dir / f"pool_labels_{s}.txt").write_text("\n".join(map(str, success[order].tolist())) + "\n")
    desc = {
        "horizon_steps": int(horizon_steps),
        "n_rollouts": int(len(arrays)),
        "n_pool": int(len(eligible)),
        "n_timeout_rows_not_in_pool": int(timeout.sum()),
        "state_dim": int(states.shape[1]),
        "total_states": int(states.shape[0]),
        "note": "train.npz rows follow npz_row_names.txt; pool_order_<seed>.txt lists row ids, "
                "pool_labels_<seed>.txt the aligned 0/1 success labels.",
    }
    (out_dir / "train_description.json").write_text(json.dumps(desc, indent=2))
    desc["train_npz_bytes"] = int((out_dir / "train.npz").stat().st_size)
    return desc

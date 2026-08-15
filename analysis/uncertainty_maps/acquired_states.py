#!/usr/bin/env python
"""Recover WHERE each arm collected its data, per adaptive epoch.

`artifacts_v2.json` records `acquisition.d2_indices` -- the pool indices the arm
chose that epoch. The campaign ran in `start` candidate mode (no
`candidate_mode` override in any run config), so those are TRAJECTORY indices
and their start states come straight from the run's own data source. Building
that data source from the run's saved config, rather than re-deriving the index
space by hand, is what guarantees the indices are read in the same order the run
wrote them.

Output: one npz per run holding, for every epoch, the (theta, theta_dot) start
states that were added to training that epoch, plus their true outcome labels.

    python acquired_states.py --pred fm --level high --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.uncertainty_maps.common import ARMS, EXP, epochs_with_eval  # noqa: E402
from analysis.uncertainty_maps.compute_members import load_cfg  # noqa: E402


def data_source_for(run: str):
    """The run's own pool data source, built from its saved config."""
    from adaptive_roa.adaptive.data_source import TrajectoryDataSourceConfig
    from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource
    from adaptive_roa.adaptive.data_source import TrajectoryDataSource

    cfg = load_cfg(run)
    ds = cfg.data_source
    dsc = TrajectoryDataSourceConfig(
        trajectories_dir=ds.trajectories_dir,
        shuffled_indices_file=ds.shuffled_indices_file,
        shuffled_labels_file=ds.get("shuffled_labels_file", None),
        eval_states_file=ds.get("eval_states_file", None),
    )
    if str(ds.get("pool_format", "text")) == "npz":
        return NpzTrajectoryDataSource(dsc)
    return TrajectoryDataSource(dsc)


def acquired_for_run(run: str) -> dict:
    pred, level, arm = run.split("_", 2)
    src = data_source_for(run)
    out = {}
    for ep in epochs_with_eval(pred, level, arm):
        f = EXP / run / f"epoch_{ep:03d}" / "artifacts_v2.json"
        acq = (json.loads(f.read_text()).get("acquisition") or {})
        idx = list(acq.get("d2_indices") or []) + list(acq.get("d1_indices") or [])
        if not idx:
            continue
        states = np.asarray(src.get_start_states(idx), dtype=np.float32)
        try:
            labels = np.asarray(src.get_labels(idx), dtype=np.int64)
        except Exception:
            labels = np.full(len(idx), 0, dtype=np.int64)
        out[ep] = (states, labels, np.asarray(idx, dtype=np.int64))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, choices=("clf", "fm"))
    ap.add_argument("--level", required=True)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = ARMS if a.arms == "all" else tuple(a.arms.split(","))
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for arm in arms:
        run = f"{a.pred}_{a.level}_{arm}"
        if not (EXP / run).exists():
            print(f"{run}: absent, skip")
            continue
        got = acquired_for_run(run)
        if not got:
            print(f"{run}: no acquisition records")
            continue
        payload = {}
        for ep, (s, lab, idx) in got.items():
            payload[f"states_{ep:03d}"] = s
            payload[f"labels_{ep:03d}"] = lab
            payload[f"indices_{ep:03d}"] = idx
        payload["epochs"] = np.asarray(sorted(got), dtype=np.int64)
        np.savez_compressed(out_root / f"{run}_acquired.npz", **payload)
        n = {ep: len(v[0]) for ep, v in sorted(got.items())}
        print(f"{run}: {len(got)} epochs, sizes {sorted(set(n.values()))}")


if __name__ == "__main__":
    main()

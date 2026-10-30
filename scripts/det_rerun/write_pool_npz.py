#!/usr/bin/env python3
"""Write train.npz and integer pool orders for a splits_det_rerun dir built before npz pools.

Reads pool_trajectories.txt (plus, on pendulum, the timeout trajectories, flagged and kept
out of every pool order), replaces the filename-based pool_order_<seed>.txt and
pool_labels_<seed>.txt with row ids into train.npz, and records the npz in
split_manifest.json. cal/val/test files are not touched.

Usage:
    python scripts/det_rerun/write_pool_npz.py --system cartpole
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_splits import DATA, DEFAULT_SEEDS, RUN_SEEDS, SYSTEMS, load_names_labels, log  # noqa: E402

from adaptive_roa.det_rerun.pool_npz import parse_rows, write_pool_npz  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    args = ap.parse_args()
    spec = SYSTEMS[args.system]
    root = DATA / spec["dir"]
    out = root / "splits_det_rerun"
    manifest_path = out / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if "pool_npz" in manifest:
        raise SystemExit(f"{out} already has an npz pool")

    names, labels = load_names_labels(root, spec)
    label_of = dict(zip(names, labels.tolist()))
    pool = (out / "pool_trajectories.txt").read_text().split()
    timeouts = [n for n in names if label_of[n] < 0]
    rows = pool + timeouts
    log(f"{args.system}: reading {len(pool)} pool + {len(timeouts)} timeout trajectories")
    arrays = []
    for i, n in enumerate(rows):
        with open(root / "trajectories" / n) as f:
            arrays.append(parse_rows(f.read()))
        if (i + 1) % 50000 == 0:
            log(f"  {i + 1}/{len(rows)}")
    # Replace the filename pool orders (row ids now index train.npz, not sequence numbers).
    for s in RUN_SEEDS + DEFAULT_SEEDS:
        for stem in ("pool_order", "pool_labels"):
            p = out / f"{stem}_{s}.txt"
            if p.exists():
                p.unlink()
    npz = write_pool_npz(out, rows, arrays, [int(label_of[n] == 1) for n in rows],
                         [label_of[n] < 0 for n in rows], seeds=RUN_SEEDS + DEFAULT_SEEDS,
                         horizon_steps=spec["horizon"])
    manifest["pool_npz"] = npz
    manifest["format"] = ("cal/val/test rows are copied verbatim from the eval file: start, end, label (0/1); "
                          "train.npz is the pool for NpzTrajectoryDataSource (rows follow npz_row_names.txt); "
                          "pool_order_<seed>.txt lists its row ids for data_source.shuffled_indices_file, "
                          "pool_labels_<seed>.txt the aligned 0/1 labels.")
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)
    log(json.dumps(npz, indent=1))


if __name__ == "__main__":
    main()

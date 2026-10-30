#!/usr/bin/env python3
"""Build disjoint pool / cal / val / test splits for one deterministic dataset.

One pass reads every stored row of every trajectory and looks it up among the eval
states, so the source of each eval state is known exactly (a start on pendulum,
cartpole and quad2d; any row on quad3d, whose eval states are picked along flights).
Trajectories are then partitioned; an eval state is kept only if no pool trajectory
contains it, and cal, val and test are split by source trajectory. Outputs go to
<dataset>/splits_det_rerun/; the source files are only read.

Usage:
    python scripts/det_rerun/build_splits.py --system quad3d [--validate 1000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from adaptive_roa.det_rerun.pool_npz import parse_rows, write_pool_npz
from adaptive_roa.det_rerun.splits import assert_disjoint, group_split, partition_ids, row_key

DATA = Path(os.environ.get("DATA_DIR", "/common/users/shared/pracsys/genMoPlan/data_trajectories")) / "deterministic"
PARTITION_SEED = 20261029
RUN_SEEDS = (99512, 33407, 24475)   # paper seeds (tracker D04)
DEFAULT_SEEDS = (42,)               # configs' default seed; tests and ad hoc runs only

SYSTEMS = {
    # name: dataset dir, state dim, eval file, sizes, whether eval states are trajectory starts
    "pendulum": dict(dir="pendulum/lqr_failset", k=2, eval="eval_states.txt",
                     labels="trajectory_labels.txt", sizes=dict(pool=25000, cal=2000, val=2000),
                     starts=True, float_keys=True, horizon=502),
    "cartpole": dict(dir="cartpole_pybullet", k=4, eval="eval_states.txt", labels=None,
                     sizes=dict(pool=50000, cal=2000, val=2000), starts=True, float_keys=False, horizon=1000),
    "quad2d": dict(dir="quadrotor2D_rl", k=6, eval="eval_states.txt", labels=None,
                   sizes=dict(pool=300000, cal=10000, val=10000), starts=True, float_keys=False, horizon=1200),
    "quad3d": dict(dir="quadrotor3D_lqr", k=13, eval="eval_states.txt", labels=None,
                   sizes=dict(pool=600000, cal=10000, val=10000), starts=False, float_keys=False, horizon=100000),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_names_labels(root: Path, spec: dict):
    if spec["labels"]:
        rows = [ln.strip().split(",") for ln in open(root / spec["labels"]) if ln.strip()]
        return [r[0] for r in rows], np.array([int(r[1]) for r in rows])
    names = [ln.strip() for ln in open(root / "train_test_splits/all_shuffled_indices.txt") if ln.strip()]
    labels = np.loadtxt(root / "train_test_splits/all_shuffled_labels.txt", dtype=int)
    return names, labels


def key_of(line: str, k: int, float_keys: bool) -> str:
    if float_keys:
        return row_key(line.split(",")[:k])
    if line.count(",") == k - 1:
        return line
    return ",".join(line.split(",")[:k])


def scan(root: Path, names, eval_keys: dict, k: int, float_keys: bool):
    """Return n_rows per trajectory, (eval_row, traj, row) matches, and every trajectory as float32."""
    n_rows = np.zeros(len(names), dtype=np.int32)
    matches = []
    arrays = []
    tdir = root / "trajectories"
    for t, n in enumerate(names):
        with open(tdir / n) as f:
            text = f.read()
        arrays.append(parse_rows(text))
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        n_rows[t] = len(lines)
        for r, ln in enumerate(lines):
            hit = eval_keys.get(key_of(ln, k, float_keys))
            if hit is not None:
                for e in hit:
                    matches.append((e, t, r))
        if (t + 1) % 50000 == 0:
            log(f"  scanned {t + 1}/{len(names)} trajectories, {len(matches)} matches")
    return n_rows, np.array(matches, dtype=np.int64).reshape(-1, 3), arrays


def validate(root: Path, names, eval_start: np.ndarray, matches: np.ndarray, k: int, n: int) -> dict:
    """KD-tree check on n random trajectories: every near-exact match must be in `matches`."""
    from scipy.spatial import cKDTree
    tree = cKDTree(eval_start)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(names), size=min(n, len(names)), replace=False)
    have = set(map(tuple, matches[:, 1:].tolist()))
    missed = found = 0
    for t in pick:
        X = np.loadtxt(root / "trajectories" / names[t], delimiter=",", ndmin=2)[:, :k]
        d, _ = tree.query(X)
        for r in np.where(d < 1e-5)[0]:
            found += 1
            missed += (int(t), int(r)) not in have
    return {"trajectories_checked": int(len(pick)), "kdtree_matches": found, "missed_by_string_scan": missed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--validate", type=int, default=1000, help="trajectories for the KD-tree cross-check")
    ap.add_argument("--sizes", default=None, help="override, e.g. pool=8,cal=2,val=2 (tests only)")
    args = ap.parse_args()
    spec = dict(SYSTEMS[args.system])
    if args.sizes:
        spec["sizes"] = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.sizes.split(",")}
    root = DATA / spec["dir"]
    out = root / "splits_det_rerun"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    k = spec["k"]

    names, labels = load_names_labels(root, spec)
    log(f"{args.system}: {len(names)} trajectories")
    eval_lines = [ln.strip() for ln in open(root / spec["eval"]) if ln.strip()]
    eval_keys: dict = defaultdict(list)
    for e, ln in enumerate(eval_lines):
        eval_keys[key_of(ln, k, spec["float_keys"])].append(e)
    eval_label = np.array([int(float(ln.rsplit(",", 1)[1])) for ln in eval_lines])
    dup_eval = sum(len(v) - 1 for v in eval_keys.values() if len(v) > 1)
    log(f"{len(eval_lines)} eval rows, {len(eval_keys)} distinct states ({dup_eval} duplicate rows)")

    n_rows, matches, arrays = scan(root, names, dict(eval_keys), k, spec["float_keys"])
    log(f"scan done: {len(matches)} matches")
    eval_start = np.array([[float(v) for v in ln.split(",")[:k]] for ln in eval_lines])
    check = validate(root, names, eval_start, matches, k, args.validate) if args.validate else {}
    log(f"validation {check}")
    if check.get("missed_by_string_scan"):
        raise SystemExit("string scan missed KD-tree matches; fix the key format before splitting")

    # Sources of each eval row: every trajectory containing that state at any row.
    sources = defaultdict(set)
    start_src = defaultdict(set)
    for e, t, r in matches.tolist():
        sources[e].add(t)
        if r == 0:
            start_src[e].add(t)
    n_src = np.array([len(sources.get(e, ())) for e in range(len(eval_lines))])
    stats = {
        "eval_rows": len(eval_lines),
        "eval_rows_with_0_sources": int(np.sum(n_src == 0)),
        "eval_rows_with_1_source": int(np.sum(n_src == 1)),
        "eval_rows_with_2plus_sources": int(np.sum(n_src >= 2)),
        "eval_rows_matching_a_start": int(sum(1 for e in range(len(eval_lines)) if start_src.get(e))),
        "eval_rows_matching_only_later_rows": int(sum(1 for e in range(len(eval_lines)) if sources.get(e) and not start_src.get(e))),
        "trajectories_contributing_eval_rows": int(len({t for s in sources.values() for t in s})),
        "trajectory_rows_scanned": int(n_rows.sum()),
    }

    usable = np.where(labels >= 0)[0]  # pendulum timeouts (label -1) are left out entirely
    stats["trajectories_excluded_timeout"] = int(len(names) - len(usable))

    if spec["starts"]:
        # Each eval row should be the start of exactly one trajectory.
        home = {}
        for e in range(len(eval_lines)):
            s = start_src.get(e, set())
            if len(s) == 1:
                home[e] = next(iter(s))
        traj_eval = defaultdict(list)
        for e, t in home.items():
            traj_eval[t].append(e)
        stats["eval_rows_without_unique_start_match"] = int(len(eval_lines) - len(home))
        stats["trajectories_with_2plus_eval_rows"] = int(sum(1 for v in traj_eval.values() if len(v) > 1))
        mism = sum(1 for e, t in home.items() if (eval_label[e] == 1) != (labels[t] == 1))
        stats["eval_label_vs_trajectory_label_mismatch"] = int(mism)
        parts = partition_ids(usable, spec["sizes"], rest="test", seed=PARTITION_SEED)
        pool_set = set(parts["pool"].tolist())
        rows_of = {}
        dropped = 0
        for name in ("cal", "val", "test"):
            rows = []
            for t in parts[name].tolist():
                for e in traj_eval.get(t, []):
                    if sources[e] & pool_set:   # state also lies on a pool trajectory
                        dropped += 1
                        continue
                    rows.append(e)
            rows_of[name] = np.array(sorted(rows), dtype=np.int64)
        stats["eval_rows_dropped_on_pool_trajectory"] = dropped
        eval_trajs = {name: parts[name] for name in ("cal", "val", "test")}
    else:
        tparts = partition_ids(usable, {"pool": spec["sizes"]["pool"]}, rest="eval", seed=PARTITION_SEED)
        parts = {"pool": tparts["pool"]}
        pool_set = set(parts["pool"].tolist())
        keep = [e for e in range(len(eval_lines)) if not (sources.get(e, set()) & pool_set)]
        stats["eval_rows_dropped_on_pool_trajectory"] = int(len(eval_lines) - len(keep))
        # Group rows that share any source trajectory (union-find over sources).
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in keep:
            src = sorted(sources.get(e, set()))
            node = ("t", src[0]) if src else ("e", e)
            for t in src[1:]:
                parent[find(("t", t))] = find(node)
            find(node)
        root_id: dict = {}
        group = np.array([root_id.setdefault(find(("t", min(sources[e]))) if sources.get(e) else find(("e", e)),
                                             len(root_id)) for e in keep])
        mism = sum(1 for e in keep for t in sources.get(e, ()) if (eval_label[e] == 1) != (labels[t] == 1))
        stats["eval_label_vs_source_label_mismatch"] = int(mism)
        gs = group_split(group, {"cal": spec["sizes"]["cal"], "val": spec["sizes"]["val"]}, rest="test", seed=PARTITION_SEED)
        keep = np.array(keep, dtype=np.int64)
        rows_of = {name: np.sort(keep[idx]) for name, idx in gs.items()}
        eval_trajs = {name: np.array(sorted({t for e in rows_of[name].tolist() for t in sources.get(e, ())}), dtype=np.int64)
                      for name in ("cal", "val", "test")}
        stats["eval_trajectories_unused_by_pool"] = int(len(tparts["eval"]))

    # Overlap checks: trajectory ids, eval states, and pool trajectories vs kept eval rows.
    assert_disjoint({"pool": pool_set, **{n: set(v.tolist()) for n, v in eval_trajs.items()}})
    keyset = {n: {key_of(eval_lines[e], k, spec["float_keys"]) for e in rows.tolist()} for n, rows in rows_of.items()}
    assert_disjoint(keyset)
    for name, rows in rows_of.items():
        bad = sum(1 for e in rows.tolist() if sources.get(e, set()) & pool_set)
        if bad:
            raise SystemExit(f"{name}: {bad} kept eval rows lie on pool trajectories")
    overlap = {"pool_vs_eval_trajectories": 0, "eval_states_between_splits": 0,
               "kept_eval_states_on_any_pool_trajectory_row": 0}

    out.mkdir()
    pool_names = [names[t] for t in sorted(pool_set)]
    pool_lab = {names[t]: int(labels[t] == 1) for t in pool_set}
    with open(out / "pool_trajectories.txt", "w") as f:
        f.write("\n".join(pool_names) + "\n")
    # npz rows: the pool, then any timeout trajectories (flagged, never in a pool order).
    timeout_ids = [t for t in range(len(names)) if labels[t] < 0]
    npz_rows = sorted(pool_set) + timeout_ids
    npz = write_pool_npz(out, [names[t] for t in npz_rows], [arrays[t] for t in npz_rows],
                         [int(labels[t] == 1) for t in npz_rows], [bool(labels[t] < 0) for t in npz_rows],
                         seeds=RUN_SEEDS + DEFAULT_SEEDS, horizon_steps=spec["horizon"])
    for name, rows in rows_of.items():
        with open(out / f"{name}_set.txt", "w") as f:
            f.write("\n".join(eval_lines[e] for e in rows.tolist()) + "\n")
        with open(out / f"{name}_trajectories.txt", "w") as f:
            f.write("\n".join(names[t] for t in eval_trajs[name].tolist()) + "\n")

    lab = lambda rows: {"n": int(len(rows)), "success": int(np.sum(eval_label[rows] == 1))}
    manifest = {
        "system": args.system,
        "dataset": str(root),
        "built": str(date.today()),
        "script": "scripts/det_rerun/build_splits.py",
        "partition_seed": PARTITION_SEED,
        "run_seeds": list(RUN_SEEDS),
        "default_seed_files_not_for_paper": list(DEFAULT_SEEDS),
        "source_files": {spec["eval"]: sha1(root / spec["eval"])},
        "pool_trajectories": {"n": len(pool_names), "success": int(sum(pool_lab.values()))},
        "pool_rows": int(n_rows[sorted(pool_set)].sum()),
        "pool_npz": npz,
        "splits": {name: lab(rows) for name, rows in rows_of.items()},
        "split_trajectories": {name: int(len(v)) for name, v in eval_trajs.items()},
        "scan": stats,
        "validation": check,
        "overlap_check": overlap,
        "format": "cal/val/test rows are copied verbatim from the eval file: start, end, label (0/1); "
                  "train.npz is the pool for NpzTrajectoryDataSource (rows follow npz_row_names.txt); "
                  "pool_order_<seed>.txt lists its row ids for data_source.shuffled_indices_file, "
                  "pool_labels_<seed>.txt the aligned 0/1 labels.",
    }
    with open(out / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

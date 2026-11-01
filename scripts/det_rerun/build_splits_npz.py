#!/usr/bin/env python3
"""Disjoint pool / cal / val / test splits for an npz-backed dataset with a recorded eval source map.

build_splits.py finds each eval state's source by scanning per-trajectory text files. The
stochastic-tree quad3d PPO datasets ship one `train.npz` instead, plus `eval_fps.npz`,
which records for every eval row the flight it was picked from (`tid`) and the global
row in `states` (`row`). This builder reads that map, checks it against the stored rows,
and then applies the same rules as build_splits.py's picked-states branch: partition
flights into pool and the rest, keep only eval rows whose source flight is not in the
pool, split them into cal / val / test by source flight, and refuse any overlap. One
check goes further than the source map: no kept eval state may equal any stored row of
any pool flight (hashed float32 rows, candidates compared exactly).

Split files use the det layout `start(k), end(k), label`: the start is the eval file's
text, the end is the eval rollout's terminal row from `eval_success_prob.npz`.

Usage:
    python scripts/det_rerun/build_splits_npz.py --system quad3d_rl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np

from adaptive_roa.det_rerun.pool_npz import write_pool_npz
from adaptive_roa.det_rerun.splits import assert_disjoint, group_split, partition_ids

DATA = Path(os.environ.get("DATA_DIR", "/common/users/shared/pracsys/genMoPlan/data_trajectories"))
PARTITION_SEED = 20261029
RUN_SEEDS = (99512, 33407, 24475)   # paper seeds (tracker D04)
DEFAULT_SEEDS = (42,)               # configs' default seed; tests and ad hoc runs only

REFLY = Path(os.environ.get("DET_REFLY_DIR", "/common/users/st1122/det_rerun_refly"))

SYSTEMS = {
    # Pool and cal / val sizes as quad3d LQR (tracker section 5). test is capped at about the
    # LQR test size so per-epoch eval cost matches; the remaining eligible rows stay unused.
    "quad3d_rl": dict(dir="stochastic/quadrotor3D/corridor_sine_ambient/ppo_1500K/f_0.00", k=13,
                      sizes=dict(pool=600000, cal=10000, val=10000, test=240000), horizon=1000, eval="fps"),
    # Eval states are a grid flown separately from the train starts, so the pool can be every
    # train flight; end states come from the eval re-fly (eval_refly/cp_rl_refly.py), which also
    # showed no eval flight timed out. cal / val as the det cartpole (section 5).
    "cartpole_ppo": dict(dir="stochastic/cartpole/gaussian_signal/safe_explorer_ppo/baseline", k=4,
                         sizes=dict(pool=116242, cal=2000, val=2000, test=112603), horizon=150,
                         eval="refly", refly="cartpole"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def row_hash(x: np.ndarray) -> np.ndarray:
    """64-bit hash of each float32 row's bits; equal rows hash equal, collisions are checked exactly."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    h = np.zeros(len(u), dtype=np.uint64)
    for j in range(u.shape[1]):
        h = (h ^ u[:, j]) * np.uint64(0x100000001B3)
    return h


def rows_on_flights(states: np.ndarray, offsets: np.ndarray, flights: np.ndarray, query: np.ndarray) -> int:
    """How many query rows equal some stored row of the given flights."""
    idx = np.concatenate([np.arange(offsets[t], offsets[t + 1]) for t in flights]) if len(flights) else np.zeros(0, int)
    hq = row_hash(query)
    hp = row_hash(states[idx])
    cand = np.where(np.isin(hq, hp))[0]
    if not len(cand):
        return 0
    stored = {states[i].tobytes() for i in idx[np.isin(hp, hq[cand])]}
    return int(sum(query[c].astype(np.float32).tobytes() in stored for c in cand))


def _fps_sources(root, spec, states, offsets, labels, timeout, usable, eval_lines, text):
    """Eval states picked along train flights: eval_fps.npz names each one's source flight."""
    fps = np.load(root / "eval_fps.npz")
    tid, row, fstates = fps["tid"], fps["row"], fps["states"]
    if len(eval_lines) != len(tid):
        raise SystemExit(f"eval_states.txt has {len(eval_lines)} rows, eval_fps.npz {len(tid)}")
    ev = np.load(root / "eval_success_prob.npz")
    finals, p_success = ev["finals"][:, 0, :], ev["p_success"]
    if not (ev["trials"] == 1).all():
        raise SystemExit("eval rows carry more than one trial; this builder is for deterministic data")
    # The source map must be exact: row inside flight tid, stored row equal to the eval state.
    if not np.all((row >= offsets[tid]) & (row < offsets[tid + 1])):
        raise SystemExit("eval_fps rows fall outside their source flights")
    if not np.array_equal(states[row], fstates):
        raise SystemExit("eval_fps states differ from the stored rows they cite")
    gap = float(np.abs(text - fstates).max())
    if gap > 5.1e-7 * max(1.0, float(np.abs(fstates).max())):
        raise SystemExit(f"eval_states.txt and eval_fps.npz disagree by {gap}")
    eval_label = (p_success >= 0.5).astype(int)
    stats = {
        "eval_rows": len(tid),
        "distinct_source_flights": int(len(np.unique(tid))),
        "eval_rows_at_a_start": int(np.sum(row == offsets[tid])),
        "eval_label_vs_source_label_mismatch": int(np.sum(eval_label != labels[tid])),
        "eval_text_vs_npz_max_abs_gap": gap,
    }
    tparts = partition_ids(usable, {"pool": spec["sizes"]["pool"]}, rest="eval", seed=PARTITION_SEED)
    pool = np.sort(tparts["pool"])
    in_pool = np.zeros(len(labels), dtype=bool)
    in_pool[pool] = True
    keep = np.where(~in_pool[tid] & ~timeout[tid])[0]
    stats["eval_rows_dropped_on_pool_flight"] = int(len(tid) - len(keep))
    return pool, keep, tid[keep], fstates, finals, eval_label, stats, lambda r: np.unique(tid[r])


def load_refly(path: Path, n_rows: int):
    """The eval re-fly's shards, ordered by eval row; refuses a partial or disagreeing re-fly."""
    files = sorted(path.glob("shard_*.npz"))
    d = {key: np.concatenate([np.load(f)[key] for f in files]) for key in
         ("index", "stored_label", "success", "terminated", "final")}
    order = np.argsort(d["index"])
    d = {key: v[order] for key, v in d.items()}
    if not np.array_equal(d["index"], np.arange(n_rows)):
        raise SystemExit(f"re-fly in {path} covers {len(d['index'])} rows, not all {n_rows}")
    bad = int(np.sum(d["success"] != d["stored_label"]))
    if bad:
        raise SystemExit(f"re-fly in {path} disagrees with {bad} stored labels")
    return d


def _refly_sources(spec, n, usable, eval_lines, text):
    """Eval states flown on their own (a grid): every train flight may be pool; a flight that
    timed out in the re-fly is dropped from the eval rows."""
    rf = load_refly(REFLY / spec["refly"], len(eval_lines))
    stored = np.array([float(ln.rsplit(",", 1)[1]) for ln in eval_lines])
    eval_label = (stored >= 0.5).astype(int)
    tparts = partition_ids(usable, {"pool": min(spec["sizes"]["pool"], len(usable))}, rest="eval",
                           seed=PARTITION_SEED)
    pool = np.sort(tparts["pool"])
    keep = np.where(rf["terminated"] == 1)[0]
    stats = {"eval_rows": len(eval_lines), "eval_rows_timed_out_in_refly": int(len(eval_lines) - len(keep)),
             "refly_label_agreement": 1.0}
    fstates = text.astype(np.float32)
    return pool, keep, keep, fstates, rf["final"], eval_label, stats, lambda r: np.zeros(0, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--sizes", default=None, help="override, e.g. pool=8,cal=2,val=2,test=4 (tests only)")
    args = ap.parse_args()
    spec = dict(SYSTEMS[args.system])
    if args.sizes:
        spec["sizes"] = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.sizes.split(",")}
    root = DATA / spec["dir"]
    out = root / "splits_det_rerun"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    k = spec["k"]

    tr = np.load(root / "train.npz")
    states, offsets, labels = tr["states"], tr["offsets"], tr["labels"].astype(int)
    n = len(labels)
    timeout = tr["timeout"].astype(bool) if "timeout" in tr.files else np.zeros(n, dtype=bool)
    log(f"{args.system}: {n} flights, {len(states)} stored rows, {int(timeout.sum())} timeouts")
    eval_lines = [ln.strip() for ln in open(root / "eval_states.txt") if ln.strip()]
    text = np.array([[float(v) for v in ln.split(",")[:k]] for ln in eval_lines])
    usable = np.where(~timeout)[0]
    if spec["eval"] == "fps":
        pool, keep, groups, fstates, finals, eval_label, stats, sources = _fps_sources(
            root, spec, states, offsets, labels, timeout, usable, eval_lines, text)
        source_files = ("eval_states.txt", "eval_fps.npz")
    else:
        pool, keep, groups, fstates, finals, eval_label, stats, sources = _refly_sources(
            spec, n, usable, eval_lines, text)
        source_files = ("eval_states.txt",)
    on_pool = rows_on_flights(states, offsets, pool, fstates[keep])
    stats["kept_eval_states_equal_to_a_pool_row"] = on_pool
    if on_pool:
        raise SystemExit(f"{on_pool} kept eval states equal a stored row of a pool flight")
    log(f"{len(keep)} eval rows off the pool; none equals a pool row")

    sizes = {name: spec["sizes"][name] for name in ("cal", "val", "test")}
    gs = group_split(groups, sizes, rest="unused", seed=PARTITION_SEED)
    rows_of = {name: np.sort(keep[gs[name]]) for name in ("cal", "val", "test")}
    stats["eval_rows_unused"] = int(len(gs["unused"]))
    eval_trajs = {name: sources(r) for name, r in rows_of.items()}

    assert_disjoint({"pool": set(pool.tolist()), **{nm: set(v.tolist()) for nm, v in eval_trajs.items()}})
    assert_disjoint({nm: {fstates[e].tobytes() for e in r.tolist()} for nm, r in rows_of.items()})

    out.mkdir()
    names = [f"sequence_{t}.txt" for t in range(n)]
    (out / "pool_trajectories.txt").write_text("\n".join(names[t] for t in pool) + "\n")
    npz = write_pool_npz(out, [names[t] for t in pool], [states[offsets[t]:offsets[t + 1]] for t in pool],
                         labels[pool].tolist(), [False] * len(pool), seeds=RUN_SEEDS + DEFAULT_SEEDS,
                         horizon_steps=spec["horizon"])
    fmt = lambda v: ",".join(f"{x:.6f}" for x in v)
    for name, rows in rows_of.items():
        lines = [",".join(eval_lines[e].split(",")[:k]) + "," + fmt(finals[e]) + f",{eval_label[e]}"
                 for e in rows.tolist()]
        (out / f"{name}_set.txt").write_text("\n".join(lines) + "\n")
        (out / f"{name}_trajectories.txt").write_text("\n".join(names[t] for t in eval_trajs[name].tolist()) + "\n")

    manifest = {
        "system": args.system,
        "dataset": str(root),
        "built": str(date.today()),
        "script": "scripts/det_rerun/build_splits_npz.py",
        "partition_seed": PARTITION_SEED,
        "run_seeds": list(RUN_SEEDS),
        "default_seed_files_not_for_paper": list(DEFAULT_SEEDS),
        "source_files": {f: sha1(root / f) for f in source_files},
        "pool_trajectories": {"n": int(len(pool)), "success": int(labels[pool].sum())},
        "pool_rows": int(npz["total_states"]),
        "pool_npz": npz,
        "splits": {nm: {"n": int(len(r)), "success": int(eval_label[r].sum())} for nm, r in rows_of.items()},
        "split_trajectories": {nm: int(len(v)) for nm, v in eval_trajs.items()},
        "scan": stats,
        "overlap_check": {"pool_vs_eval_trajectories": 0, "eval_states_between_splits": 0,
                          "kept_eval_states_on_any_pool_trajectory_row": 0},
        "format": "cal/val/test rows: start (eval_states.txt text), end (the eval rollout's terminal row), "
                  "label (0/1); train.npz is the pool for NpzTrajectoryDataSource (rows follow "
                  "npz_row_names.txt); pool_order_<seed>.txt lists its row ids, pool_labels_<seed>.txt "
                  "the aligned 0/1 labels.",
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

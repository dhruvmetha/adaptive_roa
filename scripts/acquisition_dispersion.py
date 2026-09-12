#!/usr/bin/env python
"""Measure the spatial dispersion and label composition of each epoch's acquired batch.

Answers "does this selection rule clump in state space, and does the clumping depend
on the level's base rate" by recovering the exact set of trajectories each epoch
acquired and comparing it against a random batch of the same size.

WHY THE ID RECOVERY IS NOT OBVIOUS
----------------------------------
``datasets/train_trajectories.txt`` is written by
``DatasetBuilder.build_datasets`` -> ``build_trajectory_index`` and holds
``data_source.trajectory_name(idx)`` for each index. For the npz source that is
``rollout_ids[idx]``, i.e. a direct row index into ``train.npz:starts``. So the
values in the file index the start-state array with no further mapping.

The ordering is the trap. The file is the acquisition-ordered ``train_split``
with the VAL SLICE REMOVED FROM THE FRONT, and the val slice is
``n_val = max(1, int(val_ratio * len(train_split)))`` -- recomputed at every
rebuild, so it GROWS as the run deepens. A fixed offset gives a plausible,
silently wrong answer. Epoch k's acquired ids live at

    [initial_train_size - n_val + samples_per_epoch*k,  + samples_per_epoch)

which this script asserts rather than assumes: the line count must satisfy
``L + int(val_ratio*total) == total`` with ``total = initial + k*per_epoch``
exactly, or it refuses to proceed.

Cross-check two arms of the same level with --verify-pair. They share their
initial pool exactly, so once you shift by the difference in their val prefixes
the shared portion must be identical id-for-id. That catches an ordering or
offset error that the arithmetic alone would pass.

USAGE
    python scripts/acquisition_dispersion.py \
        --npz .../train.npz \
        --run <run-dir> [<run-dir> ...] \
        [--epochs 0-5] [--views position,velocity,quaternion,all]

    python scripts/acquisition_dispersion.py --npz ... \
        --verify-pair <run-dir-A> <run-dir-B>

State layout is read from --layout (default 'quadrotor3d': pos 0:3, quat 3:7,
lin vel 7:10, ang vel 10:13). Pass 'generic' for an unstructured state, which
disables the position/velocity/quaternion views and leaves only 'all'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

LAYOUTS = {
    # name: (position slice, quaternion slice, velocity slice, required state dim)
    # The dim is enforced, not assumed. Slicing a 13-dim layout over a 6-dim q2d
    # state does NOT fail loudly: 7:13 comes back empty and prints nan, but 3:7
    # comes back as 3 columns that nn_quat will normalise and report as perfectly
    # plausible geodesic angles for a system that has no quaternion at all.
    "quadrotor3d": (slice(0, 3), slice(3, 7), slice(7, 13), 13),
    "generic": (None, None, None, None),
}


def read_split_params(run_dir: Path) -> dict:
    """Pull the split geometry out of an arm's resolved Hydra config.

    Only plain scalars are read. dataset_root is left alone on purpose: it is an
    unresolved interpolation in config.yaml and resolving it needs the launching
    environment, so the caller passes --npz instead.
    """
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no .hydra/config.yaml under {run_dir}")
    cfg = yaml.safe_load(cfg_path.read_text())

    def find(key, node=cfg):
        """Depth-first search for a key; the split params sit at different depths
        across config versions."""
        if isinstance(node, dict):
            if key in node and not isinstance(node[key], (dict, list)):
                return node[key]
            for v in node.values():
                got = find(key, v)
                if got is not None:
                    return got
        return None

    params = {
        "initial": find("initial_train_size"),
        "per_epoch": find("samples_per_epoch"),
        "val_ratio": find("val_ratio"),
        "selection_rule": find("selection_rule"),
    }
    missing = [k for k, v in params.items() if v is None and k != "selection_rule"]
    if missing:
        raise ValueError(f"{run_dir.name}: config is missing {missing}")
    return params


def load_ids(run_dir: Path) -> np.ndarray:
    f = run_dir / "datasets" / "train_trajectories.txt"
    if not f.exists():
        raise FileNotFoundError(
            f"{run_dir.name}: no datasets/train_trajectories.txt. This file holds only "
            "the CURRENT rebuilt state and is not synced by the usual epoch-dir rsync, "
            "so read it on the box the arm actually runs on."
        )
    return np.loadtxt(f, dtype=np.int64, ndmin=1)


def epoch_blocks(ids: np.ndarray, p: dict,
                 allow_truncated: bool = False) -> tuple[dict[int, np.ndarray], int, int]:
    """Split the id list into per-epoch acquired batches.

    Raises rather than guessing: a silently wrong offset here would produce
    dispersion numbers that look entirely reasonable.

    On a small-budget campaign the val prefix can outgrow the whole initial pool
    (q2d: initial 2000, +500 x 24 -> pool 14000, prefix 2800), which eats the
    earliest batches off the front of the file. The later epochs are still at a
    well-defined offset, so ``allow_truncated`` recovers them -- but it NEVER
    returns a partial batch. A batch clipped from 500 to 200 members would still
    produce a perfectly plausible 1-NN number, and mean nearest-neighbour distance
    falls steeply with n, so a short batch compared against a full-size random
    baseline reads as far more dispersed than it is. Only whole batches are
    comparable, so the cut is at the first FULLY surviving epoch.
    """
    L = len(ids)
    initial, per_epoch, val_ratio = p["initial"], p["per_epoch"], p["val_ratio"]
    total = int(round(L / (1.0 - val_ratio)))
    n_val = max(1, int(total * val_ratio))
    if total - n_val != L:
        raise ValueError(
            f"val-prefix arithmetic does not close: {L} lines implies total={total}, "
            f"n_val={n_val}, which leaves {total - n_val}. Check val_ratio={val_ratio}."
        )
    if (total - initial) % per_epoch:
        raise ValueError(
            f"total {total} is not {initial} + k*{per_epoch}; the run may use a "
            "different acquisition budget than its config claims."
        )
    n_ep = (total - initial) // per_epoch
    off = initial - n_val

    k_min = 0
    if off < 0:
        # First epoch whose whole batch still sits at a non-negative file position.
        k_min = (-off + per_epoch - 1) // per_epoch  # ceil(-off / per_epoch)
        if k_min >= n_ep:
            raise ValueError(
                f"val prefix ({n_val}) has consumed the initial pool ({initial}) and "
                f"every one of the {n_ep} acquired batches; nothing is recoverable."
            )
        if not allow_truncated:
            raise ValueError(
                f"val prefix ({n_val}) has grown past the initial pool ({initial}) by "
                f"{-off}, so epochs 0-{k_min - 1} are not recoverable from this file. "
                f"Epochs {k_min}-{n_ep - 1} are exactly locatable; pass "
                f"--allow-truncated-history to measure those and drop the rest."
            )

    blocks = {}
    for k in range(k_min, n_ep):
        start = off + per_epoch * k
        block = ids[start: start + per_epoch]
        if len(block) != per_epoch:  # never hand back a short batch
            continue
        blocks[k] = block
    return blocks, n_ep, n_val


def verify_pair(a: Path, b: Path, allow_truncated: bool = False) -> None:
    """Two arms of the same level share their initial pool exactly. Shift by the
    difference in val prefixes and the shared portion must match id-for-id."""
    pa, pb = read_split_params(a), read_split_params(b)
    ia, ib = load_ids(a), load_ids(b)
    _, ka, na = epoch_blocks(ia, pa, allow_truncated)
    _, kb, nb = epoch_blocks(ib, pb, allow_truncated)
    print(f"{a.name}: {len(ia)} ids, depth {ka}, val prefix {na}")
    print(f"{b.name}: {len(ib)} ids, depth {kb}, val prefix {nb}")
    if pa["initial"] != pb["initial"]:
        print(f"REFUSING: different initial_train_size ({pa['initial']} vs {pb['initial']})")
        sys.exit(1)

    initial = pa["initial"]
    # Each file starts at initial-pool member n_val. The later-running arm has the
    # longer prefix, so align on the larger one.
    lo, hi = (na, nb) if na <= nb else (nb, na)
    shared = initial - hi
    if shared <= 0:
        print(f"CANNOT VERIFY: the larger val prefix ({hi}) has consumed the whole initial "
              f"pool ({initial}), so the two files share no initial-pool ids to compare. "
              "The join is not disproved, it is untestable this way -- on a small-budget "
              "campaign fall back to checking that both files' epoch blocks are the "
              "declared size and that their acquisitions overlap above chance.")
        sys.exit(2)
    A = (ia if na <= nb else ib)[hi - lo: hi - lo + shared]
    B = (ib if na <= nb else ia)[:shared]
    ok = np.array_equal(A, B)
    print(f"\nshared initial-pool tail: {shared} ids, identical = {ok}")
    if not ok:
        n_diff = int((A != B).sum())
        print(f"  MISMATCH in {n_diff}/{shared} positions -- the ordering assumption is wrong,")
        print("  do NOT trust per-epoch blocks from these files.")
        sys.exit(1)

    ba, _, _ = epoch_blocks(ia, pa, allow_truncated)
    bb, _, _ = epoch_blocks(ib, pb, allow_truncated)
    common = sorted(set(ba) & set(bb))
    if common:
        k = common[0]
        ov = len(np.intersect1d(ba[k], bb[k]))
        print(f"epoch-{k} batch overlap: {ov}/{len(ba[k])} ids "
              f"(identical rules on identical models would be ~{len(ba[k])})")
    print("\njoin verified")


def nn_euclid(X: np.ndarray) -> float:
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0))
    np.fill_diagonal(D, np.inf)
    return float(D.min(1).mean())


def nn_quat(Q: np.ndarray) -> float:
    """Mean 1-NN geodesic angle on SO(3). Euclidean distance between quaternions is
    not a rotation metric and double-covers, so use 2*arccos|<qi,qj>|."""
    if Q.shape[1] != 4:
        raise ValueError(
            f"nn_quat needs exactly 4 columns, got {Q.shape[1]}: normalising a "
            "non-quaternion slice returns believable angles that mean nothing."
        )
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    C = np.abs(Qn @ Qn.T).clip(0.0, 1.0)
    A = 2.0 * np.arccos(C)
    np.fill_diagonal(A, np.inf)
    return float(A.min(1).mean())


def build_views(layout: str, state_dim: int):
    """Views for this layout, validated against the actual state dimension.

    Refuses a mismatch instead of slicing anyway. An over-long layout fails in two
    different ways and only one of them is visible: an out-of-range slice comes back
    empty and prints nan, while a short slice comes back with the wrong number of
    columns and prints numbers that look entirely reasonable.
    """
    pos, quat, vel, need = LAYOUTS[layout]
    if need is not None and state_dim != need:
        raise SystemExit(
            f"--layout {layout} describes a {need}-dim state but this pool's states are "
            f"{state_dim}-dim. Slicing anyway would emit nan for the out-of-range view "
            f"and plausible-looking nonsense for the short one. Use --layout generic, "
            f"which measures the whole state vector and nothing else."
        )
    views = {}
    if pos is not None:
        views["position"] = lambda Z, S, s=pos: nn_euclid(Z[:, s])
        views["velocity"] = lambda Z, S, s=vel: nn_euclid(Z[:, s])
        views["quaternion"] = lambda Z, S, s=quat: nn_quat(S[:, s])
    views["all"] = lambda Z, S: nn_euclid(Z)
    return views


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True,
                    help="train.npz for the level (needs 'starts' and 'labels')")
    ap.add_argument("--run", nargs="+", default=[], help="run directories to measure")
    ap.add_argument("--verify-pair", nargs=2, metavar=("A", "B"),
                    help="verify the id join on two arms of the same level, then exit")
    ap.add_argument("--epochs", default=None,
                    help="epoch range to measure, e.g. '0-5'. Default: every epoch present.")
    ap.add_argument("--layout", default="quadrotor3d", choices=sorted(LAYOUTS))
    ap.add_argument("--views", default=None,
                    help="comma-separated subset of the layout's views")
    ap.add_argument("--baseline-draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--allow-truncated-history", action="store_true",
                    help="When the val prefix has outgrown the initial pool, measure the "
                         "epochs that remain exactly locatable instead of refusing. Never "
                         "returns a partial batch. The dropped range is printed per arm.")
    args = ap.parse_args()

    if args.verify_pair:
        verify_pair(Path(args.verify_pair[0]), Path(args.verify_pair[1]),
                    args.allow_truncated_history)
        return
    if not args.run:
        ap.error("--run is required unless --verify-pair is given")

    npz = np.load(args.npz)
    starts = npz["starts"].astype(np.float64)
    labels = npz["labels"]
    mu, sd = starts.mean(0), starts.std(0)
    sd[sd == 0] = 1.0
    Z = (starts - mu) / sd
    pool_br = float(labels.mean())

    views = build_views(args.layout, starts.shape[1])
    if args.views:
        want = [v.strip() for v in args.views.split(",")]
        unknown = [v for v in want if v not in views]
        if unknown:
            ap.error(f"unknown view(s) {unknown}; available: {sorted(views)}")
        views = {k: views[k] for k in want}

    lo, hi = 0, None
    if args.epochs:
        a, _, b = args.epochs.partition("-")
        lo, hi = int(a), int(b) if b else int(a)

    # One random baseline per level: batch size is fixed by samples_per_epoch, and
    # every arm on a level draws from the same start-state array, so the baseline
    # is shared and makes levels comparable to each other.
    rng = np.random.default_rng(args.seed)
    n_batch = read_split_params(Path(args.run[0]))["per_epoch"]
    base, degenerate = {}, {}
    for name, fn in views.items():
        vals = []
        for _ in range(args.baseline_draws):
            i = rng.choice(len(Z), n_batch, replace=False)
            vals.append(fn(Z[i], starts[i]))
        b = float(np.mean(vals))
        # A zero baseline means the pool has no spread in this view at all, so every
        # ratio would be a division by zero. Drop it with a reason: a nan column reads
        # as a broken run rather than as a constant coordinate.
        (degenerate if b <= 1e-12 else base)[name] = b
    for name in degenerate:
        del views[name]
    if not views:
        raise SystemExit("every view is degenerate on this pool; nothing to measure")
    print(f"pool base rate {pool_br:.4f}   random-{n_batch} 1-NN baselines: "
          + "  ".join(f"{k} {v:.4f}" for k, v in base.items()))
    if degenerate:
        print(f"  DROPPED degenerate view(s) {sorted(degenerate)}: zero spread across the "
              f"whole pool, so every state shares these coordinates and a dispersion "
              f"ratio is undefined. Not a failure, and not evidence of clustering.")

    out = []
    for rd in args.run:
        run_dir = Path(rd)
        try:
            p = read_split_params(run_dir)
            ids = load_ids(run_dir)
            blocks, n_ep, n_val = epoch_blocks(ids, p, args.allow_truncated_history)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n{run_dir.name}: SKIP ({e})")
            continue
        if p["per_epoch"] != n_batch:
            print(f"\n{run_dir.name}: SKIP (samples_per_epoch {p['per_epoch']} does not "
                  f"match the baseline's {n_batch}; baselines would not be comparable)")
            continue
        end = n_ep - 1 if hi is None else min(hi, n_ep - 1)
        ks = [k for k in range(lo, end + 1) if k in blocks]
        if not ks:
            print(f"\n{run_dir.name}: SKIP (depth {n_ep}, nothing in requested range)")
            continue
        dropped = [k for k in range(n_ep) if k not in blocks]
        if dropped:
            # Loud, never silent: a narrowed range must be visible next to the numbers.
            print(f"\n{run_dir.name}: DROPPED epochs {dropped[0]}-{dropped[-1]} "
                  f"({len(dropped)} of {n_ep}) -- val prefix {n_val} exceeds initial pool "
                  f"{p['initial']}; measuring {ks[0]}-{ks[-1]} only")
        acc = {v: [] for v in views}
        brs = []
        for k in ks:
            idx = blocks[k]
            for v, fn in views.items():
                acc[v].append(fn(Z[idx], starts[idx]))
            brs.append(float(labels[idx].mean()))
        rule = p["selection_rule"] or "?"
        print(f"\n{run_dir.name}  rule={rule}  depth={n_ep}  epochs {ks[0]}-{ks[-1]}")
        print("   " + "  ".join(f"{v} {np.mean(acc[v])/base[v]:.3f}x" for v in views)
              + f"   batch success rate {np.mean(brs):.3f} (pool {pool_br:.3f})")
        out.append(dict(run=run_dir.name, rule=rule, depth=n_ep, epochs=ks,
                        pool_br=pool_br, batch_br=float(np.mean(brs)),
                        ratio={v: float(np.mean(acc[v]) / base[v]) for v in views},
                        per_epoch={v: [float(x) for x in acc[v]] for v in views},
                        baseline=base))

    if args.json_out and out:
        Path(args.json_out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

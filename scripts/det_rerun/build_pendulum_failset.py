#!/usr/bin/env python3
"""Build deterministic/pendulum/lqr_failset from deterministic/pendulum/lqr.

Adds a failure set (a ball around each of the two failure equilibria) and recuts every
trajectory at its first entry into success or failure. Unchanged trajectories are
symlinked to the source; recut ones are new files holding the source's text lines up to
and including the hitting row. The source dataset is only read.

Usage:
    python scripts/det_rerun/build_pendulum_failset.py \
        --src $DATA_DIR/deterministic/pendulum/lqr \
        --dst $DATA_DIR/deterministic/pendulum/lqr_failset
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import numpy as np

from adaptive_roa.det_rerun.failset import first_hit, pendulum_membership, wrap_angle
from adaptive_roa.det_rerun.splits import row_key

SUCCESS_RADIUS = 0.075   # collector rule, unchanged
FAILURE_RADIUS = 0.05


def read_lines(path: Path) -> list[str]:
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")

    names = [ln.strip() for ln in open(src / "train_test_splits/all_shuffled_indices.txt")]
    stored = np.loadtxt(src / "train_test_splits/all_shuffled_labels.txt", dtype=int)
    assert len(names) == len(stored)

    print(f"reading {len(names)} trajectories", flush=True)
    lines = []
    for i, n in enumerate(names):
        lines.append(read_lines(src / "trajectories" / n))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}", flush=True)
    arrays = [np.array([[float(v) for v in ln.split(",")] for ln in L]) for L in lines]

    # Failure centres: median wrapped angle of settled failure endpoints on each side.
    last = np.array([a[-1] for a in arrays])
    settled = (stored == 0) & (np.abs(last[:, 1]) < 1e-3)
    th = wrap_angle(last[settled, 0])
    centers = [(float(np.median(th[th > 0])), 0.0), (float(np.median(th[th < 0])), 0.0)]
    spread = {f"{c[0]:+.4f}": float(np.std(th[th > 0] if c[0] > 0 else th[th < 0])) for c in centers}
    print(f"failure centres {centers}, settled n={int(settled.sum())}, std {spread}", flush=True)

    hit_row, label = np.empty(len(names), dtype=object), np.empty(len(names), dtype=int)
    for i, a in enumerate(arrays):
        in_s, in_f = pendulum_membership(a, SUCCESS_RADIUS, centers, FAILURE_RADIUS)
        hit_row[i], label[i] = first_hit(in_s, in_f)

    n_rows = np.array([len(a) for a in arrays])
    success_src = stored == 1
    stats = {
        "source_success": int(success_src.sum()),
        "source_failure": int((stored == 0).sum()),
        # The collector stops a success at its first success row, so this should be 0.
        "source_success_not_ending_at_first_success_row": int(sum(
            1 for i in np.where(success_src)[0]
            if label[i] == 1 and hit_row[i] != n_rows[i] - 1)),
        "source_success_relabelled_failure": int(np.sum(success_src & (label == 0))),
        "source_failure_relabelled_success": int(np.sum(~success_src & (label == 1))),
        "success": int(np.sum(label == 1)),
        "failure": int(np.sum(label == 0)),
        "timeout": int(np.sum(label == -1)),
        "failure_hit_on_last_row": int(sum(1 for i in np.where(label == 0)[0] if hit_row[i] == n_rows[i] - 1)),
    }
    print(json.dumps(stats, indent=1), flush=True)

    (dst / "trajectories").mkdir(parents=True)
    n_link = n_new = 0
    for i, n in enumerate(names):
        out = dst / "trajectories" / n
        keep = n_rows[i] if hit_row[i] is None else hit_row[i] + 1
        if keep == n_rows[i]:
            os.symlink(src.resolve() / "trajectories" / n, out)
            n_link += 1
        else:
            with open(out, "w") as f:
                f.write("\n".join(lines[i][:keep]) + "\n")
            n_new += 1
    stats["files_symlinked"], stats["files_recut"] = n_link, n_new

    with open(dst / "trajectory_labels.txt", "w") as f:
        for n, l in zip(names, label):
            f.write(f"{n},{l}\n")
    with open(dst / "timeouts.txt", "w") as f:
        for n, l in zip(names, label):
            if l == -1:
                f.write(n + "\n")
    # Every resolved trajectory as start, first-hit endpoint, label; timeouts left out.
    with open(dst / "eval_states.txt", "w") as f:
        for i, n in enumerate(names):
            if label[i] == -1:
                continue
            end = arrays[i][hit_row[i]]
            f.write(f"{row_key(arrays[i][0])},{row_key(end)},{label[i]}\n")

    desc = json.load(open(src / "dataset_description.json"))
    desc["derived_from"] = {
        "source": str(src),
        "script": "scripts/det_rerun/build_pendulum_failset.py",
        "built": str(date.today()),
        "note": "Trajectories recut at the first entry into the success or failure set; "
                "unchanged files are symlinks into the source.",
    }
    tc = desc["generation_parameters"]["termination_conditions"]
    tc["failure"] = {
        "name": "Failure equilibrium reached",
        "condition": f"wrapped-angle L2 distance to either failure centre below {FAILURE_RADIUS}",
        "centers_theta_thetadot": centers,
        "radius": FAILURE_RADIUS,
        "centers_measured_from": f"median wrapped theta of {int(settled.sum())} source failure "
                                 "endpoints with |theta_dot| < 1e-3",
        "early_termination": True,
        "termination_rule": "Recut at the first row inside either failure ball; that row is the final state.",
    }
    tc["first_hit_rule"] = "label = whichever of success or failure is entered first; neither within the horizon = timeout (label -1, listed in timeouts.txt, excluded from eval_states.txt)"
    # The loader treats rows == horizon_steps + 1 as a timeout. Timeouts are excluded from
    # every split, and a failure can first hit on the last stored row, so set the horizon
    # to the stored row count: no pool trajectory is then read as a timeout.
    desc["horizon_steps"] = int(n_rows.max())
    desc["horizon_note"] = ("Stored trajectories have at most this many rows. Timeouts (label -1) are "
                            "excluded from all splits, so timeout_intermediates=drop is a no-op here; "
                            "a horizon of rows - 1 would misread failures that first hit on the last row "
                            f"({stats['failure_hit_on_last_row']} trajectories) as timeouts.")
    desc["first_hit_statistics"] = stats
    with open(dst / "dataset_description.json", "w") as f:
        json.dump(desc, f, indent=2)
    print(f"wrote {dst}: {n_new} recut, {n_link} symlinked; horizon_steps={desc['horizon_steps']}", flush=True)


if __name__ == "__main__":
    main()

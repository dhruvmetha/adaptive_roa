#!/usr/bin/env python3
"""Summarize an eval re-fly (q2d_refly.py or cp_rl_refly.py shards).

Reports, over every re-flown eval state: agreement with the stored label, flights
that never terminated (a timeout the shipped label would have called a failure),
the longest flight, and whether each final state lies in the success ball or past
a wall. Writes <out_dir>/summary.json and, when any flight timed out, the indices
of those eval rows to <out_dir>/timeout_rows.txt.

    reduce_refly.py --system q2d|cartpole --out-dir <dir with shard_*.npz> --expect N
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


def classify(system: str, F: np.ndarray):
    if system == "cartpole":      # file order [x, theta, x_dot, theta_dot], goal x = 0.7
        e = np.stack([F[:, 0] - 0.7, wrap(F[:, 1]), F[:, 2], F[:, 3]], 1)
        ball = np.linalg.norm(e, axis=1) < 0.05
        wall = (np.abs(F[:, 0]) >= 3.0) | (np.abs(F[:, 2]) >= 3.0) | (np.abs(F[:, 1]) >= 0.48) | (np.abs(F[:, 3]) >= 3.0)
    else:                         # file order [x, z, theta, x_dot, z_dot, theta_dot], goal (0, 1)
        e = np.stack([F[:, 0], F[:, 1] - 1.0, wrap(F[:, 2]), F[:, 3], F[:, 4], F[:, 5]], 1)
        ball = np.linalg.norm(e, axis=1) < 0.2
        wall = ((np.abs(F[:, 0]) >= 1.0) | (F[:, 1] <= 0.1) | (F[:, 1] >= 1.5) | (np.abs(F[:, 3]) >= 1.0)
                | (np.abs(F[:, 4]) >= 1.0) | (np.abs(F[:, 5]) >= 8.0))
    return ball, wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["q2d", "cartpole"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expect", type=int, required=True, help="number of eval rows")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.out_dir}/shard_*.npz"))
    d = {k: np.concatenate([np.load(f)[k] for f in files]) for k in
         ("index", "stored_label", "success", "steps", "terminated", "final")}
    horizon = int(np.load(files[0])["horizon"])
    covered = np.unique(d["index"])
    ball, wall = classify(a.system, d["final"])
    timeout = d["terminated"] == 0
    s = {
        "system": a.system, "shards": len(files), "rows": int(len(d["index"])), "expected_rows": a.expect,
        "complete": bool(len(covered) == a.expect and covered[0] == 0 and covered[-1] == a.expect - 1),
        "horizon_steps": horizon, "steps_max": int(d["steps"].max()),
        "not_terminated": int(timeout.sum()),
        "label_agreement": float(np.mean(d["success"] == d["stored_label"])),
        "label_disagreements": int(np.sum(d["success"] != d["stored_label"])),
        "final_in_ball": int(ball.sum()), "final_past_wall": int(wall.sum()),
        "final_neither": int((~ball & ~wall).sum()),
        "timeouts_stored_as_failure": int(np.sum(timeout & (d["stored_label"] == 0))),
    }
    Path(a.out_dir, "summary.json").write_text(json.dumps(s, indent=2))
    if timeout.any():
        Path(a.out_dir, "timeout_rows.txt").write_text("\n".join(map(str, d["index"][timeout].tolist())) + "\n")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()

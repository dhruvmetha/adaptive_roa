#!/usr/bin/env python
"""Stopping-time and cutoff table per det system (tracker H08; REVIEW_TODO R09).

From each system's pool train.npz: trajectory count, success share, hitting-time
quantiles (stored rows, which is steps + 1) by outcome, and the timeouts the
loader would flag (rows == horizon + 1 and not a success).
"""
import json
from pathlib import Path

import numpy as np

D = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic")
SYSTEMS = {"pendulum": "pendulum/lqr_failset", "cartpole": "cartpole_pybullet",
           "quad2d": "quadrotor2D_rl", "quad3d": "quadrotor3D_lqr"}


def main():
    print("| System | Pool trajectories | Success share | Rows, success (p50 / p90 / max) | Rows, failure (p50 / p90 / max) | Horizon | Timeouts |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, rel in SYSTEMS.items():
        sd = D / rel / "splits_det_rerun"
        with np.load(sd / "train.npz") as z:
            n = np.diff(z["offsets"])
            y = z["labels"].astype(int)
        h = json.loads((sd / "train_description.json").read_text())["horizon_steps"]
        q = lambda a: " / ".join(str(int(v)) for v in np.quantile(a, [0.5, 0.9, 1.0])) if len(a) else "-"
        timeouts = int(np.sum((n == h + 1) & (y != 1)))
        print(f"| {name} | {len(n):,} | {y.mean():.3f} | {q(n[y == 1])} | {q(n[y != 1])} | {h} | {timeouts} |")


if __name__ == "__main__":
    main()

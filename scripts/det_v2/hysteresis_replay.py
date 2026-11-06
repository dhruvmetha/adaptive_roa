#!/usr/bin/env python
"""Replay the A9 tau* regime hysteresis over a run's stored threshold-fit rows.

Feeds each epoch's predictions_thrfit.npz, in order, through
HysteresisJSelThresholdBackend with a fixed margin (no bootstrap), and prints per run
the raw regime of each round's global argmin (I = tau_s + tau_f >= 1, O = < 1), and
for each margin the regime flips and the J_sel cost of holding (chosen tau* minus
the round's global best; 0 on a switch round, since the switch takes the better regime).

Stored p is float32, so k/K reads back off the grid (0.2 -> 0.200000003) and flips
strict-> ties; p is restored to round(p K) / K first.

    hysteresis_replay.py <run_dir> [<run_dir> ...] [--margins 0.002,0.005,0.01]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adaptive_roa.adaptive_v2.eval import det_v2_rule as R
from adaptive_roa.adaptive_v2.threshold.hysteresis import HysteresisJSelThresholdBackend


def rounds(run_dir: Path):
    """(p_s, p_f, y, stored tau*) per epoch, with p restored to multiples of 1/K."""
    out = []
    for f in sorted(glob.glob(str(run_dir / "epoch_*" / "predictions_thrfit.npz"))):
        z = np.load(f)
        k = int(z["num_mc_samples"])
        ps, pf = (np.round(z[key].astype(float) * k) / k for key in ("p_success", "p_failure")) if k > 1 \
            else (z["p_success"].astype(float), z["p_failure"].astype(float))
        ts = json.loads((Path(f).parent / "artifacts_v2.json").read_text())["threshold_state"]
        out.append((ps, pf, z["true_labels"], (ts["tau_s_star"], ts["tau_f_star"])))
    return out


def backend(margin: float, w: float = 0.1) -> HysteresisJSelThresholdBackend:
    d = {"predictor_type": "endpoint", "objective": "j_sel", "w": w, "num_mc_samples": 100,
         "verbose": False, "margin": margin}
    return HysteresisJSelThresholdBackend(SimpleNamespace(**d, get=lambda k, v=None: d.get(k, v)), None, "cpu")


def replay(run_dir: Path, margins: list[float]) -> dict:
    rs = rounds(run_dir)
    raw, match, glob_j = [], [], []
    for ps, pf, y, stored in rs:
        g = R.fit(ps, pf, y, w=0.1)
        raw.append(R.region_of(g["tau_s"], g["tau_f"]))
        match.append(abs(g["tau_s"] - stored[0]) < 1e-9 and abs(g["tau_f"] - stored[1]) < 1e-9)
        glob_j.append(g["j"])
    res = {"n": len(rs), "match": all(match), "regimes": "".join("I" if r == "inv" else "O" for r in raw),
           "raw_flips": sum(a != b for a, b in zip(raw, raw[1:])), "margins": {}}
    for m in margins:
        b = backend(m)
        regions, costs = [], []
        for (ps, pf, y, _), gj in zip(rs, glob_j):
            st = b.optimize_probs(None, y, ps, pf, 1 - ps - pf, 100)
            regions.append(st.threshold_region)
            jm = R.objectives(R.grid_counts(ps, pf, y), 0.1)["j_sel"]
            costs.append(float(jm[round(st.tau_s_star * 100) - 1, round(st.tau_f_star * 100) - 1]) - gj)
        res["margins"][m] = {"flips": sum(a != b for a, b in zip(regions, regions[1:])),
                             "mean_cost": float(np.mean(costs[1:])), "max_cost": float(np.max(costs[1:]))}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--margins", default="0.002,0.005,0.01")
    a = ap.parse_args()
    margins = [float(x) for x in a.margins.split(",")]
    for run in a.runs:
        r = replay(run, margins)
        if r["n"] < 2:
            continue
        line = f"{run.name[:24]:24} n={r['n']:2} match={r['match']} regimes={r['regimes']} flips={r['raw_flips']}"
        for m, v in r["margins"].items():
            line += f" | eps={m}: flips={v['flips']} meanJcost={v['mean_cost']:.4f} maxJcost={v['max_cost']:.4f}"
        print(line)


if __name__ == "__main__":
    main()

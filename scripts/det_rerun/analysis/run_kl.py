#!/usr/bin/env python
"""KL (log score of p_success) before and after the common post-hoc step, per regime.

The step, identical for every method and system:
  1. form p from the samples two ways: raw (n_s + 0.5) / (K + 1) and renorm
     (n_s + 0.5) / (n_s + n_f + 1), which drops invalid samples from the vote
     (identical for a classifier, which has no invalid mass);
  2. fit Platt scaling sigmoid(a * logit(p) + b) on val for each;
  3. keep the variant with the lower NLL on the held-out cal split;
  4. score that on the test grid.
Usage: run_kl.py --mode ranked [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib  # noqa: E402
from cells import LEARNED, SETTINGS, SHORT, load_cell, matched_epoch, run_dirs  # noqa: E402


def _cal(d: Path, epoch: int):
    z = np.load(d / f"epoch_{epoch:03d}" / "predictions_cal.npz")
    if "n_success" in z.files:
        k = float(z["num_mc_samples"])
        return z["n_success"] / k, z["n_failure"] / k, z["true_labels"], int(k)
    p = np.asarray(z["p_success"], float)
    return p, 1 - p, z["true_labels"], None


def kl_row(d: Path, E: int) -> dict:
    c = load_cell(d, E)
    ps, pf, _, y = c["test"]
    vs, vf, _, vy = c["val"]
    cs, cf, cy, kc = _cal(d, E)
    k, kv = c["k_test"], (100 if c["counts"] else None)
    out = {"raw": calib.nll(calib.raw(ps, k), y)}
    best = None
    for name, ft, fv, fc in (("raw", calib.raw(ps, k), calib.raw(vs, kv), calib.raw(cs, kc)),
                             ("renorm", calib.renorm(ps, pf, k), calib.renorm(vs, vf, kv), calib.renorm(cs, cf, kc))):
        ab = calib.fit_platt(fv, vy)
        sel = calib.nll(calib.apply_platt(fc, ab), cy)
        test = calib.nll(calib.apply_platt(ft, ab), y)
        out[f"{name}+platt"] = test
        if best is None or sel < best[0]:
            best = (sel, name, test, ab)
    out["chosen"], out["after"], out["platt_ab"] = best[1], best[2], list(best[3])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ranked")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dirs = run_dirs()
    rows = []
    for s in SETTINGS:
        E = matched_epoch(dirs, s, a.mode) if a.mode != "uniform" else None
        for arm in LEARNED:
            key = (s, arm, a.mode)
            if key not in dirs:
                continue
            e = E
            if e is None:      # uniform: each cell at its own latest complete epoch
                from cells import complete_epochs
                eps = complete_epochs(dirs[key])
                if not eps:
                    continue
                e = max(eps)
            r = {"setting": s, "arm": arm, "mode": a.mode, "epoch": e, **kl_row(dirs[key], e)}
            rows.append(r)
            print(f"{s:7s} {SHORT[arm]:8s} ep{e:<2d} raw {r['raw']:.4f} -> {r['after']:.4f} ({r['chosen']}+platt)")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()

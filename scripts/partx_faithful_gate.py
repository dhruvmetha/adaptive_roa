#!/usr/bin/env python
"""Pendulum gate for the partx_faithful arm (spec §8.3). Criteria were fixed in the spec before any run:

  C1  every epoch directory exists and hits its budget, or the run records unspent budget
  C2  (low only) classified leaves hold >= 50% of the eval grid at the last checkpoint
  C3  <= 5% of eval-grid points in classified leaves are on the wrong side of p = 0.5 (oracle)
  C4  acquired points are ambiguous (0.2 <= p <= 0.8) more often than the initial set
  C5  n0 = 10 and n0 = 30 reported side by side (no pass/fail)

Usage:
  python scripts/partx_faithful_gate.py --low-key low --out docs/partx_faithful_pendulum_gate.md \
      --run low=<run_dir>:<dataset_root> --run med=... --run high=... --run med_n030=...
"""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spm = importlib.import_module("stoch_prob_metrics")


def leaf_types_at(X, lows, highs, rtypes, root_low, root_high) -> np.ndarray:
    X = np.clip(np.asarray(X, dtype=float), root_low, root_high)
    out = np.full(len(X), "", dtype=object)
    todo = np.ones(len(X), dtype=bool)
    for lo, hi, t in zip(lows, highs, rtypes):
        m = todo & np.all((X >= lo) & (X <= hi), axis=1)
        out[m] = t
        todo &= ~m
    return out


def coverage_and_wrong_side(types, p):
    types = np.asarray(types, dtype=object)
    classified = np.isin(types, ["+", "-"])
    wrong = ((types == "+") & (p < 0.5)) | ((types == "-") & (p > 0.5))
    n_c = int(classified.sum())
    return float(classified.mean()), (float(wrong.sum() / n_c) if n_c else 0.0), n_c


def ambiguous_fraction(p, lo=0.2, hi=0.8) -> float:
    p = np.asarray(p, dtype=float)
    return float(np.mean((p >= lo) & (p <= hi))) if len(p) else float("nan")


def check_run(name: str, run_dir: Path, dataset_root: Path, is_low: bool) -> dict:
    cfg = yaml.safe_load((run_dir / ".hydra" / "config.yaml").read_text())
    initial, spe, n_epochs = int(cfg["initial_train_size"]), int(cfg["samples_per_epoch"]), int(cfg["n_epochs"])
    eps = [run_dir / f"epoch_{e:03d}" for e in range(n_epochs)]
    complete = all((d / "artifacts_v2.json").exists() and (d / "full_roa_per_point.npz").exists() for d in eps)
    arts = [json.loads((d / "artifacts_v2.json").read_text()) for d in eps] if complete else []
    on_budget = complete and all(a["train_trajectories"] == initial + e * spe for e, a in enumerate(arts))
    unspent = arts[-1]["acquisition"]["diagnostics"]["unspent"] if complete else None
    c1 = bool(complete and (on_budget or (unspent or 0) > 0))
    if not complete:
        return dict(name=name, c1=False, note="incomplete run")
    last = eps[-1]
    leaves = np.load(last / "partx_leaves.npz")
    with np.load(last / "full_roa_per_point.npz") as z:
        X = z["start_states"].astype(float)
    gt = spm.load_ground_truth(dataset_root)
    p = gt[1][spm.match_to_truth(X, gt[0])]
    types = leaf_types_at(X, leaves["low"], leaves["high"], leaves["rtype"].astype(str),
                          leaves["root_low"], leaves["root_high"])
    cov, wrong, n_c = coverage_and_wrong_side(types, p)
    with open(run_dir / "partx_state.pkl", "rb") as f:
        px = pickle.load(f)["px"]
    tree = cKDTree(gt[0])
    p_init = gt[1][tree.query(px.store.X(np.arange(px.n_init)))[1]]
    p_acq = gt[1][tree.query(px.store.X(px.log))[1]]
    a_init, a_acq = ambiguous_fraction(p_init), ambiguous_fraction(p_acq)
    row, _ = spm.score_epoch_full(last, gt, None, None)
    return dict(name=name, n0=arts[-1]["extra"].get("n0_resolved"), c1=c1, coverage=cov,
                c2=(cov >= 0.50) if is_low else None, wrong=wrong, n_classified=n_c,
                c3=wrong <= 0.05, amb_init=a_init, amb_acq=a_acq, c4=a_acq > a_init,
                unspent=unspent, KL=row["KL"], sAUROC=row.get("sAUROC"))


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "pass" if v else "FAIL"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=run_dir:dataset_root")
    ap.add_argument("--low-key", default="low")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for spec in args.run:
        name, rest = spec.split("=", 1)
        run_dir, droot = rest.split(":", 1)
        rows.append(check_run(name, Path(run_dir), Path(droot), name == args.low_key))
    cols = ["name", "n0", "c1", "coverage", "c2", "wrong", "n_classified", "c3", "amb_init", "amb_acq",
            "c4", "unspent", "KL", "sAUROC"]
    lines = ["# partx_faithful pendulum gate", "", "Criteria from spec §8.3, fixed before any run.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    gated = [r for r in rows if r.get("n0") in (10, None) and not r["name"].endswith("_n030")]
    ok = all(r.get("c1") and r.get("c3") and r.get("c4") and r.get("c2") in (True, None) for r in gated)
    lines += ["", f"**Gate (n0 = 10 runs): {'PASS' if ok else 'FAIL'}**",
              "", "C5 is the n0 = 10 vs n0 = 30 comparison in the table above; it has no pass/fail."]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

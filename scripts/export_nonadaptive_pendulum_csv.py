#!/usr/bin/env python
"""CSV of probability-comparison metrics for the non-adaptive pendulum campaign.

Emits, per (level, arm, train_size): KL, Brier (raw + doubly-debiased), sAUROC
and F1. Metric DEFINITIONS are imported from scripts/stoch_prob_metrics.py --
the same module behind docs/stoch_compare -- rather than reimplemented, so these
numbers are comparable with that work.

Two ground-truth regimes, deliberately kept distinct:

  stochastic (tau_*)  p_success is estimated from M=100 rollouts per grid cell,
                      read from eval_success_prob.npz. F1 is `f1@0.5`:
                      BOTH sides thresholded at 0.5, i.e. the model's field is
                      scored against the true field binarised the same way.
  deterministic       ground truth is exact, so there is no M-sample noise to
                      debias (trials=1 -> _noise() returns 0) and p in {0,1}.
                      F1 is therefore the ordinary binary F1 against the real
                      label, which is what `f1@0.5` reduces to when p is already
                      binary -- the same code path, no special-casing needed.

The M-sample debias term is what makes brier_debiased comparable ACROSS levels:
without it a noisier grid looks like a worse model. The deterministic row gets a
zero correction, which is correct (exact truth) but means its raw and debiased
Brier coincide -- do not read that as the anchor being better calibrated.

K (the model's own sample count) is taken from the run: count-based arms report
`log_score_smoothing == "kt_count"` and get K = num_mc_samples; a classifier
emits p directly and gets K = None, so no model-side noise is subtracted.

Usage:
    ./env/bin/python scripts/export_nonadaptive_pendulum_csv.py -o docs/nonadaptive_pendulum_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stoch_prob_metrics import (  # noqa: E402
    all_metrics,
    load_ground_truth,
    match_to_truth,
)

EXP = "/common/users/shared/pracsys/adaptive_roa_experiments"
DATA = "/common/users/shared/pracsys/genMoPlan/data_trajectories"

LEVELS = [
    ("det",      "adaptive_pendulum_det_anchor_lqr",           f"{DATA}/deterministic/pendulum/lqr"),
    ("tau_0.10", "adaptive_pendulum_noisy_torque_tau_0.10",    f"{DATA}/stochastic/pendulum/noisy_torque/lqr/tau_0.10"),
    ("tau_0.15", "adaptive_pendulum_noisy_torque_tau_0.15",    f"{DATA}/stochastic/pendulum/noisy_torque/lqr/tau_0.15"),
    ("tau_0.30", "adaptive_pendulum_noisy_torque_tau_0.30",    f"{DATA}/stochastic/pendulum/noisy_torque/lqr/tau_0.30"),
    ("tau_0.50", "adaptive_pendulum_noisy_torque_tau_0.50",    f"{DATA}/stochastic/pendulum/noisy_torque/lqr/tau_0.50"),
]
ARMS = ["fm", "fm_outcome", "mlp"]
SIZES = [500, 1000, 2000, 4000, 6000]


# The deterministic anchor's ground truth is EXACT, not an M-draw estimate, so
# the correct treatment is the M -> infinity limit: the debias term
# p(1-p)/(M-1) vanishes and the sharpness factor M/(M-1) tends to 1. Taken
# numerically with a large sentinel because np.inf breaks M/(M-1) (inf/inf =
# nan), and M=1 divides by zero. The CSV reports M as "inf" rather than the
# sentinel so no reader mistakes it for a real rollout count.
M_EXACT = 1e12


def ground_truth(dataset_root: Path, npz) -> tuple:
    """(starts, p, successes, trials); exact and binary for the deterministic anchor."""
    if (dataset_root / "eval_success_prob.npz").exists():
        return load_ground_truth(dataset_root)
    # Deterministic: the label IS the truth. true_labels is {-1,+1} -- mapped
    # with `> 0`, NOT astype(bool), which would score -1 as True.
    y = np.asarray(npz["true_labels"])
    p = (y > 0).astype(np.float64)
    return (np.asarray(npz["start_states"], dtype=np.float64), p,
            p * M_EXACT, np.full_like(p, M_EXACT))


def k_for(art: dict) -> float | None:
    """Model-side sample count, or None when the arm emits p directly."""
    em = art.get("eval_metrics", {}) or {}
    tf = em.get("threshold_free") or {}
    if tf.get("log_score_smoothing") != "kt_count":
        return None
    try:
        k = float(em.get("num_mc_samples"))
        return k if np.isfinite(k) and k > 1 else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="docs/nonadaptive_pendulum_metrics.csv")
    args = ap.parse_args()

    rows, missing = [], []
    for level, root, ds in LEVELS:
        base, dsp = Path(EXP) / root, Path(ds)
        gt_cache = None
        for arm in ARMS:
            for n in SIZES:
                ep = base / "outputs" / arm / f"train_{n}_seed_42_d2_0.0_epochs_1" / "epoch_000"
                npz_p, art_p = ep / "full_roa_per_point.npz", ep / "artifacts_v2.json"
                if not (npz_p.exists() and art_p.exists()):
                    missing.append(f"{level}/{arm}/n={n}")
                    continue
                with np.load(npz_p) as z:
                    states = np.asarray(z["start_states"], dtype=np.float64)
                    p_hat = np.asarray(z["p_success"], dtype=np.float64)
                    if gt_cache is None:
                        gt_cache = ground_truth(dsp, z)
                art = json.loads(art_p.read_text())
                gs, gp, gsucc, gtr = gt_cache
                idx = match_to_truth(states, gs)
                m = all_metrics(p_hat, gp[idx], gsucc[idx], gtr[idx], k=k_for(art))
                rows.append({
                    "level": level, "arm": arm, "train_size": n,
                    "n_points": m["n_points"], "K": m["K"],
                    "M": "inf" if float(m["M"]) >= M_EXACT else round(float(m["M"]), 1),
                    "KL": round(m["KL"], 6),
                    "brier_raw": round(m["brier_raw"], 6),
                    "brier_debiased": round(m["brier_debiased"], 6),
                    "brier_debiased_se": round(m["brier_debiased_se"], 6),
                    "sAUROC": round(m["sAUROC"], 6),
                    "F1": round(m["f1@0.5"], 6),
                    "precision": round(m["prec@0.5"], 6),
                    "recall": round(m["rec@0.5"], 6),
                    "skill_score": round(m["skill_score"], 6),
                    "MAE": round(m["MAE"], 6),
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    if missing:
        # Named, not just counted: a silently short table is the failure mode here.
        print(f"NOT YET COMPLETE ({len(missing)}): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

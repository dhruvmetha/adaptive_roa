#!/usr/bin/env python
"""One row per cell of the non-adaptive cartpole campaign.

Combines the two metric families the campaign needs:

  * thresholded / threshold-free scores read from each run's own
    ``artifacts_v2.json`` (f1, accuracy, AUC, Brier) -- the same columns the
    pendulum CSV carries, so the two studies line up.
  * graded probabilistic scores computed here against the eval grid's
    ``eval_success_prob.npz`` via scripts/stoch_prob_metrics.py -- **KL** and
    **sAUROC**, plus the debiased Brier and skill score that come with them.

The stochastic levels take their ground truth from eval_success_prob.npz
(successes/trials over M~95 rollouts per cell). The DETERMINISTIC anchor has no
such file, but it does not need one: a deterministic system's p_success is
exactly 0 or 1, and that IS the label in its 9-column eval file. So it is scored
against p in {0,1} with trials=1, which is correct rather than a substitute:

  * sAUROC reduces exactly to standard AUROC when the truth is binary -- each
    cell contributes p positive and (1-p) negative mass, which for p in {0,1} is
    one unit on one side.
  * KL becomes the log loss against a certain outcome; it has no M dependence.
  * the debiasing does the right thing automatically: `_noise` returns zero for
    n <= 1 ("not a proportion over draws"), so brier_debiased and skill_score
    subtract the model's K-sample noise but NO grid noise -- correct, because
    the truth here is exact, not estimated.

ONE ASYMMETRY REMAINS, and it is reported in the `truth` column rather than
hidden: the sigma levels carry an irreducible KL/Brier floor from their M~95
sampling, while the deterministic level's floor is zero. A deterministic KL is
therefore measured against a stricter reference and is NOT on the same footing
as a stochastic one. Compare within a column, not across the `truth` boundary.

Predicted probabilities are read from full_roa_per_point.npz, so this never
retrains and can be re-run at will.

Usage:
    ./env/bin/python scripts/cartpole_nonadapt_csv.py --out docs/cartpole_nonadapt.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stoch_prob_metrics import (  # noqa: E402
    calibration_scores, debiased_brier, load_ground_truth, match_to_truth,
    skill_score, soft_auroc,
)


def graded_metrics(p_hat, p, succ, trials, k):
    """The CSV's graded block, assembled from the individual scorers.

    Deliberately NOT stoch_prob_metrics.all_metrics: that bundle includes
    `sharpness`, which computes m/(m-1) and so divides by zero when the ground
    truth is exact (trials=1, as for the deterministic level). Sharpness is a
    statement about SAMPLING spread and is meaningless without sampling, so the
    fix is to not ask for it rather than to fake an m>1.

    Everything here is well defined for both truth kinds. m = mean(trials)
    feeds the debiasing, and `_noise` already returns zero for m <= 1, so the
    exact-truth case correctly subtracts the model's K-noise and no grid noise.
    """
    import numpy as _np
    m = float(_np.mean(trials))
    out = {
        "brier_raw": float(_np.mean((p_hat - p) ** 2)),
        "brier_debiased": debiased_brier(p_hat, p, k, m),
        "skill_score": skill_score(p_hat, p, k, m),
        "sAUROC": soft_auroc(p_hat, p),
        "M": m,
    }
    out.update(calibration_scores(p_hat, p, succ, trials))
    return out

DATA = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories")
LEVELS = [
    # (label, experiment dir, dataset root or None when there is no graded truth)
    ("deterministic", "adaptive_cartpole_pybullet_nonadapt",
     DATA / "deterministic/cartpole_pybullet"),
    ("sigma_015.0", "adaptive_cartpole_stoch_sigma_015.0_nonadapt",
     DATA / "stochastic/cartpole/noisy_action/lqr/sigma_015.0"),
    ("sigma_020.0", "adaptive_cartpole_stoch_sigma_020.0_nonadapt",
     DATA / "stochastic/cartpole/noisy_action/lqr/sigma_020.0"),
    ("sigma_030.0", "adaptive_cartpole_stoch_sigma_030.0_nonadapt",
     DATA / "stochastic/cartpole/noisy_action/lqr/sigma_030.0"),
    ("sigma_040.0", "adaptive_cartpole_stoch_sigma_040.0_nonadapt",
     DATA / "stochastic/cartpole/noisy_action/lqr/sigma_040.0"),
]
ARMS = ["fm", "fm_outcome", "mlp"]
SIZES = [1000, 2000, 4000, 6000, 8000]

FIELDS = ["level", "arm", "train_size", "n_eval", "binary_outcomes", "truth",
          "KL", "sAUROC", "brier_debiased", "skill_score", "brier_raw",
          "f1", "accuracy", "precision", "recall", "auc", "brier", "log_score",
          "n_invalid", "K", "M"]


def deterministic_truth(dataset_root: Path):
    """Exact p_success in {0,1} from the deterministic eval file's label column.

    state_dim=4 is load-bearing: the file is 9-column (4 start + 4 end + label)
    and without it the loader would read it as 2 start + 2 end + label and
    silently truncate cartpole to 2-D.
    """
    from adaptive_roa.adaptive_v2.eval.full_roa import load_eval_states
    X, _end, y = load_eval_states(str(dataset_root / "eval_states.txt"), state_dim=4)
    p = (np.asarray(y) == 1).astype(np.float64)      # {-1,1} -> {0,1}
    return np.asarray(X, dtype=np.float64), p, p.copy(), np.ones_like(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="/common/users/shared/pracsys/adaptive_roa_experiments")
    ap.add_argument("--out", default="docs/cartpole_nonadapt.csv")
    args = ap.parse_args()

    gt_cache: dict[str, tuple] = {}
    rows, n_graded, n_missing = [], 0, 0

    for label, expdir, root in LEVELS:
        for arm in ARMS:
            for n in SIZES:
                ep = (Path(args.exp_dir) / expdir / "outputs" / arm /
                      f"train_{n}_seed_42_d2_0.0_epochs_1" / "epoch_000")
                npz, art = ep / "full_roa_per_point.npz", ep / "artifacts_v2.json"
                if not (npz.exists() and art.exists()):
                    n_missing += 1
                    continue

                d = np.load(str(npz))
                em = json.loads(art.read_text()).get("eval_metrics", {}) or {}
                ld = em.get("lambda_delta") or {}
                tf = em.get("threshold_free") or {}
                p_hat = np.asarray(d["p_success"], dtype=np.float64)

                row = {f: "" for f in FIELDS}
                row.update(
                    level=label, arm=arm, train_size=n, n_eval=len(p_hat),
                    binary_outcomes=bool(np.max(d["p_invalid"]) == 0),
                    f1=ld.get("f1"), accuracy=ld.get("accuracy"),
                    precision=ld.get("precision"), recall=ld.get("recall"),
                    auc=tf.get("auc"), brier=tf.get("brier"),
                    n_invalid=ld.get("n_invalid"), K=em.get("num_mc_samples"),
                )

                if root is not None:
                    key = str(root)
                    if key not in gt_cache:
                        gt_cache[key] = (deterministic_truth(root) if label == "deterministic"
                                         else load_ground_truth(root))
                    starts, p_true, succ, trials = gt_cache[key]
                    row["truth"] = "exact" if label == "deterministic" else "mc"
                    try:
                        idx = match_to_truth(np.asarray(d["start_states"], dtype=np.float64), starts)
                    except ValueError as e:
                        print(f"  [{label}/{arm}/n={n}] grid mismatch, graded metrics skipped: {e}")
                        rows.append(row)
                        continue
                    m = graded_metrics(p_hat, p_true[idx], succ[idx], trials[idx],
                                       k=em.get("num_mc_samples"))
                    row.update(KL=m["KL"], sAUROC=m["sAUROC"],
                               brier_debiased=m["brier_debiased"],
                               skill_score=m["skill_score"], brier_raw=m["brier_raw"],
                               log_score=m["log_score"], M=m["M"])
                    n_graded += 1
                rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}  ({len(rows)} rows; {n_graded} with graded KL/sAUROC, "
          f"{len(rows)-n_graded} without; {n_missing} cells missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

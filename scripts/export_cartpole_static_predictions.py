#!/usr/bin/env python3
"""Export static CartPole predictions and primary probability metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from stoch_prob_metrics import calibration_scores, soft_auroc


def stochastic_truth(dataset_root: Path, states: np.ndarray):
    with np.load(dataset_root / "eval_success_prob.npz") as truth:
        starts = truth["starts"].astype(np.float64)
        distance, index = cKDTree(starts).query(states.astype(np.float64), k=1)
        if float(distance.max()) > 1e-3:
            raise ValueError(
                f"evaluation states do not match stochastic grid: max distance={distance.max()}"
            )
        return (
            truth["p_success"].astype(np.float64)[index],
            truth["successes"].astype(np.float64)[index],
            truth["trials"].astype(np.float64)[index],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    per_point = args.run_dir / "epoch_000" / "full_roa_per_point.npz"
    if not per_point.is_file():
        raise FileNotFoundError(per_point)
    with np.load(per_point) as result:
        states = result["start_states"]
        p_pred = result["p_success"].astype(np.float64)
        p_failure = result["p_failure"].astype(np.float64)
        p_invalid = result["p_invalid"].astype(np.float64)
        true_labels = result["true_labels"].astype(np.int64)

    if args.regime == "deterministic":
        p_true = (true_labels == 1).astype(np.float64)
        successes = p_true.copy()
        trials = np.ones_like(p_true)
    else:
        p_true, successes, trials = stochastic_truth(args.dataset_root, states)

    metric = calibration_scores(p_pred, p_true, successes, trials)
    metric.update(
        sAUROC=float(soft_auroc(p_pred, p_true)),
        n_points=int(len(states)),
        regime=args.regime,
        method=args.method,
        budget=args.budget,
        seed=args.seed,
    )

    np.savez_compressed(
        args.run_dir / "eval_predictions.npz",
        start_states=states,
        p_success_pred=p_pred,
        p_success_true=p_true,
        p_failure_pred=p_failure,
        p_invalid_pred=p_invalid,
        successes=successes,
        trials=trials,
        true_labels=true_labels,
        regime=np.array(args.regime),
        method=np.array(args.method),
        budget=np.int64(args.budget),
        seed=np.int64(args.seed),
        binary_outcomes=np.bool_(True),
    )
    (args.run_dir / "probability_metrics.json").write_text(
        json.dumps(metric, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"exported {len(states)} probabilities: KL={metric['KL']:.6f}, "
        f"sAUROC={metric['sAUROC']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

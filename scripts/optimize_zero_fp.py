#!/usr/bin/env python3
"""
Post-hoc threshold optimization: maximize TP subject to FP=0.

For each experiment at a specified epoch, grid-searches over (lambda, delta)
to find the thresholds that give the largest certifiable safe region (max TP)
with zero false positives. Relaxes the FP budget if no solution achieves FP=0.

Loads cached predictions (full_roa_per_point.npz or mc_cache) — no GPU needed.

Usage:
    python scripts/optimize_zero_fp.py
    python scripts/optimize_zero_fp.py --output_dir docs/zero_fp_results
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from adaptive_roa.adaptive_v2.eval.full_roa import (
    _classification_metrics_from_predictions,
    _conservative_metrics,
    _predict_lambda_delta,
)
from adaptive_roa.adaptive.data_source import load_eval_states


EXP = Path("/common/users/shared/pracsys/adaptive_roa_experiments")
LOCAL = Path("/common/home/st1122/Projects/adaptive_roa/outputs")
DATA = Path("/common/users/shared/pracsys/genMoPlan")


@dataclass
class ExperimentSpec:
    system: str
    method: str
    exp_dir: Path
    target_epoch: int
    decision_rule: str
    data_source: str        # "per_point" or "mc_cache"
    test_set_file: str = ""


EXPERIMENTS = [
    # ── Pendulum (target: 500 trajs) ─────────────────────────────────────
    ExperimentSpec("pendulum", "fm_ranked",
        EXP / "dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-20_01-55-55",
        8, "one_sided", "mc_cache",
        str(DATA / "lyapunov_data_trajectories/pendulum_lqr_50k/test_set.txt")),
    ExperimentSpec("pendulum", "fm_direct",
        EXP / "dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-20_01-56-07",
        8, "one_sided", "mc_cache",
        str(DATA / "lyapunov_data_trajectories/pendulum_lqr_50k/test_set.txt")),
    ExperimentSpec("pendulum", "fm_nonadapt",
        EXP / "dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_0.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-28_13-15-25",
        9, "one_sided", "mc_cache",
        str(DATA / "lyapunov_data_trajectories/pendulum_lqr_50k/test_set.txt")),
    ExperimentSpec("pendulum", "clf_adaptive",
        EXP / "dhruv/adaptive_classification/pendulum/adaptive",
        9, "one_sided", "per_point"),
    ExperimentSpec("pendulum", "clf_nonadapt",
        EXP / "dhruv/adaptive_classification/pendulum/random",
        9, "one_sided", "per_point"),
    ExperimentSpec("pendulum", "partx",
        EXP / "adaptive_pendulum_dhruv/outputs/training_index_0_warm_start_False_adapt_iter_10/2026-07-09_17-37-51",
        9, "one_sided", "per_point"),

    # ── CartPole (target: 1000 trajs) ────────────────────────────────────
    ExperimentSpec("cartpole", "fm_ranked",
        EXP / "dhruv/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked_w_0.9/2026-02-18_19-22-14",
        14, "two_sided", "mc_cache",
        str(DATA / "data_trajectories/deterministic/cartpole_pybullet/test_set.txt")),
    ExperimentSpec("cartpole", "fm_direct",
        EXP / "dhruv/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9/2026-02-18_14-26-32",
        14, "two_sided", "mc_cache",
        str(DATA / "data_trajectories/deterministic/cartpole_pybullet/test_set.txt")),
    ExperimentSpec("cartpole", "fm_nonadapt",
        EXP / "dhruv/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked_w_0.9/2026-02-18_14-26-34",
        14, "two_sided", "mc_cache",
        str(DATA / "data_trajectories/deterministic/cartpole_pybullet/test_set.txt")),
    ExperimentSpec("cartpole", "clf_adaptive",
        EXP / "dhruv/adaptive_classification/cartpole/adaptive",
        7, "two_sided", "per_point"),
    ExperimentSpec("cartpole", "clf_nonadapt",
        EXP / "dhruv/adaptive_classification/cartpole/random",
        7, "two_sided", "per_point"),
    ExperimentSpec("cartpole", "partx",
        EXP / "adaptive_cartpole_pybullet/outputs/training_index_0_warm_start_False_manifold_False_adapt_iter_15/2026-07-09_17-37-51",
        14, "two_sided", "per_point"),

    # ── Quad2D (target: 12000 trajs) ─────────────────────────────────────
    ExperimentSpec("quad2d", "fm_ranked",
        EXP / "adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_ranked/newcrit_20260708",
        9, "two_sided", "per_point"),
    ExperimentSpec("quad2d", "fm_direct",
        EXP / "adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/newcrit_20260708",
        9, "two_sided", "per_point"),
    ExperimentSpec("quad2d", "fm_nonadapt",
        EXP / "adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/newcrit_20260708",
        9, "two_sided", "per_point"),
    ExperimentSpec("quad2d", "clf_adaptive",
        EXP / "dhruv/adaptive_classification/quad2d/adaptive",
        9, "two_sided", "per_point"),
    ExperimentSpec("quad2d", "clf_nonadapt",
        EXP / "dhruv/adaptive_classification/quad2d/random",
        9, "two_sided", "per_point"),
    ExperimentSpec("quad2d", "partx",
        EXP / "adaptive_quadrotor2d/outputs/training_index_0_warm_start_False_manifold_False_adapt_iter_10/2026-07-09_23-13-21",
        9, "two_sided", "per_point"),

    # ── Quad3D (target: 25000 trajs) ─────────────────────────────────────
    ExperimentSpec("quad3d", "fm_ranked",
        EXP / "adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/newcrit_20260708",
        15, "two_sided", "per_point"),
    ExperimentSpec("quad3d", "fm_direct",
        EXP / "adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/newcrit_20260708",
        15, "two_sided", "per_point"),
    ExperimentSpec("quad3d", "fm_nonadapt",
        EXP / "adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/newcrit_20260708",
        14, "two_sided", "per_point"),
    ExperimentSpec("quad3d", "clf_adaptive",
        LOCAL / "clf_quad3d_match_fm/adaptive",
        14, "two_sided", "per_point"),
    ExperimentSpec("quad3d", "clf_nonadapt",
        LOCAL / "clf_quad3d_match_fm_slurm/random",
        14, "two_sided", "per_point"),
    ExperimentSpec("quad3d", "partx",
        EXP / "adaptive_quadrotor3d/outputs/training_index_0_warm_start_False_manifold_False_adapt_iter_15/2026-07-10_09-10-18",
        14, "two_sided", "per_point"),
]


def load_predictions_per_point(epoch_dir: Path):
    """Load from full_roa_per_point.npz (CLF, part-X, newcrit FM)."""
    npz = np.load(str(epoch_dir / "full_roa_per_point.npz"))
    return (
        npz["p_success"],
        npz["p_failure"],
        npz["p_invalid"],
        npz["true_labels"],
    )


def load_predictions_mc_cache(exp_dir: Path, epoch_num: int, test_set_file: str):
    """Load from mc_cache (Dhruv FM pendulum/cartpole)."""
    cache_path = exp_dir / "mc_cache" / f"epoch_{epoch_num:03d}_test.npz"
    mc = np.load(str(cache_path))
    mc_labels = mc["mc_labels"]  # [N, K] int8: {-1, 0, 1}
    K = int(mc["num_mc_samples"])
    p_success = (mc_labels == 1).sum(axis=1).astype(np.float64) / K
    p_failure = (mc_labels == -1).sum(axis=1).astype(np.float64) / K
    p_invalid = (mc_labels == 0).sum(axis=1).astype(np.float64) / K

    _, _, y_true = load_eval_states(test_set_file)
    assert len(y_true) == len(p_success), (
        f"Test set size mismatch: {len(y_true)} vs {len(p_success)}"
    )
    return p_success, p_failure, p_invalid, y_true


def load_original_thresholds(epoch_dir: Path):
    """Load original (lambda, delta) from artifacts_v2.json or results.json."""
    artifacts_path = epoch_dir / "artifacts_v2.json"
    if artifacts_path.exists():
        with open(artifacts_path) as f:
            a = json.load(f)
        ts = a.get("threshold_state", {})
        lam = ts.get("lambda_star")
        delta = ts.get("delta_star")
        if lam is not None and delta is not None:
            return float(lam), float(delta)

    results_path = epoch_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            r = json.load(f)
        lam = r.get("lambda_star")
        delta = r.get("delta_star")
        if lam is not None and delta is not None:
            return float(lam), float(delta)

    raise FileNotFoundError(f"No threshold state found in {epoch_dir}")


def load_train_trajectories(epoch_dir: Path) -> int:
    """Load train_trajectories count from results.json."""
    results_path = epoch_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            r = json.load(f)
        return r.get("train_trajectories", -1)
    return -1


def optimize_zero_fp(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    p_invalid: np.ndarray,
    y_true: np.ndarray,
    decision_rule: str,
    n_lambda_steps: int = 200,
    n_delta_steps: int = 100,
    lambda_range: tuple = (0.05, 0.95),
    delta_range: tuple = (0.005, 0.49),
) -> dict:
    """
    Grid search: maximize TP subject to FP=0.

    If no (lambda, delta) achieves FP=0, relaxes to FP<=1, FP<=2, etc.
    Returns the best result with both standard and conservative metrics.
    """
    lambda_vals = np.linspace(lambda_range[0], lambda_range[1], n_lambda_steps)
    delta_vals = np.linspace(delta_range[0], delta_range[1], n_delta_steps)

    candidates = []

    for lam in lambda_vals:
        for d in delta_vals:
            pred, _ = _predict_lambda_delta(
                p_success, p_failure, p_invalid,
                lambda_star=float(lam), delta=float(d),
                decision_rule=decision_rule,
                invalid_threshold=None,
            )
            metrics = _classification_metrics_from_predictions(pred, y_true)
            cons = _conservative_metrics(pred, y_true)
            candidates.append({
                "lambda_star": float(lam),
                "delta": float(d),
                "fp": metrics["fp"],
                "tp": metrics["tp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "accuracy": metrics["accuracy"],
                "separatrix_pct": metrics["separatrix_pct"],
                "n_confident": metrics["n_confident"],
                "n_uncertain": metrics["n_uncertain"],
                "n_invalid": metrics["n_invalid"],
                "conservative_f1": cons["f1"],
                "conservative_precision": cons["precision"],
                "conservative_recall": cons["recall"],
                "conservative_fp": cons["fp"],
                "conservative_tp": cons["tp"],
            })

    # Require TP > 0 to avoid the trivial all-uncertain solution
    nontrivial = [c for c in candidates if c["tp"] > 0]
    if not nontrivial:
        # Everything is degenerate — fall back to the candidate with max TP
        best = max(candidates, key=lambda c: c["tp"])
        best["fp_budget"] = best["fp"]
        best["fp_zero_achievable"] = False
        return {"best": best, "pareto_front": [], "grid_size": n_lambda_steps * n_delta_steps}

    # Among non-trivial: minimize FP, then maximize TP, then minimize separatrix
    min_fp = min(c["fp"] for c in nontrivial)
    best_candidates = [c for c in nontrivial if c["fp"] == min_fp]
    best = max(best_candidates, key=lambda c: (c["tp"], -c["separatrix_pct"]))

    best["fp_budget"] = min_fp
    best["fp_zero_achievable"] = (min_fp == 0)

    # Pareto frontier: for each FP level (non-trivial only), max TP
    fp_levels = sorted(set(c["fp"] for c in nontrivial))
    pareto = []
    for fp in fp_levels[:8]:
        pool = [c for c in nontrivial if c["fp"] == fp]
        winner = max(pool, key=lambda c: (c["tp"], -c["separatrix_pct"]))
        pareto.append({
            "fp": fp,
            "tp": winner["tp"],
            "f1": winner["f1"],
            "separatrix_pct": winner["separatrix_pct"],
            "lambda_star": winner["lambda_star"],
            "delta": winner["delta"],
            "conservative_f1": winner["conservative_f1"],
        })

    return {
        "best": best,
        "pareto_front": pareto,
        "grid_size": n_lambda_steps * n_delta_steps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Threshold optimization: maximize TP subject to FP=0"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("docs/zero_fp_results"),
        help="Output directory for results (default: docs/zero_fp_results)",
    )
    parser.add_argument(
        "--system", type=str, default=None,
        help="Run only for a specific system (pendulum/cartpole/quad2d/quad3d)",
    )
    parser.add_argument(
        "--method", type=str, default=None,
        help="Run only for a specific method (fm_ranked/fm_direct/...)",
    )
    parser.add_argument(
        "--n_lambda", type=int, default=200,
        help="Number of lambda grid points (default: 200)",
    )
    parser.add_argument(
        "--n_delta", type=int, default=100,
        help="Number of delta grid points (default: 100)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.system:
        experiments = [e for e in experiments if e.system == args.system]
    if args.method:
        experiments = [e for e in experiments if e.method == args.method]

    if not experiments:
        print("No experiments matched filters.")
        return 1

    print("=" * 90)
    print("THRESHOLD OPTIMIZATION: Maximize TP subject to FP=0")
    print("=" * 90)
    print(f"Grid: {args.n_lambda} lambda × {args.n_delta} delta = {args.n_lambda * args.n_delta} points")
    print(f"Output: {output_dir}")
    print(f"Experiments: {len(experiments)}")
    print()

    all_results = {}

    for spec in experiments:
        epoch_dir = spec.exp_dir / f"epoch_{spec.target_epoch:03d}"
        key = f"{spec.system}/{spec.method}"

        print(f"{'─' * 90}")
        print(f"  {key}  (epoch {spec.target_epoch})")
        print(f"  {epoch_dir}")

        if not epoch_dir.exists():
            print(f"  SKIP: epoch directory not found")
            continue

        # Load trajectory count
        n_trajs = load_train_trajectories(epoch_dir)
        print(f"  Trajectories: {n_trajs}")

        # Load predictions
        try:
            if spec.data_source == "per_point":
                p_s, p_f, p_i, y = load_predictions_per_point(epoch_dir)
            else:
                p_s, p_f, p_i, y = load_predictions_mc_cache(
                    spec.exp_dir, spec.target_epoch, spec.test_set_file
                )
        except Exception as e:
            print(f"  SKIP: failed to load predictions: {e}")
            continue

        n_total = len(y)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == -1))
        print(f"  Test set: {n_total} ({n_pos} success, {n_neg} failure)")

        # Load original thresholds and compute original metrics
        try:
            orig_lam, orig_delta = load_original_thresholds(epoch_dir)
        except FileNotFoundError:
            orig_lam, orig_delta = 0.5, 0.05

        orig_pred, _ = _predict_lambda_delta(
            p_s, p_f, p_i, orig_lam, orig_delta,
            decision_rule=spec.decision_rule, invalid_threshold=None,
        )
        orig_metrics = _classification_metrics_from_predictions(orig_pred, y)
        orig_cons = _conservative_metrics(orig_pred, y)

        print(f"  Original: λ={orig_lam:.4f} δ={orig_delta:.4f} → "
              f"TP={orig_metrics['tp']} FP={orig_metrics['fp']} "
              f"F1={orig_metrics['f1']:.4f} Sep={orig_metrics['separatrix_pct']:.1%}")
        print(f"  Original conservative: F1={orig_cons['f1']:.4f} "
              f"FP={orig_cons['fp']} TP={orig_cons['tp']}")

        # Run optimization
        result = optimize_zero_fp(
            p_s, p_f, p_i, y,
            decision_rule=spec.decision_rule,
            n_lambda_steps=args.n_lambda,
            n_delta_steps=args.n_delta,
        )

        best = result["best"]
        print(f"  Optimized: λ={best['lambda_star']:.4f} δ={best['delta']:.4f} → "
              f"TP={best['tp']} FP={best['fp']} "
              f"F1={best['f1']:.4f} Sep={best['separatrix_pct']:.1%}")
        print(f"  Optimized conservative: F1={best['conservative_f1']:.4f} "
              f"FP={best['conservative_fp']} TP={best['conservative_tp']}")

        if not best["fp_zero_achievable"]:
            print(f"  ⚠ FP=0 not achievable; best FP={best['fp_budget']}")

        # Pareto summary
        print(f"  Pareto (FP→TP): ", end="")
        for p in result["pareto_front"][:4]:
            print(f"FP={p['fp']}→TP={p['tp']}  ", end="")
        print()

        all_results[key] = {
            "system": spec.system,
            "method": spec.method,
            "epoch": spec.target_epoch,
            "train_trajectories": n_trajs,
            "decision_rule": spec.decision_rule,
            "test_set_size": n_total,
            "test_n_pos": n_pos,
            "test_n_neg": n_neg,
            "original": {
                "lambda_star": orig_lam,
                "delta": orig_delta,
                "fp": orig_metrics["fp"],
                "tp": orig_metrics["tp"],
                "fn": orig_metrics["fn"],
                "tn": orig_metrics["tn"],
                "f1": orig_metrics["f1"],
                "precision": orig_metrics["precision"],
                "recall": orig_metrics["recall"],
                "separatrix_pct": orig_metrics["separatrix_pct"],
                "conservative_f1": orig_cons["f1"],
                "conservative_fp": orig_cons["fp"],
                "conservative_tp": orig_cons["tp"],
            },
            "optimized": best,
            "pareto_front": result["pareto_front"],
        }

    # Save results
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "grid": {"n_lambda": args.n_lambda, "n_delta": args.n_delta},
        "objective": "maximize TP subject to FP=0 (relax if impossible)",
        "experiments": all_results,
    }

    with open(output_dir / "zero_fp_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print()
    print("=" * 130)
    print(f"{'System':<10} {'Method':<15} {'Trajs':>6} │ {'Orig FP':>7} {'Orig TP':>7} {'Orig F1':>7} "
          f"│ {'Opt FP':>6} {'Opt TP':>6} {'Opt F1':>6} {'Opt Sep%':>8} │ {'Cons F1':>7} {'Cons FP':>7}")
    print("─" * 130)

    for key, r in all_results.items():
        o = r["original"]
        b = r["optimized"]
        print(f"{r['system']:<10} {r['method']:<15} {r['train_trajectories']:>6} │ "
              f"{o['fp']:>7} {o['tp']:>7} {o['f1']:>7.4f} │ "
              f"{b['fp']:>6} {b['tp']:>6} {b['f1']:>6.4f} {b['separatrix_pct']:>7.1%} │ "
              f"{b['conservative_f1']:>7.4f} {b['conservative_fp']:>7}")

    print("─" * 130)
    print(f"\nResults saved to: {output_dir / 'zero_fp_results.json'}")

    # Also save a CSV for easy import
    csv_path = output_dir / "zero_fp_summary.csv"
    with open(csv_path, "w") as f:
        f.write("system,method,trajs,orig_lambda,orig_delta,orig_fp,orig_tp,orig_f1,orig_sep_pct,"
                "opt_lambda,opt_delta,opt_fp,opt_tp,opt_f1,opt_sep_pct,opt_precision,opt_recall,"
                "cons_f1,cons_fp,cons_tp,fp_zero_achievable\n")
        for key, r in all_results.items():
            o = r["original"]
            b = r["optimized"]
            f.write(f"{r['system']},{r['method']},{r['train_trajectories']},"
                    f"{o['lambda_star']:.6f},{o['delta']:.6f},{o['fp']},{o['tp']},{o['f1']:.6f},{o['separatrix_pct']:.6f},"
                    f"{b['lambda_star']:.6f},{b['delta']:.6f},{b['fp']},{b['tp']},{b['f1']:.6f},{b['separatrix_pct']:.6f},"
                    f"{b['precision']:.6f},{b['recall']:.6f},"
                    f"{b['conservative_f1']:.6f},{b['conservative_fp']},{b['conservative_tp']},"
                    f"{b['fp_zero_achievable']}\n")

    print(f"CSV saved to: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

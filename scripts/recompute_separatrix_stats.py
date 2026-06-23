#!/usr/bin/env python3
"""Recompute mc_sample_errors with separatrix region from existing MC cache + artifacts.

This script reconstructs the qhat prediction masks from cached MC data and
calibration parameters, then computes error statistics for the separatrix
(uncertain | invalid) region. It outputs all 7 regions (full, certain,
certain_success, certain_failure, uncertain, invalid, separatrix) with all
16 MC sample error stats across all epochs up to the cutoff.

Output format: one table per (run, region) with trajectory count on rows
and statistics on columns.

Usage:
    python scripts/recompute_separatrix_stats.py
    python scripts/recompute_separatrix_stats.py --output docs/mc_sample_errors_report.md
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_roa.adaptive.data_source import load_eval_states
from adaptive_roa.adaptive_v2.eval.full_roa import (
    _compute_mc_sample_error_stats,
    _predict_qhat_prediction_sets,
    _predict_lambda_delta,
)
from adaptive_roa.adaptive_v2.eval.mc_cache import load_mc_cache


BASE = "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv"
DATA_DIR = "/common/users/shared/pracsys/genMoPlan/data_trajectories"

# Trajectory count formulas: initial + samples_per_epoch * epoch_num
TRAJ_FORMULAS = {
    "pendulum": (100, 50),       # 100 + 50*epoch
    "cartpole": (300, 50),       # 300 + 50*epoch
    "quadrotor2d": (3000, 1000), # 3000 + 1000*epoch
    "quadrotor3d": (10000, 1000),# 10000 + 1000*epoch
}

RUNS = [
    {
        "name": "Pendulum Non-adaptive",
        "system": "pendulum",
        "path": f"{BASE}/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_0.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked/2026-02-20_01-56-20",
        "eval": "radius_0.075_alpha_0.1_mc_20_batch_100000",
        "max_epoch": 8,
        "test_set": f"{DATA_DIR}/pendulum_lqr_50k/test_set.txt",
    },
    {
        "name": "Pendulum Adaptive",
        "system": "pendulum",
        "path": f"{BASE}/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-20_01-56-07",
        "eval": "radius_0.075_alpha_0.1_mc_20_batch_100000",
        "max_epoch": 8,
        "test_set": f"{DATA_DIR}/pendulum_lqr_50k/test_set.txt",
    },
    {
        "name": "CartPole Non-adaptive",
        "system": "cartpole",
        "path": f"{BASE}/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_ranked_w_0.9/2026-02-18_14-26-34",
        "eval": "radius_0.2_alpha_0.1_mc_20_batch_100000",
        "max_epoch": 14,
        "test_set": f"{DATA_DIR}/cartpole_pybullet/test_set.txt",
    },
    {
        "name": "CartPole Adaptive",
        "system": "cartpole",
        "path": f"{BASE}/adaptive_cartpole_pybullet/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct_w_0.9/2026-02-18_14-26-32",
        "eval": "radius_0.2_alpha_0.1_mc_20_batch_100000",
        "max_epoch": 14,
        "test_set": f"{DATA_DIR}/cartpole_pybullet/test_set.txt",
    },
    {
        "name": "Quadrotor2D Non-adaptive",
        "system": "quadrotor2d",
        "path": f"{BASE}/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-17-55",
        "eval": "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000",
        "max_epoch": 9,
        "test_set": f"{DATA_DIR}/quadrotor2D_rl/test_set.txt",
    },
    {
        "name": "Quadrotor2D Adaptive",
        "system": "quadrotor2d",
        "path": f"{BASE}/adaptive_quadrotor2d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_10_alpha_0.1_sampling_mode_direct/2026-02-28_10-05-32",
        "eval": "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000",
        "max_epoch": 9,
        "test_set": f"{DATA_DIR}/quadrotor2D_rl/test_set.txt",
    },
    {
        "name": "Quadrotor3D Non-adaptive",
        "system": "quadrotor3d",
        "path": f"{BASE}/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_0.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_15_alpha_0.1_sampling_mode_direct/2026-02-28_11-49-54",
        "eval": "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000",
        "max_epoch": 14,
        "test_set": f"{DATA_DIR}/quadrotor3D_lqr/test_set.txt",
    },
    {
        "name": "Quadrotor3D Adaptive",
        "system": "quadrotor3d",
        "path": f"{BASE}/adaptive_quadrotor3d/outputs/training_index_0_d2_ratio_1.0_warm_start_False_manifold_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-26_12-02-36",
        "eval": "radius_0.3_alpha_0.1_mc_10_batch_100000_calk_1000",
        "max_epoch": 15,
        "test_set": f"{DATA_DIR}/quadrotor3D_lqr/test_set.txt",
    },
]

REGIONS = ["full", "certain", "certain_success", "certain_failure", "uncertain", "invalid", "separatrix"]
STATS = [
    "mean_of_means", "median_of_means", "p90_of_means", "p99_of_means",
    "mean_of_medians", "median_of_medians", "p90_of_medians", "p99_of_medians",
    "mean_of_p90s", "median_of_p90s", "p90_of_p90s", "p99_of_p90s",
    "mean_of_p99s", "median_of_p99s", "p90_of_p99s", "p99_of_p99s",
]


def recompute_epoch(run: dict, epoch_num: int, end_states_all: np.ndarray, y_all: np.ndarray) -> dict[str, dict[str, Any]]:
    """Recompute all mc_sample_error stats for a single epoch, including separatrix."""
    path = run["path"]
    eval_name = run["eval"]
    epoch_str = f"epoch_{epoch_num:03d}"

    # Load MC cache
    cache_path = Path(path) / "mc_cache" / f"epoch_{epoch_num:03d}_test.npz"
    cache = load_mc_cache(cache_path)

    # Compute mc_errors: norm(mc_endpoints - true_endpoints) per sample
    mc_errors = np.linalg.norm(
        cache.mc_endpoints - end_states_all[:, np.newaxis, :],
        axis=2,
    ).astype(np.float32)

    # Load calibration params from artifacts
    art_path = Path(path) / "evaluations" / eval_name / epoch_str / "artifacts_v2.json"
    with open(art_path) as f:
        artifacts = json.load(f)

    em = artifacts["eval_metrics"]
    lambda_star = em["lambda_star"]
    delta = em["delta"]
    q_hat = em["q_hat"]

    # Load eval config for decision rule
    eval_config_path = Path(path) / "evaluations" / eval_name / "eval_config.json"
    with open(eval_config_path) as f:
        eval_config = json.load(f)
    decision_rule = eval_config.get("decision_rule", "one_sided")
    invalid_threshold = eval_config.get("parameters", {}).get("invalid_threshold")

    # Compute p_success, p_failure, p_invalid from mc_labels
    num_mc_samples = cache.num_mc_samples
    mc_labels = cache.mc_labels
    p_success = (mc_labels == 1).sum(axis=1) / num_mc_samples
    p_failure = (mc_labels == -1).sum(axis=1) / num_mc_samples
    p_invalid = (mc_labels == 0).sum(axis=1) / num_mc_samples

    # Reconstruct qhat prediction labels
    if q_hat is not None:
        pred_qhat, _ = _predict_qhat_prediction_sets(
            p_success, p_failure, p_invalid, y_all,
            lambda_star=float(lambda_star),
            delta=float(delta),
            q_hat=float(q_hat),
            decision_rule=decision_rule,
            invalid_threshold=invalid_threshold,
        )
    else:
        pred_qhat, _ = _predict_lambda_delta(
            p_success, p_failure, p_invalid,
            lambda_star=float(lambda_star),
            delta=float(delta),
            decision_rule=decision_rule,
            invalid_threshold=invalid_threshold,
        )

    # Build masks
    q_mask_invalid = pred_qhat == -2
    q_mask_uncertain = pred_qhat == -1
    q_mask_certain_success = pred_qhat == 1
    q_mask_certain_failure = pred_qhat == 0
    q_mask_certain = q_mask_certain_success | q_mask_certain_failure
    q_mask_separatrix = q_mask_uncertain | q_mask_invalid

    masks = {
        "full": None,
        "certain": q_mask_certain,
        "certain_success": q_mask_certain_success,
        "certain_failure": q_mask_certain_failure,
        "uncertain": q_mask_uncertain,
        "invalid": q_mask_invalid,
        "separatrix": q_mask_separatrix,
    }

    results = {}
    for region, mask in masks.items():
        results[region] = _compute_mc_sample_error_stats(mc_errors, mask)

    return results


def format_markdown(all_run_results: list[tuple[dict, list[tuple[int, int, dict]]]]) -> str:
    """Format results as markdown with one table per (run, region).

    Each table has trajectory count on the y-axis and statistics on the x-axis.
    """
    lines = ["# MC Sample Errors (qhat regions) — All Runs, All Epochs", ""]
    lines.append("Each table shows statistics (columns) across trajectory counts (rows).")
    lines.append("Separatrix = uncertain + invalid combined.")
    lines.append("")

    # Abbreviated stat names for table headers
    stat_short = {
        "mean_of_means": "mean_means",
        "median_of_means": "med_means",
        "p90_of_means": "p90_means",
        "p99_of_means": "p99_means",
        "mean_of_medians": "mean_meds",
        "median_of_medians": "med_meds",
        "p90_of_medians": "p90_meds",
        "p99_of_medians": "p99_meds",
        "mean_of_p90s": "mean_p90s",
        "median_of_p90s": "med_p90s",
        "p90_of_p90s": "p90_p90s",
        "p99_of_p90s": "p99_p90s",
        "mean_of_p99s": "mean_p99s",
        "median_of_p99s": "med_p99s",
        "p90_of_p99s": "p90_p99s",
        "p99_of_p99s": "p99_p99s",
    }

    for run, epoch_results in all_run_results:
        lines.append(f"## {run['name']}")
        lines.append("")

        for region in REGIONS:
            lines.append(f"### {region}")
            lines.append("")

            # Header
            header = "| n_traj | n_pts | " + " | ".join(stat_short[s] for s in STATS) + " |"
            sep_line = "|--------|-------|" + "|".join("-------" for _ in STATS) + "|"
            lines.append(header)
            lines.append(sep_line)

            for epoch_num, n_traj, results in epoch_results:
                r = results.get(region, {})
                n = r.get("n_points", 0)
                if n == 0:
                    vals = " | ".join("—" for _ in STATS)
                else:
                    vals = " | ".join(f"{r.get(s, 0):.4f}" for s in STATS)
                lines.append(f"| {n_traj} | {n} | {vals} |")

            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Recompute MC sample error stats with separatrix region")
    parser.add_argument("--output", type=str, default="docs/mc_sample_errors_report.md",
                        help="Output markdown file path")
    parser.add_argument("--json-output", type=str, default=None,
                        help="Also save raw results as JSON")
    args = parser.parse_args()

    all_run_results = []  # list of (run, [(epoch_num, n_traj, results), ...])

    for run in RUNS:
        print(f"\nProcessing: {run['name']}...")
        init_traj, step_traj = TRAJ_FORMULAS[run["system"]]
        max_epoch = run["max_epoch"]

        # Load test set once per run
        X_all, end_states_all, y_all = load_eval_states(run["test_set"])
        print(f"  Test set: {len(y_all)} points")

        epoch_results = []
        for epoch_num in range(max_epoch + 1):
            n_traj = init_traj + step_traj * epoch_num
            epoch_str = f"epoch_{epoch_num:03d}"

            # Check that cache and artifacts exist
            cache_path = Path(run["path"]) / "mc_cache" / f"{epoch_str}_test.npz"
            art_path = Path(run["path"]) / "evaluations" / run["eval"] / epoch_str / "artifacts_v2.json"
            if not cache_path.exists() or not art_path.exists():
                print(f"  {epoch_str} ({n_traj} traj): SKIPPED (missing files)")
                continue

            try:
                results = recompute_epoch(run, epoch_num, end_states_all, y_all)
                epoch_results.append((epoch_num, n_traj, results))
                sep_n = results.get("separatrix", {}).get("n_points", 0)
                print(f"  {epoch_str} ({n_traj} traj): OK  sep_n={sep_n}")
            except Exception as e:
                print(f"  {epoch_str} ({n_traj} traj): ERROR: {e}")

        all_run_results.append((run, epoch_results))

    # Write markdown
    md = format_markdown(all_run_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)
    print(f"\nMarkdown report written to: {output_path}")

    # Write JSON
    if args.json_output:
        json_data = {}
        for run, epoch_results in all_run_results:
            run_data = []
            for epoch_num, n_traj, results in epoch_results:
                run_data.append({
                    "epoch": epoch_num,
                    "n_traj": n_traj,
                    "regions": results,
                })
            json_data[run["name"]] = run_data
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(json_data, indent=2))
        print(f"JSON data written to: {json_path}")


if __name__ == "__main__":
    main()

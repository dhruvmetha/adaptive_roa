#!/usr/bin/env python3
"""
Compile and plot metrics across adaptive ROA training epochs.

Supports both default training metrics and re-evaluations.

Usage:
    # Compile ALL: default + all re-evaluations
    python scripts/compile_adaptive_metrics.py /path/to/training/output

    # Compile only default (training) metrics
    python scripts/compile_adaptive_metrics.py /path/to/training/output --eval_dir default

    # Compile specific re-evaluation
    python scripts/compile_adaptive_metrics.py /path/to/training/output --eval_dir radius_0.25

Output structure:
    - Default metrics: {training_dir}/metrics/
    - Re-evaluation metrics: {training_dir}/evaluations/{eval_name}/metrics/
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def extract_metrics_from_epoch(results: dict[str, Any]) -> dict[str, Any]:
    """Extract all metrics from a single epoch's results.json."""
    metrics = {}

    # Basic epoch info
    metrics["epoch"] = results.get("epoch", 0)
    metrics["train_trajectories"] = results.get("train_trajectories", 0)

    # Sampling statistics
    metrics["n_d1_added"] = results.get("n_d1_added", 0)
    metrics["n_d2_added"] = results.get("n_d2_added", 0)
    metrics["n_d2_uncertain"] = results.get("n_d2_uncertain", 0)
    metrics["n_d2_invalid"] = results.get("n_d2_invalid", 0)
    metrics["n_discarded_certain"] = results.get("n_discarded_certain", 0)

    # Conformal parameters
    metrics["lambda_star"] = results.get("lambda_star", 0)
    metrics["delta_star"] = results.get("delta_star", 0)
    metrics["q_hat"] = results.get("q_hat", 0)
    metrics["q_hat_eval"] = results.get("q_hat_eval", 0)

    # Full ROA evaluation metrics
    full_roa = results.get("full_roa", {})

    # Classification metrics (lambda_delta, with fallback to old conformal_thresholds key)
    conformal = full_roa.get("lambda_delta", full_roa.get("conformal_thresholds", {}))
    metrics["f1"] = conformal.get("f1", 0)
    metrics["accuracy"] = conformal.get("accuracy", 0)
    metrics["precision"] = conformal.get("precision", 0)
    metrics["recall"] = conformal.get("recall", 0)
    metrics["specificity"] = conformal.get("specificity", 0)

    # Coverage metrics
    metrics["separatrix_pct"] = conformal.get("separatrix_pct", 0)
    metrics["invalid_pct"] = conformal.get("invalid_pct", 0)
    metrics["uncertain_pct"] = conformal.get("uncertain_pct", 0)

    # Counts
    metrics["n_confident"] = conformal.get("n_confident", 0)
    metrics["n_invalid"] = conformal.get("n_invalid", 0)
    metrics["n_uncertain"] = conformal.get("n_uncertain", 0)

    # Confusion matrix
    metrics["tp"] = conformal.get("tp", 0)
    metrics["tn"] = conformal.get("tn", 0)
    metrics["fp"] = conformal.get("fp", 0)
    metrics["fn"] = conformal.get("fn", 0)

    # Threshold-free scores (no lambda/delta operating point).
    # Default None, not 0 -- auc=0.0 is a real value meaning inverted ranking.
    threshold_free = full_roa.get("threshold_free") or {}
    for key in ("auc", "auprc", "brier", "log_score", "log_score_smoothing",
                "n_saturated", "base_rate"):
        metrics[key] = threshold_free.get(key)

    # Endpoint errors (training data)
    endpoint_error = results.get("endpoint_error", {})
    metrics["endpoint_mae_overall"] = endpoint_error.get("overall_mae", 0)
    metrics["endpoint_mae_success"] = endpoint_error.get("success_mae", 0)
    metrics["endpoint_mae_failure"] = endpoint_error.get("failure_mae", 0)

    # Per-component endpoint errors
    per_component_mae = endpoint_error.get("per_component_mae", [0, 0, 0, 0])
    if len(per_component_mae) >= 4:
        metrics["endpoint_mae_cart_pos"] = per_component_mae[0]
        metrics["endpoint_mae_pole_angle"] = per_component_mae[1]
        metrics["endpoint_mae_cart_vel"] = per_component_mae[2]
        metrics["endpoint_mae_ang_vel"] = per_component_mae[3]
    else:
        metrics["endpoint_mae_cart_pos"] = 0
        metrics["endpoint_mae_pole_angle"] = 0
        metrics["endpoint_mae_cart_vel"] = 0
        metrics["endpoint_mae_ang_vel"] = 0

    # ROA endpoint errors by region
    roa_errors = full_roa.get("endpoint_errors", {})
    metrics["roa_error_full"] = roa_errors.get("full", {}).get("mean", 0)
    metrics["roa_error_certain"] = roa_errors.get("certain", {}).get("mean", 0)
    metrics["roa_error_certain_success"] = roa_errors.get("certain_success", {}).get("mean", 0)
    metrics["roa_error_certain_failure"] = roa_errors.get("certain_failure", {}).get("mean", 0)
    metrics["roa_error_uncertain"] = roa_errors.get("uncertain", {}).get("mean", 0)
    metrics["roa_error_invalid"] = roa_errors.get("invalid", {}).get("mean", 0)

    return metrics


def compile_metrics(source_dir: Path) -> pd.DataFrame:
    """Compile metrics from all epochs into a DataFrame."""
    epoch_dirs = sorted(
        [d for d in source_dir.iterdir() if d.is_dir() and re.match(r"epoch_\d+", d.name)],
        key=lambda x: int(re.search(r"\d+", x.name).group()),
    )

    if not epoch_dirs:
        raise ValueError(f"No epoch directories found in {source_dir}")

    all_metrics = []
    for epoch_dir in epoch_dirs:
        results_file = epoch_dir / "results.json"
        if not results_file.exists():
            print(f"    Warning: {results_file} not found, skipping")
            continue

        try:
            with open(results_file) as f:
                results = json.load(f)
        except PermissionError:
            print(f"    Warning: Permission denied for {results_file}, skipping")
            continue
        except json.JSONDecodeError as e:
            print(f"    Warning: Failed to parse {results_file}: {e}, skipping")
            continue

        metrics = extract_metrics_from_epoch(results)
        all_metrics.append(metrics)

    return pd.DataFrame(all_metrics)


def generate_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate all plots."""
    plot_performance_metrics(df, plots_dir)
    plot_coverage_breakdown(df, plots_dir)
    plot_endpoint_errors(df, plots_dir)
    plot_training_progress(df, plots_dir)
    plot_stratified_region_errors(df, plots_dir)


def plot_performance_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot F1 and Separatrix % (2 subplots)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # F1 Score
    axes[0].plot(df["train_trajectories"], df["f1"], "b-o", linewidth=2, markersize=6)
    axes[0].set_ylabel("F1 Score", fontsize=12)
    axes[0].set_title("Classification Performance", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Separatrix %
    axes[1].plot(
        df["train_trajectories"], df["separatrix_pct"] * 100, "r-o", linewidth=2, markersize=6
    )
    axes[1].set_ylabel("Separatrix %", fontsize=12)
    axes[1].set_xlabel("Training Trajectories", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / "performance_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_coverage_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot Invalid % and Uncertain % breakdown."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["train_trajectories"],
        df["invalid_pct"] * 100,
        "r-o",
        linewidth=2,
        markersize=6,
        label="Invalid %",
    )
    ax.plot(
        df["train_trajectories"],
        df["uncertain_pct"] * 100,
        "orange",
        linestyle="-",
        marker="s",
        linewidth=2,
        markersize=6,
        label="Uncertain %",
    )
    ax.plot(
        df["train_trajectories"],
        df["separatrix_pct"] * 100,
        "purple",
        linestyle="--",
        marker="^",
        linewidth=2,
        markersize=6,
        label="Separatrix % (total)",
    )

    ax.set_xlabel("Training Trajectories", fontsize=12)
    ax.set_ylabel("Percentage", fontsize=12)
    ax.set_title("Coverage Breakdown", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(df["separatrix_pct"].max() * 100 * 1.1, 10)])

    plt.tight_layout()
    plt.savefig(output_dir / "coverage_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_endpoint_errors(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot Overall/Success/Failure MAE."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["train_trajectories"],
        df["endpoint_mae_overall"],
        "b-o",
        linewidth=2,
        markersize=6,
        label="Overall MAE",
    )
    ax.plot(
        df["train_trajectories"],
        df["endpoint_mae_success"],
        "g-s",
        linewidth=2,
        markersize=6,
        label="Success MAE",
    )
    ax.plot(
        df["train_trajectories"],
        df["endpoint_mae_failure"],
        "r-^",
        linewidth=2,
        markersize=6,
        label="Failure MAE",
    )

    ax.set_xlabel("Training Trajectories", fontsize=12)
    ax.set_ylabel("Mean Absolute Error", fontsize=12)
    ax.set_title("Endpoint Prediction Errors (Training Data)", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "endpoint_errors.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_progress(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot training trajectories added per epoch."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate trajectories added per epoch
    traj_added = df["train_trajectories"].diff().fillna(df["train_trajectories"].iloc[0])

    ax.bar(
        df["epoch"],
        traj_added,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Trajectories Added", fontsize=12)
    ax.set_title("Training Data Added Per Epoch", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "training_progress.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_stratified_region_errors(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot errors by prediction region."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each region
    ax.plot(
        df["train_trajectories"],
        df["roa_error_certain"],
        "b-o",
        linewidth=2,
        markersize=6,
        label="Certain (all)",
    )
    ax.plot(
        df["train_trajectories"],
        df["roa_error_certain_success"],
        "g-s",
        linewidth=2,
        markersize=6,
        label="Certain Success",
    )
    ax.plot(
        df["train_trajectories"],
        df["roa_error_certain_failure"],
        "c-^",
        linewidth=2,
        markersize=6,
        label="Certain Failure",
    )
    ax.plot(
        df["train_trajectories"],
        df["roa_error_uncertain"],
        "orange",
        linestyle="-",
        marker="d",
        linewidth=2,
        markersize=6,
        label="Uncertain",
    )
    ax.plot(
        df["train_trajectories"],
        df["roa_error_invalid"],
        "r-v",
        linewidth=2,
        markersize=6,
        label="Invalid",
    )

    ax.set_xlabel("Training Trajectories", fontsize=12)
    ax.set_ylabel("Mean Endpoint Error", fontsize=12)
    ax.set_title("Endpoint Errors by Prediction Region (Full ROA Evaluation)", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "stratified_region_errors.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def compile_default(training_dir: Path) -> None:
    """Compile metrics from default training results (epoch_XXX/results.json)."""
    print("  Source: epoch_XXX/results.json (original training)")

    # Check if epochs exist
    epoch_dirs = [d for d in training_dir.iterdir() if d.is_dir() and re.match(r"epoch_\d+", d.name)]
    if not epoch_dirs:
        print("  WARNING: No epoch directories found, skipping default compilation")
        return

    # Compile metrics
    df = compile_metrics(training_dir)
    print(f"  Found {len(df)} epochs")

    # Save to training_dir/metrics/ (not under evaluations/)
    metrics_dir = training_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    csv_path = metrics_dir / "metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    plots_dir = metrics_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    generate_plots(df, plots_dir)
    print(f"  Plots: {plots_dir}")


def compile_reeval(eval_dir: Path, training_dir: Path | None = None) -> None:
    """Compile metrics from a re-evaluation directory.

    Args:
        eval_dir: Path to the re-evaluation directory (e.g., evaluations/batch_61440/)
        training_dir: Path to the main training directory (to fetch train_trajectories).
                      If None, will be inferred as eval_dir.parent.parent.
    """
    # Load eval_config.json to show what params were used
    eval_config_path = eval_dir / "eval_config.json"
    if eval_config_path.exists():
        with open(eval_config_path) as f:
            eval_config = json.load(f)
        params = eval_config.get("parameters", {})
        print(f"  Parameters: {params}")

    # Check if epochs exist
    epoch_dirs = [d for d in eval_dir.iterdir() if d.is_dir() and re.match(r"epoch_\d+", d.name)]
    if not epoch_dirs:
        print("  WARNING: No epoch directories found, skipping")
        return

    # Compile metrics
    df = compile_metrics(eval_dir)
    print(f"  Found {len(df)} epochs")

    # Fetch train_trajectories from main training directory
    if training_dir is None:
        training_dir = eval_dir.parent.parent

    train_traj_map = {}
    for epoch_dir in training_dir.iterdir():
        if epoch_dir.is_dir() and re.match(r"epoch_\d+", epoch_dir.name):
            results_file = epoch_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        results = json.load(f)
                    epoch_num = int(re.search(r"\d+", epoch_dir.name).group())
                    train_traj_map[epoch_num] = results.get("train_trajectories", 0)
                except (json.JSONDecodeError, PermissionError):
                    pass

    # Update train_trajectories column in dataframe
    if train_traj_map:
        df["train_trajectories"] = df["epoch"].map(train_traj_map).fillna(0).astype(int)
        print(f"  Loaded train_trajectories from {len(train_traj_map)} training epochs")

    # Save to eval_dir/metrics/
    metrics_dir = eval_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    csv_path = metrics_dir / "metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    plots_dir = metrics_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    generate_plots(df, plots_dir)
    print(f"  Plots: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile and plot metrics across adaptive ROA training epochs"
    )
    parser.add_argument(
        "training_dir",
        type=Path,
        help="Path to training output directory containing epoch_XXX folders",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=None,
        help="Specific evaluation to compile: 'default', 'radius_0.25', etc. "
             "If not specified, compiles ALL evaluations (default + re-evaluations).",
    )
    args = parser.parse_args()

    training_dir = args.training_dir
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")

    evaluations_dir = training_dir / "evaluations"

    if args.eval_dir is None:
        # Compile ALL: default + re-evaluations
        print("=" * 70)
        print("COMPILING ALL EVALUATIONS")
        print("=" * 70)

        # 1. Compile DEFAULT (from training_dir/epoch_XXX/)
        print("\n--- Compiling: default ---")
        compile_default(training_dir)

        # 2. Compile all re-evaluations
        if evaluations_dir.exists():
            for eval_dir in sorted(evaluations_dir.iterdir()):
                if eval_dir.is_dir():
                    print(f"\n--- Compiling: {eval_dir.name} ---")
                    compile_reeval(eval_dir, training_dir)
        else:
            print("\n(No re-evaluations found)")

        print("\n" + "=" * 70)
        print("COMPILATION COMPLETE")
        print("=" * 70)

    elif args.eval_dir == "default":
        # Compile DEFAULT only
        print("=" * 70)
        print("COMPILING: default (original training metrics)")
        print("=" * 70)
        compile_default(training_dir)
        print("\nDone!")

    else:
        # Compile specific re-evaluation
        eval_dir = evaluations_dir / args.eval_dir
        if not eval_dir.exists():
            print(f"ERROR: {eval_dir} does not exist")
            print("\nAvailable re-evaluations:")
            if evaluations_dir.exists():
                for d in sorted(evaluations_dir.iterdir()):
                    if d.is_dir():
                        print(f"  - {d.name}")
            else:
                print("  (none)")
            return 1

        print("=" * 70)
        print(f"COMPILING: {args.eval_dir}")
        print("=" * 70)
        compile_reeval(eval_dir, training_dir)
        print("\nDone!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)

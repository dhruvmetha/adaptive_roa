#!/usr/bin/env python3
"""
Compile and plot metrics across adaptive ROA training epochs.

Usage:
    python scripts/compile_adaptive_metrics.py /path/to/training/output/dir

Output is saved to {training_output_dir}/metrics/:
    - metrics_summary.csv: Complete metrics table
    - plots/: Directory with visualization plots
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

    # Classification metrics (from conformal_thresholds or top-level)
    conformal = full_roa.get("conformal_thresholds", {})
    metrics["f1"] = full_roa.get("f1", conformal.get("f1", 0))
    metrics["accuracy"] = full_roa.get("accuracy", conformal.get("accuracy", 0))
    metrics["precision"] = full_roa.get("precision", conformal.get("precision", 0))
    metrics["recall"] = full_roa.get("recall", conformal.get("recall", 0))
    metrics["specificity"] = full_roa.get("specificity", conformal.get("specificity", 0))

    # Coverage metrics
    metrics["separatrix_pct"] = full_roa.get("separatrix_pct", conformal.get("separatrix_pct", 0))
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


def compile_metrics(output_dir: Path) -> pd.DataFrame:
    """Compile metrics from all epochs into a DataFrame."""
    epoch_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and re.match(r"epoch_\d+", d.name)],
        key=lambda x: int(re.search(r"\d+", x.name).group()),
    )

    if not epoch_dirs:
        raise ValueError(f"No epoch directories found in {output_dir}")

    all_metrics = []
    for epoch_dir in epoch_dirs:
        results_file = epoch_dir / "results.json"
        if not results_file.exists():
            print(f"Warning: {results_file} not found, skipping")
            continue

        try:
            with open(results_file) as f:
                results = json.load(f)
        except PermissionError:
            print(f"Warning: Permission denied for {results_file}, skipping")
            continue
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse {results_file}: {e}, skipping")
            continue

        metrics = extract_metrics_from_epoch(results)
        all_metrics.append(metrics)

    return pd.DataFrame(all_metrics)


def plot_performance_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot F1 and Separatrix % (2 subplots)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # F1 Score
    axes[0].plot(df["epoch"], df["f1"], "b-o", linewidth=2, markersize=6)
    axes[0].set_ylabel("F1 Score", fontsize=12)
    axes[0].set_title("Classification Performance", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Separatrix %
    axes[1].plot(
        df["epoch"], df["separatrix_pct"] * 100, "r-o", linewidth=2, markersize=6
    )
    axes[1].set_ylabel("Separatrix %", fontsize=12)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / "performance_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_coverage_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot Invalid % and Uncertain % breakdown."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["epoch"],
        df["invalid_pct"] * 100,
        "r-o",
        linewidth=2,
        markersize=6,
        label="Invalid %",
    )
    ax.plot(
        df["epoch"],
        df["uncertain_pct"] * 100,
        "orange",
        linestyle="-",
        marker="s",
        linewidth=2,
        markersize=6,
        label="Uncertain %",
    )
    ax.plot(
        df["epoch"],
        df["separatrix_pct"] * 100,
        "purple",
        linestyle="--",
        marker="^",
        linewidth=2,
        markersize=6,
        label="Separatrix % (total)",
    )

    ax.set_xlabel("Epoch", fontsize=12)
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
        df["epoch"],
        df["endpoint_mae_overall"],
        "b-o",
        linewidth=2,
        markersize=6,
        label="Overall MAE",
    )
    ax.plot(
        df["epoch"],
        df["endpoint_mae_success"],
        "g-s",
        linewidth=2,
        markersize=6,
        label="Success MAE",
    )
    ax.plot(
        df["epoch"],
        df["endpoint_mae_failure"],
        "r-^",
        linewidth=2,
        markersize=6,
        label="Failure MAE",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean Absolute Error", fontsize=12)
    ax.set_title("Endpoint Prediction Errors (Training Data)", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "endpoint_errors.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_progress(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot cumulative training trajectories."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["epoch"],
        df["train_trajectories"],
        "b-o",
        linewidth=2,
        markersize=6,
    )
    ax.fill_between(df["epoch"], 0, df["train_trajectories"], alpha=0.3)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cumulative Trajectories", fontsize=12)
    ax.set_title("Training Data Growth", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_progress.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_stratified_region_errors(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot errors by prediction region."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each region
    ax.plot(
        df["epoch"],
        df["roa_error_certain"],
        "b-o",
        linewidth=2,
        markersize=6,
        label="Certain (all)",
    )
    ax.plot(
        df["epoch"],
        df["roa_error_certain_success"],
        "g-s",
        linewidth=2,
        markersize=6,
        label="Certain Success",
    )
    ax.plot(
        df["epoch"],
        df["roa_error_certain_failure"],
        "c-^",
        linewidth=2,
        markersize=6,
        label="Certain Failure",
    )
    ax.plot(
        df["epoch"],
        df["roa_error_uncertain"],
        "orange",
        linestyle="-",
        marker="d",
        linewidth=2,
        markersize=6,
        label="Uncertain",
    )
    ax.plot(
        df["epoch"],
        df["roa_error_invalid"],
        "r-v",
        linewidth=2,
        markersize=6,
        label="Invalid",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean Endpoint Error", fontsize=12)
    ax.set_title("Endpoint Errors by Prediction Region (Full ROA Evaluation)", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "stratified_region_errors.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compile and plot metrics across adaptive ROA training epochs"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to training output directory containing epoch_NNN folders",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Create metrics output directory inside training output dir
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    plots_dir = metrics_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Compiling metrics from {output_dir}")

    # Compile metrics
    df = compile_metrics(output_dir)
    print(f"Found {len(df)} epochs")

    # Save CSV
    csv_path = metrics_dir / "metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

    # Generate plots
    print("Generating plots...")
    plot_performance_metrics(df, plots_dir)
    plot_coverage_breakdown(df, plots_dir)
    plot_endpoint_errors(df, plots_dir)
    plot_training_progress(df, plots_dir)
    plot_stratified_region_errors(df, plots_dir)

    print(f"Plots saved to {plots_dir}")
    print("Done!")


if __name__ == "__main__":
    main()

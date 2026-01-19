"""
Analyze and visualize results from adaptive sampling runs.

This script provides post-hoc analysis and visualization of adaptive sampling results,
including:
- Metrics progression across epochs (F1, accuracy, separatrix %)
- Lambda/delta progression for different optimization modes
- Phase space plot regeneration from saved NPZ data
- Multi-run comparison
- Video animation of phase space evolution across epochs

Usage:
    # Analyze a single run
    python adaptive_roa/adaptive/analyze_adaptive_results.py \
        --results_dir /path/to/adaptive_run/2025-12-19_03-01-24

    # Compare multiple runs
    python adaptive_roa/adaptive/analyze_adaptive_results.py \
        --results_dir /path/to/run1 /path/to/run2 /path/to/run3 \
        --compare

    # Regenerate phase space plots for specific epochs
    python adaptive_roa/adaptive/analyze_adaptive_results.py \
        --results_dir /path/to/run \
        --regenerate_plots --epochs 0 5 10 19

    # Create animation video of phase space evolution
    python adaptive_roa/adaptive/analyze_adaptive_results.py \
        --results_dir /path/to/run \
        --animation --framerate 3
"""

import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import glob
import subprocess
import shutil


def load_final_results(results_dir: Path) -> Dict:
    """
    Load final_results.json from a run directory.

    Tries multiple file names in order:
    1. final_results.json (original)
    2. final_results_reconstructed.json (from reconstruct_results.py)
    """
    # Try original file first
    final_results_path = results_dir / "final_results.json"
    if final_results_path.exists():
        with open(final_results_path, 'r') as f:
            return json.load(f)

    # Try reconstructed file
    reconstructed_path = results_dir / "final_results_reconstructed.json"
    if reconstructed_path.exists():
        print(f"Note: Using reconstructed results from {reconstructed_path.name}")
        with open(reconstructed_path, 'r') as f:
            return json.load(f)

    raise FileNotFoundError(
        f"Neither final_results.json nor final_results_reconstructed.json found in {results_dir}\n"
        f"Run `python adaptive_roa/adaptive/reconstruct_results.py -r {results_dir}` to reconstruct from checkpoints."
    )


def get_initial_train_size(results_dir: Path) -> int:
    """
    Get the initial training size from the Hydra config.

    The train_trajectories field in results.json is recorded AFTER adding new samples,
    so we need the initial_train_size to correctly compute the actual training size
    for each epoch.
    """
    # Try to load from Hydra config in epoch_000
    hydra_config = results_dir / "epoch_000" / ".hydra" / "config.yaml"
    if hydra_config.exists():
        import yaml
        with open(hydra_config, 'r') as f:
            cfg = yaml.safe_load(f)
            return cfg.get('initial_train_size', 50)

    # Fallback: try root .hydra
    hydra_config = results_dir / ".hydra" / "config.yaml"
    if hydra_config.exists():
        import yaml
        with open(hydra_config, 'r') as f:
            cfg = yaml.safe_load(f)
            return cfg.get('initial_train_size', 50)

    # Default fallback
    return 50


def correct_train_trajectories(epoch_results: List[Dict], initial_train_size: int) -> List[int]:
    """
    Correct the train_trajectories count for each epoch.

    The stored train_trajectories is recorded AFTER adding samples for the next epoch.
    The actual training size for epoch N is:
    - For epoch 0: initial_train_size
    - For epoch N (N>0): train_trajectories from epoch N-1

    Args:
        epoch_results: List of epoch result dicts
        initial_train_size: Initial training set size before any adaptive sampling

    Returns:
        List of corrected training sizes for each epoch
    """
    corrected = []
    for i, r in enumerate(epoch_results):
        if i == 0:
            corrected.append(initial_train_size)
        else:
            # Use previous epoch's stored value (which is what was actually trained on)
            corrected.append(epoch_results[i-1]['train_trajectories'])
    return corrected


def load_epoch_npz(results_dir: Path, epoch: int) -> Optional[Dict]:
    """Load per-point NPZ data for a specific epoch."""
    npz_path = results_dir / f"epoch_{epoch:03d}" / "full_roa_evaluation_per_point.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    return {
        'start_states': data['start_states'],
        'probabilities': data['probabilities'],
        'true_labels': data['true_labels'],
        'lambda_star': float(data['lambda_star']),
        'delta': float(data['delta']),
    }


def get_run_name(results_dir: Path) -> str:
    """Extract a descriptive name from the results directory."""
    # Try to get optimize_mode from first epoch
    epoch0_results = results_dir / "epoch_000" / "results.json"
    if epoch0_results.exists():
        with open(epoch0_results, 'r') as f:
            data = json.load(f)
            mode = data.get('optimize_mode', 'unknown')
            return f"{results_dir.name} ({mode})"
    return results_dir.name


def export_metrics_to_csv(
    results: Dict,
    output_dir: Path,
    run_name: str = "",
    initial_train_size: int = 50
) -> None:
    """
    Export metrics to CSV files.

    Creates two CSV files:
    - metrics_conformal.csv: Metrics using conformal thresholds
    - metrics_fixed.csv: Metrics using fixed 0.4/0.6 thresholds

    Args:
        results: Loaded final_results.json data
        output_dir: Directory to save CSV files
        run_name: Optional name for the run
        initial_train_size: Initial training set size (for correcting trajectory counts)
    """
    epoch_results = results['epoch_results']

    # Correct the train_trajectories count
    corrected_sizes = correct_train_trajectories(epoch_results, initial_train_size)

    for threshold_type in ['conformal', 'fixed']:
        key = f"{threshold_type}_thresholds"

        rows = []
        for i, r in enumerate(epoch_results):
            has_delta_star = 'delta_star' in r
            delta_star = r.get('delta_star', r['full_roa']['delta']) if has_delta_star else r['full_roa']['delta']

            row = {
                'epoch': r['epoch'],
                'train_trajectories': corrected_sizes[i],  # Use corrected value
                'lambda_star': r['lambda_star'],
                'delta_star': delta_star,
                'q_hat': r['q_hat'],
                'f1': r['full_roa'][key]['f1'],
                'accuracy': r['full_roa'][key]['accuracy'],
                'precision': r['full_roa'][key]['precision'],
                'recall': r['full_roa'][key]['recall'],
                'specificity': r['full_roa'][key]['specificity'],
                'separatrix_pct': r['full_roa'][key]['separatrix_pct'],
                'n_confident': r['full_roa'][key]['n_confident'],
                'n_uncertain': r['full_roa'][key]['n_uncertain'],
                'tp': r['full_roa'][key]['tp'],
                'tn': r['full_roa'][key]['tn'],
                'fp': r['full_roa'][key]['fp'],
                'fn': r['full_roa'][key]['fn'],
            }
            rows.append(row)

        output_file = output_dir / f"metrics_{threshold_type}.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved: {output_file}")


def export_summary_csv(
    results: Dict,
    output_dir: Path,
    run_label: str,
    initial_train_size: int = 50
) -> None:
    """
    Export clean summary CSV files with key metrics.

    Creates two CSV files per run:
    - {run_label}_conformal.csv
    - {run_label}_fixed.csv

    With columns: epoch, data, lambda_star, delta_star, f1, sep_pct

    Args:
        results: Loaded final_results.json data
        output_dir: Directory to save CSV files
        run_label: Label for the run (used in filename)
        initial_train_size: Initial training set size (for correcting trajectory counts)
    """
    epoch_results = results['epoch_results']
    corrected_sizes = correct_train_trajectories(epoch_results, initial_train_size)

    # Clean the run_label for filename
    safe_label = run_label.replace('[', '').replace(']', '').replace(', ', '_').replace(' ', '_')

    for threshold_type in ['conformal', 'fixed']:
        key = f"{threshold_type}_thresholds"

        rows = []
        for i, r in enumerate(epoch_results):
            delta_star = r.get('delta_star', r['full_roa']['delta'])

            row = {
                'epoch': r['epoch'],
                'data': corrected_sizes[i],
                'lambda_star': round(r['lambda_star'], 3),
                'delta_star': round(delta_star, 3),
                'f1': round(r['full_roa'][key]['f1'] * 100, 2),
                'sep_pct': round(r['full_roa'][key]['separatrix_pct'] * 100, 2),
            }
            rows.append(row)

        output_file = output_dir / f"{safe_label}_{threshold_type}.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'data', 'lambda_star', 'delta_star', 'f1', 'sep_pct'])
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved: {output_file}")


def export_comparison_to_csv(
    results_dirs: List[Path],
    output_dir: Path,
    threshold_type: str = "conformal"
) -> None:
    """
    Export comparison metrics to a single CSV with all runs.
    """
    key = f"{threshold_type}_thresholds"
    all_rows = []

    for results_dir in results_dirs:
        try:
            results = load_final_results(results_dir)
        except FileNotFoundError:
            continue

        run_name = get_run_name(results_dir)
        epoch_results = results['epoch_results']

        # Get corrected training sizes
        initial_train_size = get_initial_train_size(results_dir)
        corrected_sizes = correct_train_trajectories(epoch_results, initial_train_size)

        for i, r in enumerate(epoch_results):
            has_delta_star = 'delta_star' in r
            delta_star = r.get('delta_star', r['full_roa']['delta']) if has_delta_star else r['full_roa']['delta']

            row = {
                'run': run_name,
                'epoch': r['epoch'],
                'train_trajectories': corrected_sizes[i],  # Use corrected value
                'lambda_star': r['lambda_star'],
                'delta_star': delta_star,
                'f1': r['full_roa'][key]['f1'],
                'accuracy': r['full_roa'][key]['accuracy'],
                'separatrix_pct': r['full_roa'][key]['separatrix_pct'],
            }
            all_rows.append(row)

    if all_rows:
        output_file = output_dir / f"comparison_{threshold_type}.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"Saved: {output_file}")


def plot_metrics_progression(
    results: Dict,
    output_dir: Path,
    title_prefix: str = "",
    threshold_type: str = "conformal",
    initial_train_size: int = 50
) -> None:
    """
    Plot metrics progression across epochs.

    Args:
        results: Loaded final_results.json data
        output_dir: Directory to save plots
        title_prefix: Optional prefix for plot titles
        threshold_type: "conformal" or "fixed" thresholds
        initial_train_size: Initial training set size (for correcting trajectory counts)
    """
    epoch_results = results['epoch_results']
    epochs = [r['epoch'] for r in epoch_results]

    # Extract metrics based on threshold type
    key = f"{threshold_type}_thresholds"

    f1_scores = [r['full_roa'][key]['f1'] for r in epoch_results]
    accuracy = [r['full_roa'][key]['accuracy'] for r in epoch_results]
    precision = [r['full_roa'][key]['precision'] for r in epoch_results]
    recall = [r['full_roa'][key]['recall'] for r in epoch_results]
    separatrix_pct = [r['full_roa'][key]['separatrix_pct'] for r in epoch_results]
    # Use corrected train_trajectories
    train_trajectories = correct_train_trajectories(epoch_results, initial_train_size)

    # Create figure with subplots (extra height for legends below)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: F1 Score
    ax1 = axes[0, 0]
    ax1.plot(epochs, f1_scores, 'b-o', linewidth=2, markersize=6, label='F1 Score')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('F1 Score', fontsize=11)
    ax1.set_title('F1 Score Progression', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=10)

    # Plot 2: Accuracy, Precision, Recall
    ax2 = axes[0, 1]
    ax2.plot(epochs, accuracy, 'g-o', label='Accuracy', linewidth=2, markersize=5)
    ax2.plot(epochs, precision, 'b-s', label='Precision', linewidth=2, markersize=5)
    ax2.plot(epochs, recall, 'r-^', label='Recall', linewidth=2, markersize=5)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Accuracy, Precision, Recall', fontsize=12)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Plot 3: Separatrix Percentage
    ax3 = axes[1, 0]
    ax3.plot(epochs, [s * 100 for s in separatrix_pct], 'm-o', linewidth=2, markersize=6, label='Separatrix %')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Separatrix %', fontsize=11)
    ax3.set_title('Separatrix (Uncertain) Region', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=10)

    # Plot 4: Training Set Size
    ax4 = axes[1, 1]
    ax4.plot(epochs, train_trajectories, 'c-o', linewidth=2, markersize=6, label='# Trajectories')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('# Trajectories', fontsize=11)
    ax4.set_title('Training Set Size', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=10)

    threshold_label = "Conformal (λ*±δ*)" if threshold_type == "conformal" else "Fixed (0.4/0.6)"
    plt.suptitle(f'{title_prefix}Metrics Progression - {threshold_label}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    output_file = output_dir / f"metrics_progression_{threshold_type}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_metrics_vs_dataset_size(
    results: Dict,
    output_dir: Path,
    title_prefix: str = "",
    threshold_type: str = "conformal",
    initial_train_size: int = 50
) -> None:
    """
    Plot F1 and Separatrix % against training dataset size.

    Args:
        results: Loaded final_results.json data
        output_dir: Directory to save plots
        title_prefix: Optional prefix for plot titles
        threshold_type: "conformal" or "fixed" thresholds
        initial_train_size: Initial training set size (for correcting trajectory counts)
    """
    epoch_results = results['epoch_results']

    # Extract metrics based on threshold type
    key = f"{threshold_type}_thresholds"

    f1_scores = [r['full_roa'][key]['f1'] for r in epoch_results]
    separatrix_pct = [r['full_roa'][key]['separatrix_pct'] for r in epoch_results]
    # Use corrected train_trajectories
    train_trajectories = correct_train_trajectories(epoch_results, initial_train_size)

    # Create figure with 2 subplots side by side (extra height for legends below)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: F1 vs Dataset Size
    ax1 = axes[0]
    ax1.plot(train_trajectories, f1_scores, 'b-o', linewidth=2, markersize=6, label='F1 Score')
    ax1.set_xlabel('Training Set Size (# trajectories)', fontsize=11)
    ax1.set_ylabel('F1 Score', fontsize=11)
    ax1.set_title('F1 Score vs Dataset Size', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Add epoch annotations for key points
    for i, (x, y) in enumerate(zip(train_trajectories, f1_scores)):
        if i == 0 or i == len(train_trajectories) - 1:
            ax1.annotate(f'E{i}', (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, alpha=0.7)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=10)

    # Plot 2: Separatrix % vs Dataset Size
    ax2 = axes[1]
    ax2.plot(train_trajectories, [s * 100 for s in separatrix_pct], 'm-o', linewidth=2, markersize=6, label='Separatrix %')
    ax2.set_xlabel('Training Set Size (# trajectories)', fontsize=11)
    ax2.set_ylabel('Separatrix %', fontsize=11)
    ax2.set_title('Separatrix (Uncertain) % vs Dataset Size', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add epoch annotations for key points
    for i, (x, y) in enumerate(zip(train_trajectories, separatrix_pct)):
        if i == 0 or i == len(train_trajectories) - 1:
            ax2.annotate(f'E{i}', (x, y * 100), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, alpha=0.7)

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=10)

    threshold_label = "Conformal (λ*±δ*)" if threshold_type == "conformal" else "Fixed (0.4/0.6)"
    plt.suptitle(f'{title_prefix}Metrics vs Dataset Size - {threshold_label}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    output_file = output_dir / f"metrics_vs_dataset_size_{threshold_type}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_conformal_params_progression(
    results: Dict,
    output_dir: Path,
    title_prefix: str = ""
) -> None:
    """
    Plot lambda* and delta* progression across epochs.

    Handles both optimization modes:
    - "lambda": lambda_star varies, delta is fixed
    - "delta": delta_star varies, lambda is fixed at 0.5
    """
    epoch_results = results['epoch_results']
    epochs = [r['epoch'] for r in epoch_results]

    lambda_stars = [r['lambda_star'] for r in epoch_results]
    q_hats = [r['q_hat'] for r in epoch_results]

    # Check if delta_star exists (delta optimization mode)
    has_delta_star = 'delta_star' in epoch_results[0]
    if has_delta_star:
        delta_stars = [r['delta_star'] for r in epoch_results]
    else:
        # Lambda optimization mode - delta is fixed
        delta_stars = [r['full_roa']['delta'] for r in epoch_results]

    optimize_mode = epoch_results[0].get('optimize_mode', 'lambda')

    # Compute success/failure thresholds
    success_thresholds = [l + d for l, d in zip(lambda_stars, delta_stars)]
    failure_thresholds = [l - d for l, d in zip(lambda_stars, delta_stars)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Lambda* (or threshold center)
    ax1 = axes[0, 0]
    ax1.plot(epochs, lambda_stars, 'b-o', linewidth=2, markersize=6, label='λ*')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('λ*', fontsize=11)
    ax1.set_title(f'λ* (Decision Boundary Center) - {"optimized" if optimize_mode == "lambda" else "fixed=0.5"}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=10)

    # Plot 2: Delta*
    ax2 = axes[0, 1]
    ax2.plot(epochs, delta_stars, 'r-o', linewidth=2, markersize=6, label='δ*')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('δ*', fontsize=11)
    ax2.set_title(f'δ* (Uncertainty Half-width) - {"optimized" if optimize_mode == "delta" else "fixed"}', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.5])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=10)

    # Plot 3: Success/Failure Thresholds
    ax3 = axes[1, 0]
    ax3.plot(epochs, success_thresholds, 'g-o', label='Success (λ*+δ*)', linewidth=2, markersize=5)
    ax3.plot(epochs, failure_thresholds, 'r-s', label='Failure (λ*-δ*)', linewidth=2, markersize=5)
    ax3.plot(epochs, lambda_stars, 'b--', label='λ*', linewidth=1, alpha=0.7)
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Threshold', fontsize=11)
    ax3.set_title('Decision Thresholds', fontsize=12)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # Plot 4: q_hat
    ax4 = axes[1, 1]
    ax4.plot(epochs, q_hats, 'm-o', linewidth=2, markersize=6, label='q̂')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('q̂', fontsize=11)
    ax4.set_title('Calibrated Quantile (q̂)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=10)

    plt.suptitle(f'{title_prefix}Conformal Parameters - optimize_mode={optimize_mode}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    output_file = output_dir / "conformal_params_progression.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_phase_space(
    start_states: np.ndarray,
    success_rate: np.ndarray,
    true_labels: np.ndarray,
    lambda_star: float,
    delta: float,
    output_file: Path,
    title: str = "Pendulum ROA Phase Space",
    threshold_type: str = "conformal",
) -> Dict:
    """
    Plot the Region of Attraction (ROA) as a phase space diagram.

    Args:
        threshold_type: "conformal" uses lambda* +/- delta, "fixed" uses 0.4/0.6

    Returns statistics dictionary.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    theta = start_states[:, 0]
    theta_dot = start_states[:, 1]

    # --- Left plot: Predicted ROA ---
    ax1 = axes[0]

    # Classify based on threshold type
    pred_labels = np.zeros(len(success_rate))
    if threshold_type == "conformal":
        success_thresh = lambda_star + delta
        failure_thresh = lambda_star - delta
        thresh_label = f"λ*={lambda_star:.3f}, δ={delta:.3f}"
    else:  # fixed
        success_thresh = 0.6
        failure_thresh = 0.4
        thresh_label = "fixed 0.4/0.6"

    pred_labels[success_rate > success_thresh] = 1   # Success
    pred_labels[success_rate < failure_thresh] = -1  # Failure

    # Create color array
    colors = np.array(['gold'] * len(pred_labels))
    colors[pred_labels == 1] = 'blue'
    colors[pred_labels == -1] = 'red'

    # Sort for visualization
    order = np.argsort(np.abs(pred_labels))[::-1]

    ax1.scatter(
        theta[order], theta_dot[order],
        c=colors[order],
        s=3, alpha=0.6, edgecolors='none'
    )

    ax1.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax1.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
    ax1.set_title(f'Predicted ROA ({thresh_label})', fontsize=12)
    ax1.set_xlim(-np.pi, np.pi)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax1.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    n_success = np.sum(pred_labels == 1)
    n_failure = np.sum(pred_labels == -1)
    n_separatrix = np.sum(pred_labels == 0)

    legend_elements = [
        mpatches.Patch(color='blue', label=f'Success ({n_success:,})'),
        mpatches.Patch(color='red', label=f'Failure ({n_failure:,})'),
        mpatches.Patch(color='gold', label=f'Separatrix ({n_separatrix:,})'),
    ]
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Right plot: Ground Truth ROA ---
    ax2 = axes[1]

    gt_colors = np.array(['gold'] * len(true_labels))
    gt_colors[true_labels == 1] = 'blue'
    gt_colors[true_labels == -1] = 'red'

    gt_order = np.argsort(np.abs(true_labels))[::-1]

    ax2.scatter(
        theta[gt_order], theta_dot[gt_order],
        c=gt_colors[gt_order],
        s=3, alpha=0.6, edgecolors='none'
    )

    ax2.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax2.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
    ax2.set_title('Ground Truth ROA', fontsize=12)
    ax2.set_xlim(-np.pi, np.pi)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax2.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    gt_n_success = np.sum(true_labels == 1)
    gt_n_failure = np.sum(true_labels == -1)
    gt_n_separatrix = np.sum(true_labels == 0)

    legend_elements_gt = [
        mpatches.Patch(color='blue', label=f'Success ({gt_n_success:,})'),
        mpatches.Patch(color='red', label=f'Failure ({gt_n_failure:,})'),
        mpatches.Patch(color='gold', label=f'Separatrix ({gt_n_separatrix:,})'),
    ]
    ax2.legend(handles=legend_elements_gt, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'n_pred_success': int(n_success),
        'n_pred_failure': int(n_failure),
        'n_pred_separatrix': int(n_separatrix),
        'n_gt_success': int(gt_n_success),
        'n_gt_failure': int(gt_n_failure),
    }


def plot_probability_heatmap(
    start_states: np.ndarray,
    success_rate: np.ndarray,
    lambda_star: float,
    delta: float,
    output_file: Path,
    title: str = "Pendulum ROA - Success Probability",
) -> None:
    """Plot the ROA as a heatmap of success probability."""
    fig, ax = plt.subplots(figsize=(10, 8))

    theta = start_states[:, 0]
    theta_dot = start_states[:, 1]

    scatter = ax.scatter(
        theta, theta_dot,
        c=success_rate,
        cmap='RdYlBu',
        s=3, alpha=0.7, edgecolors='none',
        vmin=0, vmax=1
    )

    cbar = plt.colorbar(scatter, ax=ax, label='p(success)')

    # Mark decision boundaries on colorbar
    cbar.ax.axhline(y=lambda_star + delta, color='black', linestyle='--', linewidth=1.5)
    cbar.ax.axhline(y=lambda_star - delta, color='black', linestyle='--', linewidth=1.5)
    cbar.ax.axhline(y=lambda_star, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
    ax.set_title(f'{title}\n(λ*={lambda_star:.3f}, bounds: [{lambda_star-delta:.3f}, {lambda_star+delta:.3f}])', fontsize=12)
    ax.set_xlim(-np.pi, np.pi)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def regenerate_epoch_plots(
    results_dir: Path,
    epochs: List[int],
    output_dir: Optional[Path] = None
) -> None:
    """Regenerate phase space plots from saved NPZ data for specified epochs."""
    if output_dir is None:
        output_dir = results_dir / "regenerated_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in epochs:
        npz_data = load_epoch_npz(results_dir, epoch)
        if npz_data is None:
            print(f"Warning: No NPZ data found for epoch {epoch}")
            continue

        # Phase space plot
        phase_space_file = output_dir / f"epoch_{epoch:03d}_phase_space.png"
        plot_phase_space(
            start_states=npz_data['start_states'],
            success_rate=npz_data['probabilities'],
            true_labels=npz_data['true_labels'],
            lambda_star=npz_data['lambda_star'],
            delta=npz_data['delta'],
            output_file=phase_space_file,
            title=f"Pendulum ROA - Epoch {epoch}"
        )
        print(f"Saved: {phase_space_file}")

        # Probability heatmap
        heatmap_file = output_dir / f"epoch_{epoch:03d}_heatmap.png"
        plot_probability_heatmap(
            start_states=npz_data['start_states'],
            success_rate=npz_data['probabilities'],
            lambda_star=npz_data['lambda_star'],
            delta=npz_data['delta'],
            output_file=heatmap_file,
            title=f"Pendulum ROA - Epoch {epoch}"
        )
        print(f"Saved: {heatmap_file}")


def plot_multi_run_comparison(
    results_dirs: List[Path],
    output_dir: Path,
    threshold_type: str = "conformal",
    labels: Optional[List[str]] = None
) -> None:
    """
    Compare metrics across multiple runs.

    Args:
        results_dirs: List of run directories to compare
        output_dir: Directory to save comparison plots
        threshold_type: "conformal" or "fixed" thresholds
        labels: Optional custom labels for each run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dirs)))

    key = f"{threshold_type}_thresholds"

    for idx, results_dir in enumerate(results_dirs):
        try:
            results = load_final_results(results_dir)
        except FileNotFoundError:
            print(f"Warning: Could not load {results_dir}")
            continue

        epoch_results = results['epoch_results']
        epochs = [r['epoch'] for r in epoch_results]

        # Get corrected training sizes
        initial_train_size = get_initial_train_size(results_dir)
        train_trajectories = correct_train_trajectories(epoch_results, initial_train_size)

        f1_scores = [r['full_roa'][key]['f1'] for r in epoch_results]
        accuracy = [r['full_roa'][key]['accuracy'] for r in epoch_results]
        separatrix_pct = [r['full_roa'][key]['separatrix_pct'] for r in epoch_results]

        # Use custom label if provided, otherwise generate from run name
        if labels and idx < len(labels):
            label = labels[idx]
        else:
            label = get_run_name(results_dir)
        color = colors[idx]

        axes[0, 0].plot(epochs, f1_scores, '-o', color=color, label=label, linewidth=2, markersize=4)
        axes[0, 1].plot(epochs, accuracy, '-o', color=color, label=label, linewidth=2, markersize=4)
        axes[1, 0].plot(epochs, [s * 100 for s in separatrix_pct], '-o', color=color, label=label, linewidth=2, markersize=4)
        axes[1, 1].plot(epochs, train_trajectories, '-o', color=color, label=label, linewidth=2, markersize=4)

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('F1 Score')
    axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Separatrix %')
    axes[1, 0].set_title('Separatrix (Uncertain) %')
    axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('# Trajectories')
    axes[1, 1].set_title('Training Set Size')
    axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    threshold_label = "Conformal (λ*±δ*)" if threshold_type == "conformal" else "Fixed (0.4/0.6)"
    plt.suptitle(f'Multi-Run Comparison - {threshold_label}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    output_file = output_dir / f"multi_run_comparison_{threshold_type}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_multi_run_vs_dataset_size(
    results_dirs: List[Path],
    output_dir: Path,
    threshold_type: str = "conformal",
    labels: Optional[List[str]] = None
) -> None:
    """
    Compare F1 and Separatrix % vs dataset size across multiple runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dirs)))

    key = f"{threshold_type}_thresholds"

    for idx, results_dir in enumerate(results_dirs):
        try:
            results = load_final_results(results_dir)
        except FileNotFoundError:
            print(f"Warning: Could not load {results_dir}")
            continue

        epoch_results = results['epoch_results']

        # Get corrected training sizes
        initial_train_size = get_initial_train_size(results_dir)
        train_trajectories = correct_train_trajectories(epoch_results, initial_train_size)

        f1_scores = [r['full_roa'][key]['f1'] for r in epoch_results]
        separatrix_pct = [r['full_roa'][key]['separatrix_pct'] for r in epoch_results]

        # Use custom label if provided
        if labels and idx < len(labels):
            label = labels[idx]
        else:
            label = get_run_name(results_dir)
        color = colors[idx]

        axes[0].plot(train_trajectories, f1_scores, '-o', color=color, label=label, linewidth=2, markersize=4)
        axes[1].plot(train_trajectories, [s * 100 for s in separatrix_pct], '-o', color=color, label=label, linewidth=2, markersize=4)

    axes[0].set_xlabel('Training Set Size (# trajectories)')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score vs Dataset Size')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(results_dirs), fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    axes[1].set_xlabel('Training Set Size (# trajectories)')
    axes[1].set_ylabel('Separatrix %')
    axes[1].set_title('Separatrix (Uncertain) % vs Dataset Size')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(results_dirs), fontsize=9)
    axes[1].grid(True, alpha=0.3)

    threshold_label = "Conformal (λ*±δ*)" if threshold_type == "conformal" else "Fixed (0.4/0.6)"
    plt.suptitle(f'Metrics vs Dataset Size - {threshold_label}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    output_file = output_dir / f"multi_run_vs_dataset_size_{threshold_type}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_conformal_params_comparison(
    results_dirs: List[Path],
    output_dir: Path,
    labels: Optional[List[str]] = None
) -> None:
    """Compare conformal parameters (lambda*, delta*) across multiple runs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dirs)))

    for idx, results_dir in enumerate(results_dirs):
        try:
            results = load_final_results(results_dir)
        except FileNotFoundError:
            continue

        epoch_results = results['epoch_results']
        epochs = [r['epoch'] for r in epoch_results]

        lambda_stars = [r['lambda_star'] for r in epoch_results]
        has_delta_star = 'delta_star' in epoch_results[0]
        if has_delta_star:
            delta_stars = [r['delta_star'] for r in epoch_results]
        else:
            delta_stars = [r['full_roa']['delta'] for r in epoch_results]

        success_thresholds = [l + d for l, d in zip(lambda_stars, delta_stars)]
        failure_thresholds = [l - d for l, d in zip(lambda_stars, delta_stars)]

        # Use custom label if provided
        if labels and idx < len(labels):
            label = labels[idx]
        else:
            label = get_run_name(results_dir)
        color = colors[idx]

        axes[0, 0].plot(epochs, lambda_stars, '-o', color=color, label=label, linewidth=2, markersize=4)
        axes[0, 1].plot(epochs, delta_stars, '-o', color=color, label=label, linewidth=2, markersize=4)
        axes[1, 0].plot(epochs, success_thresholds, '-o', color=color, label=label, linewidth=2, markersize=4)
        axes[1, 1].plot(epochs, failure_thresholds, '-o', color=color, label=label, linewidth=2, markersize=4)

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('λ*')
    axes[0, 0].set_title('λ* (Decision Boundary Center)')
    axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('δ*')
    axes[0, 1].set_title('δ* (Uncertainty Half-width)')
    axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 0.5])

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('λ* + δ*')
    axes[1, 0].set_title('Success Threshold (λ* + δ*)')
    axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('λ* - δ*')
    axes[1, 1].set_title('Failure Threshold (λ* - δ*)')
    axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])

    plt.suptitle('Conformal Parameters Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    output_file = output_dir / "conformal_params_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_epoch_animation_frames(
    results_dir: Path,
    output_dir: Optional[Path] = None,
    max_epochs: Optional[int] = None,
    framerate: int = 2,
    create_video: bool = True,
    threshold_types: List[str] = None,
) -> Dict[str, Optional[Path]]:
    """
    Create phase space frames for all epochs and optionally compile into video.

    Args:
        results_dir: Path to the adaptive run results directory
        output_dir: Directory to save frames (default: {results_dir}/analysis/animation_frames)
        max_epochs: Maximum number of epochs to include (default: all)
        framerate: Frames per second for the video (default: 2)
        create_video: Whether to compile frames into MP4 video (default: True)
        threshold_types: List of threshold types to generate ("conformal", "fixed", or both)

    Returns:
        Dict mapping threshold_type to video path (or None if failed/skipped)
    """
    if threshold_types is None:
        threshold_types = ["conformal", "fixed"]

    if output_dir is None:
        output_dir = results_dir / "analysis"

    # Find all epoch directories
    epoch_dirs = sorted(glob.glob(str(results_dir / "epoch_*")))

    if max_epochs:
        epoch_dirs = epoch_dirs[:max_epochs]

    video_paths = {}

    for threshold_type in threshold_types:
        frames_dir = output_dir / f"animation_frames_{threshold_type}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {len(epoch_dirs)} frames ({threshold_type} thresholds)...")

        for epoch_dir in epoch_dirs:
            epoch_dir = Path(epoch_dir)
            epoch_num = int(epoch_dir.name.split('_')[1])

            npz_data = load_epoch_npz(results_dir, epoch_num)
            if npz_data is None:
                continue

            # Load epoch results for training size and metrics
            results_file = epoch_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    epoch_results = json.load(f)
                train_size = epoch_results.get('train_trajectories', '?')
                # Get F1 score for display based on threshold type
                f1_key = f"{threshold_type}_thresholds"
                f1 = epoch_results.get('full_roa', {}).get(f1_key, {}).get('f1', None)
                if f1 is not None:
                    title = f"Epoch {epoch_num} | Train: {train_size} | F1: {f1:.1%}"
                else:
                    title = f"Epoch {epoch_num} | Training: {train_size} trajectories"
            else:
                train_size = '?'
                title = f"Epoch {epoch_num} | Training: {train_size} trajectories"

            phase_space_file = frames_dir / f"frame_{epoch_num:03d}.png"
            plot_phase_space(
                start_states=npz_data['start_states'],
                success_rate=npz_data['probabilities'],
                true_labels=npz_data['true_labels'],
                lambda_star=npz_data['lambda_star'],
                delta=npz_data['delta'],
                output_file=phase_space_file,
                title=title,
                threshold_type=threshold_type
            )

        print(f"Saved {len(epoch_dirs)} frames to {frames_dir}")

        # Create video if requested
        if create_video:
            video_path = create_video_from_frames(
                frames_dir,
                output_path=output_dir / f"phase_space_evolution_{threshold_type}.mp4",
                framerate=framerate
            )
            video_paths[threshold_type] = video_path

    return video_paths


def create_video_from_frames(
    frames_dir: Path,
    output_path: Optional[Path] = None,
    framerate: int = 2,
    pattern: str = "frame_%03d.png"
) -> Optional[Path]:
    """
    Create MP4 video from PNG frames using ffmpeg.

    Args:
        frames_dir: Directory containing frame images
        output_path: Output video path (default: {frames_dir}/../phase_space_evolution.mp4)
        framerate: Frames per second
        pattern: Frame filename pattern for ffmpeg

    Returns:
        Path to created video, or None if ffmpeg is not available
    """
    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        print("Warning: ffmpeg not found. Skipping video creation.")
        print(f"To create video manually: cd {frames_dir} && ffmpeg -framerate {framerate} -i {pattern} -c:v libx264 -pix_fmt yuv420p video.mp4")
        return None

    if output_path is None:
        output_path = frames_dir.parent / "phase_space_evolution.mp4"

    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-framerate', str(framerate),
        '-i', str(frames_dir / pattern),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Created video: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e.stderr}")
        return None
    except Exception as e:
        print(f"Error creating video: {e}")
        return None


def print_summary(results_dir: Path) -> None:
    """Print a summary of the run."""
    results = load_final_results(results_dir)
    epoch_results = results['epoch_results']

    print("\n" + "=" * 70)
    print(f"SUMMARY: {results_dir.name}")
    print("=" * 70)

    # Get optimization mode
    optimize_mode = epoch_results[0].get('optimize_mode', 'lambda')
    print(f"Optimization Mode: {optimize_mode}")
    print(f"Number of Epochs: {len(epoch_results)}")

    first = epoch_results[0]
    last = epoch_results[-1]

    print(f"\n--- Training Set ---")
    print(f"  Initial: {first['train_trajectories']} trajectories")
    print(f"  Final:   {last['train_trajectories']} trajectories")

    print(f"\n--- Conformal Parameters ---")
    print(f"  λ*: {first['lambda_star']:.4f} -> {last['lambda_star']:.4f}")
    if 'delta_star' in first:
        print(f"  δ*: {first['delta_star']:.4f} -> {last['delta_star']:.4f}")
    else:
        print(f"  δ:  {first['full_roa']['delta']:.4f} (fixed)")
    print(f"  q̂:  {first['q_hat']:.4f} -> {last['q_hat']:.4f}")

    print(f"\n--- Full ROA Metrics (Conformal Thresholds) ---")
    conf_first = first['full_roa']['conformal_thresholds']
    conf_last = last['full_roa']['conformal_thresholds']
    print(f"  F1:          {conf_first['f1']:.2%} -> {conf_last['f1']:.2%}")
    print(f"  Accuracy:    {conf_first['accuracy']:.2%} -> {conf_last['accuracy']:.2%}")
    print(f"  Precision:   {conf_first['precision']:.2%} -> {conf_last['precision']:.2%}")
    print(f"  Recall:      {conf_first['recall']:.2%} -> {conf_last['recall']:.2%}")
    print(f"  Separatrix:  {conf_first['separatrix_pct']:.2%} -> {conf_last['separatrix_pct']:.2%}")

    print(f"\n--- Full ROA Metrics (Fixed 0.4/0.6 Thresholds) ---")
    fixed_first = first['full_roa']['fixed_thresholds']
    fixed_last = last['full_roa']['fixed_thresholds']
    print(f"  F1:          {fixed_first['f1']:.2%} -> {fixed_last['f1']:.2%}")
    print(f"  Accuracy:    {fixed_first['accuracy']:.2%} -> {fixed_last['accuracy']:.2%}")
    print(f"  Separatrix:  {fixed_first['separatrix_pct']:.2%} -> {fixed_last['separatrix_pct']:.2%}")

    print("=" * 70)


def analyze_single_run(
    results_dir: Path,
    output_dir: Optional[Path] = None,
    regenerate_plots: bool = False,
    epochs_to_regenerate: Optional[List[int]] = None,
    create_animation: bool = False,
    framerate: int = 2,
) -> None:
    """Full analysis of a single run."""
    if output_dir is None:
        output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print summary
    print_summary(results_dir)

    # Load results
    results = load_final_results(results_dir)

    # Get initial train size for correcting trajectory counts
    initial_train_size = get_initial_train_size(results_dir)

    # Export metrics to CSV (detailed)
    export_metrics_to_csv(results, output_dir, initial_train_size=initial_train_size)

    # Export summary CSV (clean format with key metrics)
    run_label = get_run_name(results_dir)
    export_summary_csv(results, output_dir, run_label, initial_train_size)

    # Create progression plots
    plot_metrics_progression(results, output_dir, threshold_type="conformal", initial_train_size=initial_train_size)
    plot_metrics_progression(results, output_dir, threshold_type="fixed", initial_train_size=initial_train_size)
    plot_conformal_params_progression(results, output_dir)

    # Create metrics vs dataset size plots
    plot_metrics_vs_dataset_size(results, output_dir, threshold_type="conformal", initial_train_size=initial_train_size)
    plot_metrics_vs_dataset_size(results, output_dir, threshold_type="fixed", initial_train_size=initial_train_size)

    # Regenerate epoch plots if requested
    if regenerate_plots:
        if epochs_to_regenerate is None:
            # Default: first, middle, last epochs
            n_epochs = len(results['epoch_results'])
            epochs_to_regenerate = [0, n_epochs // 2, n_epochs - 1]
        regenerate_epoch_plots(results_dir, epochs_to_regenerate, output_dir / "epoch_plots")

    # Create animation frames and videos if requested (both threshold types)
    if create_animation:
        video_paths = create_epoch_animation_frames(
            results_dir,
            output_dir=output_dir,
            framerate=framerate,
            create_video=True,
            threshold_types=["conformal", "fixed"]
        )
        for thresh_type, video_path in video_paths.items():
            if video_path:
                print(f"Video saved ({thresh_type}): {video_path}")

    print(f"\nAnalysis complete. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze adaptive sampling results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single run
    python adaptive_roa/adaptive/analyze_adaptive_results.py \\
        --results_dir /path/to/run/2025-12-19_03-01-24

    # Compare multiple runs
    python adaptive_roa/adaptive/analyze_adaptive_results.py \\
        --results_dir /path/to/run1 /path/to/run2 \\
        --compare

    # Regenerate phase space plots
    python adaptive_roa/adaptive/analyze_adaptive_results.py \\
        --results_dir /path/to/run \\
        --regenerate_plots --epochs 0 5 10 19

    # Create animation frames
    python adaptive_roa/adaptive/analyze_adaptive_results.py \\
        --results_dir /path/to/run \\
        --animation
        """
    )

    parser.add_argument(
        '--results_dir', '-r',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to adaptive sampling results directory'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory for analysis plots (default: {results_dir}/analysis)'
    )
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Compare multiple runs (requires multiple --results_dir paths)'
    )
    parser.add_argument(
        '--regenerate_plots',
        action='store_true',
        help='Regenerate phase space plots from saved NPZ data'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        nargs='*',
        default=None,
        help='Specific epochs to regenerate plots for (default: first, middle, last)'
    )
    parser.add_argument(
        '--animation',
        action='store_true',
        help='Create animation frames and video for all epochs'
    )
    parser.add_argument(
        '--framerate', '-f',
        type=int,
        default=2,
        help='Frames per second for animation video (default: 2)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=str,
        choices=['conformal', 'fixed', 'both'],
        default='both',
        help='Which threshold type to use for comparison plots'
    )
    parser.add_argument(
        '--labels', '-l',
        type=str,
        nargs='*',
        default=None,
        help='Custom short labels for runs in comparison (e.g., --labels "λ-opt" "δ-fixed" "δ-uncertain")'
    )

    args = parser.parse_args()

    results_dirs = [Path(d) for d in args.results_dir]
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.compare and len(results_dirs) > 1:
        # Multi-run comparison
        if output_dir is None:
            output_dir = results_dirs[0].parent / "comparison"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get labels (use provided or None for auto-generated)
        labels = args.labels

        # Export comparison CSVs and plots
        if args.threshold in ['conformal', 'both']:
            export_comparison_to_csv(results_dirs, output_dir, threshold_type="conformal")
            plot_multi_run_comparison(results_dirs, output_dir, threshold_type="conformal", labels=labels)
            plot_multi_run_vs_dataset_size(results_dirs, output_dir, threshold_type="conformal", labels=labels)
        if args.threshold in ['fixed', 'both']:
            export_comparison_to_csv(results_dirs, output_dir, threshold_type="fixed")
            plot_multi_run_comparison(results_dirs, output_dir, threshold_type="fixed", labels=labels)
            plot_multi_run_vs_dataset_size(results_dirs, output_dir, threshold_type="fixed", labels=labels)

        plot_conformal_params_comparison(results_dirs, output_dir, labels=labels)

        # Export summary CSVs for each run
        for idx, results_dir in enumerate(results_dirs):
            try:
                results = load_final_results(results_dir)
                initial_train_size = get_initial_train_size(results_dir)

                # Determine label for this run
                if labels and idx < len(labels):
                    run_label = labels[idx]
                else:
                    run_label = get_run_name(results_dir)

                export_summary_csv(results, output_dir, run_label, initial_train_size)
            except FileNotFoundError:
                print(f"Warning: Could not load {results_dir}")

        # Print summaries for all runs
        for results_dir in results_dirs:
            print_summary(results_dir)

        print(f"\nComparison plots and CSVs saved to: {output_dir}")
    else:
        # Single run analysis (or analyze each run separately)
        for results_dir in results_dirs:
            analyze_single_run(
                results_dir=results_dir,
                output_dir=output_dir,
                regenerate_plots=args.regenerate_plots,
                epochs_to_regenerate=args.epochs,
                create_animation=args.animation,
                framerate=args.framerate
            )


if __name__ == "__main__":
    main()

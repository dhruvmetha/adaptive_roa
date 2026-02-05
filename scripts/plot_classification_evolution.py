#!/usr/bin/env python3
"""
Plot Classification Evolution Across Adaptive Epochs

This script visualizes how the conformal classification evolves across
adaptive sampling epochs. It takes 4 epochs (first, last, and 2 evenly spaced)
and creates high-quality phase space images showing:
- TP (True Positive): Green
- TN (True Negative): Blue
- FP (False Positive): Red
- FN (False Negative): Orange
- Uncertain: Gray

Usage:
    python scripts/plot_classification_evolution.py \
        --output_dir /path/to/adaptive/output/timestamp \
        --eval_states /path/to/eval_states.txt \
        --save_dir ./classification_plots
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adaptive_roa.flow_matching.pendulum.latent_conditional.flow_matcher import (
    PendulumLatentConditionalFlowMatcher
)
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.conformal.config import ConformalConfig
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator


def find_best_checkpoint(epoch_dir: Path) -> Path:
    """Find the best checkpoint in an epoch directory."""
    checkpoint_dir = epoch_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory in {epoch_dir}")

    # Look for best checkpoint (pattern: best-*.ckpt)
    best_checkpoints = list(checkpoint_dir.glob("best-*.ckpt"))
    if best_checkpoints:
        return best_checkpoints[0]

    # Fall back to last.ckpt
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def load_conformal_state(epoch_dir: Path) -> Dict:
    """Load conformal state from epoch directory."""
    conformal_state_path = epoch_dir / "conformal_state.json"
    if not conformal_state_path.exists():
        raise FileNotFoundError(f"No conformal_state.json in {epoch_dir}")

    with open(conformal_state_path) as f:
        return json.load(f)


def load_eval_states(eval_states_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load evaluation states from file.

    Format: comma-separated with columns: θ_start, θ̇_start, θ_end, θ̇_end, label

    Returns:
        start_states: [N, 2] array of start states
        end_states: [N, 2] array of end states
        labels: [N] array of labels (0 -> -1, 1 -> 1)
    """
    data = np.loadtxt(eval_states_path, delimiter=',')

    start_states = data[:, :2].astype(np.float32)
    end_states = data[:, 2:4].astype(np.float32)
    raw_labels = data[:, 4].astype(int)

    # Convert labels: 0 -> -1 (failure), 1 -> 1 (success)
    labels = np.where(raw_labels == 0, -1, 1)

    return start_states, end_states, labels


def select_epochs(output_dir: Path, n_epochs: int = 4) -> List[int]:
    """
    Select n_epochs from available epochs: first, last, and evenly spaced middle ones.

    Args:
        output_dir: Path to output directory containing epoch_XXX subdirs
        n_epochs: Number of epochs to select (default: 4)

    Returns:
        List of epoch indices to use
    """
    # Find all epoch directories
    epoch_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith("epoch_")
    ])

    # Filter to only complete epochs (with checkpoints AND conformal_state.json)
    complete_epochs = []
    for epoch_dir in epoch_dirs:
        checkpoint_dir = epoch_dir / "checkpoints"
        conformal_state = epoch_dir / "conformal_state.json"
        if (checkpoint_dir.exists() and
            any(checkpoint_dir.glob("*.ckpt")) and
            conformal_state.exists()):
            epoch_idx = int(epoch_dir.name.split("_")[1])
            complete_epochs.append(epoch_idx)

    if len(complete_epochs) < n_epochs:
        print(f"Warning: Only {len(complete_epochs)} complete epochs found, using all")
        return complete_epochs

    # Select first, last, and evenly spaced middle epochs
    first = complete_epochs[0]
    last = complete_epochs[-1]

    if n_epochs == 2:
        return [first, last]
    elif n_epochs == 3:
        middle_idx = len(complete_epochs) // 2
        return [first, complete_epochs[middle_idx], last]
    else:  # n_epochs >= 4
        # For 4 epochs: first, 1/3, 2/3, last
        n_middle = n_epochs - 2
        step = (len(complete_epochs) - 1) / (n_epochs - 1)
        selected = [first]
        for i in range(1, n_epochs - 1):
            idx = int(round(i * step))
            selected.append(complete_epochs[idx])
        selected.append(last)
        return selected


def classify_with_thresholds(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    p_invalid: np.ndarray,
    true_labels: np.ndarray,
    lambda_star: float,
    delta_star: float,
    invalid_threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict]:
    """
    Classify points using simple threshold-based approach.

    Classification logic:
    1. If p_invalid >= invalid_threshold: classify as INVALID (separatrix)
    2. Else if p_success > lambda + delta: classify as SUCCESS
    3. Else if p_success < lambda - delta: classify as FAILURE
    4. Else: classify as UNCERTAIN

    Returns:
        categories: [N] array with values:
            0: Uncertain (gray)
            1: TP (green)
            2: TN (blue)
            3: FP (red)
            4: FN (orange)
        stats: Dict with classification statistics
    """
    N = len(p_success)

    success_threshold = lambda_star + delta_star
    failure_threshold = lambda_star - delta_star

    # Initialize predicted labels
    pred_labels = np.zeros(N, dtype=np.int32)  # 0 = uncertain by default

    # Classify based on thresholds
    # Step 1: Handle invalid (separatrix) predictions
    invalid_mask = p_invalid >= invalid_threshold

    # Step 2: Classify non-invalid points
    success_mask = (~invalid_mask) & (p_success > success_threshold)
    failure_mask = (~invalid_mask) & (p_success < failure_threshold)
    uncertain_mask = (~invalid_mask) & (~success_mask) & (~failure_mask)

    pred_labels[success_mask] = 1
    pred_labels[failure_mask] = -1
    pred_labels[invalid_mask] = 0  # Treat invalid as uncertain for now
    pred_labels[uncertain_mask] = 0

    # Compute categories (TP, TN, FP, FN, Uncertain)
    categories = np.zeros(N, dtype=np.int32)

    # Uncertain: invalid, explicitly uncertain, or empty prediction
    uncertain_total = invalid_mask | uncertain_mask
    categories[uncertain_total] = 0  # Uncertain

    # For confident predictions
    confident_mask = success_mask | failure_mask

    # TP: predicted success, true success
    tp_mask = (pred_labels == 1) & (true_labels == 1)
    categories[tp_mask] = 1

    # TN: predicted failure, true failure
    tn_mask = (pred_labels == -1) & (true_labels == -1)
    categories[tn_mask] = 2

    # FP: predicted success, true failure
    fp_mask = (pred_labels == 1) & (true_labels == -1)
    categories[fp_mask] = 3

    # FN: predicted failure, true success
    fn_mask = (pred_labels == -1) & (true_labels == 1)
    categories[fn_mask] = 4

    # Compute statistics
    stats = {
        'n_total': N,
        'n_uncertain': np.sum(categories == 0),
        'n_tp': np.sum(categories == 1),
        'n_tn': np.sum(categories == 2),
        'n_fp': np.sum(categories == 3),
        'n_fn': np.sum(categories == 4),
        'n_invalid': np.sum(invalid_mask),
        'lambda_star': lambda_star,
        'delta_star': delta_star,
        'success_threshold': success_threshold,
        'failure_threshold': failure_threshold,
    }

    stats['uncertain_rate'] = stats['n_uncertain'] / N
    n_confident = N - stats['n_uncertain']
    if n_confident > 0:
        stats['accuracy'] = (stats['n_tp'] + stats['n_tn']) / n_confident
        stats['precision'] = stats['n_tp'] / (stats['n_tp'] + stats['n_fp']) if (stats['n_tp'] + stats['n_fp']) > 0 else 0
        stats['recall'] = stats['n_tp'] / (stats['n_tp'] + stats['n_fn']) if (stats['n_tp'] + stats['n_fn']) > 0 else 0
    else:
        stats['accuracy'] = 0
        stats['precision'] = 0
        stats['recall'] = 0

    return categories, stats


def classify_points(
    start_states: np.ndarray,
    true_labels: np.ndarray,
    flow_matcher: PendulumLatentConditionalFlowMatcher,
    system: PendulumSystem,
    conformal_config: ConformalConfig,
    lambda_star: float,
    delta_star: float,
    device: str = "cuda"
) -> Tuple[np.ndarray, Dict]:
    """
    Classify points using probability estimation and threshold-based classification.

    Returns:
        categories: [N] array with values:
            0: Uncertain (gray)
            1: TP (green)
            2: TN (blue)
            3: FP (red)
            4: FN (orange)
        stats: Dict with classification statistics
    """
    # Create probability estimator
    prob_estimator = ProbabilityEstimator(flow_matcher, system, conformal_config, device)

    # Estimate probabilities for evaluation set
    print("    Estimating p(success|x) for evaluation set...")
    p_success, p_failure, p_invalid = prob_estimator.estimate(start_states)

    # Classify using thresholds
    print("    Classifying with thresholds...")
    categories, stats = classify_with_thresholds(
        p_success, p_failure, p_invalid, true_labels,
        lambda_star, delta_star
    )

    return categories, stats


def plot_classification(
    start_states: np.ndarray,
    categories: np.ndarray,
    epoch: int,
    stats: Dict,
    save_path: Path,
    dpi: int = 300
):
    """
    Create high-quality phase space classification plot.

    Colors:
        0: Uncertain -> Gray
        1: TP -> Green
        2: TN -> Blue
        3: FP -> Red
        4: FN -> Orange
    """
    # Define colors for each category
    colors = {
        0: '#808080',  # Gray (Uncertain)
        1: '#2ecc71',  # Green (TP)
        2: '#3498db',  # Blue (TN)
        3: '#e74c3c',  # Red (FP)
        4: '#e67e22',  # Orange (FN)
    }

    labels = {
        0: 'Uncertain',
        1: 'TP (True Positive)',
        2: 'TN (True Negative)',
        3: 'FP (False Positive)',
        4: 'FN (False Negative)',
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    # Plot each category (uncertain first, TP last to be on top)
    for cat in [0, 2, 3, 4, 1]:
        mask = categories == cat
        if np.sum(mask) > 0:
            ax.scatter(
                start_states[mask, 0],  # θ
                start_states[mask, 1],  # θ̇
                c=colors[cat],
                s=2,  # Small point size for density
                alpha=0.7,
                label=f"{labels[cat]} ({np.sum(mask):,})",
                rasterized=True  # Rasterize for PDF/high quality
            )

    # Set axis labels with π formatting
    ax.set_xlabel(r'$\theta$ (rad)', fontsize=14)
    ax.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=14)

    # Set x-axis ticks in terms of π
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

    # Set limits
    ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)

    # Add title with stats
    f1 = 2*stats['precision']*stats['recall']/(stats['precision']+stats['recall']) if (stats['precision']+stats['recall']) > 0 else 0
    title = f"Epoch {epoch}: Classification Results\n"
    title += f"δ*={stats['delta_star']:.3f} | Uncertain: {stats['uncertain_rate']:.1%} | "
    title += f"Accuracy: {stats['accuracy']:.1%} | F1: {f1:.1%}"
    ax.set_title(title, fontsize=12)

    # Add legend
    ax.legend(loc='upper right', fontsize=9, markerscale=3)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_combined_evolution(
    all_start_states: np.ndarray,
    all_categories: List[np.ndarray],
    epochs: List[int],
    all_stats: List[Dict],
    save_path: Path,
    dpi: int = 300
):
    """
    Create a combined 2x2 grid showing evolution across epochs.
    """
    # Define colors for each category
    colors = {
        0: '#808080',  # Gray (Uncertain)
        1: '#2ecc71',  # Green (TP)
        2: '#3498db',  # Blue (TN)
        3: '#e74c3c',  # Red (FP)
        4: '#e67e22',  # Orange (FN)
    }

    labels = {
        0: 'Uncertain',
        1: 'TP',
        2: 'TN',
        3: 'FP',
        4: 'FN',
    }

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=dpi)
    axes = axes.flatten()

    for idx, (epoch, categories, stats) in enumerate(zip(epochs, all_categories, all_stats)):
        ax = axes[idx]

        # Plot each category
        for cat in [0, 2, 3, 4, 1]:  # Plot uncertain first, TP last
            mask = categories == cat
            if np.sum(mask) > 0:
                ax.scatter(
                    all_start_states[mask, 0],
                    all_start_states[mask, 1],
                    c=colors[cat],
                    s=1,
                    alpha=0.7,
                    label=f"{labels[cat]}: {np.sum(mask):,}",
                    rasterized=True
                )

        # Axis settings
        ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
        ax.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
        ax.grid(True, alpha=0.3)

        # Title with stats
        f1 = 2*stats['precision']*stats['recall']/(stats['precision']+stats['recall']) if (stats['precision']+stats['recall']) > 0 else 0
        title = f"Epoch {epoch} (δ*={stats['delta_star']:.3f})\n"
        title += f"Uncertain: {stats['uncertain_rate']:.1%} | Acc: {stats['accuracy']:.1%} | F1: {f1:.1%}"
        ax.set_title(title, fontsize=11)

        # Legend
        ax.legend(loc='upper right', fontsize=8, markerscale=4)

    # Overall title
    fig.suptitle('Classification Evolution Across Adaptive Epochs', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved combined plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot classification evolution across adaptive epochs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to adaptive output directory (containing epoch_XXX subdirs)"
    )
    parser.add_argument(
        "--eval_states",
        type=str,
        default="/common/users/shared/pracsys/genMoPlan/data_trajectories/pendulum_lqr_50k/eval_states.txt",
        help="Path to eval_states.txt file"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./classification_evolution",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=4,
        help="Number of epochs to plot (default: 4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (cuda/cpu)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output images"
    )
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=50,
        help="Number of MC samples for probability estimation"
    )

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check paths exist
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    if not Path(args.eval_states).exists():
        print(f"Error: Eval states file not found: {args.eval_states}")
        sys.exit(1)

    # Select epochs
    print(f"\n{'='*60}")
    print("CLASSIFICATION EVOLUTION PLOTTER")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Eval states: {args.eval_states}")

    epochs = select_epochs(output_dir, args.n_epochs)
    print(f"Selected epochs: {epochs}")

    # Load evaluation data
    print("\nLoading evaluation data...")
    start_states, end_states, true_labels = load_eval_states(args.eval_states)
    print(f"  Loaded {len(start_states):,} evaluation points")
    print(f"  Labels: {np.sum(true_labels == 1):,} success, {np.sum(true_labels == -1):,} failure")

    # Initialize system
    print("\nInitializing pendulum system...")
    system = PendulumSystem()

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Process each epoch
    all_categories = []
    all_stats = []

    for epoch in epochs:
        print(f"\n{'='*60}")
        print(f"PROCESSING EPOCH {epoch}")
        print(f"{'='*60}")

        epoch_dir = output_dir / f"epoch_{epoch:03d}"

        # Load checkpoint
        checkpoint_path = find_best_checkpoint(epoch_dir)
        print(f"  Loading checkpoint: {checkpoint_path.name}")

        flow_matcher = PendulumLatentConditionalFlowMatcher.load_from_checkpoint(
            str(checkpoint_path),
            device=device
        )
        flow_matcher.eval()

        # Load conformal state
        conformal_state = load_conformal_state(epoch_dir)
        lambda_star = conformal_state['lambda_star']
        delta_star = conformal_state['delta_star']
        print(f"  λ* = {lambda_star:.4f}, δ* = {delta_star:.4f}")

        # Create conformal config
        config_dict = conformal_state.get('config', {})
        conformal_config = ConformalConfig(
            delta=config_dict.get('delta', 0.05),
            w=config_dict.get('w', 0.9),
            alpha=config_dict.get('alpha', 0.1),
            num_mc_samples=args.num_mc_samples,
            attractor_radius=config_dict.get('attractor_radius', 0.1),
            optimize_mode=config_dict.get('optimize_mode', 'delta'),
            decision_rule=config_dict.get('decision_rule', 'one_sided'),
        )

        # Classify points
        print(f"\n  Classifying {len(start_states):,} points...")
        categories, stats = classify_points(
            start_states,
            true_labels,
            flow_matcher,
            system,
            conformal_config,
            lambda_star,
            delta_star,
            device
        )

        all_categories.append(categories)
        all_stats.append(stats)

        # Print stats
        print(f"\n  Results for Epoch {epoch}:")
        print(f"    Thresholds: success > {stats['success_threshold']:.3f}, failure < {stats['failure_threshold']:.3f}")
        print(f"    Uncertain: {stats['n_uncertain']:,} ({stats['uncertain_rate']:.1%})")
        print(f"    TP: {stats['n_tp']:,}")
        print(f"    TN: {stats['n_tn']:,}")
        print(f"    FP: {stats['n_fp']:,}")
        print(f"    FN: {stats['n_fn']:,}")
        print(f"    Accuracy: {stats['accuracy']:.2%}")
        print(f"    Precision: {stats['precision']:.2%}")
        print(f"    Recall: {stats['recall']:.2%}")

        # Plot individual epoch
        plot_path = save_dir / f"classification_epoch_{epoch:03d}.png"
        plot_classification(start_states, categories, epoch, stats, plot_path, args.dpi)

        # Clear GPU memory
        del flow_matcher
        if device == "cuda":
            torch.cuda.empty_cache()

    # Create combined plot
    combined_path = save_dir / "classification_evolution_combined.png"
    plot_combined_evolution(
        start_states, all_categories, epochs, all_stats,
        combined_path, args.dpi
    )

    # Save statistics to JSON (convert numpy types to Python types)
    stats_path = save_dir / "classification_stats.json"

    def convert_to_python_types(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(i) for i in obj]
        return obj

    stats_to_save = {
        'epochs': epochs,
        'stats': convert_to_python_types(all_stats)
    }
    with open(stats_path, 'w') as f:
        json.dump(stats_to_save, f, indent=2)
    print(f"\nSaved statistics: {stats_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Output saved to: {save_dir}")


if __name__ == "__main__":
    main()

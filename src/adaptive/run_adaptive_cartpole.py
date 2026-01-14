"""
Run Adaptive Sampling Pipeline for CartPole PyBullet.

This script demonstrates the full adaptive sampling loop:
1. Load trajectory data source
2. Build initial endpoint dataset
3. Train flow matcher
4. Run conformal prediction to find uncertain regions
5. Add uncertain trajectories to training set
6. Repeat

Usage:
    python src/adaptive/run_adaptive_cartpole.py
    python src/adaptive/run_adaptive_cartpole.py --config-name=adaptive_cartpole_pybullet
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from pathlib import Path
import json
import shutil
import glob
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from typing import Dict

from src.adaptive.data_source import TrajectoryDataSource, TrajectoryDataSourceConfig
from src.adaptive.dataset_builder import AdaptiveDatasetBuilder
from src.adaptive.balanced_sampler import BalancedUncertainSampler
from src.conformal import ConformalConfig, ConformalPredictor
from src.conformal.probability_estimator import ProbabilityEstimator
from src.data.cartpole_endpoint_data import CartPoleEndpointDataModule

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_cartpole_roa_projections(
    start_states: np.ndarray,
    success_rate: np.ndarray,
    true_labels: np.ndarray,
    lambda_star: float,
    delta: float,
    output_file: str,
    title: str = "CartPole ROA Phase Space Projections",
):
    """
    Plot the Region of Attraction (ROA) for CartPole as 2D projections.

    Since CartPole has 4D state (x, θ, ẋ, θ̇), we create multiple 2D projections:
    - (x, θ): Position vs angle
    - (ẋ, θ̇): Velocity vs angular velocity
    - (θ, θ̇): Pole phase space (similar to pendulum)
    - (x, ẋ): Cart phase space

    Args:
        start_states: [N, 4] array of (x, θ, ẋ, θ̇)
        success_rate: [N] array of p(success) from MC sampling
        true_labels: [N] array of ground truth labels (1=success, -1=failure)
        lambda_star: Optimal decision boundary from conformal prediction
        delta: Unknown region half-width
        output_file: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # State ordering: (x, θ, ẋ, θ̇) at indices 0, 1, 2, 3

    # Classify based on lambda* +/- delta
    pred_labels = np.zeros(len(success_rate))
    pred_labels[success_rate > lambda_star + delta] = 1   # Success
    pred_labels[success_rate < lambda_star - delta] = -1  # Failure

    # Create color arrays
    pred_colors = np.array(['gold'] * len(pred_labels))
    pred_colors[pred_labels == 1] = 'blue'
    pred_colors[pred_labels == -1] = 'red'

    gt_colors = np.array(['gold'] * len(true_labels))
    gt_colors[true_labels == 1] = 'blue'
    gt_colors[true_labels == -1] = 'red'

    # Sort for visualization (uncertain on top)
    pred_order = np.argsort(np.abs(pred_labels))[::-1]
    gt_order = np.argsort(np.abs(true_labels))[::-1]

    # Define projections: (x_idx, y_idx, x_label, y_label, x_lim, y_lim)
    projections = [
        (0, 1, 'x (cart pos)', r'$\theta$ (pole angle)', None, (-np.pi, np.pi)),
        (2, 3, r'$\dot{x}$ (cart vel)', r'$\dot{\theta}$ (pole ang vel)', None, None),
        (1, 3, r'$\theta$ (pole angle)', r'$\dot{\theta}$ (pole ang vel)', (-np.pi, np.pi), None),
        (0, 2, 'x (cart pos)', r'$\dot{x}$ (cart vel)', None, None),
    ]

    for col, (xi, yi, xlabel, ylabel, xlim, ylim) in enumerate(projections):
        # Top row: Predictions
        ax_pred = axes[0, col]
        ax_pred.scatter(
            start_states[pred_order, xi],
            start_states[pred_order, yi],
            c=pred_colors[pred_order],
            s=2, alpha=0.5, edgecolors='none'
        )
        ax_pred.set_xlabel(xlabel, fontsize=10)
        ax_pred.set_ylabel(ylabel, fontsize=10)
        ax_pred.set_title(f'Predicted ({xlabel} vs {ylabel})', fontsize=10)
        if xlim:
            ax_pred.set_xlim(xlim)
        if ylim:
            ax_pred.set_ylim(ylim)
        ax_pred.grid(True, alpha=0.3)
        ax_pred.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax_pred.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

        # Bottom row: Ground Truth
        ax_gt = axes[1, col]
        ax_gt.scatter(
            start_states[gt_order, xi],
            start_states[gt_order, yi],
            c=gt_colors[gt_order],
            s=2, alpha=0.5, edgecolors='none'
        )
        ax_gt.set_xlabel(xlabel, fontsize=10)
        ax_gt.set_ylabel(ylabel, fontsize=10)
        ax_gt.set_title(f'Ground Truth ({xlabel} vs {ylabel})', fontsize=10)
        if xlim:
            ax_gt.set_xlim(xlim)
        if ylim:
            ax_gt.set_ylim(ylim)
        ax_gt.grid(True, alpha=0.3)
        ax_gt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax_gt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # Count labels
    n_pred_success = np.sum(pred_labels == 1)
    n_pred_failure = np.sum(pred_labels == -1)
    n_pred_sep = np.sum(pred_labels == 0)
    n_gt_success = np.sum(true_labels == 1)
    n_gt_failure = np.sum(true_labels == -1)

    # Add legend to first column
    legend_elements = [
        mpatches.Patch(color='blue', label=f'Success'),
        mpatches.Patch(color='red', label=f'Failure'),
        mpatches.Patch(color='gold', label=f'Separatrix'),
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper right', fontsize=8)
    axes[1, 0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.suptitle(f'{title}\n(λ*={lambda_star:.3f}, δ={delta:.3f}) | '
                 f'Pred: S={n_pred_success}, F={n_pred_failure}, Sep={n_pred_sep} | '
                 f'GT: S={n_gt_success}, F={n_gt_failure}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved CartPole ROA projections to: {output_file}")

    return {
        'n_pred_success': int(n_pred_success),
        'n_pred_failure': int(n_pred_failure),
        'n_pred_separatrix': int(n_pred_sep),
        'n_gt_success': int(n_gt_success),
        'n_gt_failure': int(n_gt_failure),
    }


def plot_cartpole_probability_heatmap(
    start_states: np.ndarray,
    success_rate: np.ndarray,
    lambda_star: float,
    delta: float,
    output_file: str,
    title: str = "CartPole ROA - Success Probability",
):
    """
    Plot the ROA as probability heatmaps for each 2D projection.

    Args:
        start_states: [N, 4] array of (x, θ, ẋ, θ̇)
        success_rate: [N] array of p(success) from MC sampling
        lambda_star: Optimal decision boundary
        delta: Unknown region half-width
        output_file: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    projections = [
        (0, 1, 'x (cart pos)', r'$\theta$ (pole angle)', None, (-np.pi, np.pi)),
        (2, 3, r'$\dot{x}$ (cart vel)', r'$\dot{\theta}$ (pole ang vel)', None, None),
        (1, 3, r'$\theta$ (pole angle)', r'$\dot{\theta}$ (pole ang vel)', (-np.pi, np.pi), None),
        (0, 2, 'x (cart pos)', r'$\dot{x}$ (cart vel)', None, None),
    ]

    for col, (xi, yi, xlabel, ylabel, xlim, ylim) in enumerate(projections):
        ax = axes[col]
        scatter = ax.scatter(
            start_states[:, xi],
            start_states[:, yi],
            c=success_rate,
            cmap='RdYlBu',
            s=2, alpha=0.7, edgecolors='none',
            vmin=0, vmax=1
        )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{xlabel} vs {ylabel}', fontsize=10)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

        if col == 3:  # Add colorbar to last plot
            cbar = plt.colorbar(scatter, ax=ax, label='p(success)')
            cbar.ax.axhline(y=lambda_star + delta, color='black', linestyle='--', linewidth=1)
            cbar.ax.axhline(y=lambda_star - delta, color='black', linestyle='--', linewidth=1)
            cbar.ax.axhline(y=lambda_star, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle(f'{title}\n(λ*={lambda_star:.3f}, bounds: [{lambda_star-delta:.3f}, {lambda_star+delta:.3f}])',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved CartPole probability heatmap to: {output_file}")


def compute_metrics_at_threshold(success_rate: np.ndarray, y_all: np.ndarray,
                                  success_thresh: float, failure_thresh: float) -> Dict:
    """Helper to compute metrics at given thresholds (p_success only, old style)."""
    n_total = len(y_all)
    pred_labels = np.zeros(n_total)
    pred_labels[success_rate > success_thresh] = 1
    pred_labels[success_rate < failure_thresh] = -1

    n_uncertain = np.sum(pred_labels == 0)
    separatrix_pct = n_uncertain / n_total

    confident_mask = pred_labels != 0
    n_confident = np.sum(confident_mask)

    y_pred_conf = pred_labels[confident_mask]
    y_true_conf = y_all[confident_mask]

    tp = int(np.sum((y_pred_conf == 1) & (y_true_conf == 1)))
    tn = int(np.sum((y_pred_conf == -1) & (y_true_conf == -1)))
    fp = int(np.sum((y_pred_conf == 1) & (y_true_conf == -1)))
    fn = int(np.sum((y_pred_conf == -1) & (y_true_conf == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n_confident if n_confident > 0 else 0.0

    return {
        'n_confident': int(n_confident),
        'n_uncertain': int(n_uncertain),
        'separatrix_pct': float(separatrix_pct),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'success_threshold': float(success_thresh),
        'failure_threshold': float(failure_thresh),
    }


def compute_metrics_notebook_style(p_success: np.ndarray, p_failure: np.ndarray, 
                                   y_all: np.ndarray, threshold: float = 0.6) -> Dict:
    """
    Compute metrics using notebook-style evaluation (separate success/failure thresholds).
    
    Decision rule (like notebooks/cartpole_eval.ipynb):
    - SUCCESS if p_success > threshold (e.g., 0.6)
    - FAILURE if p_failure > threshold (e.g., 0.6)
    - SEPARATRIX otherwise
    
    This properly handles CartPole's three-way classification where:
    - p_success + p_failure + p_separatrix = 1
    - A point can have low p_success without being failure (high p_separatrix)
    
    Args:
        p_success: [N] array of P(MC label == 1)
        p_failure: [N] array of P(MC label == -1)
        y_all: [N] ground truth labels (-1=failure, 1=success)
        threshold: Confidence threshold (default 0.6)
        
    Returns:
        Dict with evaluation metrics
    """
    n_total = len(y_all)
    
    # Notebook-style classification: use BOTH p_success and p_failure
    pred_labels = np.zeros(n_total)  # Default: separatrix/unknown
    pred_labels[p_success > threshold] = 1   # Success if p_s > threshold
    pred_labels[p_failure > threshold] = -1  # Failure if p_f > threshold
    
    # Handle edge case: if both p_success > threshold and p_failure > threshold
    # This is a confusing/uncertain situation - treat as separatrix
    both_high = (p_success > threshold) & (p_failure > threshold)
    pred_labels[both_high] = 0  # Separatrix
    
    n_uncertain = np.sum(pred_labels == 0)
    separatrix_pct = n_uncertain / n_total
    
    confident_mask = pred_labels != 0
    n_confident = np.sum(confident_mask)
    
    y_pred_conf = pred_labels[confident_mask]
    y_true_conf = y_all[confident_mask]
    
    tp = int(np.sum((y_pred_conf == 1) & (y_true_conf == 1)))
    tn = int(np.sum((y_pred_conf == -1) & (y_true_conf == -1)))
    fp = int(np.sum((y_pred_conf == 1) & (y_true_conf == -1)))
    fn = int(np.sum((y_pred_conf == -1) & (y_true_conf == 1)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n_confident if n_confident > 0 else 0.0
    
    return {
        'n_confident': int(n_confident),
        'n_uncertain': int(n_uncertain),
        'separatrix_pct': float(separatrix_pct),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'threshold': float(threshold),
        'n_pred_success': int(np.sum(pred_labels == 1)),
        'n_pred_failure': int(np.sum(pred_labels == -1)),
        'n_pred_separatrix': int(n_uncertain),
    }


def compute_metrics_lambda_delta_cartpole(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    y_all: np.ndarray,
    lambda_star: float,
    delta: float,
) -> Dict:
    """
    CartPole-specific λ/δ evaluation using BOTH p_success and p_failure.

    Decision rule:
      - SUCCESS if p_success > λ + δ
      - FAILURE if (1 - p_failure) < λ - δ   (equivalently p_failure > 1 - (λ - δ))
      - SEPARATRIX otherwise

    This differs from the legacy p_success-only rule and avoids treating
    low p_success (due to high separatrix probability) as failure.
    """
    n_total = len(y_all)

    success_thresh = float(lambda_star + delta)
    failure_thresh_one_minus_pf = float(lambda_star - delta)

    pred_labels = np.zeros(n_total)  # Default: separatrix/unknown
    pred_labels[p_success > success_thresh] = 1
    pred_labels[(1.0 - p_failure) < failure_thresh_one_minus_pf] = -1

    # If both conditions trigger, treat as separatrix/unknown (rare but possible numerically)
    both = (p_success > success_thresh) & ((1.0 - p_failure) < failure_thresh_one_minus_pf)
    pred_labels[both] = 0

    n_uncertain = int(np.sum(pred_labels == 0))
    separatrix_pct = n_uncertain / n_total

    confident_mask = pred_labels != 0
    n_confident = int(np.sum(confident_mask))

    y_pred_conf = pred_labels[confident_mask]
    y_true_conf = y_all[confident_mask]

    tp = int(np.sum((y_pred_conf == 1) & (y_true_conf == 1)))
    tn = int(np.sum((y_pred_conf == -1) & (y_true_conf == -1)))
    fp = int(np.sum((y_pred_conf == 1) & (y_true_conf == -1)))
    fn = int(np.sum((y_pred_conf == -1) & (y_true_conf == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n_confident if n_confident > 0 else 0.0

    return {
        'n_confident': int(n_confident),
        'n_uncertain': int(n_uncertain),
        'separatrix_pct': float(separatrix_pct),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'lambda_star': float(lambda_star),
        'delta': float(delta),
        'success_threshold': float(success_thresh),
        'failure_threshold_one_minus_pfailure': float(failure_thresh_one_minus_pf),
    }


def evaluate_full_roa_fast(
    flow_matcher,
    system,
    data_source: TrajectoryDataSource,
    num_mc_samples: int = 20,
    batch_size: int = 2048,
    lambda_star: float = None,
    delta: float = 0.05,
    attractor_radius: float = 0.2,
    device: str = 'cuda',
    output_file: str = None,
    verbose: bool = True
) -> Dict:
    """
    Fast batched evaluation on the ENTIRE roa_labels.txt dataset.

    Uses direct batched inference instead of conformal predictor for speed.
    Processes 2048 points at a time with num_mc_samples forward passes.

    Saves per-point data and computes metrics for BOTH threshold schemes:
    1. λ*±δ from conformal prediction (using p_success only)
    2. Notebook-style: success if p_s > 0.6, failure if p_f > 0.6

    Args:
        flow_matcher: Trained flow matcher model
        system: System for classify_attractor
        data_source: TrajectoryDataSource with all labels
        num_mc_samples: Number of MC samples per point
        batch_size: Batch size for GPU inference
        lambda_star: Optimized λ* from conformal prediction (if None, use 0.5)
        delta: Unknown region half-width from conformal config
        attractor_radius: Radius for attractor classification
        device: Device for inference
        output_file: Optional path to save results JSON (also saves .npz with same base name)
        verbose: Print results

    Returns:
        Dict with evaluation metrics for both threshold schemes
    """
    from tqdm import tqdm

    # Get ALL start states and labels from roa_labels.txt
    all_indices = list(range(data_source.n_trajectories))
    X_all = data_source.get_start_states(all_indices)
    y_all = data_source.get_labels(all_indices)

    n_total = len(y_all)

    # Default λ* if not provided
    if lambda_star is None:
        lambda_star = 0.5

    if verbose:
        print(f"\n{'='*60}")
        print("FULL ROA EVALUATION (fast batched)")
        print(f"{'='*60}")
        print(f"Total trajectories: {n_total}")
        print(f"  Success (y=1): {np.sum(y_all == 1)}")
        print(f"  Failure (y=-1): {np.sum(y_all == -1)}")
        print(f"MC samples: {num_mc_samples}, batch_size: {batch_size}")
        print(f"Attractor radius: {attractor_radius}")

    # Convert to tensor
    X_tensor = torch.from_numpy(X_all).float().to(device)

    # Collect MC labels for each point (stores -1, 0, 1 for each sample)
    mc_labels = np.zeros((n_total, num_mc_samples), dtype=np.int8)

    flow_matcher.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, n_total, batch_size), desc="Evaluating", disable=not verbose):
            batch_end = min(batch_start + batch_size, n_total)
            batch_inp = X_tensor[batch_start:batch_end]

            for sample_idx in range(num_mc_samples):
                pred = flow_matcher.predict_endpoint(batch_inp)
                attractor_labels = system.classify_attractor(pred, attractor_radius).cpu().numpy()
                mc_labels[batch_start:batch_end, sample_idx] = attractor_labels

    # Compute probabilities for each class
    p_success = (mc_labels == 1).sum(axis=1) / num_mc_samples   # P(label == 1)
    p_failure = (mc_labels == -1).sum(axis=1) / num_mc_samples  # P(label == -1)
    p_separatrix = (mc_labels == 0).sum(axis=1) / num_mc_samples  # P(label == 0)

    if verbose:
        print(f"\nMC probability statistics:")
        print(f"  p_success:   mean={p_success.mean():.4f}, min={p_success.min():.4f}, max={p_success.max():.4f}")
        print(f"  p_failure:   mean={p_failure.mean():.4f}, min={p_failure.min():.4f}, max={p_failure.max():.4f}")
        print(f"  p_separatrix: mean={p_separatrix.mean():.4f}, min={p_separatrix.min():.4f}, max={p_separatrix.max():.4f}")

    # Compute metrics for BOTH threshold schemes
    # 1. λ*±δ from conformal prediction (CartPole: uses BOTH p_success and p_failure)
    metrics_conformal = compute_metrics_lambda_delta_cartpole(
        p_success=p_success,
        p_failure=p_failure,
        y_all=y_all,
        lambda_star=lambda_star,
        delta=delta,
    )

    # 2. Notebook-style: success if p_s > 0.6, failure if p_f > 0.6
    metrics_notebook = compute_metrics_notebook_style(
        p_success, p_failure, y_all,
        threshold=0.6
    )

    if verbose:
        print(f"\n{'='*60}")
        print("METRICS WITH λ*±δ THRESHOLDS (p_s AND p_f)")
        print(f"{'='*60}")
        print(f"λ*={lambda_star:.4f}, δ={delta:.4f}")
        print(f"  Success if p_success > {lambda_star + delta:.4f}")
        print(f"  Failure if (1 - p_failure) < {lambda_star - delta:.4f}")
        print(f"Separatrix %:    {metrics_conformal['separatrix_pct']:.2%}")
        print(f"Confident:       {metrics_conformal['n_confident']} predictions")
        print(f"Accuracy:        {metrics_conformal['accuracy']:.2%}")
        print(f"F1 Score:        {metrics_conformal['f1']:.2%}")
        print(f"Precision:       {metrics_conformal['precision']:.2%}")
        print(f"Recall:          {metrics_conformal['recall']:.2%}")
        print(f"Specificity:     {metrics_conformal['specificity']:.2%}")

        print(f"\n{'='*60}")
        print("METRICS WITH NOTEBOOK-STYLE THRESHOLDS (p_s AND p_f)")
        print(f"{'='*60}")
        print(f"  Success if p_success > 0.6")
        print(f"  Failure if p_failure > 0.6")
        print(f"Separatrix %:    {metrics_notebook['separatrix_pct']:.2%}")
        print(f"Confident:       {metrics_notebook['n_confident']} predictions")
        print(f"  Pred success:  {metrics_notebook['n_pred_success']}")
        print(f"  Pred failure:  {metrics_notebook['n_pred_failure']}")
        print(f"Accuracy:        {metrics_notebook['accuracy']:.2%}")
        print(f"F1 Score:        {metrics_notebook['f1']:.2%}")
        print(f"Precision:       {metrics_notebook['precision']:.2%}")
        print(f"Recall:          {metrics_notebook['recall']:.2%}")
        print(f"Specificity:     {metrics_notebook['specificity']:.2%}")

    # Combine metrics
    metrics = {
        'n_total': n_total,
        'num_mc_samples': num_mc_samples,
        'attractor_radius': float(attractor_radius),
        'lambda_star': float(lambda_star),
        'delta': float(delta),
        'conformal_thresholds': metrics_conformal,
        'notebook_thresholds': metrics_notebook,
        # Keep top-level metrics for backward compatibility (using notebook-style now)
        'separatrix_pct': metrics_notebook['separatrix_pct'],
        'accuracy': metrics_notebook['accuracy'],
        'precision': metrics_notebook['precision'],
        'recall': metrics_notebook['recall'],
        'specificity': metrics_notebook['specificity'],
        'f1': metrics_notebook['f1'],
    }

    # Save to files if specified
    if output_file:
        # Save JSON metrics
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        if verbose:
            print(f"\nSaved metrics to: {output_file}")

        # Save per-point data as NPZ for analysis (now includes both p_s and p_f)
        npz_file = output_file.replace('.json', '_per_point.npz')
        np.savez(
            npz_file,
            start_states=X_all,           # [N, 4] - (x, θ, ẋ, θ̇)
            p_success=p_success,          # [N] - P(MC label == 1)
            p_failure=p_failure,          # [N] - P(MC label == -1)
            p_separatrix=p_separatrix,    # [N] - P(MC label == 0)
            true_labels=y_all,            # [N] - ground truth labels (1=success, -1=failure)
            lambda_star=lambda_star,
            delta=delta,
            attractor_radius=attractor_radius
        )
        if verbose:
            print(f"Saved per-point data to: {npz_file}")
            print(f"  Arrays: start_states [{X_all.shape}], p_success [{p_success.shape}], p_failure [{p_failure.shape}], true_labels [{y_all.shape}]")

        # Generate ROA phase space projection plots
        phase_space_file = output_file.replace('.json', '_roa_projections.png')
        plot_cartpole_roa_projections(
            start_states=X_all,
            success_rate=p_success,
            true_labels=y_all,
            lambda_star=lambda_star,
            delta=delta,
            output_file=phase_space_file,
            title="CartPole ROA - Epoch Evaluation"
        )

        # Generate probability heatmap
        heatmap_file = output_file.replace('.json', '_roa_heatmap.png')
        plot_cartpole_probability_heatmap(
            start_states=X_all,
            success_rate=p_success,
            lambda_star=lambda_star,
            delta=delta,
            output_file=heatmap_file,
            title="CartPole ROA - Success Probability"
        )

    return metrics


def train_flow_matcher(
    cfg: DictConfig,
    train_file: str,
    val_file: str,
    output_dir: str,
    max_epochs: int = 500,
    resume_checkpoint: str = None
):
    """
    Train a flow matcher on the given dataset files.

    Args:
        cfg: Hydra config with system and model settings
        train_file: Path to training endpoint dataset
        val_file: Path to validation endpoint dataset
        output_dir: Directory for checkpoints and logs
        max_epochs: Maximum training epochs
        resume_checkpoint: Path to checkpoint to resume from (for warm start)

    Returns:
        Trained flow matcher model
    """
    # Instantiate system
    system = hydra.utils.instantiate(cfg.system)

    # Create data module with current dataset files
    data_module = CartPoleEndpointDataModule(
        data_file=train_file,
        validation_file=val_file,
        test_file=val_file,  # Use val as test for now
        batch_size=cfg.get('batch_size', 256),
        val_batch_size=cfg.get('val_batch_size', 2048),
        num_workers=cfg.get('num_workers', 4),
    )

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model)

    # Instantiate flow matcher (matching existing cartpole FM training)
    flow_matcher = hydra.utils.instantiate(
        cfg.flow_matcher,
        system=system,
        model=model,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        latent_dim=cfg.flow_matching.latent_dim,
        mae_val_frequency=cfg.flow_matching.mae_val_frequency,
        _recursive_=False
    )

    # Load weights from previous checkpoint if warm starting
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"🔥 Warm start: Loading weights from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
        state_dict = checkpoint["state_dict"]
        # Load only model weights (not optimizer state)
        model_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        flow_matcher.model.load_state_dict(model_state_dict)
        print(f"   ✓ Loaded model weights ({len(model_state_dict)} tensors)")

    # Setup trainer
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get trainer config
    trainer_cfg = cfg.get('trainer', {})

    # Instantiate callbacks from config
    callbacks = []
    for cb_cfg in trainer_cfg.get('callbacks', []):
        cb = hydra.utils.instantiate(cb_cfg)
        # Set dirpath for ModelCheckpoint (changes per epoch)
        if isinstance(cb, ModelCheckpoint):
            cb.dirpath = str(checkpoint_dir)
        callbacks.append(cb)

    # Fallback if no callbacks in config
    if not callbacks:
        callbacks = [
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                save_last=True,
                filename='best-{epoch:02d}-{val_loss:.4f}'
            ),
        ]

    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="",
        version=None
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,  # Override from function arg
        accelerator=trainer_cfg.get('accelerator', 'gpu') if torch.cuda.is_available() else 'cpu',
        devices=trainer_cfg.get('devices', 1),
        precision=trainer_cfg.get('precision', 32),
        gradient_clip_val=trainer_cfg.get('gradient_clip_val', 1.0),
        log_every_n_steps=trainer_cfg.get('log_every_n_steps', 10),
        check_val_every_n_epoch=trainer_cfg.get('check_val_every_n_epoch', 1),
        enable_progress_bar=trainer_cfg.get('enable_progress_bar', True),
        enable_model_summary=trainer_cfg.get('enable_model_summary', True),
        callbacks=callbacks,
        logger=logger,
    )

    # Train
    trainer.fit(flow_matcher, data_module)

    # Load best checkpoint using custom load_from_checkpoint method
    import glob
    ckpts = glob.glob(str(checkpoint_dir / "best*.ckpt"))
    if ckpts:
        # Use the custom load_from_checkpoint which only takes path and device
        flow_matcher = type(flow_matcher).load_from_checkpoint(
            ckpts[0],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    return flow_matcher


@hydra.main(config_path="../../configs", config_name="adaptive_cartpole_pybullet", version_base=None)
def main(cfg: DictConfig):
    """Main adaptive sampling loop."""
    print("=" * 70)
    print("ADAPTIVE SAMPLING PIPELINE - CartPole PyBullet")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    seed = cfg.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trajectory data source
    data_source_config = TrajectoryDataSourceConfig(
        trajectories_dir=cfg.data_source.trajectories_dir,
        shuffled_indices_file=cfg.data_source.shuffled_indices_file,
        roa_labels_file=cfg.data_source.roa_labels_file,
    )
    data_source = TrajectoryDataSource(data_source_config)

    # Initialize dataset builder
    # Note: No seed needed - sampling is sequential from shuffled_indices.txt
    # Val/test are subsets of training (with overlap)
    dataset_builder = AdaptiveDatasetBuilder(
        data_source=data_source,
        output_dir=str(output_dir / "datasets"),
        val_ratio=cfg.get('val_ratio', 0.1),
        test_ratio=cfg.get('test_ratio', 0.1),
    )

    # Get initial training set
    initial_size = cfg.get('initial_train_size', 100)
    initial_indices = dataset_builder.get_initial_training_set(initial_size)
    print(f"\nInitial training set: {len(initial_indices)} trajectories")

    # Build initial datasets
    dataset_files = dataset_builder.build_all_datasets()
    print(f"Built datasets: {dataset_files}")

    # Conformal prediction config
    conformal_config = ConformalConfig(
        delta=cfg.conformal.get('delta', 0.05),
        w=cfg.conformal.get('w', 0.9),
        alpha=cfg.conformal.get('alpha', 0.1),
        num_mc_samples=cfg.conformal.get('num_mc_samples', 100),
        attractor_radius=cfg.conformal.get('attractor_radius', 0.2),
        # Optimization mode: "lambda" or "delta"
        optimize_mode=cfg.conformal.get('optimize_mode', 'lambda'),
        # Decision rule: "one_sided" (p_s only) or "two_sided" (p_s and p_f)
        decision_rule=cfg.conformal.get('decision_rule', 'two_sided'),  # Default two_sided for CartPole
        lambda_grid_size=cfg.conformal.get('lambda_grid_size', 100),
        delta_grid_size=cfg.conformal.get('delta_grid_size', 100),
        delta_min=cfg.conformal.get('delta_min', 0.01),
        delta_max=cfg.conformal.get('delta_max', 0.49),
    )

    # Instantiate system for conformal prediction
    system = hydra.utils.instantiate(cfg.system)

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Adaptive sampling loop
    n_epochs = cfg.get('n_epochs', 10)
    samples_per_epoch = cfg.get('samples_per_epoch', 50)  # Legacy, used for fixed sampling
    adaptive_data_max = cfg.get('adaptive_data_max', 50)  # Total samples per epoch (D1 + D2)
    d2_ratio = cfg.get('d2_ratio', 0.5)  # Fraction for uncertainty-filtered sampling
    warm_start = cfg.get('warm_start', False)

    epoch_results = []
    previous_best_checkpoint = None  # Track previous epoch's best checkpoint for warm start

    for epoch in range(n_epochs):
        print("\n" + "=" * 70)
        print(f"EPOCH {epoch}")
        print("=" * 70)

        # Step 1: Train flow matcher on current dataset
        print(f"\n[1] Training flow matcher on {len(dataset_builder.train_split)} trajectories...")
        epoch_output_dir = output_dir / f"epoch_{epoch:03d}"
        epoch_output_dir.mkdir(parents=True, exist_ok=True)

        # Determine resume checkpoint for warm start
        resume_ckpt = None
        if warm_start and previous_best_checkpoint:
            resume_ckpt = previous_best_checkpoint
            print(f"    (warm_start=true, resuming from previous epoch)")

        flow_matcher = train_flow_matcher(
            cfg,
            train_file=dataset_files['train'],
            val_file=dataset_files['val'],
            output_dir=str(epoch_output_dir),
            max_epochs=cfg.trainer.get('max_epochs', 1000),
            resume_checkpoint=resume_ckpt,
        )
        flow_matcher.eval()
        flow_matcher.to(device)

        # Track this epoch's best checkpoint for next epoch's warm start
        epoch_ckpts = glob.glob(str(epoch_output_dir / "checkpoints" / "best*.ckpt"))
        if epoch_ckpts:
            previous_best_checkpoint = epoch_ckpts[0]

        # Step 2: Create conformal predictor
        print(f"\n[2] Creating conformal predictor...")
        conformal_predictor = ConformalPredictor(
            flow_matcher=flow_matcher,
            system=system,
            config=conformal_config,
            device=device,
        )

        # Step 3: Get training labels for lambda optimization
        X_train, y_train = dataset_builder.get_train_labels()

        # Split training data for calibration
        n_train = len(y_train)
        cal_ratio = cfg.conformal.get('calibration_ratio', 0.3)
        n_cal = int(n_train * cal_ratio)
        perm = np.random.permutation(n_train)
        cal_idx = perm[:n_cal]
        train_idx = perm[n_cal:]

        X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]
        X_opt, y_opt = X_train[train_idx], y_train[train_idx]

        # Step 4: Fit conformal predictor
        print(f"\n[3] Fitting conformal predictor...")
        conformal_predictor.fit(X_opt, y_opt, X_cal, y_cal, verbose=True)

        # Step 5: Evaluate on test set (subset of training)
        print(f"\n[4] Evaluating on test set...")
        X_test, y_test = dataset_builder.get_test_labels()
        test_metrics = conformal_predictor.evaluate(X_test, y_test, verbose=True)

        # Step 5b: Evaluate on FULL roa_labels.txt (fast batched)
        # Use λ* and δ* from conformal predictor for consistent classification
        num_mc_samples_eval = cfg.conformal.get('num_mc_samples_eval', 20)
        conformal_state = conformal_predictor.get_state()
        lambda_star = conformal_state['lambda_star']
        delta_star = conformal_state['delta_star']  # Use optimized delta (may differ from config if optimize_mode="delta")

        print(f"\n[5] Evaluating on FULL roa_labels.txt ({num_mc_samples_eval} MC samples, fast batched)...")
        print(f"    Using λ*={lambda_star:.4f} ± δ*={delta_star:.4f} from conformal prediction")
        full_roa_output_file = epoch_output_dir / "full_roa_evaluation.json"
        full_roa_metrics = evaluate_full_roa_fast(
            flow_matcher=flow_matcher,
            system=system,
            data_source=data_source,
            num_mc_samples=num_mc_samples_eval,
            batch_size=cfg.get('val_batch_size', 2048),
            lambda_star=lambda_star,
            delta=delta_star,
            attractor_radius=cfg.conformal.get('attractor_radius', 0.2),
            device=device,
            output_file=str(full_roa_output_file),
            verbose=True
        )

        # Step 6: Sample candidates based on strategy
        sampling_strategy = cfg.get('sampling_strategy', 'fixed')

        if sampling_strategy == 'balanced_uncertain':
            # Balanced Uncertain Sampling: sample until |uncertain| == |D1|
            print(f"\n[6] Balanced Uncertain Sampling...")

            # Create probability estimator for uncertainty evaluation
            prob_estimator = ProbabilityEstimator(
                flow_matcher=flow_matcher,
                system=system,
                config=conformal_config,
                device=device
            )

            # Create balanced sampler
            balanced_sampler = BalancedUncertainSampler(
                dataset_builder=dataset_builder,
                adaptive_data_max=adaptive_data_max,
                d2_ratio=d2_ratio,
                batch_size=cfg.get('batch_size_sampling', 50),
                max_samples=cfg.get('max_samples_per_epoch', 50000),
            )

            # Sample epoch (use same decision_rule as conformal predictor)
            decision_rule = cfg.conformal.get('decision_rule', 'two_sided')
            sample_result = balanced_sampler.sample_epoch(
                prob_estimator=prob_estimator,
                lambda_star=lambda_star,
                delta_star=delta_star,
                decision_rule=decision_rule,
                verbose=True
            )

            if len(sample_result.d1_indices) == 0 and len(sample_result.d2_indices) == 0:
                print("No more trajectories available!")
                break

            # Add D1 and D2 to training
            d1_indices = sample_result.d1_indices
            d2_indices = sample_result.d2_indices

            print(f"\n[7] Adding to training...")
            dataset_builder.add_to_training_balanced(d1_indices)
            dataset_builder.add_to_training_balanced(d2_indices)

            n_d2 = len(d2_indices)
            n_confident = sample_result.n_discarded_certain
            n_total_sampled = sample_result.n_total_sampled

            print(f"    Added: D1={len(d1_indices)}, D2={n_d2}, Total={len(d1_indices) + n_d2}")
            print(f"    Discarded (certain, stay in pool): {n_confident}")
            print(f"    Total evaluated: {n_total_sampled} over {sample_result.n_batches} batches")

        else:
            # Fixed sampling (original behavior)
            print(f"\n[6] Fixed Sampling: {samples_per_epoch} candidate trajectories...")
            candidate_states, candidate_indices = dataset_builder.get_candidate_states(samples_per_epoch)

            if len(candidate_indices) == 0:
                print("No more trajectories available!")
                break

            # Split into D1 (calibration) and D2 (selection pool)
            n_d1 = int(len(candidate_indices) * (1 - d2_ratio))
            d1_indices = candidate_indices[:n_d1]
            d2_indices = candidate_indices[n_d1:]
            d2_states = candidate_states[n_d1:]

            print(f"    D1 (always add): {len(d1_indices)} trajectories")
            print(f"    D2 (selective): {len(d2_indices)} trajectories")

            # Always add D1 to training
            dataset_builder.add_selected_to_training(d1_indices)

            # Evaluate D2 for uncertainty
            if len(d2_indices) > 0:
                print(f"\n[7] Evaluating D2 for uncertain points...")
                uncertain_mask, uncertain_idx, p_success, p_failure = conformal_predictor.select_uncertain(d2_states)

                n_uncertain = np.sum(uncertain_mask)
                n_confident = len(d2_indices) - n_uncertain
                print(f"    Uncertain: {n_uncertain} trajectories")
                print(f"    Confident: {n_confident} trajectories (skipped)")

                # Add only uncertain trajectories from D2
                uncertain_traj_indices = [d2_indices[i] for i in range(len(d2_indices)) if uncertain_mask[i]]
                dataset_builder.add_selected_to_training(uncertain_traj_indices)
            else:
                n_uncertain = 0
                n_confident = 0

        # Rebuild datasets with new data
        print(f"\n[8] Rebuilding datasets...")
        dataset_files = dataset_builder.build_all_datasets()

        # Record epoch results
        epoch_result = {
            'epoch': epoch,
            'train_trajectories': len(dataset_builder.train_split),
            'n_d1_added': len(d1_indices),
            'n_d2_added': int(n_d2),
            'n_discarded_certain': int(n_confident),
            # Conformal parameters
            'lambda_star': float(conformal_predictor.lambda_star),
            'delta_star': float(conformal_predictor.delta_star),
            'q_hat': float(conformal_predictor.q_hat),
            'optimize_mode': cfg.conformal.get('optimize_mode', 'lambda'),
            # Test set metrics (subset of training)
            'test_coverage': test_metrics['coverage'],
            'test_f1': test_metrics['f1'],
            'test_unknown_rate': test_metrics['unknown_rate'],
            # Full ROA metrics (entire roa_labels.txt)
            'full_roa': full_roa_metrics,
        }
        epoch_results.append(epoch_result)

        print("\n" + "-" * 70)
        print(f"EPOCH {epoch} SUMMARY")
        print("-" * 70)
        print(f"  Training trajectories: {epoch_result['train_trajectories']}")
        print(f"  Added this epoch: {len(d1_indices) + n_d2} (D1={len(d1_indices)}, D2={n_d2})")
        print(f"  Skipped (confident): {n_confident}")
        print(f"  λ* = {epoch_result['lambda_star']:.4f}, δ* = {epoch_result['delta_star']:.4f}, q_hat = {epoch_result['q_hat']:.4f}")
        print(f"  --- Full ROA (all {full_roa_metrics['n_total']} trajectories) ---")
        conf_m = full_roa_metrics['conformal_thresholds']
        notebook_m = full_roa_metrics['notebook_thresholds']
        print(f"  [λ*±δ] Sep%={conf_m['separatrix_pct']:.1%}, F1={conf_m['f1']:.2%}, Acc={conf_m['accuracy']:.2%}")
        print(f"  [Notebook p_s/p_f>0.6] Sep%={notebook_m['separatrix_pct']:.1%}, F1={notebook_m['f1']:.2%}, Acc={notebook_m['accuracy']:.2%}")

        # Save epoch results
        with open(epoch_output_dir / "results.json", 'w') as f:
            json.dump(epoch_result, f, indent=2)

        # Save conformal predictor state (convert numpy arrays to lists for JSON)
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        conformal_state = conformal_predictor.get_state()
        conformal_state_json = convert_numpy(conformal_state)
        with open(epoch_output_dir / "conformal_state.json", 'w') as f:
            json.dump(conformal_state_json, f, indent=2)

        # Copy Hydra config to epoch directory for reproducibility
        hydra_config_src = Path(cfg.output_dir) / ".hydra"
        if hydra_config_src.exists():
            hydra_config_dst = epoch_output_dir / ".hydra"
            if not hydra_config_dst.exists():
                shutil.copytree(hydra_config_src, hydra_config_dst)

        # Save dataset builder state
        dataset_builder.save_state(str(output_dir / "dataset_builder_state.json"))

    # Final summary
    print("\n" + "=" * 70)
    print("ADAPTIVE SAMPLING COMPLETE")
    print("=" * 70)
    stats = dataset_builder.get_statistics()
    print(f"Final training set: {stats['train_trajectories']} trajectories")
    print(f"Available remaining: {stats['available_trajectories']} trajectories")

    if epoch_results:
        print(f"\n--- Full ROA Metrics Progression (Initial → Final) ---")
        first = epoch_results[0]['full_roa']
        last = epoch_results[-1]['full_roa']
        print(f"\n  [λ*±δ* Thresholds]")
        print(f"  Separatrix %:  {first['conformal_thresholds']['separatrix_pct']:.2%} → {last['conformal_thresholds']['separatrix_pct']:.2%}")
        print(f"  F1 Score:      {first['conformal_thresholds']['f1']:.2%} → {last['conformal_thresholds']['f1']:.2%}")
        print(f"  Accuracy:      {first['conformal_thresholds']['accuracy']:.2%} → {last['conformal_thresholds']['accuracy']:.2%}")
        print(f"\n  [Notebook-Style Thresholds (p_s/p_f > 0.6)]")
        print(f"  Separatrix %:  {first['notebook_thresholds']['separatrix_pct']:.2%} → {last['notebook_thresholds']['separatrix_pct']:.2%}")
        print(f"  F1 Score:      {first['notebook_thresholds']['f1']:.2%} → {last['notebook_thresholds']['f1']:.2%}")
        print(f"  Accuracy:      {first['notebook_thresholds']['accuracy']:.2%} → {last['notebook_thresholds']['accuracy']:.2%}")
        print(f"\n  λ*:            {epoch_results[0]['lambda_star']:.4f} → {epoch_results[-1]['lambda_star']:.4f}")
        print(f"  δ*:            {epoch_results[0]['delta_star']:.4f} → {epoch_results[-1]['delta_star']:.4f}")
        print(f"  q_hat:         {epoch_results[0]['q_hat']:.4f} → {epoch_results[-1]['q_hat']:.4f}")

    # Save final results
    with open(output_dir / "final_results.json", 'w') as f:
        json.dump({
            'epoch_results': epoch_results,
            'final_stats': stats,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

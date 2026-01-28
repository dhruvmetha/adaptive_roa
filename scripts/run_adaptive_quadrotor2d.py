"""
Run Adaptive Sampling Pipeline for Quadrotor 2D.

This script demonstrates the full adaptive sampling loop:
1. Load trajectory data source
2. Build initial endpoint dataset
3. Train flow matcher
4. Run conformal prediction to find uncertain regions
5. Add uncertain trajectories to training set
6. Repeat

Usage:
    python adaptive_roa/adaptive/run_adaptive_quadrotor2d.py
    python adaptive_roa/adaptive/run_adaptive_quadrotor2d.py --config-name=adaptive_quadrotor2d
"""
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# Register custom Hydra resolvers for environment-based paths
from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_data_dir, get_env_config

# Register resolvers before Hydra processes configs
if not OmegaConf.has_resolver("net_id"):
    OmegaConf.register_new_resolver("net_id", lambda: get_net_id())
if not OmegaConf.has_resolver("exp_dir"):
    OmegaConf.register_new_resolver("exp_dir", lambda: get_exp_dir())
if not OmegaConf.has_resolver("data_dir"):
    OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
if not OmegaConf.has_resolver("env"):
    OmegaConf.register_new_resolver("env", lambda key, default="": os.environ.get(key, get_env_config().get(key, default)))
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

from adaptive_roa.adaptive.data_source import TrajectoryDataSource, TrajectoryDataSourceConfig, load_eval_states
from adaptive_roa.adaptive.dataset_builder import AdaptiveDatasetBuilder
from adaptive_roa.adaptive.balanced_sampler import UncertainSampler, RankedSamplingResult
from adaptive_roa.conformal import ConformalConfig, ConformalPredictor
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator
from adaptive_roa.conformal.calibrator import nonconformity_scores_batch_two_sided
from adaptive_roa.adaptive.endpoint_evaluation import (
    sample_endpoint_data_for_optimization,
    compute_endpoint_prediction_error,
)
from adaptive_roa.data.quadrotor2d_endpoint_data import Quadrotor2DEndpointDataModule

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_quadrotor2d_roa_projections(
    start_states: np.ndarray,
    success_rate: np.ndarray,
    true_labels: np.ndarray,
    lambda_star: float,
    delta: float,
    output_file: str,
    title: str = "Quadrotor 2D ROA Phase Space Projections",
):
    """
    Plot the Region of Attraction (ROA) for Quadrotor 2D as 2D projections.

    Since Quadrotor 2D has 6D state (x, z, theta, x_dot, z_dot, theta_dot),
    we create multiple 2D projections:
    - (x, z): Position in XZ plane
    - (x_dot, z_dot): Velocity in XZ plane
    - (theta, theta_dot): Pitch phase space
    - (x, x_dot): Horizontal phase space
    - (z, z_dot): Vertical phase space
    - (x, theta): Position vs angle

    Args:
        start_states: [N, 6] array of (x, z, theta, x_dot, z_dot, theta_dot)
        success_rate: [N] array of p(success) from MC sampling
        true_labels: [N] array of ground truth labels (1=success, -1=failure)
        lambda_star: Optimal decision boundary from conformal prediction
        delta: Unknown region half-width
        output_file: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 6, figsize=(30, 10))

    # State ordering: (x, z, theta, x_dot, z_dot, theta_dot) at indices 0, 1, 2, 3, 4, 5

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
        (0, 1, 'x (horizontal pos)', 'z (vertical pos)', None, None),
        (3, 4, r'$\dot{x}$ (horiz vel)', r'$\dot{z}$ (vert vel)', None, None),
        (2, 5, r'$\theta$ (pitch angle)', r'$\dot{\theta}$ (pitch ang vel)', (-np.pi, np.pi), None),
        (0, 3, 'x (horizontal pos)', r'$\dot{x}$ (horiz vel)', None, None),
        (1, 4, 'z (vertical pos)', r'$\dot{z}$ (vert vel)', None, None),
        (0, 2, 'x (horizontal pos)', r'$\theta$ (pitch angle)', None, (-np.pi, np.pi)),
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

    plt.suptitle(f'{title}\n(lambda*={lambda_star:.3f}, delta={delta:.3f}) | '
                 f'Pred: S={n_pred_success}, F={n_pred_failure}, Sep={n_pred_sep} | '
                 f'GT: S={n_gt_success}, F={n_gt_failure}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Quadrotor 2D ROA projections to: {output_file}")

    return {
        'n_pred_success': int(n_pred_success),
        'n_pred_failure': int(n_pred_failure),
        'n_pred_separatrix': int(n_pred_sep),
        'n_gt_success': int(n_gt_success),
        'n_gt_failure': int(n_gt_failure),
    }


def plot_quadrotor2d_probability_heatmap(
    start_states: np.ndarray,
    success_rate: np.ndarray,
    lambda_star: float,
    delta: float,
    output_file: str,
    title: str = "Quadrotor 2D ROA - Success Probability",
):
    """
    Plot the ROA as probability heatmaps for each 2D projection.

    Args:
        start_states: [N, 6] array of (x, z, theta, x_dot, z_dot, theta_dot)
        success_rate: [N] array of p(success) from MC sampling
        lambda_star: Optimal decision boundary
        delta: Unknown region half-width
        output_file: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))

    projections = [
        (0, 1, 'x (horizontal pos)', 'z (vertical pos)', None, None),
        (3, 4, r'$\dot{x}$ (horiz vel)', r'$\dot{z}$ (vert vel)', None, None),
        (2, 5, r'$\theta$ (pitch angle)', r'$\dot{\theta}$ (pitch ang vel)', (-np.pi, np.pi), None),
        (0, 3, 'x (horizontal pos)', r'$\dot{x}$ (horiz vel)', None, None),
        (1, 4, 'z (vertical pos)', r'$\dot{z}$ (vert vel)', None, None),
        (0, 2, 'x (horizontal pos)', r'$\theta$ (pitch angle)', None, (-np.pi, np.pi)),
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

        if col == 5:  # Add colorbar to last plot
            cbar = plt.colorbar(scatter, ax=ax, label='p(success)')
            cbar.ax.axhline(y=lambda_star + delta, color='black', linestyle='--', linewidth=1)
            cbar.ax.axhline(y=lambda_star - delta, color='black', linestyle='--', linewidth=1)
            cbar.ax.axhline(y=lambda_star, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle(f'{title}\n(lambda*={lambda_star:.3f}, bounds: [{lambda_star-delta:.3f}, {lambda_star+delta:.3f}])',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Quadrotor 2D probability heatmap to: {output_file}")


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
                                   y_all: np.ndarray, threshold: float = 0.6,
                                   p_invalid: np.ndarray = None) -> Dict:
    """
    Compute metrics using notebook-style evaluation (separate success/failure thresholds).

    Two-step classification:
    1. INVALID if p_invalid >= 0.5 (majority of MC samples were neither success nor failure)
    2. From remaining points:
       - SUCCESS if p_success > threshold (e.g., 0.6)
       - FAILURE if p_failure > threshold (e.g., 0.6)
       - UNCERTAIN otherwise (multi-modal)

    This properly handles three-way classification where:
    - p_success + p_failure + p_invalid = 1
    - A point can have low p_success without being failure (high p_invalid)

    Args:
        p_success: [N] array of P(MC label == 1)
        p_failure: [N] array of P(MC label == -1)
        y_all: [N] ground truth labels (-1=failure, 1=success)
        threshold: Confidence threshold (default 0.6)
        p_invalid: [N] array of P(MC label == 0), optional for backward compatibility

    Returns:
        Dict with evaluation metrics
    """
    n_total = len(y_all)

    # Step 1: Identify invalid points (p_invalid >= 0.5)
    if p_invalid is not None:
        is_invalid = p_invalid >= 0.5
        n_invalid = int(np.sum(is_invalid))
    else:
        is_invalid = np.zeros(n_total, dtype=bool)
        n_invalid = 0

    # Step 2: For non-invalid points, apply confidence threshold
    # Initialize all as uncertain (-1 for uncertain, -2 for invalid)
    pred_labels = np.full(n_total, -1)  # Default: uncertain
    pred_labels[is_invalid] = -2  # Mark invalid

    # Only classify non-invalid points
    non_invalid = ~is_invalid
    pred_labels[(p_success > threshold) & non_invalid] = 1   # Success
    pred_labels[(p_failure > threshold) & non_invalid] = -1  # Failure (but not if already success)

    # Handle edge case: if both p_success > threshold and p_failure > threshold
    # This is a confusing/uncertain situation - treat as uncertain
    both_high = (p_success > threshold) & (p_failure > threshold) & non_invalid
    pred_labels[both_high] = -1  # Uncertain

    # Count categories
    n_uncertain = int(np.sum(pred_labels == -1))
    n_confident = int(np.sum((pred_labels == 1) | (pred_labels == 0)))  # success or failure predictions
    # Note: pred_labels uses -1 for failure in ground truth but 0 for uncertain in old code
    # Let me fix: success=1, failure should be marked differently

    # Actually let's be clearer: 1=success, 0=failure prediction, -1=uncertain, -2=invalid
    pred_labels = np.full(n_total, -1)  # Default: uncertain
    pred_labels[is_invalid] = -2  # Mark invalid
    pred_labels[(p_success > threshold) & non_invalid] = 1   # Success
    pred_labels[(p_failure > threshold) & non_invalid] = 0   # Failure

    # Handle edge case: if both thresholds met, mark as uncertain
    both_high = (p_success > threshold) & (p_failure > threshold) & non_invalid
    pred_labels[both_high] = -1  # Uncertain

    n_uncertain = int(np.sum(pred_labels == -1))
    invalid_pct = n_invalid / n_total
    uncertain_pct = n_uncertain / n_total

    # Valid predictions: success (1) or failure (0)
    confident_mask = (pred_labels == 1) | (pred_labels == 0)
    n_confident = int(np.sum(confident_mask))

    # Map predictions to ground truth space: pred 1 -> 1, pred 0 -> -1
    y_pred_conf = np.where(pred_labels[confident_mask] == 1, 1, -1)
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
        'n_invalid': int(n_invalid),
        'n_uncertain': int(n_uncertain),
        'invalid_pct': float(invalid_pct),
        'uncertain_pct': float(uncertain_pct),
        'separatrix_pct': float(invalid_pct + uncertain_pct),  # backward compat
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'threshold': float(threshold),
        'n_pred_success': int(np.sum(pred_labels == 1)),
        'n_pred_failure': int(np.sum(pred_labels == 0)),
        'n_pred_invalid': int(n_invalid),
        'n_pred_uncertain': int(n_uncertain),
    }


def compute_geodesic_error_stats(errors: np.ndarray, mask: np.ndarray = None) -> Dict:
    """
    Compute mean, median, variance for geodesic errors (overall and per-dim).

    Args:
        errors: [N, D] array of geodesic distances per dimension
        mask: Optional boolean mask to select a subset of points

    Returns:
        Dict with n_points, mean/median/variance (overall and per-dim)
    """
    if mask is not None:
        errors = errors[mask]

    if len(errors) == 0:
        return {'n_points': 0}

    # Overall stats: L2 norm across dimensions, then stats over samples
    norms = np.linalg.norm(errors, axis=1)

    return {
        'n_points': int(len(errors)),
        # Overall stats (L2 norm of geodesic error vector)
        'mean': float(np.mean(norms)),
        'median': float(np.median(norms)),
        'variance': float(np.var(norms)),
        # Per-dimension stats
        'mean_per_dim': [float(x) for x in errors.mean(axis=0)],
        'median_per_dim': [float(x) for x in np.median(errors, axis=0)],
        'variance_per_dim': [float(x) for x in np.var(errors, axis=0)],
    }


def compute_hierarchical_error_stats(mc_errors: np.ndarray, mask: np.ndarray = None) -> Dict:
    """
    Compute hierarchical error statistics with all 16 combinations.

    Two-level hierarchy:
    - Level 1: For each start state, aggregate across MC samples (mean, median, P90, P99)
    - Level 2: Aggregate across start states (mean, median, P90, P99)

    Args:
        mc_errors: [N, num_mc_samples] array of L2 norm errors per MC sample
        mask: Optional boolean mask to select a subset of start states

    Returns:
        Dict with n_points and all 16 combinations of statistics
    """
    if mask is not None:
        mc_errors = mc_errors[mask]

    if len(mc_errors) == 0:
        return {'n_points': 0}

    n_points = len(mc_errors)

    # Level 1: Per start state, aggregate across MC samples
    per_state_mean = np.mean(mc_errors, axis=1)      # [N]
    per_state_median = np.median(mc_errors, axis=1)  # [N]
    per_state_p90 = np.percentile(mc_errors, 90, axis=1)  # [N]
    per_state_p99 = np.percentile(mc_errors, 99, axis=1)  # [N]

    level1_stats = {
        'mean': per_state_mean,
        'median': per_state_median,
        'p90': per_state_p90,
        'p99': per_state_p99,
    }

    # Level 2: Aggregate across start states
    result = {'n_points': int(n_points)}

    for l1_name, l1_arr in level1_stats.items():
        # Mean over start states
        result[f'mean_of_{l1_name}s'] = float(np.mean(l1_arr))
        # Median over start states
        result[f'median_of_{l1_name}s'] = float(np.median(l1_arr))
        # P90 over start states
        result[f'p90_of_{l1_name}s'] = float(np.percentile(l1_arr, 90))
        # P99 over start states
        result[f'p99_of_{l1_name}s'] = float(np.percentile(l1_arr, 99))

    return result


def compute_metrics_lambda_delta_quadrotor2d(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    y_all: np.ndarray,
    lambda_star: float,
    delta: float,
    p_invalid: np.ndarray = None,
) -> Dict:
    """
    Quadrotor 2D-specific lambda/delta evaluation using BOTH p_success and p_failure.

    Two-step classification:
    1. INVALID if p_invalid >= 0.5 (majority of MC samples were neither success nor failure)
    2. From remaining points:
       - SUCCESS if p_success > lambda + delta
       - FAILURE if (1 - p_failure) < lambda - delta   (equivalently p_failure > 1 - (lambda - delta))
       - UNCERTAIN otherwise (multi-modal)

    This differs from the legacy p_success-only rule and avoids treating
    low p_success (due to high separatrix probability) as failure.
    """
    n_total = len(y_all)

    # Step 1: Identify invalid points (p_invalid >= 0.5)
    if p_invalid is not None:
        is_invalid = p_invalid >= 0.5
        n_invalid = int(np.sum(is_invalid))
    else:
        is_invalid = np.zeros(n_total, dtype=bool)
        n_invalid = 0

    success_thresh = float(lambda_star + delta)
    failure_thresh_one_minus_pf = float(lambda_star - delta)

    # Step 2: For non-invalid points, apply lambda/delta thresholds
    # pred_labels: 1=success, 0=failure, -1=uncertain, -2=invalid
    pred_labels = np.full(n_total, -1)  # Default: uncertain
    pred_labels[is_invalid] = -2  # Mark invalid

    non_invalid = ~is_invalid
    pred_labels[(p_success > success_thresh) & non_invalid] = 1
    pred_labels[((1.0 - p_failure) < failure_thresh_one_minus_pf) & non_invalid] = 0

    # If both conditions trigger, treat as uncertain (rare but possible numerically)
    both = (p_success > success_thresh) & ((1.0 - p_failure) < failure_thresh_one_minus_pf) & non_invalid
    pred_labels[both] = -1

    n_uncertain = int(np.sum(pred_labels == -1))
    invalid_pct = n_invalid / n_total
    uncertain_pct = n_uncertain / n_total

    # Valid predictions: success (1) or failure (0)
    confident_mask = (pred_labels == 1) | (pred_labels == 0)
    n_confident = int(np.sum(confident_mask))

    # Map predictions to ground truth space: pred 1 -> 1, pred 0 -> -1
    y_pred_conf = np.where(pred_labels[confident_mask] == 1, 1, -1)
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
        'n_invalid': int(n_invalid),
        'n_uncertain': int(n_uncertain),
        'invalid_pct': float(invalid_pct),
        'uncertain_pct': float(uncertain_pct),
        'separatrix_pct': float(invalid_pct + uncertain_pct),  # backward compat
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


def compute_metrics_qhat_conformal(
    p_success: np.ndarray,
    p_failure: np.ndarray,
    y_all: np.ndarray,
    lambda_star: float,
    delta: float,
    q_hat: float,
    p_invalid: np.ndarray = None,
) -> Dict:
    """
    Compute metrics using q_hat-based conformal prediction sets.

    This uses the calibrated q_hat threshold to construct prediction sets,
    providing coverage guarantees. A label is included in the prediction set
    if its non-conformity score is <= q_hat.

    Classification:
    1. INVALID if p_invalid >= 0.5
    2. For non-invalid points, compute prediction sets:
       - Confident SUCCESS: prediction set == {1}
       - Confident FAILURE: prediction set == {-1}
       - Uncertain: prediction set has multiple labels or is empty

    Args:
        p_success: [N] array of P(MC label == 1)
        p_failure: [N] array of P(MC label == -1)
        y_all: [N] ground truth labels (-1=failure, 1=success)
        lambda_star: Optimal decision boundary from conformal prediction
        delta: Uncertainty half-width
        q_hat: Calibration threshold from conformal prediction
        p_invalid: [N] array of P(MC label == 0), optional

    Returns:
        Dict with evaluation metrics including coverage
    """
    n_total = len(y_all)

    # Step 1: Identify invalid points (p_invalid >= 0.5)
    if p_invalid is not None:
        is_invalid = p_invalid >= 0.5
        n_invalid = int(np.sum(is_invalid))
    else:
        is_invalid = np.zeros(n_total, dtype=bool)
        n_invalid = 0

    # Step 2: Compute non-conformity scores for each candidate label
    # For efficiency, compute scores for all labels at once using vectorized function

    # Create arrays for each candidate label
    n = n_total
    scores_success = nonconformity_scores_batch_two_sided(
        p_success, p_failure, np.ones(n, dtype=int), lambda_star, delta
    )
    scores_failure = nonconformity_scores_batch_two_sided(
        p_success, p_failure, -np.ones(n, dtype=int), lambda_star, delta
    )
    scores_unknown = nonconformity_scores_batch_two_sided(
        p_success, p_failure, np.zeros(n, dtype=int), lambda_star, delta
    )

    # Step 3: Build prediction sets (label included if score <= q_hat)
    in_set_success = scores_success <= q_hat  # [N] bool
    in_set_failure = scores_failure <= q_hat  # [N] bool
    in_set_unknown = scores_unknown <= q_hat  # [N] bool

    # Prediction set sizes
    set_sizes = in_set_success.astype(int) + in_set_failure.astype(int) + in_set_unknown.astype(int)

    # Step 4: Classify based on prediction sets (for non-invalid points)
    # pred_labels: 1=success, 0=failure, -1=uncertain, -2=invalid
    pred_labels = np.full(n_total, -1)  # Default: uncertain
    pred_labels[is_invalid] = -2  # Mark invalid

    non_invalid = ~is_invalid

    # Confident SUCCESS: only {1} in prediction set
    confident_success = in_set_success & ~in_set_failure & ~in_set_unknown & non_invalid
    pred_labels[confident_success] = 1

    # Confident FAILURE: only {-1} in prediction set
    confident_failure = ~in_set_success & in_set_failure & ~in_set_unknown & non_invalid
    pred_labels[confident_failure] = 0  # Using 0 to represent failure prediction

    # Everything else (multiple labels or empty set) remains uncertain (-1)

    n_uncertain = int(np.sum(pred_labels == -1))
    invalid_pct = n_invalid / n_total
    uncertain_pct = n_uncertain / n_total

    # Valid predictions: success (1) or failure (0)
    confident_mask = (pred_labels == 1) | (pred_labels == 0)
    n_confident = int(np.sum(confident_mask))

    # Map predictions to ground truth space: pred 1 -> 1, pred 0 -> -1
    y_pred_conf = np.where(pred_labels[confident_mask] == 1, 1, -1)
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

    # Compute coverage: fraction of points where true label is in prediction set
    # Map y_all to the in_set arrays
    true_in_set = np.zeros(n_total, dtype=bool)
    true_in_set[y_all == 1] = in_set_success[y_all == 1]
    true_in_set[y_all == -1] = in_set_failure[y_all == -1]
    # For y_all == 0 (if any), check in_set_unknown
    true_in_set[y_all == 0] = in_set_unknown[y_all == 0]

    coverage = np.mean(true_in_set)

    # Average prediction set size (excluding invalid)
    avg_set_size = np.mean(set_sizes[non_invalid]) if np.sum(non_invalid) > 0 else 0.0

    return {
        'n_confident': int(n_confident),
        'n_invalid': int(n_invalid),
        'n_uncertain': int(n_uncertain),
        'invalid_pct': float(invalid_pct),
        'uncertain_pct': float(uncertain_pct),
        'separatrix_pct': float(invalid_pct + uncertain_pct),  # backward compat
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'lambda_star': float(lambda_star),
        'delta': float(delta),
        'q_hat': float(q_hat),
        'coverage': float(coverage),
        'avg_set_size': float(avg_set_size),
        'n_pred_success': int(np.sum(pred_labels == 1)),
        'n_pred_failure': int(np.sum(pred_labels == 0)),
    }


def evaluate_full_roa_fast(
    flow_matcher,
    system,
    eval_states_file: str,
    num_mc_samples: int = 20,
    batch_size: int = 2048,
    lambda_star: float = None,
    delta: float = 0.05,
    q_hat: float = None,
    attractor_radius: float = 0.2,
    device: str = 'cuda',
    output_file: str = None,
    verbose: bool = True
) -> Dict:
    """
    Fast batched evaluation on the ENTIRE eval_states.txt dataset.

    Uses direct batched inference instead of conformal predictor for speed.
    Processes 2048 points at a time with num_mc_samples forward passes.

    Saves per-point data and computes metrics for THREE threshold schemes:
    1. lambda*+/-delta from conformal prediction (using p_success and p_failure)
    2. Notebook-style: success if p_s > 0.6, failure if p_f > 0.6
    3. q_hat conformal: uses calibrated q_hat to construct prediction sets (if q_hat provided)

    Args:
        flow_matcher: Trained flow matcher model
        system: System for classify_attractor
        eval_states_file: Path to eval_states.txt (CSV: start_state..., end_state..., label)
        num_mc_samples: Number of MC samples per point
        batch_size: Batch size for GPU inference
        lambda_star: Optimized lambda* from conformal prediction (if None, use 0.5)
        delta: Unknown region half-width from conformal config
        q_hat: Calibration threshold from conformal prediction (if None, skip q_hat-based eval)
        attractor_radius: Radius for attractor classification
        device: Device for inference
        output_file: Optional path to save results JSON (also saves .npz with same base name)
        verbose: Print results

    Returns:
        Dict with evaluation metrics for all threshold schemes
    """
    from tqdm import tqdm

    # Load eval_states file (contains start_states, end_states, labels)
    X_all, end_states_all, y_all = load_eval_states(eval_states_file)

    n_total = len(y_all)

    # Default lambda* if not provided
    if lambda_star is None:
        lambda_star = 0.5

    if verbose:
        print(f"\n{'='*60}")
        print("FULL ROA EVALUATION (fast batched) - Quadrotor 2D")
        print(f"{'='*60}")
        print(f"Total trajectories: {n_total}")
        print(f"  Success (y=1): {np.sum(y_all == 1)}")
        print(f"  Failure (y=-1): {np.sum(y_all == -1)}")
        print(f"  Separatrix (y=0): {np.sum(y_all == 0)}")
        print(f"MC samples: {num_mc_samples}, batch_size: {batch_size}")
        print(f"Attractor radius: {attractor_radius}")

    # Convert to tensors
    X_tensor = torch.from_numpy(X_all).float().to(device)
    end_states_tensor = torch.from_numpy(end_states_all).float().to(device)

    # Collect MC labels for each point (stores -1, 0, 1 for each sample)
    mc_labels = np.zeros((n_total, num_mc_samples), dtype=np.int8)

    # Collect predicted endpoints for endpoint error computation
    # Store mean prediction per point (average across MC samples)
    pred_endpoints_sum = np.zeros((n_total, end_states_all.shape[1]), dtype=np.float64)

    # Store per-MC-sample errors for hierarchical statistics [N, num_mc_samples]
    mc_errors = np.zeros((n_total, num_mc_samples), dtype=np.float32)

    flow_matcher.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, n_total, batch_size), desc="Evaluating", disable=not verbose):
            batch_end = min(batch_start + batch_size, n_total)
            batch_inp = X_tensor[batch_start:batch_end]
            batch_actual = end_states_tensor[batch_start:batch_end]

            for sample_idx in range(num_mc_samples):
                pred = flow_matcher.predict_endpoint(batch_inp)
                attractor_labels = system.classify_attractor(pred, attractor_radius).cpu().numpy()
                mc_labels[batch_start:batch_end, sample_idx] = attractor_labels
                # Accumulate predictions for mean endpoint computation
                pred_endpoints_sum[batch_start:batch_end] += pred.cpu().numpy()
                # Compute per-sample geodesic error (L2 norm across dimensions)
                geodesic_dist = flow_matcher.manifold.dist(pred, batch_actual).cpu().numpy()
                mc_errors[batch_start:batch_end, sample_idx] = np.linalg.norm(geodesic_dist, axis=1)

    # Compute mean predicted endpoints
    pred_endpoints_mean = pred_endpoints_sum / num_mc_samples

    # Compute endpoint errors using manifold's geodesic distance (for legacy stats)
    pred_tensor = torch.from_numpy(pred_endpoints_mean).float().to(device)
    actual_tensor = torch.from_numpy(end_states_all).float().to(device)
    geodesic_errors = flow_matcher.manifold.dist(pred_tensor, actual_tensor).cpu().numpy()

    # Get component names for reporting
    component_names = flow_matcher.get_manifold_component_names()

    # Compute probabilities for each class
    p_success = (mc_labels == 1).sum(axis=1) / num_mc_samples   # P(label == 1)
    p_failure = (mc_labels == -1).sum(axis=1) / num_mc_samples  # P(label == -1)
    p_invalid = (mc_labels == 0).sum(axis=1) / num_mc_samples   # P(label == 0)

    # Classification masks using conformal thresholds (lambda* +/- delta)
    success_thresh = lambda_star + delta
    failure_thresh = lambda_star - delta

    mask_invalid = p_invalid >= 0.5
    mask_success = (p_success > success_thresh) & ~mask_invalid
    mask_failure = ((1 - p_failure) < failure_thresh) & ~mask_invalid
    mask_certain = mask_success | mask_failure
    mask_uncertain = ~mask_invalid & ~mask_success & ~mask_failure

    # Compute endpoint error stats for each category (legacy: using mean prediction)
    error_stats_full = compute_geodesic_error_stats(geodesic_errors)
    error_stats_certain = compute_geodesic_error_stats(geodesic_errors, mask_certain)
    error_stats_certain_success = compute_geodesic_error_stats(geodesic_errors, mask_success)
    error_stats_certain_failure = compute_geodesic_error_stats(geodesic_errors, mask_failure)
    error_stats_uncertain = compute_geodesic_error_stats(geodesic_errors, mask_uncertain)
    error_stats_invalid = compute_geodesic_error_stats(geodesic_errors, mask_invalid)

    # Compute hierarchical error stats (all 16 combinations) for each category
    hierarchical_stats_full = compute_hierarchical_error_stats(mc_errors)
    hierarchical_stats_certain = compute_hierarchical_error_stats(mc_errors, mask_certain)
    hierarchical_stats_certain_success = compute_hierarchical_error_stats(mc_errors, mask_success)
    hierarchical_stats_certain_failure = compute_hierarchical_error_stats(mc_errors, mask_failure)
    hierarchical_stats_uncertain = compute_hierarchical_error_stats(mc_errors, mask_uncertain)
    hierarchical_stats_invalid = compute_hierarchical_error_stats(mc_errors, mask_invalid)

    if verbose:
        print(f"\nMC probability statistics:")
        print(f"  p_success:   mean={p_success.mean():.4f}, min={p_success.min():.4f}, max={p_success.max():.4f}")
        print(f"  p_failure:   mean={p_failure.mean():.4f}, min={p_failure.min():.4f}, max={p_failure.max():.4f}")
        print(f"  p_invalid:   mean={p_invalid.mean():.4f}, min={p_invalid.min():.4f}, max={p_invalid.max():.4f}")

        print(f"\n{'='*60}")
        print("GEODESIC ENDPOINT ERROR METRICS (predicted vs actual)")
        print(f"{'='*60}")
        print(f"Component names: {component_names}")

        for cat_name, stats in [("FULL", error_stats_full),
                                 ("CERTAIN", error_stats_certain),
                                 ("  CERTAIN_SUCCESS", error_stats_certain_success),
                                 ("  CERTAIN_FAILURE", error_stats_certain_failure),
                                 ("UNCERTAIN", error_stats_uncertain),
                                 ("INVALID", error_stats_invalid)]:
            print(f"\n  [{cat_name}] (n={stats['n_points']})")
            if stats['n_points'] > 0:
                print(f"    Overall: mean={stats['mean']:.6f}, median={stats['median']:.6f}, var={stats['variance']:.6f}")
                print(f"    Per-dim mean:   {[f'{x:.4f}' for x in stats['mean_per_dim']]}")
                print(f"    Per-dim median: {[f'{x:.4f}' for x in stats['median_per_dim']]}")
                print(f"    Per-dim var:    {[f'{x:.4f}' for x in stats['variance_per_dim']]}")

        print(f"\n{'='*60}")
        print("HIERARCHICAL ERROR STATS (16 combinations: L1 over MC samples, L2 over start states)")
        print(f"{'='*60}")

        for cat_name, stats in [("FULL", hierarchical_stats_full),
                                 ("CERTAIN", hierarchical_stats_certain),
                                 ("  CERTAIN_SUCCESS", hierarchical_stats_certain_success),
                                 ("  CERTAIN_FAILURE", hierarchical_stats_certain_failure),
                                 ("UNCERTAIN", hierarchical_stats_uncertain),
                                 ("INVALID", hierarchical_stats_invalid)]:
            print(f"\n  [{cat_name}] (n={stats['n_points']})")
            if stats['n_points'] > 0:
                # Print in a table format for clarity
                print(f"    {'L2/L1':<8} {'mean':>12} {'median':>12} {'p90':>12} {'p99':>12}")
                print(f"    {'-'*56}")
                for l2_agg in ['mean', 'median', 'p90', 'p99']:
                    row = f"    {l2_agg:<8}"
                    for l1_agg in ['mean', 'median', 'p90', 'p99']:
                        key = f"{l2_agg}_of_{l1_agg}s"
                        row += f" {stats[key]:>12.6f}"
                    print(row)

    # Compute metrics for BOTH threshold schemes
    # 1. lambda*+/-delta from conformal prediction (Quadrotor 2D: uses BOTH p_success and p_failure)
    metrics_conformal = compute_metrics_lambda_delta_quadrotor2d(
        p_success=p_success,
        p_failure=p_failure,
        y_all=y_all,
        lambda_star=lambda_star,
        delta=delta,
        p_invalid=p_invalid,
    )

    # 2. Notebook-style: success if p_s > 0.6, failure if p_f > 0.6
    metrics_notebook = compute_metrics_notebook_style(
        p_success, p_failure, y_all,
        threshold=0.6,
        p_invalid=p_invalid,
    )

    # 3. q_hat-based conformal prediction sets (if q_hat provided)
    if q_hat is not None:
        metrics_qhat_conformal = compute_metrics_qhat_conformal(
            p_success=p_success,
            p_failure=p_failure,
            y_all=y_all,
            lambda_star=lambda_star,
            delta=delta,
            q_hat=q_hat,
            p_invalid=p_invalid,
        )

        # Compute q_hat-based classification masks for geodesic error stats
        # Recompute non-conformity scores to get prediction sets
        n = len(y_all)
        scores_success_qhat = nonconformity_scores_batch_two_sided(
            p_success, p_failure, np.ones(n, dtype=int), lambda_star, delta
        )
        scores_failure_qhat = nonconformity_scores_batch_two_sided(
            p_success, p_failure, -np.ones(n, dtype=int), lambda_star, delta
        )
        scores_unknown_qhat = nonconformity_scores_batch_two_sided(
            p_success, p_failure, np.zeros(n, dtype=int), lambda_star, delta
        )

        # Build prediction set membership
        in_set_success_qhat = scores_success_qhat <= q_hat
        in_set_failure_qhat = scores_failure_qhat <= q_hat
        in_set_unknown_qhat = scores_unknown_qhat <= q_hat

        # Classification masks based on q_hat prediction sets
        mask_invalid_qhat = p_invalid >= 0.5
        mask_success_qhat = in_set_success_qhat & ~in_set_failure_qhat & ~in_set_unknown_qhat & ~mask_invalid_qhat
        mask_failure_qhat = ~in_set_success_qhat & in_set_failure_qhat & ~in_set_unknown_qhat & ~mask_invalid_qhat
        mask_certain_qhat = mask_success_qhat | mask_failure_qhat
        mask_uncertain_qhat = ~mask_invalid_qhat & ~mask_success_qhat & ~mask_failure_qhat

        # Compute geodesic error stats for q_hat-based classes
        error_stats_certain_qhat = compute_geodesic_error_stats(geodesic_errors, mask_certain_qhat)
        error_stats_certain_success_qhat = compute_geodesic_error_stats(geodesic_errors, mask_success_qhat)
        error_stats_certain_failure_qhat = compute_geodesic_error_stats(geodesic_errors, mask_failure_qhat)
        error_stats_uncertain_qhat = compute_geodesic_error_stats(geodesic_errors, mask_uncertain_qhat)
        error_stats_invalid_qhat = compute_geodesic_error_stats(geodesic_errors, mask_invalid_qhat)

        # Compute hierarchical error stats for q_hat-based classes
        hierarchical_stats_certain_qhat = compute_hierarchical_error_stats(mc_errors, mask_certain_qhat)
        hierarchical_stats_certain_success_qhat = compute_hierarchical_error_stats(mc_errors, mask_success_qhat)
        hierarchical_stats_certain_failure_qhat = compute_hierarchical_error_stats(mc_errors, mask_failure_qhat)
        hierarchical_stats_uncertain_qhat = compute_hierarchical_error_stats(mc_errors, mask_uncertain_qhat)
        hierarchical_stats_invalid_qhat = compute_hierarchical_error_stats(mc_errors, mask_invalid_qhat)
    else:
        metrics_qhat_conformal = None
        error_stats_certain_qhat = None
        error_stats_certain_success_qhat = None
        error_stats_certain_failure_qhat = None
        error_stats_uncertain_qhat = None
        error_stats_invalid_qhat = None
        hierarchical_stats_certain_qhat = None
        hierarchical_stats_certain_success_qhat = None
        hierarchical_stats_certain_failure_qhat = None
        hierarchical_stats_uncertain_qhat = None
        hierarchical_stats_invalid_qhat = None

    if verbose:
        print(f"\n{'='*60}")
        print("METRICS WITH lambda*+/-delta THRESHOLDS (p_s AND p_f)")
        print(f"{'='*60}")
        print(f"lambda*={lambda_star:.4f}, delta={delta:.4f}")
        print(f"  Success if p_success > {lambda_star + delta:.4f}")
        print(f"  Failure if (1 - p_failure) < {lambda_star - delta:.4f}")
        print(f"Invalid %:       {metrics_conformal['invalid_pct']:.2%} (p_invalid >= 0.5)")
        print(f"Uncertain %:     {metrics_conformal['uncertain_pct']:.2%} (multi-modal)")
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
        print(f"Invalid %:       {metrics_notebook['invalid_pct']:.2%} (p_invalid >= 0.5)")
        print(f"Uncertain %:     {metrics_notebook['uncertain_pct']:.2%} (multi-modal)")
        print(f"Confident:       {metrics_notebook['n_confident']} predictions")
        print(f"  Pred success:  {metrics_notebook['n_pred_success']}")
        print(f"  Pred failure:  {metrics_notebook['n_pred_failure']}")
        print(f"Accuracy:        {metrics_notebook['accuracy']:.2%}")
        print(f"F1 Score:        {metrics_notebook['f1']:.2%}")
        print(f"Precision:       {metrics_notebook['precision']:.2%}")
        print(f"Recall:          {metrics_notebook['recall']:.2%}")
        print(f"Specificity:     {metrics_notebook['specificity']:.2%}")

        if metrics_qhat_conformal is not None:
            print(f"\n{'='*60}")
            print("METRICS WITH q_hat CONFORMAL PREDICTION SETS")
            print(f"{'='*60}")
            print(f"lambda*={lambda_star:.4f}, delta={delta:.4f}, q_hat={q_hat:.4f}")
            print(f"  Label in prediction set if non-conformity score <= q_hat")
            print(f"  Confident SUCCESS: prediction set == {{1}}")
            print(f"  Confident FAILURE: prediction set == {{-1}}")
            print(f"Invalid %:       {metrics_qhat_conformal['invalid_pct']:.2%} (p_invalid >= 0.5)")
            print(f"Uncertain %:     {metrics_qhat_conformal['uncertain_pct']:.2%} (multi-label prediction set)")
            print(f"Confident:       {metrics_qhat_conformal['n_confident']} predictions")
            print(f"  Pred success:  {metrics_qhat_conformal['n_pred_success']}")
            print(f"  Pred failure:  {metrics_qhat_conformal['n_pred_failure']}")
            print(f"Accuracy:        {metrics_qhat_conformal['accuracy']:.2%}")
            print(f"F1 Score:        {metrics_qhat_conformal['f1']:.2%}")
            print(f"Precision:       {metrics_qhat_conformal['precision']:.2%}")
            print(f"Recall:          {metrics_qhat_conformal['recall']:.2%}")
            print(f"Specificity:     {metrics_qhat_conformal['specificity']:.2%}")
            print(f"Coverage:        {metrics_qhat_conformal['coverage']:.2%} (true label in pred set)")
            print(f"Avg set size:    {metrics_qhat_conformal['avg_set_size']:.2f}")

            # Print q_hat-based geodesic error stats
            print(f"\n{'='*60}")
            print("GEODESIC ENDPOINT ERROR METRICS (q_hat classification)")
            print(f"{'='*60}")
            print(f"Component names: {component_names}")

            for cat_name, stats in [("CERTAIN (q_hat)", error_stats_certain_qhat),
                                     ("  CERTAIN_SUCCESS (q_hat)", error_stats_certain_success_qhat),
                                     ("  CERTAIN_FAILURE (q_hat)", error_stats_certain_failure_qhat),
                                     ("UNCERTAIN (q_hat)", error_stats_uncertain_qhat),
                                     ("INVALID (q_hat)", error_stats_invalid_qhat)]:
                print(f"\n  [{cat_name}] (n={stats['n_points']})")
                if stats['n_points'] > 0:
                    print(f"    Overall: mean={stats['mean']:.6f}, median={stats['median']:.6f}, var={stats['variance']:.6f}")
                    print(f"    Per-dim mean:   {[f'{x:.4f}' for x in stats['mean_per_dim']]}")
                    print(f"    Per-dim median: {[f'{x:.4f}' for x in stats['median_per_dim']]}")
                    print(f"    Per-dim var:    {[f'{x:.4f}' for x in stats['variance_per_dim']]}")

            # Print q_hat-based hierarchical error stats
            print(f"\n{'='*60}")
            print("HIERARCHICAL ERROR STATS (q_hat classification)")
            print(f"{'='*60}")

            for cat_name, stats in [("CERTAIN (q_hat)", hierarchical_stats_certain_qhat),
                                     ("  CERTAIN_SUCCESS (q_hat)", hierarchical_stats_certain_success_qhat),
                                     ("  CERTAIN_FAILURE (q_hat)", hierarchical_stats_certain_failure_qhat),
                                     ("UNCERTAIN (q_hat)", hierarchical_stats_uncertain_qhat),
                                     ("INVALID (q_hat)", hierarchical_stats_invalid_qhat)]:
                print(f"\n  [{cat_name}] (n={stats['n_points']})")
                if stats['n_points'] > 0:
                    print(f"    {'L2/L1':<8} {'mean':>12} {'median':>12} {'p90':>12} {'p99':>12}")
                    print(f"    {'-'*56}")
                    for l2_agg in ['mean', 'median', 'p90', 'p99']:
                        row = f"    {l2_agg:<8}"
                        for l1_agg in ['mean', 'median', 'p90', 'p99']:
                            key = f"{l2_agg}_of_{l1_agg}s"
                            row += f" {stats[key]:>12.6f}"
                        print(row)

    # Combine metrics
    metrics = {
        'n_total': n_total,
        'num_mc_samples': num_mc_samples,
        'attractor_radius': float(attractor_radius),
        'lambda_star': float(lambda_star),
        'delta': float(delta),
        'conformal_thresholds': metrics_conformal,
        'notebook_thresholds': metrics_notebook,
        'qhat_conformal_thresholds': metrics_qhat_conformal,
        'q_hat': float(q_hat) if q_hat is not None else None,
        # Keep top-level metrics for backward compatibility (using notebook-style now)
        'separatrix_pct': metrics_notebook['separatrix_pct'],
        'accuracy': metrics_notebook['accuracy'],
        'precision': metrics_notebook['precision'],
        'recall': metrics_notebook['recall'],
        'specificity': metrics_notebook['specificity'],
        'f1': metrics_notebook['f1'],
        # Endpoint error metrics (geodesic distances by category) - lambda*+/-delta classification
        'endpoint_errors': {
            'component_names': component_names,
            'full': error_stats_full,
            'certain': error_stats_certain,
            'certain_success': error_stats_certain_success,
            'certain_failure': error_stats_certain_failure,
            'uncertain': error_stats_uncertain,
            'invalid': error_stats_invalid,
        },
        # Hierarchical error metrics (16 combinations) - lambda*+/-delta classification
        'hierarchical_errors': {
            'full': hierarchical_stats_full,
            'certain': hierarchical_stats_certain,
            'certain_success': hierarchical_stats_certain_success,
            'certain_failure': hierarchical_stats_certain_failure,
            'uncertain': hierarchical_stats_uncertain,
            'invalid': hierarchical_stats_invalid,
        },
        # Endpoint error metrics using q_hat-based classification
        'endpoint_errors_qhat': {
            'component_names': component_names,
            'full': error_stats_full,  # Same as above (full dataset)
            'certain': error_stats_certain_qhat,
            'certain_success': error_stats_certain_success_qhat,
            'certain_failure': error_stats_certain_failure_qhat,
            'uncertain': error_stats_uncertain_qhat,
            'invalid': error_stats_invalid_qhat,
        } if q_hat is not None else None,
        # Hierarchical error metrics using q_hat-based classification
        'hierarchical_errors_qhat': {
            'full': hierarchical_stats_full,  # Same as above (full dataset)
            'certain': hierarchical_stats_certain_qhat,
            'certain_success': hierarchical_stats_certain_success_qhat,
            'certain_failure': hierarchical_stats_certain_failure_qhat,
            'uncertain': hierarchical_stats_uncertain_qhat,
            'invalid': hierarchical_stats_invalid_qhat,
        } if q_hat is not None else None,
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
            start_states=X_all,           # [N, 6] - (x, z, theta, x_dot, z_dot, theta_dot)
            p_success=p_success,          # [N] - P(MC label == 1)
            p_failure=p_failure,          # [N] - P(MC label == -1)
            p_invalid=p_invalid,          # [N] - P(MC label == 0)
            true_labels=y_all,            # [N] - ground truth labels (1=success, -1=failure, 0=separatrix)
            lambda_star=lambda_star,
            delta=delta,
            attractor_radius=attractor_radius
        )
        if verbose:
            print(f"Saved per-point data to: {npz_file}")
            print(f"  Arrays: start_states [{X_all.shape}], p_success [{p_success.shape}], p_failure [{p_failure.shape}], true_labels [{y_all.shape}]")

        # Generate ROA phase space projection plots
        phase_space_file = output_file.replace('.json', '_roa_projections.png')
        plot_quadrotor2d_roa_projections(
            start_states=X_all,
            success_rate=p_success,
            true_labels=y_all,
            lambda_star=lambda_star,
            delta=delta,
            output_file=phase_space_file,
            title="Quadrotor 2D ROA - Epoch Evaluation"
        )

        # Generate probability heatmap
        heatmap_file = output_file.replace('.json', '_roa_heatmap.png')
        plot_quadrotor2d_probability_heatmap(
            start_states=X_all,
            success_rate=p_success,
            lambda_star=lambda_star,
            delta=delta,
            output_file=heatmap_file,
            title="Quadrotor 2D ROA - Success Probability"
        )

    return metrics


def train_flow_matcher(
    cfg: DictConfig,
    train_file: str,
    val_file: str,
    output_dir: str,
    max_epochs: int = 500,
    resume_checkpoint: str = None,
    val_error_log_file: str = None
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
        val_error_log_file: Path to text file for logging validation errors

    Returns:
        Trained flow matcher model
    """
    # Instantiate system
    system = hydra.utils.instantiate(cfg.system)

    # Create data module with current dataset files
    data_module = Quadrotor2DEndpointDataModule(
        data_file=train_file,
        validation_file=val_file,
        test_file=val_file,  # Use val as test for now
        batch_size=cfg.get('batch_size', 256),
        val_batch_size=cfg.get('val_batch_size', 2048),
        num_workers=cfg.get('num_workers', 4),
    )

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model)

    # Instantiate flow matcher (matching existing FM training)
    use_loss_weights = cfg.flow_matching.get('use_loss_weights', False)
    use_manifold = cfg.flow_matching.get('use_manifold', True)
    use_log_loss_weights = cfg.flow_matching.get('use_log_loss_weights', False)
    clamp_noise = cfg.flow_matching.get('clamp_noise', True)
    zero_latent = cfg.flow_matching.get('zero_latent', False)
    noise_scale = cfg.flow_matching.get('noise_scale', 1.0)
    flow_matcher = hydra.utils.instantiate(
        cfg.flow_matcher,
        system=system,
        model=model,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        latent_dim=cfg.flow_matching.latent_dim,
        mae_val_frequency=cfg.flow_matching.mae_val_frequency,
        use_loss_weights=use_loss_weights,
        use_manifold=use_manifold,
        use_log_loss_weights=use_log_loss_weights,
        clamp_noise=clamp_noise,
        zero_latent=zero_latent,
        noise_scale=noise_scale,
        val_error_log_file=val_error_log_file,
        _recursive_=False
    )

    # Print flow matching parameters for verification
    if use_loss_weights:
        weight_type = "1+log(limit)" if use_log_loss_weights else "limit"
        print(f"   Loss weights: ENABLED ({weight_type})")
    print(f"   Use manifold: {use_manifold}")
    print(f"   Clamp noise: {clamp_noise}")
    print(f"   Noise scale: {noise_scale}")
    print(f"   Zero latent: {zero_latent}")
    print(f"   Latent dim: {cfg.flow_matching.latent_dim}")

    # Load weights from previous checkpoint if warm starting
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"Warm start: Loading weights from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
        state_dict = checkpoint["state_dict"]
        # Load only model weights (not optimizer state)
        model_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        flow_matcher.model.load_state_dict(model_state_dict)
        print(f"   Loaded model weights ({len(model_state_dict)} tensors)")

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


@hydra.main(config_path="../configs", config_name="adaptive_quadrotor2d", version_base=None)
def main(cfg: DictConfig):
    """Main adaptive sampling loop."""
    print("=" * 70)
    print("ADAPTIVE SAMPLING PIPELINE - Quadrotor 2D")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))

    # Set seed (covers random, numpy, torch CPU/CUDA, and DataLoader workers)
    seed = cfg.get('seed', 42)
    pl.seed_everything(seed, workers=True)

    # Setup output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trajectory data source
    data_source_config = TrajectoryDataSourceConfig(
        trajectories_dir=cfg.data_source.trajectories_dir,
        shuffled_indices_file=cfg.data_source.shuffled_indices_file,
        # Use shuffled_labels (aligned with shuffled_indices) for training
        shuffled_labels_file=cfg.data_source.get('shuffled_labels_file', None),
        # Use eval_states for full ROA evaluation (contains start, end, labels)
        eval_states_file=cfg.data_source.get('eval_states_file', None),
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
        alpha=cfg.conformal.alpha_sampling,
        num_mc_samples=cfg.conformal.get('num_mc_samples', 100),
        attractor_radius=cfg.conformal.get('attractor_radius', 0.2),
        # Optimization mode: "lambda" or "delta"
        optimize_mode=cfg.conformal.get('optimize_mode', 'lambda'),
        # Decision rule: "one_sided" (p_s only) or "two_sided" (p_s and p_f)
        decision_rule=cfg.conformal.get('decision_rule', 'two_sided'),  # Default two_sided for Quadrotor 2D
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
    samples_per_epoch = cfg.get('samples_per_epoch', 50)  # Total samples per epoch (D1 + D2)
    d2_ratio = cfg.get('d2_ratio', 0.5)  # Fraction for uncertainty-filtered sampling
    warm_start = cfg.get('warm_start', False)

    epoch_results = []
    previous_best_checkpoint = None  # Track previous epoch's best checkpoint for warm start

    for epoch in range(n_epochs):
        print("\n" + "=" * 70)
        print(f"EPOCH {epoch}")
        print("=" * 70)

        # Step 1: Train flow matcher on current dataset
        # Capture training set size BEFORE adding new data (for accurate logging)
        train_trajectories_this_epoch = len(dataset_builder.train_split)
        print(f"\n[1] Training flow matcher on {train_trajectories_this_epoch} trajectories...")
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
            val_error_log_file=str(epoch_output_dir / "validation_errors.txt"),
        )
        flow_matcher.eval()
        flow_matcher.to(device)

        # Track this epoch's best checkpoint for next epoch's warm start
        epoch_ckpts = glob.glob(str(epoch_output_dir / "checkpoints" / "best*.ckpt"))
        if epoch_ckpts:
            previous_best_checkpoint = epoch_ckpts[0]

        # Step 2: Create conformal predictor and optimize lambda*/delta* on training data
        print(f"\n[2] Creating conformal predictor and optimizing lambda*/delta*...")
        conformal_predictor = ConformalPredictor(
            flow_matcher=flow_matcher,
            system=system,
            config=conformal_config,
            device=device,
        )

        # Sample from endpoint dataset for lambda/delta optimization
        max_opt_samples = cfg.conformal.get('max_opt_samples', 5000)
        X_train, y_train = sample_endpoint_data_for_optimization(
            dataset_builder, sample_fraction=0.1, max_samples=max_opt_samples, seed=seed + epoch
        )

        threshold_mode = cfg.conformal.get('threshold_mode', 'dynamic')
        conformal_verbose = cfg.conformal.get('verbose', True)

        if threshold_mode == "fixed":
            # Fixed mode: skip optimization, use fixed thresholds
            fixed_lambda_star = cfg.conformal.get('fixed_lambda_star', 0.5)
            fixed_delta_star = cfg.conformal.get('fixed_delta_star', 0.1)
            conformal_predictor.lambda_star = fixed_lambda_star
            conformal_predictor.delta_star = fixed_delta_star
            print(f"    Using fixed thresholds: lambda* = {fixed_lambda_star:.4f}, delta* = {fixed_delta_star:.4f}")
        else:
            # Dynamic mode: optimize lambda*/delta* on ALL training data (no split)
            lambda_star, delta_star, opt_info = conformal_predictor.optimize_thresholds(
                X_train, y_train, verbose=conformal_verbose
            )

        lambda_star = conformal_predictor.lambda_star
        delta_star = conformal_predictor.delta_star
        print(f"    lambda* = {lambda_star:.4f}, delta* = {delta_star:.4f}")

        # Compute endpoint prediction error on full training endpoint dataset
        endpoint_error = compute_endpoint_prediction_error(
            flow_matcher=flow_matcher,
            dataset_builder=dataset_builder,
            batch_size=cfg.get('val_batch_size', 512),
            device=device,
            verbose=conformal_verbose,
        )

        # Determine sampling mode (with backward compatibility)
        sampling_mode = cfg.get('sampling_mode', None)
        if sampling_mode is None:
            # Backward compat: derive from use_conformal
            use_conformal_flag = cfg.conformal.get('use_conformal', True)
            sampling_mode = "conformal" if use_conformal_flag else "direct"

        decision_rule = cfg.conformal.get('decision_rule', 'two_sided')

        # use_conformal controls evaluation-time conformal prediction (always true)
        use_conformal = True

        # Create probability estimator for uncertainty evaluation
        prob_estimator = ProbabilityEstimator(
            flow_matcher=flow_matcher,
            system=system,
            config=conformal_config,
            device=device
        )

        if sampling_mode == "ranked":
            # ========== RANKED MODE: D1 uniform + D2 ranked ==========

            # Compute target sizes (same split as other modes)
            n_d1_target = int(samples_per_epoch * (1 - d2_ratio))
            n_d2_target = samples_per_epoch - n_d1_target

            # Step 3: Sample D1 (uniform calibration set)
            print(f"\n[3] Sampling D1 (calibration set)...")
            d1_states, d1_indices = dataset_builder.sample_candidates_without_marking(n_d1_target)

            if len(d1_indices) == 0:
                print("    No more trajectories available!")
                break

            dataset_builder.mark_indices_as_used(d1_indices)
            dataset_builder.add_to_training_balanced(d1_indices)
            print(f"    D1: {len(d1_indices)} calibration points sampled and added to training")

            # Step 4: Sample D2 via ranked NC scores
            n_ranked_candidates = cfg.get('n_ranked_candidates', 1000)
            print(f"\n[4] Ranked sampling D2: evaluating {n_ranked_candidates} candidates, "
                  f"selecting {n_d2_target} lowest NC scores...")

            uncertain_sampler = UncertainSampler(
                dataset_builder=dataset_builder,
                target_count=n_d2_target,
                batch_size=cfg.get('batch_size_sampling', 50),
                max_candidates=cfg.get('max_samples_per_epoch', 50000),
            )

            ranked_result = uncertain_sampler.sample_ranked(
                prob_estimator=prob_estimator,
                calibrator=conformal_predictor.calibrator,
                lambda_star=lambda_star,
                delta_star=delta_star,
                decision_rule=decision_rule,
                n_candidates=n_ranked_candidates,
                n_select=n_d2_target,
                verbose=conformal_verbose,
            )

            d2_indices = ranked_result.selected_indices
            n_d2 = len(d2_indices)
            if n_d2 > 0:
                dataset_builder.mark_indices_as_used(d2_indices)
                dataset_builder.add_to_training_balanced(d2_indices)

            n_certain_discarded = ranked_result.n_candidates_evaluated - n_d2
            n_invalid_added = 0
            q_hat = None

            print(f"\n[5] Ranked sampling summary:")
            print(f"    D1 (calibration): {len(d1_indices)} points (added)")
            print(f"    D2 (ranked): {n_d2} points (added)")
            print(f"    Total added: {len(d1_indices) + n_d2}")
            print(f"    Candidates evaluated: {ranked_result.n_candidates_evaluated}")
            if ranked_result.n_candidates_evaluated > 0:
                print(f"    Score threshold: {ranked_result.score_threshold:.4f}")

        else:
            # ========== D1/D2 MODES (conformal / direct) ==========

            # Step 3: Sample D1 (calibration set)
            print(f"\n[3] Sampling D1 (calibration set)...")

            # Compute target sizes
            n_d1_target = int(samples_per_epoch * (1 - d2_ratio))
            n_d2_target = samples_per_epoch - n_d1_target

            # Sample D1 candidates (for calibration)
            d1_states, d1_indices = dataset_builder.sample_candidates_without_marking(n_d1_target)

            if len(d1_indices) == 0:
                print("    No more trajectories available!")
                break

            # Get TRUE LABELS for D1 from shuffled_labels.txt
            d1_labels = dataset_builder.data_source.get_labels(d1_indices)

            # Mark D1 as USED and add to training
            dataset_builder.mark_indices_as_used(d1_indices)
            dataset_builder.add_to_training_balanced(d1_indices)

            print(f"    D1: {len(d1_indices)} calibration points sampled and added to training")

            # Step 4: Calibrate q_hat using D1 (only if sampling_mode is conformal)
            if sampling_mode == "conformal":
                print(f"\n[4] Calibrating q_hat on D1...")
                q_hat = conformal_predictor.calibrate_qhat(d1_states, d1_labels, verbose=conformal_verbose)
                if conformal_verbose:
                    print(f"    q_hat = {q_hat:.4f}")
            else:
                print(f"\n[4] Skipping q_hat calibration (sampling_mode={sampling_mode})")
                q_hat = None

            # Step 5: Sample D2 (uncertain)
            if sampling_mode == "conformal":
                print(f"\n[5] Sampling D2 (uncertain) using q_hat...")
            else:
                print(f"\n[5] Sampling D2 (uncertain) using λ*/δ* thresholds directly...")

            # Create uncertain sampler
            uncertain_sampler = UncertainSampler(
                dataset_builder=dataset_builder,
                target_count=n_d2_target,
                batch_size=cfg.get('batch_size_sampling', 50),
                max_candidates=cfg.get('max_samples_per_epoch', 50000),
            )

            # Sample D2 using appropriate method
            if sampling_mode == "conformal":
                # Use q_hat-based conformal prediction sets
                sample_result = uncertain_sampler.sample(
                    prob_estimator=prob_estimator,
                    calibrator=conformal_predictor.calibrator,
                    lambda_star=lambda_star,
                    delta_star=delta_star,
                    q_hat=q_hat,
                    decision_rule=decision_rule,
                    exclude=set(d1_indices),  # Don't re-sample D1
                    verbose=conformal_verbose
                )
            else:
                # Use λ*/δ* thresholds directly (no q_hat)
                sample_result = uncertain_sampler.sample_direct(
                    prob_estimator=prob_estimator,
                    lambda_star=lambda_star,
                    delta_star=delta_star,
                    decision_rule=decision_rule,
                    exclude=set(d1_indices),  # Don't re-sample D1
                    verbose=conformal_verbose
                )

            # D2 results
            d2_indices = sample_result.uncertain_indices
            n_d2 = len(d2_indices)
            n_certain_discarded = sample_result.n_certain_discarded
            n_invalid_added = sample_result.n_invalid_added

            # Mark D2 as used and add to training
            if n_d2 > 0:
                dataset_builder.mark_indices_as_used(d2_indices)
                dataset_builder.add_to_training_balanced(d2_indices)

            print(f"\n[6] Summary of sampling this epoch...")
            print(f"    D1 (calibration): {len(d1_indices)} points (added)")
            print(f"    D2 (uncertain+invalid): {n_d2} points (added)")
            print(f"        - uncertain: {n_d2 - n_invalid_added}")
            print(f"        - invalid:   {n_invalid_added}")
            print(f"    Total added:      {len(d1_indices) + n_d2}")
            print(f"    Discarded (certain success/failure): {n_certain_discarded}")
            print(f"    D2 candidates evaluated: {sample_result.n_candidates_evaluated} over {sample_result.n_batches} batches")

        # Step 7: Evaluate on test set using D1-calibrated q_hat (only for conformal sampling)
        if sampling_mode == "conformal":
            print(f"\n[7] Evaluating on test set (D1-calibrated conformal)...")
            X_test, y_test = dataset_builder.get_test_labels()
            test_metrics = conformal_predictor.evaluate(X_test, y_test, verbose=conformal_verbose)
        else:
            print(f"\n[7] Skipping D1 conformal evaluation (sampling_mode={sampling_mode})")
            test_metrics = {'coverage': None, 'f1': None, 'unknown_rate': None}

        # Step 8: Evaluate on HELD-OUT test set with proper calibration
        num_mc_samples_eval = cfg.conformal.get('num_mc_samples_eval', 20)

        print(f"\n[8] Evaluating on HELD-OUT test set...")

        # 8a: Load held-out calibration set and calibrate NEW q_hat (only if conformal enabled)
        if use_conformal:
            print(f"    Loading calibration set: {cfg.data_source.cal_set_file}")
            X_cal_eval, _, y_cal_eval = load_eval_states(cfg.data_source.cal_set_file)
            print(f"    Calibration set size: {len(X_cal_eval)}")

            # Estimate probabilities on calibration set
            X_cal_tensor = torch.tensor(X_cal_eval, dtype=torch.float32, device=device)
            with torch.no_grad():
                p_cal_success, p_cal_failure, _ = prob_estimator.estimate(X_cal_tensor)

            # Calibrate q_hat on held-out calibration set
            from adaptive_roa.conformal.calibrator import Calibrator
            eval_conformal_config = ConformalConfig(
                delta=delta_star,
                alpha=cfg.conformal.alpha_eval,
                decision_rule=cfg.conformal.get('decision_rule', 'two_sided')
            )
            eval_calibrator = Calibrator(eval_conformal_config)
            # prob_estimator.estimate() returns numpy arrays directly
            p_cal_success_np = p_cal_success.cpu().numpy() if hasattr(p_cal_success, 'cpu') else p_cal_success
            p_cal_failure_np = p_cal_failure.cpu().numpy() if hasattr(p_cal_failure, 'cpu') else p_cal_failure
            q_hat_eval = eval_calibrator.calibrate(
                p_cal_success_np,
                y_cal_eval,
                lambda_star,
                delta_star,
                p_failure=p_cal_failure_np,
                verbose=conformal_verbose
            )
            if conformal_verbose:
                print(f"    Eval q_hat = {q_hat_eval:.4f} (calibrated on {len(X_cal_eval)} held-out points)")
                print(f"    (Training q_hat was {q_hat:.4f} from D1)")
            n_cal_eval = len(X_cal_eval)
        else:
            print(f"    Skipping held-out q_hat calibration (use_conformal=false)")
            q_hat_eval = None
            n_cal_eval = 0

        # 8b: Evaluate on held-out test set
        print(f"    Evaluating on test set: {cfg.data_source.test_set_file}")
        full_roa_output_file = epoch_output_dir / "full_roa_evaluation.json"
        full_roa_metrics = evaluate_full_roa_fast(
            flow_matcher=flow_matcher,
            system=system,
            eval_states_file=cfg.data_source.test_set_file,  # Changed: use test_set
            num_mc_samples=num_mc_samples_eval,
            batch_size=cfg.get('val_batch_size', 2048),
            lambda_star=lambda_star,
            delta=delta_star,
            q_hat=q_hat_eval,  # Changed: use eval-time q_hat
            attractor_radius=cfg.conformal.get('attractor_radius', 0.2),
            device=device,
            output_file=str(full_roa_output_file),
            verbose=conformal_verbose
        )

        # Store q_hats in results (if conformal enabled)
        full_roa_metrics['q_hat_training'] = float(q_hat) if q_hat is not None else None
        full_roa_metrics['q_hat_eval'] = float(q_hat_eval) if q_hat_eval is not None else None
        full_roa_metrics['n_cal_eval'] = n_cal_eval
        full_roa_metrics['use_conformal'] = use_conformal

        # Step 9: Rebuild datasets with new data
        print(f"\n[9] Rebuilding datasets...")
        dataset_files = dataset_builder.build_all_datasets()

        # Record epoch results
        epoch_result = {
            'epoch': epoch,
            'train_trajectories': train_trajectories_this_epoch,
            'sampling_mode': sampling_mode,
            'n_d1_added': len(d1_indices),
            'n_d2_added': int(n_d2),
            'n_d2_uncertain': int(n_d2 - n_invalid_added),
            'n_d2_invalid': int(n_invalid_added),
            'n_discarded_certain': int(n_certain_discarded),
            # Ranked sampling info (only for ranked mode)
            'n_ranked_candidates_evaluated': int(ranked_result.n_candidates_evaluated) if sampling_mode == "ranked" else None,
            'ranked_score_threshold': float(ranked_result.score_threshold) if sampling_mode == "ranked" else None,
            # Conformal parameters (training-time, used for adaptive sampling)
            'lambda_star': float(conformal_predictor.lambda_star),
            'delta_star': float(conformal_predictor.delta_star),
            'q_hat': float(q_hat) if q_hat is not None else None,  # Training-time q_hat (used for sampling)
            'q_hat_eval': float(q_hat_eval) if q_hat_eval is not None else None,  # Eval-time q_hat
            'n_cal_eval': n_cal_eval,                    # Held-out calibration set size
            'use_conformal': use_conformal,              # Whether conformal prediction is enabled
            'optimize_mode': cfg.conformal.get('optimize_mode', 'lambda'),
            # Test set metrics (subset of training)
            'test_coverage': test_metrics['coverage'],
            'test_f1': test_metrics['f1'],
            'test_unknown_rate': test_metrics['unknown_rate'],
            # Endpoint prediction error on training data
            'endpoint_error': endpoint_error,
            # Full ROA metrics (held-out test_set.txt)
            'full_roa': full_roa_metrics,
        }
        epoch_results.append(epoch_result)

        print("\n" + "-" * 70)
        print(f"EPOCH {epoch} SUMMARY")
        print("-" * 70)
        print(f"  Training trajectories: {epoch_result['train_trajectories']}")
        print(f"  Sampling mode: {sampling_mode}")
        if sampling_mode == "ranked":
            print(f"  Added this epoch: {len(d1_indices) + n_d2} (D1={len(d1_indices)}, D2={n_d2})")
            print(f"  Ranked candidates evaluated: {ranked_result.n_candidates_evaluated}")
            if ranked_result.n_candidates_evaluated > 0:
                print(f"  Score threshold: {ranked_result.score_threshold:.4f}")
        else:
            print(f"  Added this epoch: {len(d1_indices) + n_d2} (D1={len(d1_indices)}, D2={n_d2})")
            print(f"    D2 breakdown: {n_d2 - n_invalid_added} uncertain + {n_invalid_added} invalid")
            print(f"  Discarded (certain success/failure): {n_certain_discarded}")
        print(f"  lambda* = {epoch_result['lambda_star']:.4f}, delta* = {epoch_result['delta_star']:.4f}")
        if conformal_verbose and sampling_mode == "conformal":
            print(f"  q_hat (training) = {epoch_result['q_hat']:.4f}, q_hat (eval) = {epoch_result['q_hat_eval']:.4f}")
        elif conformal_verbose and q_hat_eval is not None:
            print(f"  q_hat (eval) = {epoch_result['q_hat_eval']:.4f}")
        print(f"  --- Full ROA (held-out test set: {full_roa_metrics['n_total']} trajectories) ---")
        conf_m = full_roa_metrics['conformal_thresholds']
        notebook_m = full_roa_metrics['notebook_thresholds']
        print(f"  [lambda*+/-delta] Sep%={conf_m['separatrix_pct']:.1%}, F1={conf_m['f1']:.2%}, Acc={conf_m['accuracy']:.2%}")
        print(f"  [Notebook p_s/p_f>0.6] Sep%={notebook_m['separatrix_pct']:.1%}, F1={notebook_m['f1']:.2%}, Acc={notebook_m['accuracy']:.2%}")
        print(f"  Endpoint prediction MAE: {endpoint_error['overall_mae']:.6f} ({endpoint_error['n_endpoint_pairs']} pairs)")

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
        print(f"\n--- Full ROA Metrics Progression (Initial -> Final) ---")
        first = epoch_results[0]['full_roa']
        last = epoch_results[-1]['full_roa']
        print(f"\n  [lambda*+/-delta* Thresholds]")
        print(f"  Separatrix %:  {first['conformal_thresholds']['separatrix_pct']:.2%} -> {last['conformal_thresholds']['separatrix_pct']:.2%}")
        print(f"  F1 Score:      {first['conformal_thresholds']['f1']:.2%} -> {last['conformal_thresholds']['f1']:.2%}")
        print(f"  Accuracy:      {first['conformal_thresholds']['accuracy']:.2%} -> {last['conformal_thresholds']['accuracy']:.2%}")
        print(f"\n  [Notebook-Style Thresholds (p_s/p_f > 0.6)]")
        print(f"  Separatrix %:  {first['notebook_thresholds']['separatrix_pct']:.2%} -> {last['notebook_thresholds']['separatrix_pct']:.2%}")
        print(f"  F1 Score:      {first['notebook_thresholds']['f1']:.2%} -> {last['notebook_thresholds']['f1']:.2%}")
        print(f"  Accuracy:      {first['notebook_thresholds']['accuracy']:.2%} -> {last['notebook_thresholds']['accuracy']:.2%}")
        print(f"\n  lambda*:       {epoch_results[0]['lambda_star']:.4f} -> {epoch_results[-1]['lambda_star']:.4f}")
        print(f"  delta*:        {epoch_results[0]['delta_star']:.4f} -> {epoch_results[-1]['delta_star']:.4f}")
        if cfg.conformal.get('verbose', True):
            if epoch_results[0].get('q_hat') is not None:
                print(f"  q_hat (train): {epoch_results[0]['q_hat']:.4f} -> {epoch_results[-1]['q_hat']:.4f}")
            if epoch_results[0].get('q_hat_eval') is not None:
                print(f"  q_hat (eval):  {epoch_results[0]['q_hat_eval']:.4f} -> {epoch_results[-1]['q_hat_eval']:.4f}")
                print(f"  n_cal (eval):  {epoch_results[-1]['n_cal_eval']} held-out calibration points")
        if epoch_results[0].get('endpoint_error') and epoch_results[-1].get('endpoint_error'):
            print(f"\n  Endpoint MAE:  {epoch_results[0]['endpoint_error']['overall_mae']:.6f} -> {epoch_results[-1]['endpoint_error']['overall_mae']:.6f}")

    # Save final results
    with open(output_dir / "final_results.json", 'w') as f:
        json.dump({
            'epoch_results': epoch_results,
            'final_stats': stats,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

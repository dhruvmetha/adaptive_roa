"""
Reconstruct final_results.json from checkpoints.

This script allows you to regenerate ROA evaluation results from saved checkpoints
when the original run was interrupted or final_results.json wasn't created.

Usage:
    # Reconstruct results for a run with checkpoints
    python adaptive_roa/adaptive/reconstruct_results.py \
        --results_dir /path/to/adaptive_run/2025-12-19_03-01-24

    # Specify system type explicitly (if config not found)
    python adaptive_roa/adaptive/reconstruct_results.py \
        --results_dir /path/to/run \
        --system pendulum

    # Override number of MC samples
    python adaptive_roa/adaptive/reconstruct_results.py \
        --results_dir /path/to/run \
        --num_mc_samples 20
"""

import argparse
import json
import numpy as np
import torch
import yaml
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import subprocess
import shutil


def load_hydra_config(results_dir: Path) -> Optional[Dict]:
    """
    Load Hydra config from the results directory.

    Tries multiple locations:
    1. {results_dir}/.hydra/config.yaml
    2. {results_dir}/epoch_000/.hydra/config.yaml
    """
    # Try root .hydra first
    config_path = results_dir / ".hydra" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # Try epoch_000 .hydra
    config_path = results_dir / "epoch_000" / ".hydra" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    return None


def load_conformal_state(epoch_dir: Path) -> Optional[Dict]:
    """Load conformal_state.json from an epoch directory."""
    state_path = epoch_dir / "conformal_state.json"
    if state_path.exists():
        with open(state_path, 'r') as f:
            return json.load(f)
    return None


def load_epoch_results(epoch_dir: Path) -> Optional[Dict]:
    """Load results.json from an epoch directory."""
    results_path = epoch_dir / "results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def find_best_checkpoint(epoch_dir: Path) -> Optional[Path]:
    """Find the best checkpoint in an epoch directory."""
    ckpt_dir = epoch_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    # Look for best-*.ckpt first
    best_ckpts = list(ckpt_dir.glob("best-*.ckpt"))
    if best_ckpts:
        return best_ckpts[0]

    # Fall back to last.ckpt
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    return None


def get_system_and_flow_matcher(system_type: str):
    """
    Get system and flow matcher class based on system type.

    Returns:
        Tuple of (system_instance, flow_matcher_class)
    """
    if system_type == "pendulum":
        from adaptive_roa.systems.pendulum import PendulumSystem
        from adaptive_roa.flow_matching.pendulum.latent_conditional.flow_matcher import (
            PendulumLatentConditionalFlowMatcher
        )
        return PendulumSystem(), PendulumLatentConditionalFlowMatcher
    elif system_type == "cartpole":
        from adaptive_roa.systems.cartpole import CartPoleSystem
        from adaptive_roa.flow_matching.cartpole.latent_conditional.flow_matcher import (
            CartPoleLatentConditionalFlowMatcher
        )
        return CartPoleSystem(), CartPoleLatentConditionalFlowMatcher
    else:
        raise ValueError(f"Unknown system type: {system_type}")


def compute_metrics_at_threshold(
    success_rate: np.ndarray,
    y_all: np.ndarray,
    success_thresh: float,
    failure_thresh: float
) -> Dict:
    """Compute classification metrics at given probability thresholds."""
    n_total = len(y_all)
    pred_labels = np.zeros(n_total)
    pred_labels[success_rate > success_thresh] = 1
    pred_labels[success_rate < failure_thresh] = -1

    n_uncertain = np.sum(pred_labels == 0)
    separatrix_pct = n_uncertain / n_total

    # Only evaluate on points where BOTH prediction and ground truth are confident
    # (excludes ground truth separatrix points with y=0)
    confident_mask = (pred_labels != 0) & (y_all != 0)
    n_confident = np.sum(confident_mask)

    y_pred_conf = pred_labels[confident_mask]
    y_true_conf = y_all[confident_mask]

    tp = int(np.sum((y_pred_conf == 1) & (y_true_conf == 1)))
    tn = int(np.sum((y_pred_conf == -1) & (y_true_conf == -1)))
    fp = int(np.sum((y_pred_conf == 1) & (y_true_conf == -1)))
    fn = int(np.sum((y_pred_conf == -1) & (y_true_conf == 1)))

    # Accuracy is (TP + TN) / (TP + TN + FP + FN)
    total_evaluated = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_evaluated if total_evaluated > 0 else 0.0

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


def evaluate_full_roa(
    flow_matcher,
    system,
    X_all: np.ndarray,
    y_all: np.ndarray,
    num_mc_samples: int = 20,
    batch_size: int = 2048,
    lambda_star: float = 0.5,
    delta: float = 0.05,
    attractor_radius: float = 0.1,
    device: str = 'cuda',
    verbose: bool = True
) -> Tuple[Dict, np.ndarray]:
    """
    Evaluate ROA on full dataset.

    Args:
        flow_matcher: Trained flow matcher model
        system: System for classify_attractor
        X_all: All start states [N, state_dim]
        y_all: All ground truth labels [N]
        num_mc_samples: Number of MC samples per point
        batch_size: Batch size for GPU inference
        lambda_star: Decision boundary center
        delta: Unknown region half-width
        attractor_radius: Radius for attractor classification
        device: Device for inference
        verbose: Print progress

    Returns:
        Tuple of (metrics_dict, success_rate_array)
    """
    n_total = len(y_all)

    if verbose:
        print(f"Evaluating {n_total} points with {num_mc_samples} MC samples...")

    # Convert to tensor
    X_tensor = torch.from_numpy(X_all).float().to(device)

    # Collect success counts for each point
    is_success = np.zeros((n_total, num_mc_samples))

    flow_matcher.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, n_total, batch_size), desc="Evaluating", disable=not verbose):
            batch_end = min(batch_start + batch_size, n_total)
            batch_inp = X_tensor[batch_start:batch_end]

            for sample_idx in range(num_mc_samples):
                pred = flow_matcher.predict_endpoint(batch_inp)
                attractor_labels = system.classify_attractor(pred, attractor_radius).cpu().numpy()
                is_success[batch_start:batch_end, sample_idx] = attractor_labels

    # Compute success rate per point
    success_rate = (is_success == 1).sum(axis=1) / num_mc_samples

    # Compute metrics for BOTH threshold schemes
    metrics_conformal = compute_metrics_at_threshold(
        success_rate, y_all,
        success_thresh=lambda_star + delta,
        failure_thresh=lambda_star - delta
    )

    metrics_fixed = compute_metrics_at_threshold(
        success_rate, y_all,
        success_thresh=0.6,
        failure_thresh=0.4
    )

    metrics = {
        'n_total': n_total,
        'num_mc_samples': num_mc_samples,
        'lambda_star': float(lambda_star),
        'delta': float(delta),
        'conformal_thresholds': metrics_conformal,
        'fixed_thresholds': metrics_fixed,
        # Keep top-level metrics for backward compatibility
        'separatrix_pct': metrics_conformal['separatrix_pct'],
        'accuracy': metrics_conformal['accuracy'],
        'precision': metrics_conformal['precision'],
        'recall': metrics_conformal['recall'],
        'specificity': metrics_conformal['specificity'],
        'f1': metrics_conformal['f1'],
    }

    return metrics, success_rate


def load_roa_labels(roa_labels_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ROA labels file.

    Supports both comma-separated and space-separated formats.
    Converts labels to standard format: 1 (success), -1 (failure).

    Input label formats supported:
    - 0/1 format: 0 = failure, 1 = success
    - -1/1 format: -1 = failure, 1 = success (already standard)

    Returns:
        Tuple of (start_states, labels) where labels are in {-1, 1} format
    """
    # Try to detect delimiter by reading first line
    with open(roa_labels_file, 'r') as f:
        first_line = f.readline().strip()

    if ',' in first_line:
        delimiter = ','
    else:
        delimiter = None  # whitespace

    data = np.loadtxt(roa_labels_file, delimiter=delimiter)

    # Format: theta, theta_dot, label (or x, theta, x_dot, theta_dot, label for cartpole)
    if data.shape[1] == 3:
        # Pendulum: theta, theta_dot, label
        start_states = data[:, :2]
        labels = data[:, 2].astype(int)
    elif data.shape[1] == 5:
        # CartPole: x, theta, x_dot, theta_dot, label
        start_states = data[:, :4]
        labels = data[:, 4].astype(int)
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    # Convert 0/1 format to -1/1 format if needed
    # Check if labels are in 0/1 format (no -1 values and has 0 values)
    unique_labels = np.unique(labels)
    if 0 in unique_labels and -1 not in unique_labels:
        # Convert: 0 -> -1 (failure), 1 -> 1 (success)
        labels = np.where(labels == 0, -1, labels)
        print(f"  Converted labels from 0/1 to -1/1 format")

    return start_states, labels


def plot_roa_phase_space(
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
        start_states: [N, 2] or [N, 4] array of states
        success_rate: [N] array of p(success) from MC sampling
        true_labels: [N] array of ground truth labels
        lambda_star: Optimal decision boundary
        delta: Unknown region half-width
        output_file: Path to save the plot
        title: Plot title
        threshold_type: "conformal" uses lambda* +/- delta, "fixed" uses 0.4/0.6

    Returns:
        Statistics dictionary
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # For pendulum: theta, theta_dot
    # For cartpole: use theta (col 1) and theta_dot (col 3)
    if start_states.shape[1] == 2:
        theta = start_states[:, 0]
        theta_dot = start_states[:, 1]
        xlabel = r'$\theta$ (rad)'
        ylabel = r'$\dot{\theta}$ (rad/s)'
    else:
        # CartPole: x, theta, x_dot, theta_dot
        theta = start_states[:, 1]
        theta_dot = start_states[:, 3]
        xlabel = r'$\theta$ (rad)'
        ylabel = r'$\dot{\theta}$ (rad/s)'

    # --- Left plot: Predicted ROA ---
    ax1 = axes[0]

    # Classify based on threshold type
    pred_labels = np.zeros(len(success_rate))
    if threshold_type == "conformal":
        success_thresh = lambda_star + delta
        failure_thresh = lambda_star - delta
        thresh_label = f"λ*={lambda_star:.3f}, δ={delta:.3f}"
    else:
        success_thresh = 0.6
        failure_thresh = 0.4
        thresh_label = "fixed 0.4/0.6"

    pred_labels[success_rate > success_thresh] = 1
    pred_labels[success_rate < failure_thresh] = -1

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

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
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
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
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

    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
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
    ax2.legend(handles=legend_elements_gt, loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'n_pred_success': int(n_success),
        'n_pred_failure': int(n_failure),
        'n_pred_separatrix': int(n_separatrix),
        'n_gt_success': int(gt_n_success),
        'n_gt_failure': int(gt_n_failure),
    }


def plot_roa_probability_heatmap(
    start_states: np.ndarray,
    success_rate: np.ndarray,
    lambda_star: float,
    delta: float,
    output_file: Path,
    title: str = "ROA - Success Probability",
) -> None:
    """
    Plot the ROA as a heatmap of success probability.

    Args:
        start_states: [N, 2] or [N, 4] array of states
        success_rate: [N] array of p(success) from MC sampling
        lambda_star: Optimal decision boundary
        delta: Unknown region half-width
        output_file: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # For pendulum: theta, theta_dot
    # For cartpole: use theta (col 1) and theta_dot (col 3)
    if start_states.shape[1] == 2:
        theta = start_states[:, 0]
        theta_dot = start_states[:, 1]
        xlabel = r'$\theta$ (rad)'
        ylabel = r'$\dot{\theta}$ (rad/s)'
    else:
        theta = start_states[:, 1]
        theta_dot = start_states[:, 3]
        xlabel = r'$\theta$ (rad)'
        ylabel = r'$\dot{\theta}$ (rad/s)'

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

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
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


def create_video_from_frames(
    frames_dir: Path,
    output_path: Path,
    framerate: int = 2,
    pattern: str = "frame_%03d.png"
) -> Optional[Path]:
    """
    Create MP4 video from PNG frames using ffmpeg.

    Args:
        frames_dir: Directory containing frame images
        output_path: Output video path
        framerate: Frames per second
        pattern: Frame filename pattern for ffmpeg

    Returns:
        Path to created video, or None if ffmpeg is not available
    """
    if not shutil.which('ffmpeg'):
        print("Warning: ffmpeg not found. Skipping video creation.")
        print(f"To create video manually: cd {frames_dir} && ffmpeg -framerate {framerate} -i {pattern} -c:v libx264 -pix_fmt yuv420p video.mp4")
        return None

    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(framerate),
        '-i', str(frames_dir / pattern),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
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


def create_animation_from_epochs(
    results_dir: Path,
    epoch_results: List[Dict],
    X_all: np.ndarray,
    y_all: np.ndarray,
    framerate: int = 2,
    verbose: bool = True,
) -> Dict[str, Optional[Path]]:
    """
    Create animation videos from reconstructed epoch data.

    Creates three types of videos:
    1. Phase space with conformal thresholds
    2. Phase space with fixed thresholds
    3. Probability heatmap (continuous p(success))

    Args:
        results_dir: Path to results directory
        epoch_results: List of epoch result dicts
        X_all: All start states
        y_all: All ground truth labels
        framerate: Frames per second for video
        verbose: Print progress

    Returns:
        Dict mapping video type to path
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Creating animation videos...")

    video_paths = {}

    # 1. Phase space videos (conformal and fixed thresholds)
    for threshold_type in ["conformal", "fixed"]:
        frames_dir = results_dir / f"animation_frames_{threshold_type}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Generating {len(epoch_results)} phase space frames ({threshold_type} thresholds)...")

        for result in epoch_results:
            epoch_num = result['epoch']
            epoch_dir = results_dir / f"epoch_{epoch_num:03d}"

            # Load per-point data
            npz_file = epoch_dir / "full_roa_evaluation_per_point_reconstructed.npz"
            if not npz_file.exists():
                npz_file = epoch_dir / "full_roa_evaluation_per_point.npz"
            if not npz_file.exists():
                continue

            data = np.load(npz_file)
            success_rate = data['probabilities']
            lambda_star = float(data['lambda_star'])
            delta = float(data['delta'])

            # Get train size for title
            train_size = result.get('train_trajectories', '?')
            f1_key = f"{threshold_type}_thresholds"
            f1 = result.get('full_roa', {}).get(f1_key, {}).get('f1', None)

            if f1 is not None:
                title = f"Epoch {epoch_num} | Train: {train_size} | F1: {f1:.1%}"
            else:
                title = f"Epoch {epoch_num} | Train: {train_size}"

            # Generate frame
            frame_file = frames_dir / f"frame_{epoch_num:03d}.png"
            plot_roa_phase_space(
                start_states=X_all,
                success_rate=success_rate,
                true_labels=y_all,
                lambda_star=lambda_star,
                delta=delta,
                output_file=frame_file,
                title=title,
                threshold_type=threshold_type
            )

        if verbose:
            print(f"  Saved {len(epoch_results)} frames to {frames_dir}")

        # Create video
        video_path = create_video_from_frames(
            frames_dir,
            output_path=results_dir / f"phase_space_evolution_{threshold_type}_reconstructed.mp4",
            framerate=framerate
        )
        video_paths[f"phase_space_{threshold_type}"] = video_path

    # 2. Probability heatmap video
    frames_dir = results_dir / "animation_frames_probability"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Generating {len(epoch_results)} probability heatmap frames...")

    for result in epoch_results:
        epoch_num = result['epoch']
        epoch_dir = results_dir / f"epoch_{epoch_num:03d}"

        # Load per-point data
        npz_file = epoch_dir / "full_roa_evaluation_per_point_reconstructed.npz"
        if not npz_file.exists():
            npz_file = epoch_dir / "full_roa_evaluation_per_point.npz"
        if not npz_file.exists():
            continue

        data = np.load(npz_file)
        success_rate = data['probabilities']
        lambda_star = float(data['lambda_star'])
        delta = float(data['delta'])

        # Get train size and sep% for title
        train_size = result.get('train_trajectories', '?')
        sep_pct = result.get('full_roa', {}).get('conformal_thresholds', {}).get('separatrix_pct', None)

        if sep_pct is not None:
            title = f"Epoch {epoch_num} | Train: {train_size} | Sep: {sep_pct:.1%}"
        else:
            title = f"Epoch {epoch_num} | Train: {train_size}"

        # Generate probability heatmap frame
        frame_file = frames_dir / f"frame_{epoch_num:03d}.png"
        plot_roa_probability_heatmap(
            start_states=X_all,
            success_rate=success_rate,
            lambda_star=lambda_star,
            delta=delta,
            output_file=frame_file,
            title=title
        )

    if verbose:
        print(f"  Saved {len(epoch_results)} frames to {frames_dir}")

    # Create probability video
    video_path = create_video_from_frames(
        frames_dir,
        output_path=results_dir / "probability_evolution_reconstructed.mp4",
        framerate=framerate
    )
    video_paths["probability"] = video_path

    return video_paths


def reconstruct_epoch(
    epoch_dir: Path,
    flow_matcher_class,
    system,
    X_all: np.ndarray,
    y_all: np.ndarray,
    num_mc_samples: int = 20,
    batch_size: int = 2048,
    attractor_radius: float = 0.1,
    device: str = 'cuda',
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Reconstruct results for a single epoch from checkpoint.

    Args:
        epoch_dir: Path to epoch directory
        flow_matcher_class: Class to load checkpoint with
        system: System instance
        X_all: All start states
        y_all: All labels
        num_mc_samples: MC samples for evaluation
        batch_size: Batch size
        attractor_radius: Attractor classification radius
        device: Device for inference
        verbose: Print progress

    Returns:
        Epoch result dict, or None if checkpoint not found
    """
    epoch_num = int(epoch_dir.name.split('_')[1])

    # Find checkpoint
    ckpt_path = find_best_checkpoint(epoch_dir)
    if ckpt_path is None:
        if verbose:
            print(f"  No checkpoint found in {epoch_dir}")
        return None

    if verbose:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch_num}: Loading {ckpt_path.name}")

    # Load model
    flow_matcher = flow_matcher_class.load_from_checkpoint(
        str(ckpt_path),
        device=device
    )
    flow_matcher.to(device)
    flow_matcher.eval()

    # Load conformal state if available
    conformal_state = load_conformal_state(epoch_dir)
    if conformal_state:
        lambda_star = conformal_state.get('lambda_star', 0.5)
        delta_star = conformal_state.get('delta_star', conformal_state.get('delta', 0.05))
        q_hat = conformal_state.get('q_hat', 0.0)
        if verbose:
            print(f"  Loaded conformal state: λ*={lambda_star:.4f}, δ*={delta_star:.4f}, q̂={q_hat:.4f}")
    else:
        # Use defaults
        lambda_star = 0.5
        delta_star = 0.05
        q_hat = 0.0
        if verbose:
            print(f"  No conformal_state.json found, using defaults: λ*={lambda_star}, δ*={delta_star}")

    # Load existing results for metadata (train_trajectories, etc.)
    existing_results = load_epoch_results(epoch_dir)

    # Run evaluation
    full_roa_metrics, success_rate = evaluate_full_roa(
        flow_matcher=flow_matcher,
        system=system,
        X_all=X_all,
        y_all=y_all,
        num_mc_samples=num_mc_samples,
        batch_size=batch_size,
        lambda_star=lambda_star,
        delta=delta_star,
        attractor_radius=attractor_radius,
        device=device,
        verbose=verbose,
    )

    # Build result dict
    result = {
        'epoch': epoch_num,
        'lambda_star': float(lambda_star),
        'delta_star': float(delta_star),
        'q_hat': float(q_hat),
        'full_roa': full_roa_metrics,
    }

    # Copy metadata from existing results if available
    if existing_results:
        for key in ['train_trajectories', 'n_d1_added', 'n_d2_uncertain', 'n_d2_confident',
                    'optimize_mode', 'test_coverage', 'test_f1', 'test_unknown_rate']:
            if key in existing_results:
                result[key] = existing_results[key]
    else:
        # Try to infer train_trajectories from dataset files
        result['train_trajectories'] = 0  # Unknown

    if verbose:
        conf_m = full_roa_metrics['conformal_thresholds']
        fixed_m = full_roa_metrics['fixed_thresholds']
        print(f"  [λ*±δ*] Sep%={conf_m['separatrix_pct']:.1%}, F1={conf_m['f1']:.2%}, Acc={conf_m['accuracy']:.2%}")
        print(f"  [0.4/0.6] Sep%={fixed_m['separatrix_pct']:.1%}, F1={fixed_m['f1']:.2%}, Acc={fixed_m['accuracy']:.2%}")

    # Save per-epoch outputs
    output_file = epoch_dir / "full_roa_evaluation_reconstructed.json"
    with open(output_file, 'w') as f:
        json.dump(full_roa_metrics, f, indent=2)

    # Save per-point data as NPZ
    npz_file = epoch_dir / "full_roa_evaluation_per_point_reconstructed.npz"
    np.savez(
        npz_file,
        start_states=X_all,
        probabilities=success_rate,
        true_labels=y_all,
        lambda_star=lambda_star,
        delta=delta_star
    )

    # Generate phase space plots
    phase_space_file = epoch_dir / "full_roa_evaluation_roa_phase_space_reconstructed.png"
    plot_roa_phase_space(
        start_states=X_all,
        success_rate=success_rate,
        true_labels=y_all,
        lambda_star=lambda_star,
        delta=delta_star,
        output_file=phase_space_file,
        title=f"ROA - Epoch {epoch_num}",
        threshold_type="conformal"
    )

    # Generate probability heatmap
    heatmap_file = epoch_dir / "full_roa_evaluation_roa_heatmap_reconstructed.png"
    plot_roa_probability_heatmap(
        start_states=X_all,
        success_rate=success_rate,
        lambda_star=lambda_star,
        delta=delta_star,
        output_file=heatmap_file,
        title=f"ROA - Epoch {epoch_num}"
    )

    if verbose:
        print(f"  Saved: {output_file.name}, {npz_file.name}")
        print(f"  Saved: {phase_space_file.name}, {heatmap_file.name}")

    return result


def reconstruct_results(
    results_dir: Path,
    system_type: Optional[str] = None,
    roa_labels_file: Optional[str] = None,
    num_mc_samples: int = 20,
    batch_size: int = 2048,
    attractor_radius: float = 0.1,
    device: str = 'cuda',
    max_epochs: Optional[int] = None,
    framerate: int = 2,
    create_video: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Reconstruct final_results.json from checkpoints.

    Args:
        results_dir: Path to adaptive run directory
        system_type: "pendulum" or "cartpole" (auto-detected from config if None)
        roa_labels_file: Path to roa_labels.txt (auto-detected from config if None)
        num_mc_samples: MC samples for evaluation
        batch_size: Batch size for inference
        attractor_radius: Radius for attractor classification
        device: Device for inference
        max_epochs: Maximum number of epochs to process (None = all)
        verbose: Print progress

    Returns:
        Final results dict
    """
    results_dir = Path(results_dir)

    # Load config
    cfg = load_hydra_config(results_dir)

    # Determine system type
    if system_type is None:
        if cfg and 'system' in cfg:
            # Try to infer from config
            system_target = cfg.get('system', {}).get('_target_', '')
            if 'pendulum' in system_target.lower():
                system_type = 'pendulum'
            elif 'cartpole' in system_target.lower():
                system_type = 'cartpole'

        if system_type is None:
            # Try to infer from directory name
            if 'pendulum' in results_dir.name.lower():
                system_type = 'pendulum'
            elif 'cartpole' in results_dir.name.lower():
                system_type = 'cartpole'

        if system_type is None:
            raise ValueError("Could not determine system type. Please specify --system pendulum or --system cartpole")

    if verbose:
        print(f"System type: {system_type}")

    # Get system and flow matcher class
    system, flow_matcher_class = get_system_and_flow_matcher(system_type)

    # Determine roa_labels_file
    if roa_labels_file is None:
        if cfg and 'data_source' in cfg:
            roa_labels_file = cfg['data_source'].get('roa_labels_file')

        if roa_labels_file is None:
            raise ValueError("Could not find roa_labels_file in config. Please specify --roa_labels_file")

    if verbose:
        print(f"ROA labels file: {roa_labels_file}")

    # Load ROA labels
    X_all, y_all = load_roa_labels(roa_labels_file)
    if verbose:
        print(f"Loaded {len(y_all)} points: {np.sum(y_all == 1)} success, {np.sum(y_all == -1)} failure")

    # Get attractor_radius from config if available
    if cfg and 'conformal' in cfg:
        attractor_radius = cfg['conformal'].get('attractor_radius', attractor_radius)

    # Find all epoch directories
    epoch_dirs = sorted(results_dir.glob("epoch_*"))
    if max_epochs:
        epoch_dirs = epoch_dirs[:max_epochs]

    if verbose:
        print(f"\nFound {len(epoch_dirs)} epoch directories")

    # Process each epoch
    epoch_results = []
    for epoch_dir in epoch_dirs:
        result = reconstruct_epoch(
            epoch_dir=epoch_dir,
            flow_matcher_class=flow_matcher_class,
            system=system,
            X_all=X_all,
            y_all=y_all,
            num_mc_samples=num_mc_samples,
            batch_size=batch_size,
            attractor_radius=attractor_radius,
            device=device,
            verbose=verbose,
        )
        if result is not None:
            epoch_results.append(result)

    # Build final results
    final_results = {
        'epoch_results': epoch_results,
        'final_stats': {
            'n_epochs_reconstructed': len(epoch_results),
            'system_type': system_type,
            'roa_labels_file': str(roa_labels_file),
            'num_mc_samples': num_mc_samples,
        },
        'reconstructed': True,
    }

    # Save final results
    output_file = results_dir / "final_results_reconstructed.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Create animation videos
    if create_video and len(epoch_results) > 1:
        video_paths = create_animation_from_epochs(
            results_dir=results_dir,
            epoch_results=epoch_results,
            X_all=X_all,
            y_all=y_all,
            framerate=framerate,
            verbose=verbose,
        )
        final_results['video_paths'] = {k: str(v) if v else None for k, v in video_paths.items()}

    if verbose:
        print(f"\n{'='*60}")
        print("RECONSTRUCTION COMPLETE")
        print(f"{'='*60}")
        print(f"Reconstructed {len(epoch_results)} epochs")
        print(f"Saved to: {output_file}")

        if epoch_results:
            first = epoch_results[0]['full_roa']
            last = epoch_results[-1]['full_roa']
            print(f"\n--- Full ROA Metrics Progression ---")
            print(f"[λ*±δ*] Sep%: {first['conformal_thresholds']['separatrix_pct']:.2%} -> {last['conformal_thresholds']['separatrix_pct']:.2%}")
            print(f"[λ*±δ*] F1:   {first['conformal_thresholds']['f1']:.2%} -> {last['conformal_thresholds']['f1']:.2%}")
            print(f"[0.4/0.6] F1: {first['fixed_thresholds']['f1']:.2%} -> {last['fixed_thresholds']['f1']:.2%}")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct final_results.json from checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Reconstruct results (auto-detect system from config)
    python adaptive_roa/adaptive/reconstruct_results.py \\
        --results_dir /path/to/adaptive_run/2025-12-19_03-01-24

    # Specify system type
    python adaptive_roa/adaptive/reconstruct_results.py \\
        --results_dir /path/to/run \\
        --system pendulum

    # Override evaluation parameters
    python adaptive_roa/adaptive/reconstruct_results.py \\
        --results_dir /path/to/run \\
        --num_mc_samples 50 \\
        --batch_size 4096
"""
    )

    parser.add_argument(
        '--results_dir', '-r',
        type=str,
        required=True,
        help='Path to adaptive sampling results directory'
    )
    parser.add_argument(
        '--system', '-s',
        type=str,
        choices=['pendulum', 'cartpole'],
        default=None,
        help='System type (auto-detected from config if not specified)'
    )
    parser.add_argument(
        '--roa_labels_file',
        type=str,
        default=None,
        help='Path to roa_labels.txt (auto-detected from config if not specified)'
    )
    parser.add_argument(
        '--num_mc_samples', '-n',
        type=int,
        default=20,
        help='Number of MC samples for evaluation (default: 20)'
    )
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=2048,
        help='Batch size for inference (default: 2048)'
    )
    parser.add_argument(
        '--attractor_radius',
        type=float,
        default=0.1,
        help='Radius for attractor classification (default: 0.1)'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda',
        help='Device for inference (default: cuda)'
    )
    parser.add_argument(
        '--max_epochs', '-e',
        type=int,
        default=None,
        help='Maximum number of epochs to process (default: all)'
    )
    parser.add_argument(
        '--framerate', '-f',
        type=int,
        default=2,
        help='Frames per second for animation video (default: 2)'
    )
    parser.add_argument(
        '--no_video',
        action='store_true',
        help='Skip video generation'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    reconstruct_results(
        results_dir=Path(args.results_dir),
        system_type=args.system,
        roa_labels_file=args.roa_labels_file,
        num_mc_samples=args.num_mc_samples,
        batch_size=args.batch_size,
        attractor_radius=args.attractor_radius,
        device=args.device,
        max_epochs=args.max_epochs,
        framerate=args.framerate,
        create_video=not args.no_video,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

"""
Run Adaptive Sampling Pipeline for Pendulum 2D.

This script demonstrates the full adaptive sampling loop for the pendulum system:
1. Load trajectory data source
2. Build initial endpoint dataset
3. Train flow matcher
4. Run conformal prediction to find uncertain regions
5. Add uncertain trajectories to training set
6. Repeat

Usage:
    python src/adaptive/run_adaptive_pendulum.py
    python src/adaptive/run_adaptive_pendulum.py --config-name=adaptive_pendulum
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
from src.data.pendulum_endpoint_data import PendulumEndpointDataModule

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_roa_phase_space(
    start_states: np.ndarray,
    success_rate: np.ndarray,
    true_labels: np.ndarray,
    lambda_star: float,
    delta: float,
    output_file: str,
    title: str = "Pendulum ROA Phase Space",
    show_ground_truth: bool = False,
):
    """
    Plot the Region of Attraction (ROA) as a phase space diagram.

    Args:
        start_states: [N, 2] array of (theta, theta_dot)
        success_rate: [N] array of p(success) from MC sampling
        true_labels: [N] array of ground truth labels (1=success, -1=failure)
        lambda_star: Optimal decision boundary from conformal prediction
        delta: Unknown region half-width
        output_file: Path to save the plot
        title: Plot title
        show_ground_truth: If True, plot ground truth labels instead of predictions
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    theta = start_states[:, 0]
    theta_dot = start_states[:, 1]

    # --- Left plot: Predicted ROA ---
    ax1 = axes[0]

    # Classify based on lambda* +/- delta
    pred_labels = np.zeros(len(success_rate))
    pred_labels[success_rate > lambda_star + delta] = 1   # Success
    pred_labels[success_rate < lambda_star - delta] = -1  # Failure
    # pred_labels == 0 means separatrix/uncertain

    # Create color array
    colors = np.array(['gold'] * len(pred_labels))  # Default: separatrix (yellow/gold)
    colors[pred_labels == 1] = 'blue'               # Success
    colors[pred_labels == -1] = 'red'               # Failure

    # Sort by prediction confidence for better visualization (uncertain on top)
    # Plot in order: success, failure, then separatrix (so separatrix is visible)
    order = np.argsort(np.abs(pred_labels))[::-1]  # Confident first, uncertain last

    scatter1 = ax1.scatter(
        theta[order], theta_dot[order],
        c=colors[order],
        s=3, alpha=0.6, edgecolors='none'
    )

    ax1.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax1.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
    ax1.set_title(f'Predicted ROA (λ*={lambda_star:.3f}, δ={delta:.3f})', fontsize=12)
    ax1.set_xlim(-np.pi, np.pi)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # Add pi labels on x-axis
    ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax1.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    # Count labels
    n_success = np.sum(pred_labels == 1)
    n_failure = np.sum(pred_labels == -1)
    n_separatrix = np.sum(pred_labels == 0)

    # Legend
    legend_elements = [
        mpatches.Patch(color='blue', label=f'Success ({n_success:,})'),
        mpatches.Patch(color='red', label=f'Failure ({n_failure:,})'),
        mpatches.Patch(color='gold', label=f'Separatrix ({n_separatrix:,})'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Right plot: Ground Truth ROA ---
    ax2 = axes[1]

    # Create color array for ground truth
    gt_colors = np.array(['gold'] * len(true_labels))  # Default: separatrix
    gt_colors[true_labels == 1] = 'blue'               # Success
    gt_colors[true_labels == -1] = 'red'               # Failure

    # Sort for visualization
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

    # Count ground truth labels
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

    print(f"Saved ROA phase space plot to: {output_file}")

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
    output_file: str,
    title: str = "Pendulum ROA - Success Probability",
):
    """
    Plot the ROA as a heatmap of success probability.

    Args:
        start_states: [N, 2] array of (theta, theta_dot)
        success_rate: [N] array of p(success) from MC sampling
        lambda_star: Optimal decision boundary
        delta: Unknown region half-width
        output_file: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    theta = start_states[:, 0]
    theta_dot = start_states[:, 1]

    # Create scatter plot with color based on success probability
    scatter = ax.scatter(
        theta, theta_dot,
        c=success_rate,
        cmap='RdYlBu',  # Red (low) -> Yellow (mid) -> Blue (high)
        s=3, alpha=0.7, edgecolors='none',
        vmin=0, vmax=1
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='p(success)')

    # Mark the decision boundaries on colorbar
    cbar.ax.axhline(y=lambda_star + delta, color='black', linestyle='--', linewidth=1.5)
    cbar.ax.axhline(y=lambda_star - delta, color='black', linestyle='--', linewidth=1.5)
    cbar.ax.axhline(y=lambda_star, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
    ax.set_title(f'{title}\n(λ*={lambda_star:.3f}, decision bounds: [{lambda_star-delta:.3f}, {lambda_star+delta:.3f}])', fontsize=12)
    ax.set_xlim(-np.pi, np.pi)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # Pi labels
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved ROA probability heatmap to: {output_file}")


def compute_metrics_at_threshold(success_rate: np.ndarray, y_all: np.ndarray,
                                  success_thresh: float, failure_thresh: float) -> Dict:
    """Helper to compute metrics at given thresholds."""
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


def evaluate_full_roa_fast(
    flow_matcher,
    system,
    data_source: TrajectoryDataSource,
    num_mc_samples: int = 20,
    batch_size: int = 2048,
    lambda_star: float = None,
    delta: float = 0.05,
    attractor_radius: float = 0.1,
    device: str = 'cuda',
    output_file: str = None,
    verbose: bool = True
) -> Dict:
    """
    Fast batched evaluation on the ENTIRE roa_labels.txt dataset.

    Uses direct batched inference instead of conformal predictor for speed.
    Processes batch_size points at a time with num_mc_samples forward passes.

    Saves per-point data and computes metrics for BOTH threshold schemes:
    1. lambda* +/- delta from conformal prediction
    2. Fixed 0.4/0.6 thresholds

    Args:
        flow_matcher: Trained flow matcher model
        system: System for classify_attractor
        data_source: TrajectoryDataSource with all labels
        num_mc_samples: Number of MC samples per point
        batch_size: Batch size for GPU inference
        lambda_star: Optimized lambda* from conformal prediction (if None, use 0.5)
        delta: Unknown region half-width from conformal config
        attractor_radius: Radius for attractor classification
        device: Device for inference
        output_file: Optional path to save results JSON
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

    # Default lambda* if not provided
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

    # Compute success rate per point (p_success = fraction of samples reaching success attractor)
    success_rate = (is_success == 1).sum(axis=1) / num_mc_samples

    # Compute metrics for BOTH threshold schemes
    # 1. lambda* +/- delta from conformal prediction
    metrics_conformal = compute_metrics_at_threshold(
        success_rate, y_all,
        success_thresh=lambda_star + delta,
        failure_thresh=lambda_star - delta
    )

    # 2. Fixed 0.4/0.6 thresholds (classic approach)
    metrics_fixed = compute_metrics_at_threshold(
        success_rate, y_all,
        success_thresh=0.6,
        failure_thresh=0.4
    )

    if verbose:
        print(f"\n{'='*60}")
        print("METRICS WITH lambda* +/- delta THRESHOLDS (from conformal prediction)")
        print(f"{'='*60}")
        print(f"lambda*={lambda_star:.4f}, delta={delta:.4f}")
        print(f"  Success if p > {lambda_star + delta:.4f}")
        print(f"  Failure if p < {lambda_star - delta:.4f}")
        print(f"Separatrix %:    {metrics_conformal['separatrix_pct']:.2%}")
        print(f"Confident:       {metrics_conformal['n_confident']} predictions")
        print(f"Accuracy:        {metrics_conformal['accuracy']:.2%}")
        print(f"F1 Score:        {metrics_conformal['f1']:.2%}")
        print(f"Precision:       {metrics_conformal['precision']:.2%}")
        print(f"Recall:          {metrics_conformal['recall']:.2%}")
        print(f"Specificity:     {metrics_conformal['specificity']:.2%}")

        print(f"\n{'='*60}")
        print("METRICS WITH FIXED 0.4/0.6 THRESHOLDS")
        print(f"{'='*60}")
        print(f"  Success if p > 0.6")
        print(f"  Failure if p < 0.4")
        print(f"Separatrix %:    {metrics_fixed['separatrix_pct']:.2%}")
        print(f"Confident:       {metrics_fixed['n_confident']} predictions")
        print(f"Accuracy:        {metrics_fixed['accuracy']:.2%}")
        print(f"F1 Score:        {metrics_fixed['f1']:.2%}")
        print(f"Precision:       {metrics_fixed['precision']:.2%}")
        print(f"Recall:          {metrics_fixed['recall']:.2%}")
        print(f"Specificity:     {metrics_fixed['specificity']:.2%}")

    # Combine metrics
    metrics = {
        'n_total': n_total,
        'num_mc_samples': num_mc_samples,
        'lambda_star': float(lambda_star),
        'delta': float(delta),
        'conformal_thresholds': metrics_conformal,
        'fixed_thresholds': metrics_fixed,
        # Keep top-level metrics for backward compatibility (using conformal)
        'separatrix_pct': metrics_conformal['separatrix_pct'],
        'accuracy': metrics_conformal['accuracy'],
        'precision': metrics_conformal['precision'],
        'recall': metrics_conformal['recall'],
        'specificity': metrics_conformal['specificity'],
        'f1': metrics_conformal['f1'],
    }

    # Save to files if specified
    if output_file:
        # Save JSON metrics
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        if verbose:
            print(f"\nSaved metrics to: {output_file}")

        # Save per-point data as NPZ for analysis
        npz_file = output_file.replace('.json', '_per_point.npz')
        np.savez(
            npz_file,
            start_states=X_all,           # [N, 2] - (theta, theta_dot)
            probabilities=success_rate,   # [N] - p_success
            true_labels=y_all,            # [N] - ground truth labels
            lambda_star=lambda_star,
            delta=delta
        )
        if verbose:
            print(f"Saved per-point data to: {npz_file}")
            print(f"  Arrays: start_states [{X_all.shape}], probabilities [{success_rate.shape}], true_labels [{y_all.shape}]")

        # Generate ROA phase space plots
        phase_space_file = output_file.replace('.json', '_roa_phase_space.png')
        plot_roa_phase_space(
            start_states=X_all,
            success_rate=success_rate,
            true_labels=y_all,
            lambda_star=lambda_star,
            delta=delta,
            output_file=phase_space_file,
            title=f"Pendulum ROA - Epoch Evaluation"
        )

        # Generate probability heatmap
        heatmap_file = output_file.replace('.json', '_roa_heatmap.png')
        plot_roa_probability_heatmap(
            start_states=X_all,
            success_rate=success_rate,
            lambda_star=lambda_star,
            delta=delta,
            output_file=heatmap_file,
            title="Pendulum ROA - Success Probability"
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
    data_module = PendulumEndpointDataModule(
        data_file=train_file,
        validation_file=val_file,
        test_file=val_file,  # Use val as test for now
        batch_size=cfg.get('batch_size', 64),
        val_batch_size=cfg.get('val_batch_size', 2048),
        num_workers=cfg.get('num_workers', 4),
    )

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model)

    # Instantiate flow matcher
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
        max_epochs=max_epochs,
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


@hydra.main(config_path="../../configs", config_name="adaptive_pendulum", version_base=None)
def main(cfg: DictConfig):
    """Main adaptive sampling loop for Pendulum."""
    print("=" * 70)
    print("ADAPTIVE SAMPLING PIPELINE - Pendulum 2D")
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
        attractor_radius=cfg.conformal.get('attractor_radius', 0.1),
        # Optimization mode: "lambda" or "delta"
        optimize_mode=cfg.conformal.get('optimize_mode', 'lambda'),
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
    previous_best_checkpoint = None

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
        num_mc_samples_eval = cfg.conformal.get('num_mc_samples_eval', 20)
        conformal_state = conformal_predictor.get_state()
        lambda_star = conformal_state['lambda_star']
        delta_star = conformal_state['delta_star']  # Use optimized delta (may differ from config if optimize_mode="delta")

        print(f"\n[5] Evaluating on FULL roa_labels.txt ({num_mc_samples_eval} MC samples, fast batched)...")
        print(f"    Using lambda*={lambda_star:.4f} +/- delta*={delta_star:.4f} from conformal prediction")
        full_roa_output_file = epoch_output_dir / "full_roa_evaluation.json"
        full_roa_metrics = evaluate_full_roa_fast(
            flow_matcher=flow_matcher,
            system=system,
            data_source=data_source,
            num_mc_samples=num_mc_samples_eval,
            batch_size=cfg.get('val_batch_size', 2048),
            lambda_star=lambda_star,
            delta=delta_star,
            attractor_radius=cfg.conformal.get('attractor_radius', 0.1),
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

            # Sample epoch
            sample_result = balanced_sampler.sample_epoch(
                prob_estimator=prob_estimator,
                lambda_star=lambda_star,
                delta_star=delta_star,
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
            n_d1 = int(len(candidate_indices) * d1_ratio)
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
                uncertain_mask, uncertain_idx, p_success, _ = conformal_predictor.select_uncertain(d2_states)

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
        print(f"  lambda* = {epoch_result['lambda_star']:.4f}, delta* = {epoch_result['delta_star']:.4f}, q_hat = {epoch_result['q_hat']:.4f}")
        print(f"  --- Full ROA (all {full_roa_metrics['n_total']} trajectories) ---")
        conf_m = full_roa_metrics['conformal_thresholds']
        fixed_m = full_roa_metrics['fixed_thresholds']
        print(f"  [lambda*+/-delta] Sep%={conf_m['separatrix_pct']:.1%}, F1={conf_m['f1']:.2%}, Acc={conf_m['accuracy']:.2%}")
        print(f"  [0.4/0.6] Sep%={fixed_m['separatrix_pct']:.1%}, F1={fixed_m['f1']:.2%}, Acc={fixed_m['accuracy']:.2%}")

        # Save epoch results
        with open(epoch_output_dir / "results.json", 'w') as f:
            json.dump(epoch_result, f, indent=2)

        # Save conformal predictor state
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
        print(f"\n  [Fixed 0.4/0.6 Thresholds]")
        print(f"  Separatrix %:  {first['fixed_thresholds']['separatrix_pct']:.2%} -> {last['fixed_thresholds']['separatrix_pct']:.2%}")
        print(f"  F1 Score:      {first['fixed_thresholds']['f1']:.2%} -> {last['fixed_thresholds']['f1']:.2%}")
        print(f"  Accuracy:      {first['fixed_thresholds']['accuracy']:.2%} -> {last['fixed_thresholds']['accuracy']:.2%}")
        print(f"\n  lambda*:       {epoch_results[0]['lambda_star']:.4f} -> {epoch_results[-1]['lambda_star']:.4f}")
        print(f"  delta*:        {epoch_results[0]['delta_star']:.4f} -> {epoch_results[-1]['delta_star']:.4f}")
        print(f"  q_hat:         {epoch_results[0]['q_hat']:.4f} -> {epoch_results[-1]['q_hat']:.4f}")

    # Save final results
    with open(output_dir / "final_results.json", 'w') as f:
        json.dump({
            'epoch_results': epoch_results,
            'final_stats': stats,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ROA Evaluation script for Latent Conditional Flow Matching (Facebook FM)

Integrates with Hydra config system for clean, consistent evaluation.

Usage:
    # Evaluate CartPole (auto-finds latest checkpoint)
    python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa

    # With specific checkpoint
    python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa \
        checkpoint.path=outputs/cartpole_latent_conditional_fm/.../checkpoints/best.ckpt

    # Probabilistic mode
    python src/flow_matching/evaluate_roa.py --config-name=evaluate_cartpole_roa \
        evaluation.probabilistic=true \
        evaluation.num_samples=100
"""
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import importlib


def load_roa_data(data_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ROA labeled data with automatic format detection

    Handles two formats:
    1. With index: "index θ θ̇ label" (4 columns for pendulum)
    2. Without index: "θ θ̇ label" (3 columns for pendulum)
    3. CartPole: "index x θ ẋ θ̇ label" (6 columns) or "x θ ẋ θ̇ label" (5 columns)
    """
    print(f"📂 Loading data from: {data_file}")
    data = np.loadtxt(data_file, delimiter=',')

    num_cols = data.shape[1]
    print(f"   Detected {num_cols} columns")

    # Last column is always the label
    labels = data[:, -1].astype(int)

    # Detect format based on number of columns
    if num_cols == 3:
        # Format: θ θ̇ label (no index)
        states = data[:, :-1]
        system_name = "Pendulum (2D)"
    elif num_cols == 4:
        # Format: index θ θ̇ label (with index)
        states = data[:, 1:3]  # Skip first column (index), take θ and θ̇
        system_name = "Pendulum (2D)"
    elif num_cols == 5:
        # Format: x θ ẋ θ̇ label (no index)
        states = data[:, :-1]
        states[:, 1] = np.arctan2(np.sin(states[:, 1]), np.cos(states[:, 1]))
        system_name = "CartPole (4D)"
    elif num_cols == 6:
        # Format: index x θ ẋ θ̇ label (with index)
        states = data[:, 1:5]  # Skip first column (index), take state variables
        states[:, 1] = np.arctan2(np.sin(states[:, 1]), np.cos(states[:, 1]))
        system_name = "CartPole (4D)"
    else:
        # Fallback: assume last column is label, rest are states (no index)
        states = data[:, :-1]
        system_name = f"Unknown ({states.shape[1]}D)"

    print(f"   System: {system_name}")
    print(f"   State shape: {states.shape}")
    print(f"   Loaded {len(states)} samples")
    print(f"   Success (label=1): {(labels == 1).sum()} ({(labels == 1).mean():.1%})")
    print(f"   Failure (label=0): {(labels == 0).sum()} ({(labels == 0).mean():.1%})")
    print()

    return states, labels


def find_checkpoint_in_folder(folder_path: str) -> str:
    """
    Find the best checkpoint in a given folder based on validation loss

    Args:
        folder_path: Path to training run folder (e.g., outputs/.../2025-10-10_12-30-26)

    Returns:
        Path to best checkpoint file (lowest validation loss)
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Look for checkpoints in version_0/checkpoints/
    checkpoint_dir = folder / "version_0" / "checkpoints"

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in {folder}")

    # Find all .ckpt files
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))

    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")

    # Parse validation loss from checkpoint filenames (format: epoch42-val_loss0.1234.ckpt)
    import re
    best_ckpt = None
    best_loss = float('inf')
    checkpoints_with_loss = []

    for ckpt in checkpoints:
        # Skip last.ckpt for now
        if ckpt.name == "last.ckpt":
            continue

        # Try to extract val_loss from filename (format: val_loss0.1234.ckpt or val_loss0.1234)
        match = re.search(r'val_loss(\d+\.\d+)', ckpt.name)
        if match:
            val_loss = float(match.group(1))
            checkpoints_with_loss.append((ckpt, val_loss))
            if val_loss < best_loss:
                best_loss = val_loss
                best_ckpt = ckpt

    # Print checkpoint statistics
    if checkpoints_with_loss:
        print(f"   Found {len(checkpoints_with_loss)} checkpoints with validation loss:")
        # Sort by loss for display
        sorted_ckpts = sorted(checkpoints_with_loss, key=lambda x: x[1])
        for i, (ckpt, loss) in enumerate(sorted_ckpts[:5]):  # Show top 5
            marker = " ← SELECTED" if ckpt == best_ckpt else ""
            print(f"     {i+1}. {ckpt.name} (val_loss={loss:.4f}){marker}")
        if len(sorted_ckpts) > 5:
            print(f"     ... and {len(sorted_ckpts) - 5} more")
    else:
        print(f"   Found {len(checkpoints)} checkpoints (no validation loss in filenames)")

    # If we found a checkpoint with validation loss, use it
    if best_ckpt is not None:
        return str(best_ckpt)

    # Fallback: prefer 'last.ckpt' if it exists
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)

    # Otherwise, return most recent checkpoint by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def find_latest_checkpoint(base_name: str) -> str:
    """Find the most recent checkpoint across all runs"""
    base_dir = Path("outputs") / base_name

    if not base_dir.exists():
        raise FileNotFoundError(f"No training outputs found at {base_dir}")

    # Find most recent run directory
    run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()],
                     key=lambda x: x.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        try:
            return find_checkpoint_in_folder(str(run_dir))
        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"No checkpoints found in {base_dir}")


def load_flow_matcher_class(module_path: str, class_name: str):
    """Dynamically load flow matcher class from config"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def evaluate_deterministic(flow_matcher,
                          states: np.ndarray,
                          labels: np.ndarray,
                          batch_size: int,
                          num_steps: int,
                          attractor_radius: float) -> Dict[str, Any]:
    """Deterministic evaluation: single prediction per state"""
    print("🔬 Running deterministic evaluation...")

    device = next(flow_matcher.parameters()).device
    states_tensor = torch.from_numpy(states).float().to(device)

    # Predict endpoints
    all_endpoints = []
    num_batches = int(np.ceil(len(states) / batch_size))

    for i in tqdm(range(num_batches), desc="Predicting endpoints"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(states))
        batch = states_tensor[start_idx:end_idx]

        with torch.no_grad():
            endpoints = flow_matcher.predict_endpoint(
                start_states=batch,
                num_steps=num_steps,
                latent=None
            )

        all_endpoints.append(endpoints.cpu())

    endpoints = torch.cat(all_endpoints, dim=0)

    # Wrap angles to [-π, π] before checking attractor membership
    # For CartPole: state is [x, θ, ẋ, θ̇], wrap θ at index 1
    if endpoints.shape[1] == 4:  # CartPole
        endpoints[:, 1] = torch.atan2(torch.sin(endpoints[:, 1]), torch.cos(endpoints[:, 1]))
    elif endpoints.shape[1] == 2:  # Pendulum
        endpoints[:, 0] = torch.atan2(torch.sin(endpoints[:, 0]), torch.cos(endpoints[:, 0]))

    # Check attractor membership (on wrapped angles)
    in_attractor = flow_matcher.system.is_in_attractor(endpoints, radius=attractor_radius)
    # Convert to numpy if needed (is_in_attractor might return tensor or numpy)
    if isinstance(in_attractor, torch.Tensor):
        in_attractor = in_attractor.cpu().numpy()
    predictions = in_attractor.astype(int)

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    conf_matrix = confusion_matrix(labels, predictions)

    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'predictions': predictions,
        'endpoints': endpoints,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': recall,
        'specificity': specificity,
        'confusion_matrix': conf_matrix,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def evaluate_probabilistic(flow_matcher,
                          states: np.ndarray,
                          labels: np.ndarray,
                          batch_size: int,
                          num_samples: int,
                          num_steps: int,
                          attractor_radius: float,
                          confidence_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Probabilistic evaluation with three-way classification

    Per-sample classification (system-specific):
        Pendulum:
            1: Endpoint in stable bottom attractor [0, 0] (success)
           -1: Endpoint in unstable top attractors (failure)
            0: Endpoint in separatrix (not in any attractor)

        CartPole:
            1: Endpoint in balanced attractor [0, 0, 0, 0] (success)
           -1: Endpoint exceeded termination thresholds (failure - crashed)
            0: Endpoint between attractor and failure (separatrix - uncertain)

    Per-state aggregation (over num_samples):
        Success (1): ≥confidence_threshold of samples land in success attractor
        Failure (0): ≥confidence_threshold of samples land in failure attractor
        Separatrix (-1): < confidence_threshold for both (uncertain/mixed) → EXCLUDED from metrics
    """
    print(f"🔬 Running probabilistic evaluation ({num_samples} samples/state)...")

    device = next(flow_matcher.parameters()).device
    states_tensor = torch.from_numpy(states).float().to(device)

    # Track counts for each class: [stable(1), unstable(-1), separatrix(0)]
    all_class_counts = []
    all_first_endpoints = []  # Store first sampled endpoint for each state
    num_batches = int(np.ceil(len(states) / batch_size))

    for i in tqdm(range(num_batches), desc="Sampling predictions"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(states))
        batch = states_tensor[start_idx:end_idx]

        # Count for each class: [stable_count, unstable_count, separatrix_count]
        class_counts = torch.zeros((len(batch), 3), device=device)
        first_endpoints_batch = None

        for sample_idx in range(num_samples):
            with torch.no_grad():
                endpoints = flow_matcher.predict_endpoint(
                    start_states=batch,
                    num_steps=num_steps,
                    latent=None
                )

            # Wrap angles to [-π, π] before classification
            if endpoints.shape[1] == 4:  # CartPole
                endpoints[:, 1] = torch.atan2(torch.sin(endpoints[:, 1]), torch.cos(endpoints[:, 1]))
            elif endpoints.shape[1] == 2:  # Pendulum
                endpoints[:, 0] = torch.atan2(torch.sin(endpoints[:, 0]), torch.cos(endpoints[:, 0]))

            # Save first endpoint sample (after wrapping)
            if sample_idx == 0:
                first_endpoints_batch = endpoints.cpu()

            # Get three-way classification: 1 (stable), -1 (unstable), 0 (separatrix)
            attractor_labels = flow_matcher.system.classify_attractor(endpoints, radius=attractor_radius)

            # Count each class
            class_counts[:, 0] += (attractor_labels == 1).float()   # Stable
            class_counts[:, 1] += (attractor_labels == -1).float()  # Unstable
            class_counts[:, 2] += (attractor_labels == 0).float()   # Separatrix

        all_class_counts.append(class_counts.cpu())
        all_first_endpoints.append(first_endpoints_batch)

    class_counts = torch.cat(all_class_counts, dim=0).numpy()  # [N, 3]
    first_endpoints = torch.cat(all_first_endpoints, dim=0)  # [N, state_dim]

    # Compute proportions
    p_stable = class_counts[:, 0] / num_samples      # Proportion landing in stable/success
    p_unstable = class_counts[:, 1] / num_samples    # Proportion landing in unstable/failure
    p_separatrix = class_counts[:, 2] / num_samples  # Proportion landing in separatrix (pendulum only)

    # Classify each state based on confidence threshold
    # Strategy:
    # - If ≥threshold of samples land in success attractor → label as success
    # - If ≥threshold of samples land in failure attractor → label as failure
    # - Otherwise (< threshold for both) → label as separatrix (uncertain)

    predictions = np.full(len(states), -1, dtype=int)  # Initialize all as separatrix

    # Assign success/failure based on confidence threshold
    # Only label if we're confident (≥threshold agreement)
    predictions[p_stable >= confidence_threshold] = 1      # Success: ≥threshold landed in stable/success attractor
    predictions[p_unstable >= confidence_threshold] = 0    # Failure: ≥threshold landed in unstable/failure attractor

    # States that remain -1 (separatrix):
    # - p_stable < threshold AND p_unstable < threshold
    # - These are uncertain states where predictions are split

    # Mark for exclusion
    is_separatrix = (predictions == -1)

    # Filter out separatrix points
    valid_mask = ~is_separatrix
    valid_states = states[valid_mask]
    valid_labels = labels[valid_mask]
    valid_predictions = predictions[valid_mask]

    # Report statistics
    n_total = len(states)
    n_separatrix = is_separatrix.sum()
    n_valid = valid_mask.sum()
    pct_separatrix = 100.0 * n_separatrix / n_total

    print(f"\n📊 Classification Summary:")
    print(f"   Total states: {n_total}")
    print(f"   Separatrix (excluded): {n_separatrix} ({pct_separatrix:.1f}%)")
    print(f"   Valid for evaluation: {n_valid} ({100.0 * n_valid / n_total:.1f}%)")

    # Compute metrics only on valid (non-separatrix) states
    if n_valid == 0:
        print("\n⚠️  WARNING: All states classified as separatrix! Cannot compute metrics.")
        return {
            'predictions': predictions,
            'p_stable': p_stable,
            'p_unstable': p_unstable,
            'p_separatrix': p_separatrix,
            'separatrix_count': int(n_separatrix),
            'separatrix_percentage': pct_separatrix,
            'valid_count': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    accuracy = accuracy_score(valid_labels, valid_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_predictions, average='binary', zero_division=0
    )
    conf_matrix = confusion_matrix(valid_labels, valid_predictions)
    tn, fp, fn, tp = conf_matrix.ravel()

    # AUC using probability of stable attractor
    auc = roc_auc_score(valid_labels, p_stable[valid_mask]) if len(np.unique(valid_labels)) > 1 else None
    fpr, tpr, thresholds = roc_curve(valid_labels, p_stable[valid_mask]) if len(np.unique(valid_labels)) > 1 else (None, None, None)

    # Entropy based on stable vs unstable (excluding separatrix)
    p_stable_normalized = p_stable[valid_mask] / (p_stable[valid_mask] + p_unstable[valid_mask] + 1e-9)
    p_stable_clipped = np.clip(p_stable_normalized, 1e-7, 1 - 1e-7)
    entropy = -(p_stable_clipped * np.log(p_stable_clipped) +
                (1 - p_stable_clipped) * np.log(1 - p_stable_clipped))

    return {
        'predictions': predictions,
        'endpoints': first_endpoints,  # First sampled endpoint for each state
        'valid_mask': valid_mask,
        'p_stable': p_stable,
        'p_unstable': p_unstable,
        'p_separatrix': p_separatrix,
        'separatrix_count': int(n_separatrix),
        'separatrix_percentage': pct_separatrix,
        'valid_count': int(n_valid),
        'entropy': entropy,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': recall,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'auc': auc,
        'confusion_matrix': conf_matrix,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'roc_curve': (fpr, tpr, thresholds)
    }


def print_results(results: Dict[str, Any], probabilistic: bool):
    """Print evaluation metrics"""
    print("\n" + "="*80)
    print("📊 Evaluation Results")
    print("="*80)

    # Show separatrix statistics if available
    if 'separatrix_percentage' in results:
        print(f"\n🔀 Separatrix Analysis:")
        print(f"   States on separatrix: {results['separatrix_count']} ({results['separatrix_percentage']:.1f}%)")
        print(f"   Valid for evaluation: {results['valid_count']}")

    print(f"\n📈 Performance Metrics (on valid states only):")
    print(f"   Accuracy:    {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Precision:   {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"   Recall:      {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"   F1 Score:    {results['f1']:.4f}")
    print(f"   Sensitivity: {results['sensitivity']:.4f} (same as Recall)")
    print(f"   Specificity: {results['specificity']:.4f} ({results['specificity']*100:.2f}%)")
    if probabilistic and results.get('auc'):
        print(f"   AUC:         {results['auc']:.4f}")

    print("\n📊 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Fail    Success")
    print(f"   Actual Fail   {results['tn']:5d}  {results['fp']:5d}  (Total: {results['tn'] + results['fp']})")
    print(f"        Success  {results['fn']:5d}  {results['tp']:5d}  (Total: {results['fn'] + results['tp']})")
    print(f"   Total:        {results['tn'] + results['fn']:5d}  {results['fp'] + results['tp']:5d}  (Total: {results['tn'] + results['fp'] + results['fn'] + results['tp']})")

    # Detailed error analysis
    print("\n🔍 Detailed Error Analysis:")
    total = results['tp'] + results['tn'] + results['fp'] + results['fn']
    print(f"   True Positives (TP):   {results['tp']:4d} ({results['tp']/total*100:5.2f}%) - Correctly predicted Success")
    print(f"   True Negatives (TN):   {results['tn']:4d} ({results['tn']/total*100:5.2f}%) - Correctly predicted Failure")
    print(f"   False Positives (FP):  {results['fp']:4d} ({results['fp']/total*100:5.2f}%) - Predicted Success, Actually Failure")
    print(f"   False Negatives (FN):  {results['fn']:4d} ({results['fn']/total*100:5.2f}%) - Predicted Failure, Actually Success")
    print(f"   Total Correct:         {results['tp'] + results['tn']:4d} ({(results['tp'] + results['tn'])/total*100:5.2f}%)")
    print(f"   Total Incorrect:       {results['fp'] + results['fn']:4d} ({(results['fp'] + results['fn'])/total*100:5.2f}%)")

    # Error rates
    print("\n📉 Error Rates:")
    print(f"   False Positive Rate:   {results['fp']/(results['fp'] + results['tn'])*100:5.2f}% (FP / All Actual Failures)")
    print(f"   False Negative Rate:   {results['fn']/(results['fn'] + results['tp'])*100:5.2f}% (FN / All Actual Successes)")
    if results['tp'] + results['fp'] > 0:
        print(f"   False Discovery Rate:  {results['fp']/(results['fp'] + results['tp'])*100:5.2f}% (FP / All Predicted Successes)")
    if results['tn'] + results['fn'] > 0:
        print(f"   False Omission Rate:   {results['fn']/(results['fn'] + results['tn'])*100:5.2f}% (FN / All Predicted Failures)")

    if probabilistic:
        print("\n🎲 Uncertainty Analysis:")
        print(f"   Mean entropy: {results['entropy'].mean():.4f}")
        print(f"   Std entropy:  {results['entropy'].std():.4f}")
        print(f"   Max entropy:  {results['entropy'].max():.4f}")
        print(f"   Min entropy:  {results['entropy'].min():.4f}")
    print()


def save_plots(results: Dict[str, Any], states: np.ndarray, labels: np.ndarray,
               output_dir: Path, probabilistic: bool):
    """Generate and save visualization plots"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = results['confusion_matrix']
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Failure', 'Success'])
    ax.set_yticklabels(['Failure', 'Success'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=20)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    print(f"   Saved: confusion_matrix.png")

    # Probabilistic plots
    if probabilistic:
        # Use valid_mask if available, otherwise use all points
        if 'valid_mask' in results:
            valid_mask = results['valid_mask']
            valid_labels = labels[valid_mask]
            p_stable = results['p_stable'][valid_mask]
        else:
            valid_labels = labels
            p_stable = results.get('p_stable', results.get('p_success'))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # P(success) distribution - using p_stable
        axes[0].hist(p_stable[valid_labels == 0], bins=50, alpha=0.5, label='Failure', color='red')
        axes[0].hist(p_stable[valid_labels == 1], bins=50, alpha=0.5, label='Success', color='green')
        axes[0].axvline(0.5, color='black', linestyle='--')
        axes[0].set_xlabel('P(Stable Attractor)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Probability Distribution')
        axes[0].legend()

        # Entropy
        axes[1].hist(results['entropy'], bins=50, color='purple', alpha=0.7)
        axes[1].set_xlabel('Entropy')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Uncertainty')

        plt.tight_layout()
        plt.savefig(output_dir / 'probability_distributions.png', dpi=150)
        plt.close()
        print(f"   Saved: probability_distributions.png")

        # ROC curve
        if results.get('auc'):
            fpr, tpr, _ = results['roc_curve']
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC={results["auc"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'roc_curve.png', dpi=150)
            plt.close()
            print(f"   Saved: roc_curve.png")

    # Error analysis - adapt to state dimensionality
    state_dim = states.shape[1]
    correct = results['predictions'] == labels
    incorrect = ~correct

    if state_dim == 4:
        # CartPole 4D
        dim_names = ['Cart Position', 'Pole Angle', 'Cart Velocity', 'Angular Velocity']
        dim_pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    elif state_dim == 2:
        # Pendulum 2D
        dim_names = ['Angle', 'Angular Velocity']
        dim_pairs = [(0, 1)]
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = np.array([axes])  # Make it iterable
    else:
        # Generic case
        dim_names = [f'Dim {i}' for i in range(state_dim)]
        # Plot first few dimensions
        dim_pairs = [(i, i+1) for i in range(min(state_dim-1, 3))]
        n_plots = len(dim_pairs)
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
        if n_plots == 1:
            axes = np.array([axes])

    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, (i, j) in enumerate(dim_pairs):
        ax = axes_flat[idx]
        ax.scatter(states[correct, i], states[correct, j], c=labels[correct],
                  cmap='RdYlGn', alpha=0.3, s=10, vmin=0, vmax=1)
        if incorrect.sum() > 0:
            ax.scatter(states[incorrect, i], states[incorrect, j], c=labels[incorrect],
                      cmap='RdYlGn', alpha=0.8, s=50, marker='x', linewidths=2, vmin=0, vmax=1)
        ax.set_xlabel(dim_names[i])
        ax.set_ylabel(dim_names[j])
        ax.set_title(f'{dim_names[i]} vs {dim_names[j]}')

    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=150)
    plt.close()
    print(f"   Saved: error_analysis.png")

    # Three-way classification plot (success/failure/separatrix)
    # In deterministic mode: only success/failure (no separatrix)
    # In probabilistic mode: success/failure/separatrix (uncertain states)
    preds = results['predictions']

    if state_dim == 2:
        # Pendulum 2D
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot each category with different color
        success_mask = preds == 1
        failure_mask = preds == 0
        separatrix_mask = preds == -1

        if success_mask.sum() > 0:
            ax.scatter(states[success_mask, 0], states[success_mask, 1],
                      c='green', alpha=0.5, s=15, label=f'Success ({success_mask.sum()})')
        if failure_mask.sum() > 0:
            ax.scatter(states[failure_mask, 0], states[failure_mask, 1],
                      c='red', alpha=0.5, s=15, label=f'Failure ({failure_mask.sum()})')
        if separatrix_mask.sum() > 0:
            ax.scatter(states[separatrix_mask, 0], states[separatrix_mask, 1],
                      c='orange', alpha=0.7, s=20, marker='^', label=f'Separatrix ({separatrix_mask.sum()})')

        ax.set_xlabel('Angle (rad)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('State Space Classification (Success/Failure/Separatrix)')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'state_space_classification.png', dpi=150)
        plt.close()
        print(f"   Saved: state_space_classification.png")

    elif state_dim == 4:
        # CartPole 4D - show multiple 2D projections
        dim_names = ['Cart Position', 'Pole Angle', 'Cart Velocity', 'Angular Velocity']
        dim_pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes_flat = axes.flatten()

        success_mask = preds == 1
        failure_mask = preds == 0
        separatrix_mask = preds == -1

        for idx, (i, j) in enumerate(dim_pairs):
            ax = axes_flat[idx]

            if success_mask.sum() > 0:
                ax.scatter(states[success_mask, i], states[success_mask, j],
                          c='green', alpha=0.4, s=10, label=f'Success ({success_mask.sum()})')
            if failure_mask.sum() > 0:
                ax.scatter(states[failure_mask, i], states[failure_mask, j],
                          c='red', alpha=0.4, s=10, label=f'Failure ({failure_mask.sum()})')
            if separatrix_mask.sum() > 0:
                ax.scatter(states[separatrix_mask, i], states[separatrix_mask, j],
                          c='orange', alpha=0.7, s=15, marker='^', label=f'Separatrix ({separatrix_mask.sum()})')

            ax.set_xlabel(dim_names[i])
            ax.set_ylabel(dim_names[j])
            ax.set_title(f'{dim_names[i]} vs {dim_names[j]}')
            if idx == 0:
                ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle('State Space Classification (Success/Failure/Separatrix)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'state_space_classification.png', dpi=150)
        plt.close()
        print(f"   Saved: state_space_classification.png")


@hydra.main(version_base=None, config_path="../../configs", config_name="evaluate_cartpole_roa")
def main(cfg: DictConfig):
    """Main evaluation function using Hydra config"""

    print("="*80)
    print("🔍 ROA Evaluation (Facebook FM)")
    print("="*80)
    print(f"📋 Config: {cfg.name}")
    print("="*80)
    print()

    # Load data
    print("📂 Loading Data")
    print("="*80)
    states, labels = load_roa_data(cfg.data.file)

    # Find checkpoint
    print("🤖 Loading Model")
    print("="*80)
    if cfg.checkpoint.path is None and cfg.checkpoint.auto_find:
        print(f"   🔍 Auto-finding latest checkpoint in outputs/{cfg.checkpoint.base_name}/...")
        checkpoint_path = find_latest_checkpoint(cfg.checkpoint.base_name)
        print(f"   ✓ Found: {checkpoint_path}")
    else:
        checkpoint_path = cfg.checkpoint.path
        # Check if it's a folder or a .ckpt file
        path_obj = Path(checkpoint_path)
        if path_obj.is_dir():
            print(f"   📁 Folder provided: {checkpoint_path}")
            print(f"   🔍 Finding best checkpoint in folder...")
            checkpoint_path = find_checkpoint_in_folder(checkpoint_path)
            print(f"   ✓ Found: {checkpoint_path}")
        else:
            print(f"   📄 Checkpoint file: {checkpoint_path}")

    # Dynamically load flow matcher class
    class_name = getattr(cfg.system, 'class')
    module_name = cfg.system.module
    print(f"   Loading {class_name} from {module_name}")
    FlowMatcherClass = load_flow_matcher_class(module_name, class_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    flow_matcher = FlowMatcherClass.load_from_checkpoint(
        checkpoint_path,
        device=device
    )
    flow_matcher.eval()
    print()

    # Run evaluation
    print("🔬 Running Evaluation")
    print("="*80)
    if cfg.evaluation.probabilistic:
        results = evaluate_probabilistic(
            flow_matcher, states, labels,
            cfg.evaluation.batch_size,
            cfg.evaluation.num_samples,
            cfg.evaluation.num_steps,
            cfg.evaluation.attractor_radius,
            cfg.evaluation.get('confidence_threshold', 0.6)
        )
    else:
        results = evaluate_deterministic(
            flow_matcher, states, labels,
            cfg.evaluation.batch_size,
            cfg.evaluation.num_steps,
            cfg.evaluation.attractor_radius
        )

    # Print results
    print_results(results, cfg.evaluation.probabilistic)

    # Save outputs
    output_dir = Path(cfg.output.dir)

    if cfg.output.save_plots:
        print("📊 Generating Visualizations")
        print("="*80)
        save_plots(results, states, labels, output_dir, cfg.evaluation.probabilistic)
        print()

    if cfg.output.save_data:
        save_dict = {'states': states, 'labels': labels, **results}
        if 'roc_curve' in save_dict:
            del save_dict['roc_curve']
        np.savez(output_dir / 'results.npz', **save_dict)
        print(f"💾 Saved: {output_dir / 'results.npz'}")

        # Save predicted endpoints to text file (space-separated)
        if 'endpoints' in results:
            endpoints_file = output_dir / 'predicted_endpoints.txt'
            endpoints = results['endpoints']

            # Convert to numpy if needed
            if hasattr(endpoints, 'cpu'):
                endpoints = endpoints.cpu().numpy()

            # Note: Angles already wrapped during evaluation before attractor checking

            # Save: start_state (space-separated) -> predicted_endpoint (space-separated)
            with open(endpoints_file, 'w') as f:
                for i in range(len(states)):
                    # Start state
                    start_str = ' '.join(f'{x:.6f}' for x in states[i])
                    # Predicted endpoint (with wrapped angles)
                    end_str = ' '.join(f'{x:.6f}' for x in endpoints[i])
                    # Write: start_x start_theta start_xdot start_thetadot end_x end_theta end_xdot end_thetadot
                    f.write(f'{start_str} {end_str}\n')

            print(f"💾 Saved: {endpoints_file}")
            if cfg.evaluation.probabilistic:
                print(f"   Format: start_state (space-separated) first_sampled_endpoint (space-separated)")
                print(f"   Note: In probabilistic mode, only the first of {cfg.evaluation.num_samples} samples is saved")
            else:
                print(f"   Format: start_state (space-separated) predicted_endpoint (space-separated)")
            print(f"   Note: Angles wrapped to [-π, π]")

    print("\n" + "="*80)
    print("✅ Evaluation Complete!")
    print("="*80)
    print(f"📁 Results: {output_dir}/")
    print()


if __name__ == "__main__":
    main()

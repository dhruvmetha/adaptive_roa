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
    """Load ROA labeled data (auto-detects CartPole 4D or Pendulum 2D)"""
    print(f"📂 Loading data from: {data_file}")
    data = np.loadtxt(data_file)

    # Auto-detect format: last column is label, rest are states
    states = data[:, :-1]  # All columns except last
    labels = data[:, -1].astype(int)  # Last column is label

    state_dim = states.shape[1]
    system_name = "CartPole (4D)" if state_dim == 4 else f"Pendulum (2D)" if state_dim == 2 else f"Unknown ({state_dim}D)"

    print(f"   System: {system_name}")
    print(f"   Loaded {len(states)} samples")
    print(f"   Success (label=1): {(labels == 1).sum()} ({(labels == 1).mean():.1%})")
    print(f"   Failure (label=0): {(labels == 0).sum()} ({(labels == 0).mean():.1%})")
    print()

    return states, labels


def find_checkpoint_in_folder(folder_path: str) -> str:
    """
    Find the best checkpoint in a given folder

    Args:
        folder_path: Path to training run folder (e.g., outputs/.../2025-10-10_12-30-26)

    Returns:
        Path to best checkpoint file
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

    # Prefer 'last.ckpt' if it exists
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
            result = flow_matcher.predict_endpoint(
                start_states=batch,
                num_steps=num_steps,
                latent=None
            )
            # Handle both return formats: tuple (paths, endpoints) or just endpoints
            if isinstance(result, tuple):
                _, endpoints = result
            else:
                endpoints = result

        all_endpoints.append(endpoints.cpu())

    endpoints = torch.cat(all_endpoints, dim=0)

    # Check attractor membership
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
                          attractor_radius: float) -> Dict[str, Any]:
    """Probabilistic evaluation with uncertainty"""
    print(f"🔬 Running probabilistic evaluation ({num_samples} samples/state)...")

    device = next(flow_matcher.parameters()).device
    states_tensor = torch.from_numpy(states).float().to(device)

    all_attractor_counts = []
    num_batches = int(np.ceil(len(states) / batch_size))

    for i in tqdm(range(num_batches), desc="Sampling predictions"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(states))
        batch = states_tensor[start_idx:end_idx]

        attractor_hits = torch.zeros(len(batch), device=device)

        for _ in range(num_samples):
            with torch.no_grad():
                result = flow_matcher.predict_endpoint(
                    start_states=batch,
                    num_steps=num_steps,
                    latent=None
                )
                # Handle both return formats: tuple (paths, endpoints) or just endpoints
                if isinstance(result, tuple):
                    _, endpoints = result
                else:
                    endpoints = result

            in_attractor = flow_matcher.system.is_in_attractor(endpoints, radius=attractor_radius)
            # Convert to tensor if needed (is_in_attractor might return tensor or numpy)
            if not isinstance(in_attractor, torch.Tensor):
                in_attractor = torch.from_numpy(in_attractor)
            in_attractor = in_attractor.to(device)
            attractor_hits += in_attractor.float()

        all_attractor_counts.append(attractor_hits.cpu())

    attractor_counts = torch.cat(all_attractor_counts, dim=0).numpy()
    p_success = attractor_counts / num_samples
    predictions = (p_success >= 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    conf_matrix = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = conf_matrix.ravel()

    # AUC and entropy
    auc = roc_auc_score(labels, p_success) if len(np.unique(labels)) > 1 else None
    fpr, tpr, thresholds = roc_curve(labels, p_success) if len(np.unique(labels)) > 1 else (None, None, None)

    p_safe = np.clip(p_success, 1e-7, 1 - 1e-7)
    entropy = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))

    return {
        'predictions': predictions,
        'p_success': p_success,
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
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    print(f"F1 Score:    {results['f1']:.4f}")
    print(f"Sensitivity: {results['sensitivity']:.4f}")
    print(f"Specificity: {results['specificity']:.4f}")
    if probabilistic and results.get('auc'):
        print(f"AUC:         {results['auc']:.4f}")

    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                 0      1")
    print(f"Actual  0    {results['tn']:5d}  {results['fp']:5d}")
    print(f"        1    {results['fn']:5d}  {results['tp']:5d}")

    if probabilistic:
        print("\nUncertainty:")
        print(f"  Mean entropy: {results['entropy'].mean():.4f}")
        print(f"  Max entropy:  {results['entropy'].max():.4f}")
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # P(success) distribution
        axes[0].hist(results['p_success'][labels == 0], bins=50, alpha=0.5, label='Failure', color='red')
        axes[0].hist(results['p_success'][labels == 1], bins=50, alpha=0.5, label='Success', color='green')
        axes[0].axvline(0.5, color='black', linestyle='--')
        axes[0].set_xlabel('P(Success)')
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
            cfg.evaluation.attractor_radius
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

    print("\n" + "="*80)
    print("✅ Evaluation Complete!")
    print("="*80)
    print(f"📁 Results: {output_dir}/")
    print()


if __name__ == "__main__":
    main()

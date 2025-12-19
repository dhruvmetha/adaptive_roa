#!/usr/bin/env python
"""
Evaluate all epochs from adaptive training with attractor_radius=0.1

Usage:
    python scripts/eval_all_epochs.py
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import glob

from src.flow_matching.cartpole.latent_conditional.flow_matcher import CartPoleLatentConditionalFlowMatcher
from src.systems.cartpole import CartPoleSystem


def load_roa_data(data_file: str):
    """Load ROA labels data (comma-separated: x,θ,ẋ,θ̇,label)"""
    data = np.loadtxt(data_file, delimiter=',')
    states = data[:, :4]  # x, θ, ẋ, θ̇
    labels = data[:, 4].astype(int)  # Ground truth labels
    return states, labels


def find_best_checkpoint(epoch_dir: Path) -> str:
    """Find best checkpoint in epoch directory"""
    checkpoints = list(epoch_dir.glob("checkpoints/best-*.ckpt"))
    if checkpoints:
        return str(checkpoints[0])
    # Fallback to last.ckpt
    last_ckpt = epoch_dir / "checkpoints" / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)
    raise FileNotFoundError(f"No checkpoint found in {epoch_dir}")


def evaluate_epoch(
    checkpoint_path: str,
    states: np.ndarray,
    labels: np.ndarray,
    system: CartPoleSystem,
    attractor_radius: float = 0.1,
    num_mc_samples: int = 100,
    batch_size: int = 2048,
    device: str = "cuda"
) -> dict:
    """
    Evaluate a single epoch checkpoint.

    Returns dict with metrics for both conformal and fixed thresholds.
    """
    # Load model
    flow_matcher = CartPoleLatentConditionalFlowMatcher.load_from_checkpoint(
        checkpoint_path, device=device
    )
    flow_matcher.eval()

    n_total = len(states)
    X_tensor = torch.from_numpy(states).float().to(device)

    # Collect predictions for each MC sample
    is_success = np.zeros((n_total, num_mc_samples))

    with torch.no_grad():
        for batch_start in tqdm(range(0, n_total, batch_size), desc="Evaluating", leave=False):
            batch_end = min(batch_start + batch_size, n_total)
            batch_inp = X_tensor[batch_start:batch_end]

            for sample_idx in range(num_mc_samples):
                pred = flow_matcher.predict_endpoint(batch_inp)
                attractor_labels = system.classify_attractor(pred, radius=attractor_radius).cpu().numpy()
                is_success[batch_start:batch_end, sample_idx] = attractor_labels

    # Compute success rate per point
    success_rate = (is_success == 1).sum(axis=1) / num_mc_samples

    # Compute metrics at fixed thresholds (0.4/0.6)
    metrics = compute_metrics(success_rate, labels, success_thresh=0.6, failure_thresh=0.4)

    return {
        'success_rate': success_rate,
        'metrics': metrics
    }


def compute_metrics(success_rate: np.ndarray, labels: np.ndarray,
                    success_thresh: float = 0.6, failure_thresh: float = 0.4) -> dict:
    """Compute classification metrics at given thresholds"""
    # Predictions: 1 if p > success_thresh, -1 if p < failure_thresh, 0 otherwise
    pred_success = success_rate > success_thresh
    pred_failure = success_rate < failure_thresh
    pred_uncertain = ~pred_success & ~pred_failure

    # Ground truth: 1 = success, -1 = failure, 0 = separatrix
    gt_success = labels == 1
    gt_failure = labels == -1

    # Only evaluate on confident predictions (not uncertain)
    confident_mask = ~pred_uncertain
    n_confident = confident_mask.sum()
    n_uncertain = pred_uncertain.sum()

    if n_confident == 0:
        return {'error': 'No confident predictions'}

    # TP: predicted success AND ground truth success
    tp = (pred_success & gt_success).sum()
    # TN: predicted failure AND ground truth failure
    tn = (pred_failure & gt_failure).sum()
    # FP: predicted success BUT ground truth failure
    fp = (pred_success & gt_failure).sum()
    # FN: predicted failure BUT ground truth success
    fn = (pred_failure & gt_success).sum()

    # Metrics
    accuracy = (tp + tn) / n_confident if n_confident > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'n_confident': int(n_confident),
        'n_uncertain': int(n_uncertain),
        'separatrix_pct': float(n_uncertain / len(labels)),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'success_threshold': success_thresh,
        'failure_threshold': failure_thresh
    }


def main():
    # Configuration
    base_dir = Path("/common/users/dm1487/tripods/adaptive/outputs/adaptive_cartpole_pybullet/2025-12-05_10-52-57")
    data_file = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_pybullet/roa_labels.txt"
    attractor_radius = 0.2
    num_mc_samples = 10
    batch_size = 2048
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Attractor radius: {attractor_radius}")
    print(f"MC samples: {num_mc_samples}")
    print()

    # Load data
    print("Loading ROA data...")
    states, labels = load_roa_data(data_file)
    print(f"  Loaded {len(states)} points")
    print(f"  Success: {(labels == 1).sum()}, Failure: {(labels == -1).sum()}, Sep: {(labels == 0).sum()}")
    print()

    # Initialize system
    system = CartPoleSystem()

    # Find all epoch directories
    epoch_dirs = sorted(base_dir.glob("epoch_*"))
    print(f"Found {len(epoch_dirs)} epochs")
    print()

    # Results storage
    all_results = {}

    # Evaluate each epoch
    for epoch_dir in epoch_dirs:
        epoch_name = epoch_dir.name
        epoch_num = int(epoch_name.split('_')[1])

        print(f"{'='*60}")
        print(f"Epoch {epoch_num}: {epoch_dir}")
        print(f"{'='*60}")

        try:
            checkpoint = find_best_checkpoint(epoch_dir)
            print(f"  Checkpoint: {Path(checkpoint).name}")

            result = evaluate_epoch(
                checkpoint_path=checkpoint,
                states=states,
                labels=labels,
                system=system,
                attractor_radius=attractor_radius,
                num_mc_samples=num_mc_samples,
                batch_size=batch_size,
                device=device
            )

            metrics = result['metrics']
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  Uncertain: {metrics['n_uncertain']} ({metrics['separatrix_pct']*100:.2f}%)")
            print()

            all_results[epoch_name] = metrics

        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            continue

    # Save results
    output_file = base_dir / "all_epochs_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to: {output_file}")

    # Print summary table
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Epoch':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Uncertain%':<10}")
    print("-"*80)
    for epoch_name in sorted(all_results.keys()):
        m = all_results[epoch_name]
        epoch_num = int(epoch_name.split('_')[1])
        print(f"{epoch_num:<10} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {m['separatrix_pct']*100:<10.2f}")


if __name__ == "__main__":
    main()

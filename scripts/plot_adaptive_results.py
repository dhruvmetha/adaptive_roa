#!/usr/bin/env python
"""Plot comprehensive adaptive sampling results.

Usage:
    python scripts/plot_adaptive_results.py /path/to/output/dir
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from typing import List, Dict

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')


def get_initial_train_size(output_dir: Path) -> int:
    """
    Get the initial training size from the Hydra config.

    The train_trajectories field in results.json is recorded AFTER adding new samples,
    so we need the initial_train_size to correctly compute the actual training size
    for each epoch.
    """
    # Try to load from Hydra config in epoch_000
    hydra_config = output_dir / "epoch_000" / ".hydra" / "config.yaml"
    if hydra_config.exists():
        import yaml
        with open(hydra_config, 'r') as f:
            cfg = yaml.safe_load(f)
            return cfg.get('initial_train_size', 50)

    # Fallback: try root .hydra
    hydra_config = output_dir / ".hydra" / "config.yaml"
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


def load_results(output_dir: Path) -> dict:
    """Load final_results.json from the output directory."""
    results_file = output_dir / "final_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Could not find {results_file}")

    with open(results_file, 'r') as f:
        return json.load(f)


def plot_training_progress(epoch_results: list, save_dir: Path, initial_train_size: int = 50):
    """Plot training dataset growth and sampling statistics."""
    epochs = [r['epoch'] for r in epoch_results]
    train_trajectories = correct_train_trajectories(epoch_results, initial_train_size)
    n_d1_added = [r['n_d1_added'] for r in epoch_results]
    n_d2_uncertain = [r['n_d2_uncertain'] for r in epoch_results]
    n_d2_confident = [r['n_d2_confident'] for r in epoch_results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training dataset size
    ax = axes[0, 0]
    ax.plot(epochs, train_trajectories, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Trajectories', fontsize=12)
    ax.set_title('Training Dataset Growth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Samples added per epoch
    ax = axes[0, 1]
    ax.bar(epochs, n_d1_added, alpha=0.7, label='D1 (trajectory data)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Samples Added', fontsize=12)
    ax.set_title('Trajectory Samples Added per Epoch', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Uncertain vs confident classification samples
    ax = axes[1, 0]
    width = 0.35
    x = np.array(epochs)
    ax.bar(x - width/2, n_d2_uncertain, width, label='Uncertain', alpha=0.7, color='orange')
    ax.bar(x + width/2, n_d2_confident, width, label='Confident', alpha=0.7, color='green')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Classification Samples', fontsize=12)
    ax.set_title('D2 Classification Samples per Epoch', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative samples
    ax = axes[1, 1]
    cumulative_d1 = np.cumsum(n_d1_added)
    cumulative_d2_uncertain = np.cumsum(n_d2_uncertain)
    cumulative_d2_confident = np.cumsum(n_d2_confident)
    ax.plot(epochs, cumulative_d1, 'b-o', label='D1 (trajectory)', linewidth=2)
    ax.plot(epochs, cumulative_d2_uncertain, 'o-', color='orange', label='D2 uncertain', linewidth=2)
    ax.plot(epochs, cumulative_d2_confident, 'g-o', label='D2 confident', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cumulative Samples', fontsize=12)
    ax.set_title('Cumulative Samples Added', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'training_progress.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved training_progress.png")


def plot_conformal_metrics(epoch_results: list, save_dir: Path):
    """Plot conformal prediction metrics (lambda*, q_hat, coverage)."""
    epochs = [r['epoch'] for r in epoch_results]
    lambda_star = [r['lambda_star'] for r in epoch_results]
    q_hat = [r['q_hat'] for r in epoch_results]
    test_coverage = [r['test_coverage'] for r in epoch_results]
    test_f1 = [r['test_f1'] for r in epoch_results]
    test_unknown_rate = [r['test_unknown_rate'] for r in epoch_results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Lambda star
    ax = axes[0, 0]
    ax.plot(epochs, lambda_star, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('λ*', fontsize=12)
    ax.set_title('Optimal Decision Threshold (λ*)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # q_hat
    ax = axes[0, 1]
    ax.plot(epochs, q_hat, 'r-o', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('q̂', fontsize=12)
    ax.set_title('Conformal Quantile (q̂)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Test coverage
    ax = axes[1, 0]
    ax.plot(epochs, test_coverage, 'g-o', linewidth=2, markersize=6)
    ax.axhline(y=0.95, color='r', linestyle='--', label='Target (95%)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Coverage', fontsize=12)
    ax.set_title('Test Set Coverage', fontsize=14, fontweight='bold')
    ax.set_ylim([0.7, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Test F1 and unknown rate
    ax = axes[1, 1]
    ax.plot(epochs, test_f1, 'b-o', linewidth=2, markersize=6, label='F1 Score')
    ax.plot(epochs, test_unknown_rate, 'o-', color='orange', linewidth=2, markersize=6, label='Unknown Rate')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Test Set F1 and Unknown Rate', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'conformal_metrics.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'conformal_metrics.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved conformal_metrics.png")


def plot_roa_metrics(epoch_results: list, save_dir: Path):
    """Plot ROA evaluation metrics over epochs."""
    epochs = [r['epoch'] for r in epoch_results]

    # Extract conformal threshold metrics
    accuracy_conf = [r['full_roa']['conformal_thresholds']['accuracy'] for r in epoch_results]
    precision_conf = [r['full_roa']['conformal_thresholds']['precision'] for r in epoch_results]
    recall_conf = [r['full_roa']['conformal_thresholds']['recall'] for r in epoch_results]
    f1_conf = [r['full_roa']['conformal_thresholds']['f1'] for r in epoch_results]
    specificity_conf = [r['full_roa']['conformal_thresholds']['specificity'] for r in epoch_results]
    separatrix_conf = [r['full_roa']['conformal_thresholds']['separatrix_pct'] * 100 for r in epoch_results]

    # Extract fixed threshold metrics
    accuracy_fixed = [r['full_roa']['fixed_thresholds']['accuracy'] for r in epoch_results]
    precision_fixed = [r['full_roa']['fixed_thresholds']['precision'] for r in epoch_results]
    recall_fixed = [r['full_roa']['fixed_thresholds']['recall'] for r in epoch_results]
    f1_fixed = [r['full_roa']['fixed_thresholds']['f1'] for r in epoch_results]
    specificity_fixed = [r['full_roa']['fixed_thresholds']['specificity'] for r in epoch_results]
    separatrix_fixed = [r['full_roa']['fixed_thresholds']['separatrix_pct'] * 100 for r in epoch_results]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Accuracy comparison
    ax = axes[0, 0]
    ax.plot(epochs, accuracy_conf, 'b-o', linewidth=2, markersize=5, label='Conformal')
    ax.plot(epochs, accuracy_fixed, 'r-s', linewidth=2, markersize=5, label='Fixed (0.4/0.6)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim([0.85, 1.0])
    ax.grid(True, alpha=0.3)

    # Precision comparison
    ax = axes[0, 1]
    ax.plot(epochs, precision_conf, 'b-o', linewidth=2, markersize=5, label='Conformal')
    ax.plot(epochs, precision_fixed, 'r-s', linewidth=2, markersize=5, label='Fixed (0.4/0.6)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim([0.85, 1.0])
    ax.grid(True, alpha=0.3)

    # Recall comparison
    ax = axes[0, 2]
    ax.plot(epochs, recall_conf, 'b-o', linewidth=2, markersize=5, label='Conformal')
    ax.plot(epochs, recall_fixed, 'r-s', linewidth=2, markersize=5, label='Fixed (0.4/0.6)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)

    # F1 comparison
    ax = axes[1, 0]
    ax.plot(epochs, f1_conf, 'b-o', linewidth=2, markersize=5, label='Conformal')
    ax.plot(epochs, f1_fixed, 'r-s', linewidth=2, markersize=5, label='Fixed (0.4/0.6)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim([0.3, 1.0])
    ax.grid(True, alpha=0.3)

    # Specificity comparison
    ax = axes[1, 1]
    ax.plot(epochs, specificity_conf, 'b-o', linewidth=2, markersize=5, label='Conformal')
    ax.plot(epochs, specificity_fixed, 'r-s', linewidth=2, markersize=5, label='Fixed (0.4/0.6)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Specificity', fontsize=12)
    ax.set_title('Specificity', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim([0.98, 1.0])
    ax.grid(True, alpha=0.3)

    # Separatrix percentage
    ax = axes[1, 2]
    ax.plot(epochs, separatrix_conf, 'b-o', linewidth=2, markersize=5, label='Conformal')
    ax.plot(epochs, separatrix_fixed, 'r-s', linewidth=2, markersize=5, label='Fixed (0.4/0.6)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Separatrix %', fontsize=12)
    ax.set_title('Uncertain Region Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'roa_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'roa_metrics_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved roa_metrics_comparison.png")


def plot_confusion_matrix_evolution(epoch_results: list, save_dir: Path):
    """Plot evolution of TP, TN, FP, FN over epochs."""
    epochs = [r['epoch'] for r in epoch_results]

    # Extract counts from conformal thresholds
    tp = [r['full_roa']['conformal_thresholds']['tp'] for r in epoch_results]
    tn = [r['full_roa']['conformal_thresholds']['tn'] for r in epoch_results]
    fp = [r['full_roa']['conformal_thresholds']['fp'] for r in epoch_results]
    fn = [r['full_roa']['conformal_thresholds']['fn'] for r in epoch_results]
    n_uncertain = [r['full_roa']['conformal_thresholds']['n_uncertain'] for r in epoch_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar chart of predictions
    ax = axes[0]
    width = 0.8
    ax.bar(epochs, tp, width, label='True Positive', color='#2ecc71', alpha=0.8)
    ax.bar(epochs, tn, width, bottom=tp, label='True Negative', color='#3498db', alpha=0.8)
    ax.bar(epochs, fp, width, bottom=np.array(tp) + np.array(tn), label='False Positive', color='#e74c3c', alpha=0.8)
    ax.bar(epochs, fn, width, bottom=np.array(tp) + np.array(tn) + np.array(fp), label='False Negative', color='#f39c12', alpha=0.8)
    ax.bar(epochs, n_uncertain, width, bottom=np.array(tp) + np.array(tn) + np.array(fp) + np.array(fn),
           label='Uncertain', color='#9b59b6', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Number of Points', fontsize=12)
    ax.set_title('Prediction Breakdown (Conformal)', fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3, axis='y')

    # Error rates
    ax = axes[1]
    total = np.array(tp) + np.array(tn) + np.array(fp) + np.array(fn)
    fp_rate = np.array(fp) / total * 100
    fn_rate = np.array(fn) / total * 100
    uncertain_rate = np.array(n_uncertain) / (total + np.array(n_uncertain)) * 100
    ax.plot(epochs, fp_rate, 'r-o', linewidth=2, markersize=5, label='False Positive Rate')
    ax.plot(epochs, fn_rate, 'o-', color='orange', linewidth=2, markersize=5, label='False Negative Rate')
    ax.plot(epochs, uncertain_rate, 'p-', color='purple', linewidth=2, markersize=5, label='Uncertain Rate')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('Error and Uncertainty Rates', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_evolution.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'confusion_evolution.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved confusion_evolution.png")


def plot_summary_dashboard(epoch_results: list, final_stats: dict, save_dir: Path, initial_train_size: int = 50):
    """Create a summary dashboard with key metrics."""
    epochs = [r['epoch'] for r in epoch_results]

    # Extract key metrics
    train_trajectories = correct_train_trajectories(epoch_results, initial_train_size)
    f1_conf = [r['full_roa']['conformal_thresholds']['f1'] for r in epoch_results]
    recall_conf = [r['full_roa']['conformal_thresholds']['recall'] for r in epoch_results]
    accuracy_conf = [r['full_roa']['conformal_thresholds']['accuracy'] for r in epoch_results]
    separatrix_conf = [r['full_roa']['conformal_thresholds']['separatrix_pct'] * 100 for r in epoch_results]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Training data growth
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(epochs, train_trajectories, alpha=0.3, color='blue')
    ax1.plot(epochs, train_trajectories, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Trajectories', fontsize=11)
    ax1.set_title('Dataset Growth', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # F1 Score evolution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, f1_conf, 'g-o', linewidth=2, markersize=6)
    ax2.fill_between(epochs, f1_conf, alpha=0.3, color='green')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_title('F1 Score (Conformal)', fontsize=13, fontweight='bold')
    ax2.set_ylim([0.7, 0.95])
    ax2.grid(True, alpha=0.3)

    # Accuracy evolution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, accuracy_conf, 'b-o', linewidth=2, markersize=6)
    ax3.fill_between(epochs, accuracy_conf, alpha=0.3, color='blue')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('Accuracy (Conformal)', fontsize=13, fontweight='bold')
    ax3.set_ylim([0.9, 1.0])
    ax3.grid(True, alpha=0.3)

    # Recall evolution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, recall_conf, 'o-', color='purple', linewidth=2, markersize=6)
    ax4.fill_between(epochs, recall_conf, alpha=0.3, color='purple')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Recall', fontsize=11)
    ax4.set_title('Recall (Conformal)', fontsize=13, fontweight='bold')
    ax4.set_ylim([0.6, 0.9])
    ax4.grid(True, alpha=0.3)

    # Uncertain region size
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, separatrix_conf, 'o-', color='orange', linewidth=2, markersize=6)
    ax5.fill_between(epochs, separatrix_conf, alpha=0.3, color='orange')
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Uncertain %', fontsize=11)
    ax5.set_title('Uncertain Region Size', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Summary statistics text box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    final_epoch = epoch_results[-1]
    summary_text = (
        f"Final Summary (Epoch {final_epoch['epoch']})\n"
        f"{'─' * 35}\n\n"
        f"Training Data:\n"
        f"  Trajectories: {final_stats['train_trajectories']:,}\n"
        f"  Success Rate: {final_stats['train_success_rate']:.1%}\n\n"
        f"ROA Metrics (Conformal):\n"
        f"  Accuracy: {final_epoch['full_roa']['conformal_thresholds']['accuracy']:.3f}\n"
        f"  F1 Score: {final_epoch['full_roa']['conformal_thresholds']['f1']:.3f}\n"
        f"  Precision: {final_epoch['full_roa']['conformal_thresholds']['precision']:.3f}\n"
        f"  Recall: {final_epoch['full_roa']['conformal_thresholds']['recall']:.3f}\n"
        f"  Specificity: {final_epoch['full_roa']['conformal_thresholds']['specificity']:.3f}\n"
        f"  Uncertain: {final_epoch['full_roa']['conformal_thresholds']['separatrix_pct']:.1%}\n\n"
        f"Conformal Parameters:\n"
        f"  lambda* = {final_epoch['lambda_star']:.4f}\n"
        f"  delta = {final_epoch['full_roa']['delta']}"
    )
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('Adaptive Sampling Results: CartPole PyBullet', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_dir / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'summary_dashboard.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved summary_dashboard.png")


def plot_threshold_comparison_detailed(epoch_results: list, save_dir: Path):
    """Detailed comparison of conformal vs fixed thresholds."""
    epochs = [r['epoch'] for r in epoch_results]

    # Conformal metrics
    f1_conf = [r['full_roa']['conformal_thresholds']['f1'] for r in epoch_results]
    recall_conf = [r['full_roa']['conformal_thresholds']['recall'] for r in epoch_results]
    precision_conf = [r['full_roa']['conformal_thresholds']['precision'] for r in epoch_results]

    # Fixed metrics
    f1_fixed = [r['full_roa']['fixed_thresholds']['f1'] for r in epoch_results]
    recall_fixed = [r['full_roa']['fixed_thresholds']['recall'] for r in epoch_results]
    precision_fixed = [r['full_roa']['fixed_thresholds']['precision'] for r in epoch_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # F1 improvement
    ax = axes[0]
    f1_improvement = np.array(f1_conf) - np.array(f1_fixed)
    colors = ['green' if x > 0 else 'red' for x in f1_improvement]
    ax.bar(epochs, f1_improvement, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Improvement', fontsize=12)
    ax.set_title('F1: Conformal - Fixed', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Recall improvement
    ax = axes[1]
    recall_improvement = np.array(recall_conf) - np.array(recall_fixed)
    colors = ['green' if x > 0 else 'red' for x in recall_improvement]
    ax.bar(epochs, recall_improvement, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Recall Improvement', fontsize=12)
    ax.set_title('Recall: Conformal - Fixed', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Precision comparison (fixed is usually higher)
    ax = axes[2]
    precision_diff = np.array(precision_conf) - np.array(precision_fixed)
    colors = ['green' if x > 0 else 'red' for x in precision_diff]
    ax.bar(epochs, precision_diff, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Precision Difference', fontsize=12)
    ax.set_title('Precision: Conformal - Fixed', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'threshold_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'threshold_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved threshold_comparison.png")


def plot_conformal_vs_fixed(epoch_results: list, save_dir: Path, initial_train_size: int = 50):
    """Original comparison plot: F1 and separatrix vs dataset size."""
    # Extract data
    train_sizes = correct_train_trajectories(epoch_results, initial_train_size)
    conformal_f1 = [r['full_roa']['conformal_thresholds']['f1'] for r in epoch_results]
    conformal_separatrix = [r['full_roa']['conformal_thresholds']['separatrix_pct'] * 100 for r in epoch_results]
    fixed_f1 = [r['full_roa']['fixed_thresholds']['f1'] for r in epoch_results]
    fixed_separatrix = [r['full_roa']['fixed_thresholds']['separatrix_pct'] * 100 for r in epoch_results]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: F1 Score comparison
    ax1.plot(train_sizes, conformal_f1, 'b-o', label='Conformal (λ*±δ)', linewidth=2, markersize=6)
    ax1.plot(train_sizes, fixed_f1, 'r--s', label='Fixed (0.4/0.6)', linewidth=2, markersize=6)
    ax1.set_xlabel('Training Dataset Size (trajectories)', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score vs Dataset Size\n(Full ROA Evaluation)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.0, 1.0])
    ax1.fill_between(train_sizes, fixed_f1, conformal_f1, alpha=0.2, color='green')

    # Plot 2: Separatrix comparison
    ax2.plot(train_sizes, conformal_separatrix, 'b-o', label='Conformal (λ*±δ)', linewidth=2, markersize=6)
    ax2.plot(train_sizes, fixed_separatrix, 'r--s', label='Fixed (0.4/0.6)', linewidth=2, markersize=6)
    ax2.set_xlabel('Training Dataset Size (trajectories)', fontsize=12)
    ax2.set_ylabel('Separatrix / Unknown Rate (%)', fontsize=12)
    ax2.set_title('Uncertainty Region vs Dataset Size\n(Full ROA Evaluation)', fontsize=14)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'conformal_vs_fixed_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'conformal_vs_fixed_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved conformal_vs_fixed_comparison.png")


def print_summary(epoch_results: list, final_stats: dict):
    """Print summary statistics to console."""
    conformal_f1 = [r['full_roa']['conformal_thresholds']['f1'] for r in epoch_results]
    fixed_f1 = [r['full_roa']['fixed_thresholds']['f1'] for r in epoch_results]
    conformal_separatrix = [r['full_roa']['conformal_thresholds']['separatrix_pct'] * 100 for r in epoch_results]
    fixed_separatrix = [r['full_roa']['fixed_thresholds']['separatrix_pct'] * 100 for r in epoch_results]

    print("\n" + "="*70)
    print("COMPARISON: CONFORMAL (λ*±δ) vs FIXED (0.4/0.6) THRESHOLDS")
    print("="*70)
    print(f"{'Metric':<25} {'Conformal':<15} {'Fixed':<15}")
    print("-"*70)
    print(f"{'F1 (start)':<25} {conformal_f1[0]:.3f}{'':<11} {fixed_f1[0]:.3f}")
    print(f"{'F1 (end)':<25} {conformal_f1[-1]:.3f}{'':<11} {fixed_f1[-1]:.3f}")
    print(f"{'F1 (best)':<25} {max(conformal_f1):.3f}{'':<11} {max(fixed_f1):.3f}")
    print(f"{'F1 (mean)':<25} {np.mean(conformal_f1):.3f}{'':<11} {np.mean(fixed_f1):.3f}")
    print("-"*70)
    print(f"{'Separatrix % (mean)':<25} {np.mean(conformal_separatrix):.2f}%{'':<10} {np.mean(fixed_separatrix):.2f}%")
    print("="*70)
    print(f"\nConformal F1 is {np.mean(conformal_f1)/np.mean(fixed_f1):.1f}x better on average")
    print(f"Training trajectories: {final_stats['train_trajectories']:,}")
    print(f"Training success rate: {final_stats['train_success_rate']:.1%}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_adaptive_results.py /path/to/output/dir [--save-dir DIR]")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    # Parse optional save-dir argument
    save_dir = None
    for i, arg in enumerate(sys.argv):
        if arg == '--save-dir' and i + 1 < len(sys.argv):
            save_dir = Path(sys.argv[i + 1])

    # Load results
    results = load_results(output_dir)
    epoch_results = results['epoch_results']
    final_stats = results['final_stats']

    # Get initial train size for correcting trajectory counts
    initial_train_size = get_initial_train_size(output_dir)

    print(f"Loaded results from {output_dir}")
    print(f"  - {len(epoch_results)} epochs")
    print(f"  - Initial training size: {initial_train_size}")
    print(f"  - Final training trajectories: {final_stats['train_trajectories']}")

    # Create save directory
    if save_dir is None:
        save_dir = output_dir / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to {save_dir}")

    # Generate all plots
    plot_training_progress(epoch_results, save_dir, initial_train_size)
    plot_conformal_metrics(epoch_results, save_dir)
    plot_roa_metrics(epoch_results, save_dir)
    plot_confusion_matrix_evolution(epoch_results, save_dir)
    plot_threshold_comparison_detailed(epoch_results, save_dir)
    plot_conformal_vs_fixed(epoch_results, save_dir, initial_train_size)
    plot_summary_dashboard(epoch_results, final_stats, save_dir, initial_train_size)

    # Print summary
    print_summary(epoch_results, final_stats)

    print(f"\nAll plots saved to {save_dir}")


if __name__ == '__main__':
    main()

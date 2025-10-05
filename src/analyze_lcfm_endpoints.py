#!/usr/bin/env python3
"""
Fast LCFM endpoint analysis using cached endpoints

This script performs ROA classification analysis using pre-generated endpoints,
allowing for fast repeated analysis with different parameters.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.cm as cm

from src.systems.pendulum_lcfm import PendulumSystemLCFM


def load_cached_endpoints(cache_file):
    """Load cached endpoints from NPZ file"""
    print(f"📂 Loading cached endpoints from: {cache_file}")
    
    # Load data
    data = np.load(cache_file)
    
    # Print cache info
    print(f"✅ Cache loaded successfully!")
    print(f"   Generated: {data['timestamp']}")
    print(f"   Model: {data['model_path']}")
    print(f"   Start states: {data['start_states'].shape[0]:,}")
    print(f"   Samples per state: {data['num_samples']}")
    print(f"   Total endpoints: {data['endpoints'].size // 2:,}")
    print(f"   Generation time: {data['generation_time']:.1f}s")
    
    return {
        'start_states': data['start_states'],
        'labels': data['labels'],
        'endpoints': data['endpoints'],
        'num_samples': int(data['num_samples']),
        'attractors': data['attractors'],
        'metadata': {
            'model_path': str(data['model_path']),
            'ground_truth_path': str(data['ground_truth_path']),
            'num_steps': int(data['num_steps']),
            'generation_time': float(data['generation_time']),
            'timestamp': str(data['timestamp']),
            'random_seed': int(data['random_seed'])
        }
    }


def classify_endpoint_to_attractor(endpoints, attractors, radius_threshold=0.075):
    """Classify endpoints to nearest attractor within radius threshold"""
    n_endpoints = len(endpoints)
    attractor_indices = np.full(n_endpoints, -1, dtype=int)  # -1 = no convergence
    min_distances = np.full(n_endpoints, np.inf)
    
    # Check distance to each attractor
    for i, attractor in enumerate(attractors):
        # Calculate distances to this attractor
        distances = np.linalg.norm(endpoints - attractor, axis=1)
        
        # Update closest attractor for points within threshold
        mask = (distances < radius_threshold) & (distances < min_distances)
        attractor_indices[mask] = i
        min_distances[mask] = distances[mask]
    
    return attractor_indices, min_distances


def analyze_cached_endpoints(cached_data, prob_threshold=0.6, radius_threshold=0.075, analysis_samples=None):
    """Analyze cached endpoints with given parameters
    
    Args:
        cached_data: Loaded cached endpoint data
        prob_threshold: Probability threshold for classification
        radius_threshold: Radius threshold for attractor convergence
        analysis_samples: Number of samples to use for analysis (default: use all cached samples)
    """
    
    start_states = cached_data['start_states']
    true_labels = cached_data['labels']
    all_endpoints = cached_data['endpoints']  # [N, num_samples, 2]
    attractors = cached_data['attractors']
    max_samples = cached_data['num_samples']
    
    # Determine number of samples to use
    if analysis_samples is None:
        num_samples = max_samples
        endpoints_to_use = all_endpoints
    else:
        if analysis_samples > max_samples:
            print(f"⚠️  Warning: Requested {analysis_samples} samples but only {max_samples} available. Using {max_samples}.")
            num_samples = max_samples
            endpoints_to_use = all_endpoints
        else:
            num_samples = analysis_samples
            # Use first N samples from each start state
            endpoints_to_use = all_endpoints[:, :num_samples, :]
            print(f"🔄 Using {num_samples}/{max_samples} samples per start state for analysis")
    
    n_points = len(start_states)
    n_attractors = len(attractors)
    target_attractor_idx = 0  # (0,0) attractor is at index 0
    
    print(f"\n🎯 Analyzing cached endpoints:")
    print(f"  Start states: {n_points:,}")
    print(f"  Samples per state: {num_samples}")
    print(f"  Probability threshold: {prob_threshold}")
    print(f"  Radius threshold: {radius_threshold}")
    print(f"  Target attractor: {attractors[target_attractor_idx]}")
    
    # Storage for results
    predicted_probs_all = np.zeros((n_points, n_attractors))
    predicted_labels_3class = np.zeros(n_points, dtype=int)
    predicted_labels_binary = np.zeros(n_points, dtype=int)
    predicted_probs_target = np.zeros(n_points)
    
    # Process all endpoints (vectorized where possible)
    print("🔄 Processing endpoint classifications...")
    
    for i in tqdm(range(n_points), desc="Analyzing endpoints"):
        state_endpoints = endpoints_to_use[i]  # [num_samples, 2]
        
        # Classify endpoints to attractors
        attractor_indices, distances = classify_endpoint_to_attractor(
            state_endpoints, attractors, radius_threshold
        )
        
        # Calculate probabilities for each attractor
        attractor_probs = np.zeros(n_attractors)
        for attr_idx in range(n_attractors):
            n_convergence = np.sum(attractor_indices == attr_idx)
            attractor_probs[attr_idx] = n_convergence / num_samples
        
        # Store probabilities
        predicted_probs_all[i] = attractor_probs
        predicted_probs_target[i] = attractor_probs[target_attractor_idx]
        
        # Three-class classification logic
        max_prob = np.max(attractor_probs)
        max_attractor_idx = np.argmax(attractor_probs)
        
        if max_prob > prob_threshold:
            # Convincing attractor found
            if max_attractor_idx == target_attractor_idx:
                predicted_labels_3class[i] = 1    # Target attractor (0,0)
                predicted_labels_binary[i] = 1    # Positive for binary
            else:
                predicted_labels_3class[i] = -1   # Other attractor
                predicted_labels_binary[i] = 0    # Negative for binary
        else:
            # No convincing attractor
            predicted_labels_3class[i] = 0        # No convincing attractor
            predicted_labels_binary[i] = 0        # Negative for binary
    
    # Identify separatrix points
    separatrix_mask = (predicted_labels_3class == 0)
    basin_mask = ~separatrix_mask
    
    # Separatrix analysis
    n_separatrix = np.sum(separatrix_mask)
    n_basin = np.sum(basin_mask)
    separatrix_percentage = (n_separatrix / n_points) * 100
    
    separatrix_info = {
        'n_separatrix': n_separatrix,
        'n_basin': n_basin,
        'separatrix_percentage': separatrix_percentage,
        'separatrix_states': start_states[separatrix_mask],
        'basin_states': start_states[basin_mask],
    }
    
    print(f"\n📊 Separatrix Analysis:")
    print(f"  Total points: {n_points:,}")
    print(f"  Basin points: {n_basin:,} ({100-separatrix_percentage:.1f}%)")
    print(f"  Separatrix points: {n_separatrix:,} ({separatrix_percentage:.1f}%)")
    
    # Calculate metrics for full dataset
    full_results = calculate_classification_metrics(
        true_labels, predicted_labels_binary, predicted_probs_target
    )
    
    # Calculate metrics for basin-only
    basin_results = None
    if n_basin > 0:
        basin_true_labels = true_labels[basin_mask]
        basin_pred_labels_binary = predicted_labels_binary[basin_mask]
        basin_pred_probs_target = predicted_probs_target[basin_mask]
        
        basin_results = calculate_classification_metrics(
            basin_true_labels, basin_pred_labels_binary, basin_pred_probs_target
        )
        basin_results['basin_mask'] = basin_mask
        basin_results['n_basin_points'] = n_basin
        
        print(f"\n📈 Performance Comparison:")
        print(f"  Full Dataset - Accuracy: {full_results['accuracy']:.4f}, F1: {full_results['f1_score']:.4f}")
        print(f"  Basin Only   - Accuracy: {basin_results['accuracy']:.4f}, F1: {basin_results['f1_score']:.4f}")
        print(f"  Improvement  - Accuracy: {basin_results['accuracy']-full_results['accuracy']:+.4f}, F1: {basin_results['f1_score']-full_results['f1_score']:+.4f}")
    
    return {
        'full_dataset': full_results,
        'basin_only': basin_results,
        'separatrix_info': separatrix_info,
        'separatrix_mask': separatrix_mask,
        'evaluation_params': {
            'num_samples': num_samples,
            'prob_threshold': prob_threshold,
            'radius_threshold': radius_threshold,
            'n_points': n_points,
            'attractors': attractors,
            'target_attractor_idx': target_attractor_idx,
        }
    }


def calculate_classification_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive binary classification metrics"""
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        roc_auc = 0.0
    
    return {
        'confusion_matrix': cm,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tpr': tpr, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr,
        'roc_auc': roc_auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs
    }


def create_classification_performance_plot(results, output_dir):
    """Create 5-category classification performance plot"""
    
    print("🎨 Creating 5-category classification performance plot...")
    
    # Get data from results
    separatrix_info = results['separatrix_info']
    full_results = results['full_dataset']
    basin_results = results['basin_only']
    params = results['evaluation_params']
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Specific hex colors for different categories
    SUCCESS_COLOR = "#17BECF"      # TP: Teal/Cyan
    FAILURE_COLOR = "#393B79"      # TN: Dark Blue
    FP_COLOR = "#FE0700"           # FP: Bright Red
    FN_COLOR = "#FC7F00"           # FN: Orange
    SEPARATRIX_COLOR = "#FDFD01"   # Sep: Bright Yellow
    
    # Plot separatrix points first
    if len(separatrix_info['separatrix_states']) > 0:
        ax.scatter(separatrix_info['separatrix_states'][:, 0], separatrix_info['separatrix_states'][:, 1],
                  c=SEPARATRIX_COLOR, alpha=0.8, s=2, label=f'SEPARATRIX ({separatrix_info["n_separatrix"]:,})')
    
    # For basin points, categorize them based on ground truth vs predictions
    if basin_results is not None and len(separatrix_info['basin_states']) > 0:
        basin_states = separatrix_info['basin_states']
        y_true = basin_results['y_true']
        y_pred = basin_results['y_pred']
        
        # Create masks for each category
        tp_mask = (y_true == 1) & (y_pred == 1)  # True Positives
        tn_mask = (y_true == 0) & (y_pred == 0)  # True Negatives  
        fp_mask = (y_true == 0) & (y_pred == 1)  # False Positives
        fn_mask = (y_true == 1) & (y_pred == 0)  # False Negatives
        
        # Plot each category with specific hex colors
        if np.sum(tp_mask) > 0:
            ax.scatter(basin_states[tp_mask, 0], basin_states[tp_mask, 1],
                      c=SUCCESS_COLOR, alpha=0.7, s=2, label=f'SUCCESS (TP) ({np.sum(tp_mask):,})')
        
        if np.sum(tn_mask) > 0:
            ax.scatter(basin_states[tn_mask, 0], basin_states[tn_mask, 1],
                      c=FAILURE_COLOR, alpha=0.7, s=2, label=f'FAILURE (TN) ({np.sum(tn_mask):,})')
        
        if np.sum(fp_mask) > 0:
            ax.scatter(basin_states[fp_mask, 0], basin_states[fp_mask, 1],
                      c=FP_COLOR, alpha=0.7, s=2, label=f'FALSE POSITIVE ({np.sum(fp_mask):,})')
        
        if np.sum(fn_mask) > 0:
            ax.scatter(basin_states[fn_mask, 0], basin_states[fn_mask, 1],
                      c=FN_COLOR, alpha=0.7, s=2, label=f'FALSE NEGATIVE ({np.sum(fn_mask):,})')
        
        # Print statistics
        print(f"📊 5-Category Classification Statistics:")
        print(f"  SUCCESS (TP): {np.sum(tp_mask):,} ({np.sum(tp_mask)/len(y_true)*100:.1f}%)")
        print(f"  FAILURE (TN): {np.sum(tn_mask):,} ({np.sum(tn_mask)/len(y_true)*100:.1f}%)")
        print(f"  FALSE POSITIVE: {np.sum(fp_mask):,} ({np.sum(fp_mask)/len(y_true)*100:.1f}%)")
        print(f"  FALSE NEGATIVE: {np.sum(fn_mask):,} ({np.sum(fn_mask)/len(y_true)*100:.1f}%)")
        print(f"  SEPARATRIX: {separatrix_info['n_separatrix']:,} ({separatrix_info['separatrix_percentage']:.1f}%)")
    
    # Add attractors
    for i, attractor in enumerate(params['attractors']):
        marker = 'o' if i == params['target_attractor_idx'] else 's'
        ax.scatter(attractor[0], attractor[1], c='black', marker=marker, s=100, 
                  edgecolors='white', linewidth=2, zorder=10)
    
    # Formatting
    ax.set_xlabel('θ (angle)')
    ax.set_ylabel('θ̇ (angular velocity)')
    ax.set_title('Classification Performance Analysis\n5-Category State Space Visualization')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'classification_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🎨 Classification performance analysis plot saved: classification_performance_analysis.png")


def create_separatrix_basin_plot(results, output_dir):
    """Create simple separatrix vs basin visualization"""
    
    separatrix_info = results['separatrix_info']
    params = results['evaluation_params']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Separatrix vs Basin Classification
    if len(separatrix_info['basin_states']) > 0:
        ax.scatter(separatrix_info['basin_states'][:, 0], separatrix_info['basin_states'][:, 1],
                   c='lightblue', alpha=0.6, s=1, label=f'Basin ({separatrix_info["n_basin"]:,})')
    if len(separatrix_info['separatrix_states']) > 0:
        ax.scatter(separatrix_info['separatrix_states'][:, 0], separatrix_info['separatrix_states'][:, 1],
                   c='red', alpha=0.8, s=2, label=f'Separatrix ({separatrix_info["n_separatrix"]:,})')
    
    # Add attractors
    for i, attractor in enumerate(params['attractors']):
        marker = 'o' if i == params['target_attractor_idx'] else 's'
        ax.scatter(attractor[0], attractor[1], c='black', marker=marker, s=100, 
                   edgecolors='white', linewidth=2, zorder=10)
    
    ax.set_xlabel('θ (angle)')
    ax.set_ylabel('θ̇ (angular velocity)')
    ax.set_title(f'Separatrix vs Basin Points\n{separatrix_info["separatrix_percentage"]:.1f}% Separatrix')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'separatrix_basin_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🎨 Separatrix vs basin plot saved: separatrix_basin_analysis.png")


def save_analysis_results(results, cached_data, output_dir):
    """Save comprehensive analysis results"""
    
    separatrix_info = results['separatrix_info']
    params = results['evaluation_params']
    metadata = cached_data['metadata']
    
    # Save numerical results
    np.savez(
        output_dir / 'roa_analysis_results.npz',
        # Full dataset results
        full_y_true=results['full_dataset']['y_true'],
        full_y_pred=results['full_dataset']['y_pred'],
        full_y_probs=results['full_dataset']['y_probs'],
        full_confusion_matrix=results['full_dataset']['confusion_matrix'],
        # Basin-only results
        basin_y_true=results['basin_only']['y_true'] if results['basin_only'] else np.array([]),
        basin_y_pred=results['basin_only']['y_pred'] if results['basin_only'] else np.array([]),
        basin_y_probs=results['basin_only']['y_probs'] if results['basin_only'] else np.array([]),
        basin_confusion_matrix=results['basin_only']['confusion_matrix'] if results['basin_only'] else np.array([]),
        # Separatrix analysis
        separatrix_mask=results['separatrix_mask'],
        separatrix_states=separatrix_info['separatrix_states'],
        basin_states=separatrix_info['basin_states'],
        # Parameters
        **{k: v for k, v in params.items() if k != 'attractors'},
        attractors=np.array(params['attractors'])
    )
    
    # Save detailed report
    with open(output_dir / 'roa_analysis_report.txt', 'w') as f:
        f.write("FAST LCFM ROA CLASSIFICATION ANALYSIS REPORT\n")
        f.write("Using Cached Endpoints\n")
        f.write("=" * 60 + "\n\n")
        
        # Source information
        f.write("Source Information:\n")
        f.write(f"  Cache file: Generated {metadata['timestamp']}\n")
        f.write(f"  Model: {metadata['model_path']}\n")
        f.write(f"  Ground truth: {metadata['ground_truth_path']}\n")
        f.write(f"  Generation time: {metadata['generation_time']:.1f}s\n")
        f.write(f"  Random seed: {metadata['random_seed']}\n\n")
        
        # Analysis parameters
        f.write("Analysis Parameters:\n")
        f.write(f"  Samples per state: {params['num_samples']}\n")
        f.write(f"  Probability threshold: {params['prob_threshold']}\n")
        f.write(f"  Radius threshold: {params['radius_threshold']}\n")
        f.write(f"  Total data points: {params['n_points']:,}\n")
        f.write(f"  Target attractor: {params['attractors'][params['target_attractor_idx']]}\n")
        f.write(f"  Integration steps: {metadata['num_steps']}\n\n")
        
        # Separatrix analysis
        f.write("SEPARATRIX ANALYSIS:\n")
        f.write(f"  Total points: {params['n_points']:,}\n")
        f.write(f"  Basin points: {separatrix_info['n_basin']:,} ({100-separatrix_info['separatrix_percentage']:.1f}%)\n")
        f.write(f"  Separatrix points: {separatrix_info['n_separatrix']:,} ({separatrix_info['separatrix_percentage']:.1f}%)\n\n")
        
        # Performance results
        f.write("FULL DATASET PERFORMANCE:\n")
        full = results['full_dataset']
        f.write(f"  Accuracy:   {full['accuracy']:.6f}\n")
        f.write(f"  Precision:  {full['precision']:.6f}\n")
        f.write(f"  Recall:     {full['recall']:.6f}\n")
        f.write(f"  F1-Score:   {full['f1_score']:.6f}\n")
        f.write(f"  ROC AUC:    {full['roc_auc']:.6f}\n")
        f.write(f"  TPR:        {full['tpr']:.6f}\n")
        f.write(f"  TNR:        {full['tnr']:.6f}\n")
        f.write(f"  FPR:        {full['fpr']:.6f}\n")
        f.write(f"  FNR:        {full['fnr']:.6f}\n\n")
        
        if results['basin_only'] is not None:
            f.write("BASIN-ONLY PERFORMANCE:\n")
            basin = results['basin_only']
            f.write(f"  Points analyzed: {basin['n_basin_points']:,}\n")
            f.write(f"  Accuracy:   {basin['accuracy']:.6f}\n")
            f.write(f"  Precision:  {basin['precision']:.6f}\n")
            f.write(f"  Recall:     {basin['recall']:.6f}\n")
            f.write(f"  F1-Score:   {basin['f1_score']:.6f}\n")
            f.write(f"  ROC AUC:    {basin['roc_auc']:.6f}\n")
            f.write(f"  TPR:        {basin['tpr']:.6f}\n")
            f.write(f"  TNR:        {basin['tnr']:.6f}\n")
            f.write(f"  FPR:        {basin['fpr']:.6f}\n")
            f.write(f"  FNR:        {basin['fnr']:.6f}\n\n")
    
    print(f"💾 Analysis results saved:")
    print(f"  - roa_analysis_results.npz")
    print(f"  - roa_analysis_report.txt")


def main():
    parser = argparse.ArgumentParser(description="Fast LCFM ROA analysis using cached endpoints")
    parser.add_argument("--endpoints", type=str, required=True,
                       help="Path to cached endpoints file (.npz)")
    parser.add_argument("--output_dir", type=str, default="fast_roa_analysis",
                       help="Output directory for results")
    parser.add_argument("--prob_threshold", type=float, default=0.6,
                       help="Probability threshold for classification (default: 0.6)")
    parser.add_argument("--radius_threshold", type=float, default=0.075,
                       help="Radius threshold for attractor convergence (default: 0.075)")
    parser.add_argument("--analysis_samples", type=int, default=None,
                       help="Number of samples to use for analysis (default: use all cached samples)")
    args = parser.parse_args()
    
    # Load cached endpoints
    cached_data = load_cached_endpoints(args.endpoints)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting fast ROA analysis:")
    print(f"  Endpoints file: {args.endpoints}")
    print(f"  Output directory: {output_dir}")
    print(f"  Probability threshold: {args.prob_threshold}")
    print(f"  Radius threshold: {args.radius_threshold}")
    print(f"  Analysis samples: {args.analysis_samples or 'all cached samples'}")
    
    # Analyze endpoints
    import time
    start_time = time.time()
    
    results = analyze_cached_endpoints(
        cached_data,
        prob_threshold=args.prob_threshold,
        radius_threshold=args.radius_threshold,
        analysis_samples=args.analysis_samples
    )
    
    analysis_time = time.time() - start_time
    print(f"⚡ Analysis completed in {analysis_time:.1f} seconds")
    
    # Create visualizations
    create_classification_performance_plot(results, output_dir)
    create_separatrix_basin_plot(results, output_dir)
    
    # Save results
    save_analysis_results(results, cached_data, output_dir)
    
    # Print summary
    separatrix_info = results['separatrix_info']
    print(f"\n🎉 Fast ROA analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    print(f"\n📈 Performance Summary:")
    full = results['full_dataset']
    print(f"  Full Dataset - Accuracy: {full['accuracy']:.4f}, Precision: {full['precision']:.4f}, Recall: {full['recall']:.4f}, F1: {full['f1_score']:.4f}")
    
    if results['basin_only'] is not None:
        basin = results['basin_only']
        print(f"  Basin Only   - Accuracy: {basin['accuracy']:.4f}, Precision: {basin['precision']:.4f}, Recall: {basin['recall']:.4f}, F1: {basin['f1_score']:.4f}")
        print(f"  Improvement  - Accuracy: {basin['accuracy']-full['accuracy']:+.4f}, Precision: {basin['precision']-full['precision']:+.4f}, Recall: {basin['recall']-full['recall']:+.4f}, F1: {basin['f1_score']-full['f1_score']:+.4f}")
    
    print(f"\n🧭 Separatrix Analysis:")
    print(f"  Basin points: {separatrix_info['n_basin']:,} ({100-separatrix_info['separatrix_percentage']:.1f}%)")
    print(f"  Separatrix points: {separatrix_info['n_separatrix']:,} ({separatrix_info['separatrix_percentage']:.1f}%)")
    
    print(f"\n⚡ Speed improvement: Analysis took {analysis_time:.1f}s vs {cached_data['metadata']['generation_time']:.1f}s for generation")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Fast CartPole LCFM endpoint analysis using cached endpoints

This script performs CartPole ROA classification analysis using pre-generated endpoints,
allowing for fast repeated analysis with different parameters.

NO PLOTS - ONLY NUMERICAL METRICS: TP, TN, FP, FN, accuracy, precision, recall, F1, TPR, TNR, FPR, FNR
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import time

from src.systems.cartpole_lcfm import CartPoleSystemLCFM


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
    print(f"   Total endpoints: {data['endpoints'].size // 4:,}")  # 4D for CartPole
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


def classify_cartpole_endpoint_to_attractor(endpoints, system, radius_threshold=0.2):
    """
    Classify CartPole endpoints to attractors using CartPole-specific logic
    
    Args:
        endpoints: Array of endpoints [N, 4] (x, ẋ, θ, θ̇)
        system: CartPole system object
        radius_threshold: Radius threshold for attractor membership
    
    Returns:
        attractor_indices: Array [N] with attractor indices (-1 = no convergence)
        min_distances: Array [N] with distances to closest attractor
    """
    n_endpoints = len(endpoints)
    attractor_indices = np.full(n_endpoints, -1, dtype=int)  # -1 = no convergence
    
    # Use system's is_in_attractor method (CartPoleSystemLCFM has this method)
    # Convert to torch tensor for system compatibility
    if isinstance(endpoints, np.ndarray):
        endpoints_tensor = torch.from_numpy(endpoints).float()
    else:
        endpoints_tensor = endpoints
    
    # Check which endpoints are in attractor basins (proper LCFM method)
    in_attractor_mask = system.is_in_attractor(endpoints_tensor, radius=radius_threshold)
    
    if hasattr(in_attractor_mask, 'cpu'):
        in_attractor_mask = in_attractor_mask.cpu().numpy()
    
    # For CartPole, all attractors are essentially the same (balanced state)
    # So assign attractor index 0 to all successful states
    attractor_indices[in_attractor_mask] = 0
    
    # Calculate distances (for reference, though CartPole uses different logic)
    # Use a simple combined distance metric
    distances = np.zeros(n_endpoints)
    for i in range(n_endpoints):
        x, x_dot, theta, theta_dot = endpoints[i]
        # Simple distance metric combining position, velocity, and angle deviations
        pos_dist = abs(x)  # Distance from center position
        vel_dist = abs(x_dot)  # Velocity magnitude
        angle_dist = min(abs(theta), abs(abs(theta) - np.pi))  # Distance from vertical
        ang_vel_dist = abs(theta_dot)  # Angular velocity magnitude
        
        # Combined distance (weighted)
        distances[i] = pos_dist + 0.1 * vel_dist + angle_dist + 0.1 * ang_vel_dist
    
    return attractor_indices, distances


def analyze_cached_endpoints(cached_data, prob_threshold=0.6, radius_threshold=0.2, analysis_samples=None):
    """
    Analyze cached CartPole endpoints with given parameters
    
    Args:
        cached_data: Loaded cached endpoint data
        prob_threshold: Probability threshold for classification
        radius_threshold: Radius threshold for attractor convergence
        analysis_samples: Number of samples to use for analysis (default: use all cached samples)
    """
    
    start_states = cached_data['start_states']
    true_labels = cached_data['labels']
    all_endpoints = cached_data['endpoints']  # [N, num_samples, 4]
    attractors = cached_data['attractors']
    max_samples = cached_data['num_samples']
    
    # Create CartPole system
    system = CartPoleSystemLCFM()
    
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
    
    print(f"\n🎯 Analyzing cached CartPole endpoints:")
    print(f"  Start states: {n_points:,}")
    print(f"  Samples per state: {num_samples}")
    print(f"  Probability threshold: {prob_threshold}")
    print(f"  Radius threshold: {radius_threshold}")
    print(f"  CartPole attractors: {len(attractors)} (balanced states)")
    
    # Storage for results
    predicted_probs_target = np.zeros(n_points)
    predicted_labels_binary = np.zeros(n_points, dtype=int)
    
    # Process all endpoints
    print("🔄 Processing endpoint classifications...")
    
    for i in tqdm(range(n_points), desc="Analyzing endpoints"):
        state_endpoints = endpoints_to_use[i]  # [num_samples, 4]
        
        # Classify endpoints to attractors
        attractor_indices, distances = classify_cartpole_endpoint_to_attractor(
            state_endpoints, system, radius_threshold
        )
        
        # Calculate probability of reaching attractor (successful balance)
        n_successful = np.sum(attractor_indices >= 0)  # Count endpoints that reached any attractor
        prob_success = n_successful / num_samples
        
        # Store probability
        predicted_probs_target[i] = prob_success
        
        # Binary classification: success if probability exceeds threshold
        if prob_success > prob_threshold:
            predicted_labels_binary[i] = 1  # Predict success
        else:
            predicted_labels_binary[i] = 0  # Predict failure
    
    # Calculate comprehensive metrics
    results = calculate_classification_metrics(
        true_labels, predicted_labels_binary, predicted_probs_target
    )
    
    # Add evaluation parameters
    results['evaluation_params'] = {
        'num_samples': num_samples,
        'prob_threshold': prob_threshold,
        'radius_threshold': radius_threshold,
        'n_points': n_points,
        'attractors': attractors,
    }
    
    return results


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


def save_analysis_results(results, cached_data, output_dir):
    """Save comprehensive analysis results (NO PLOTS - NUMERICAL ONLY)"""
    
    params = results['evaluation_params']
    metadata = cached_data['metadata']
    
    # Save numerical results
    np.savez(
        output_dir / 'cartpole_roa_analysis_results.npz',
        # Classification results
        y_true=results['y_true'],
        y_pred=results['y_pred'],
        y_probs=results['y_probs'],
        confusion_matrix=results['confusion_matrix'],
        # Individual metrics
        tp=results['tp'],
        tn=results['tn'], 
        fp=results['fp'],
        fn=results['fn'],
        accuracy=results['accuracy'],
        precision=results['precision'],
        recall=results['recall'],
        f1_score=results['f1_score'],
        tpr=results['tpr'],
        tnr=results['tnr'],
        fpr=results['fpr'],
        fnr=results['fnr'],
        roc_auc=results['roc_auc'],
        # Parameters
        **{k: v for k, v in params.items() if k != 'attractors'},
        attractors=np.array(params['attractors'])
    )
    
    # Save detailed report
    with open(output_dir / 'cartpole_roa_analysis_report.txt', 'w') as f:
        f.write("CARTPOLE LCFM ROA CLASSIFICATION ANALYSIS REPORT\n")
        f.write("Using Cached Endpoints - NUMERICAL METRICS ONLY\n")
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
        f.write(f"  Integration steps: {metadata['num_steps']}\n\n")
        
        # Confusion Matrix
        f.write("CONFUSION MATRIX:\n")
        cm = results['confusion_matrix']
        f.write(f"  True Negative  (TN): {results['tn']:,}\n")
        f.write(f"  False Positive (FP): {results['fp']:,}\n")
        f.write(f"  False Negative (FN): {results['fn']:,}\n")
        f.write(f"  True Positive  (TP): {results['tp']:,}\n\n")
        
        # Performance metrics
        f.write("CLASSIFICATION METRICS:\n")
        f.write(f"  Accuracy:   {results['accuracy']:.6f}\n")
        f.write(f"  Precision:  {results['precision']:.6f}\n") 
        f.write(f"  Recall:     {results['recall']:.6f}\n")
        f.write(f"  F1-Score:   {results['f1_score']:.6f}\n")
        f.write(f"  ROC AUC:    {results['roc_auc']:.6f}\n\n")
        
        f.write("RATE METRICS:\n")
        f.write(f"  TPR (True Positive Rate):  {results['tpr']:.6f}\n")
        f.write(f"  TNR (True Negative Rate):  {results['tnr']:.6f}\n")
        f.write(f"  FPR (False Positive Rate): {results['fpr']:.6f}\n") 
        f.write(f"  FNR (False Negative Rate): {results['fnr']:.6f}\n\n")
        
        # Summary statistics
        total_samples = results['tp'] + results['tn'] + results['fp'] + results['fn']
        f.write("SUMMARY STATISTICS:\n")
        f.write(f"  Total samples: {total_samples:,}\n")
        f.write(f"  Positive labels (ground truth): {results['tp'] + results['fn']:,}\n")
        f.write(f"  Negative labels (ground truth): {results['tn'] + results['fp']:,}\n")
        f.write(f"  Positive predictions: {results['tp'] + results['fp']:,}\n")
        f.write(f"  Negative predictions: {results['tn'] + results['fn']:,}\n")
    
    print(f"💾 Analysis results saved:")
    print(f"  - cartpole_roa_analysis_results.npz")
    print(f"  - cartpole_roa_analysis_report.txt")


def print_metrics_summary(results):
    """Print concise metrics summary to console"""
    print(f"\n" + "="*60)
    print(f"CARTPOLE ROA CLASSIFICATION RESULTS")
    print(f"="*60)
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"  TP: {results['tp']:,}    TN: {results['tn']:,}")
    print(f"  FP: {results['fp']:,}    FN: {results['fn']:,}")
    
    print(f"\n📈 PRIMARY METRICS:")
    print(f"  Accuracy:   {results['accuracy']:.6f}")
    print(f"  Precision:  {results['precision']:.6f}")
    print(f"  Recall:     {results['recall']:.6f}")
    print(f"  F1-Score:   {results['f1_score']:.6f}")
    
    print(f"\n🎯 RATE METRICS:")
    print(f"  TPR: {results['tpr']:.6f}    TNR: {results['tnr']:.6f}")
    print(f"  FPR: {results['fpr']:.6f}    FNR: {results['fnr']:.6f}")
    
    print(f"\n🏆 PERFORMANCE:")
    print(f"  ROC AUC: {results['roc_auc']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Fast CartPole LCFM ROA analysis using cached endpoints")
    parser.add_argument("--endpoints", type=str, required=True,
                       help="Path to cached endpoints file (.npz)")
    parser.add_argument("--output_dir", type=str, default="cartpole_roa_analysis",
                       help="Output directory for results")
    parser.add_argument("--prob_threshold", type=float, default=0.6,
                       help="Probability threshold for classification (default: 0.6)")
    parser.add_argument("--radius_threshold", type=float, default=0.2,
                       help="Radius threshold for attractor convergence (default: 0.2)")
    parser.add_argument("--analysis_samples", type=int, default=None,
                       help="Number of samples to use for analysis (default: use all cached samples)")
    args = parser.parse_args()
    
    # Load cached endpoints
    cached_data = load_cached_endpoints(args.endpoints)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting fast CartPole ROA analysis:")
    print(f"  Endpoints file: {args.endpoints}")
    print(f"  Output directory: {output_dir}")
    print(f"  Probability threshold: {args.prob_threshold}")
    print(f"  Radius threshold: {args.radius_threshold}")
    print(f"  Analysis samples: {args.analysis_samples or 'all cached samples'}")
    
    # Analyze endpoints
    start_time = time.time()
    
    results = analyze_cached_endpoints(
        cached_data,
        prob_threshold=args.prob_threshold,
        radius_threshold=args.radius_threshold,
        analysis_samples=args.analysis_samples
    )
    
    analysis_time = time.time() - start_time
    print(f"⚡ Analysis completed in {analysis_time:.1f} seconds")
    
    # Print metrics summary to console
    print_metrics_summary(results)
    
    # Save results (NO PLOTS - NUMERICAL ONLY)
    save_analysis_results(results, cached_data, output_dir)
    
    print(f"\n🎉 CartPole ROA analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"⚡ Speed improvement: Analysis took {analysis_time:.1f}s vs {cached_data['metadata']['generation_time']:.1f}s for generation")
    print(f"🚫 No plots generated (numerical metrics only as requested)")


if __name__ == "__main__":
    main()
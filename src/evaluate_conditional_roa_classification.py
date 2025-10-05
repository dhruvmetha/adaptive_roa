"""
ROA Classification Evaluation for Conditional Flow Matching

Evaluates binary classification performance of conditional flow matching model
against ground truth ROA labels using probabilistic classification approach.

Classification Logic:
- For each start state, generate num_samples endpoint predictions
- Calculate probability of convergence to (0,0) attractor based on proximity
- Classify as positive (1) if probability > prob_threshold
- Ground truth: 1 = converges to (0,0), 0 = converges to other attractors

Metrics Computed:
- Confusion Matrix: TP, TN, FP, FN
- Performance: Accuracy, Precision, Recall, F1-score
- Rates: TPR, TNR, FPR, FNR
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns

from src.flow_matching.conditional import ConditionalFlowMatchingInference
from src.systems.pendulum_config import PendulumConfig


def load_ground_truth_data(file_path):
    """
    Load ground truth ROA labels from file
    
    Args:
        file_path: Path to roa_labels.txt file
        
    Returns:
        Dictionary with start_states [N, 2] and labels [N]
    """
    print(f"Loading ground truth data from: {file_path}")
    
    # Load data: index, theta, theta_dot, label
    data = np.loadtxt(file_path)
    
    start_states = data[:, 1:3]  # theta, theta_dot columns
    labels = data[:, 3].astype(int)  # label column
    
    # Print data statistics
    n_total = len(labels)
    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)
    
    print(f"Ground truth statistics:")
    print(f"  Total samples: {n_total:,}")
    print(f"  Positive (1): {n_positive:,} ({n_positive/n_total*100:.1f}%)")
    print(f"  Negative (0): {n_negative:,} ({n_negative/n_total*100:.1f}%)")
    print(f"  State range - θ: [{np.min(start_states[:, 0]):.3f}, {np.max(start_states[:, 0]):.3f}]")
    print(f"  State range - θ̇: [{np.min(start_states[:, 1]):.3f}, {np.max(start_states[:, 1]):.3f}]")
    
    return {
        'start_states': start_states,
        'labels': labels
    }


def classify_endpoint_to_attractor(endpoints, config, radius_threshold=0.075):
    """
    Classify endpoints to nearest attractor within radius threshold
    
    Args:
        endpoints: Array of endpoint states [N, 2]
        config: PendulumConfig with attractor locations
        radius_threshold: Maximum distance to consider convergence
        
    Returns:
        attractor_indices: Index of nearest attractor (-1 if no convergence)
        distances: Distance to nearest attractor for each endpoint
    """
    n_endpoints = len(endpoints)
    attractor_indices = np.full(n_endpoints, -1, dtype=int)  # -1 = no convergence
    min_distances = np.full(n_endpoints, np.inf)
    
    # Check distance to each attractor
    for i, attractor in enumerate(config.ATTRACTORS):
        # Calculate distances to this attractor
        distances = np.linalg.norm(endpoints - attractor, axis=1)
        
        # Update closest attractor for points within threshold
        mask = (distances < radius_threshold) & (distances < min_distances)
        attractor_indices[mask] = i
        min_distances[mask] = distances[mask]
    
    return attractor_indices, min_distances


def calculate_three_class_stats(predicted_labels_3class, start_states):
    """Calculate statistics for three-class classification"""
    class_1_mask = predicted_labels_3class == 1   # Target attractor (0,0)
    class_neg1_mask = predicted_labels_3class == -1  # Other attractors
    class_0_mask = predicted_labels_3class == 0   # No convincing attractor
    
    stats = {
        'n_target_attractor': np.sum(class_1_mask),
        'n_other_attractors': np.sum(class_neg1_mask), 
        'n_no_convincing': np.sum(class_0_mask),
        'total': len(predicted_labels_3class),
        'target_states': start_states[class_1_mask] if np.sum(class_1_mask) > 0 else np.array([]),
        'other_states': start_states[class_neg1_mask] if np.sum(class_neg1_mask) > 0 else np.array([]),
        'unconvincing_states': start_states[class_0_mask] if np.sum(class_0_mask) > 0 else np.array([])
    }
    
    return stats


def evaluate_roa_classification(inferencer, ground_truth_data, num_samples=50, 
                              prob_threshold=0.6, radius_threshold=0.075, 
                              batch_size=100):
    """
    Evaluate ROA classification performance using three-class probabilistic approach
    
    Classification:
    - 1: (0,0) attractor with probability > threshold
    - -1: Other attractors with probability > threshold  
    - 0: No convincing attractor (all attractors < threshold)
    
    Args:
        inferencer: Conditional flow matching inference object
        ground_truth_data: Dictionary with start_states and labels
        num_samples: Number of samples per start state for probability estimation
        prob_threshold: Probability threshold for convincing classification
        radius_threshold: Distance threshold for attractor convergence
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with evaluation results
    """
    start_states = ground_truth_data['start_states']
    true_labels = ground_truth_data['labels']
    n_points = len(start_states)
    
    # Get pendulum configuration for attractor locations
    config = PendulumConfig()
    n_attractors = len(config.ATTRACTORS)
    target_attractor_idx = 0  # (0,0) attractor is at index 0
    
    print(f"\nEvaluating three-class ROA classification:")
    print(f"  Start states: {n_points:,}")
    print(f"  Samples per state: {num_samples}")
    print(f"  Probability threshold: {prob_threshold}")
    print(f"  Radius threshold: {radius_threshold}")
    print(f"  Attractors: {[attr.tolist() for attr in config.ATTRACTORS]}")
    print(f"  Target attractor (class 1): {config.ATTRACTORS[target_attractor_idx].tolist()}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total predictions: {n_points * num_samples:,}")
    
    # Storage for results
    predicted_probs_all = np.zeros((n_points, n_attractors))  # Probabilities for all attractors
    predicted_labels_3class = np.zeros(n_points, dtype=int)  # Three-class labels: -1, 0, 1
    predicted_labels_binary = np.zeros(n_points, dtype=int)  # Binary for compatibility: 0, 1
    predicted_probs_target = np.zeros(n_points)  # Target attractor probabilities
    
    # Process in batches
    for i in tqdm(range(0, n_points, batch_size), desc="Evaluating ROA classification"):
        end_idx = min(i + batch_size, n_points)
        batch_states = start_states[i:end_idx]
        batch_size_actual = len(batch_states)
        
        # Generate multiple samples for each start state (mega-batch approach)
        mega_batch = np.repeat(batch_states, num_samples, axis=0)
        
        # Predict endpoints
        mega_endpoints = inferencer.predict_endpoint(
            mega_batch,
            num_steps=50,
            method='rk4'
        )
        
        # Convert to numpy if needed
        if hasattr(mega_endpoints, 'cpu'):
            mega_endpoints = mega_endpoints.cpu().numpy()
        
        # Reshape to [batch_size_actual, num_samples, 2]
        batch_endpoints = mega_endpoints.reshape(batch_size_actual, num_samples, 2)
        
        # For each start state, calculate probabilities for all attractors
        for j in range(batch_size_actual):
            state_endpoints = batch_endpoints[j]  # [num_samples, 2]
            
            # Classify endpoints to attractors
            attractor_indices, distances = classify_endpoint_to_attractor(
                state_endpoints, config, radius_threshold
            )
            
            # Calculate probabilities for each attractor
            attractor_probs = np.zeros(n_attractors)
            for attr_idx in range(n_attractors):
                n_convergence = np.sum(attractor_indices == attr_idx)
                attractor_probs[attr_idx] = n_convergence / num_samples
            
            # Store probabilities
            predicted_probs_all[i + j] = attractor_probs
            predicted_probs_target[i + j] = attractor_probs[target_attractor_idx]
            
            # Three-class classification logic
            max_prob = np.max(attractor_probs)
            max_attractor_idx = np.argmax(attractor_probs)
            
            if max_prob > prob_threshold:
                # Convincing attractor found
                if max_attractor_idx == target_attractor_idx:
                    predicted_labels_3class[i + j] = 1    # Target attractor (0,0)
                    predicted_labels_binary[i + j] = 1    # Positive for binary
                else:
                    predicted_labels_3class[i + j] = -1   # Other attractor
                    predicted_labels_binary[i + j] = 0    # Negative for binary
            else:
                # No convincing attractor
                predicted_labels_3class[i + j] = 0        # No convincing attractor
                predicted_labels_binary[i + j] = 0        # Negative for binary
    
    # Calculate binary classification metrics (for compatibility with ground truth)
    binary_results = calculate_classification_metrics(true_labels, predicted_labels_binary, predicted_probs_target)
    
    # Calculate three-class statistics
    three_class_stats = calculate_three_class_stats(predicted_labels_3class, start_states)
    
    # Combine results
    results = binary_results
    results['three_class'] = {
        'predicted_labels_3class': predicted_labels_3class,
        'predicted_probs_all': predicted_probs_all,
        'stats': three_class_stats
    }
    
    # Add evaluation parameters to results
    results['evaluation_params'] = {
        'num_samples': num_samples,
        'prob_threshold': prob_threshold,
        'radius_threshold': radius_threshold,
        'n_points': n_points,
        'attractors': [attr.tolist() for attr in config.ATTRACTORS],
        'target_attractor_idx': target_attractor_idx
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
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Specificity)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        roc_auc = 0.0  # Handle case where only one class is present
    
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


def create_three_class_scatter_plot(results, output_dir):
    """Create scatter plot visualization of start states colored by three-class classification"""
    
    # Extract three-class data
    three_class_data = results['three_class']
    predicted_labels_3class = three_class_data['predicted_labels_3class']
    stats = three_class_data['stats']
    params = results['evaluation_params']
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Define colors for each class - highlight -1 (other attractors) more prominently
    colors = {-1: '#FF4500', 0: '#D3D3D3', 1: '#87CEEB'}  # Orange-red, light gray, light blue
    alphas = {-1: 0.8, 0: 0.4, 1: 0.6}  # Higher alpha for -1, lower for 0 and 1
    sizes = {-1: 2, 0: 1, 1: 1}  # Slightly larger points for -1
    labels = {-1: 'Other Attractors', 0: 'No Convincing Attractor', 1: 'Target Attractor (0,0)'}
    
    # Plot each class
    for class_val in [-1, 0, 1]:
        mask = predicted_labels_3class == class_val
        if np.sum(mask) > 0:
            states = stats['target_states'] if class_val == 1 else (
                     stats['other_states'] if class_val == -1 else 
                     stats['unconvincing_states'])
            
            if len(states) > 0:
                plt.scatter(states[:, 0], states[:, 1], 
                          c=colors[class_val], alpha=alphas[class_val], s=sizes[class_val], 
                          label=f'{labels[class_val]} (n={np.sum(mask):,})')
    
    # Add attractor locations with better visibility
    for i, attractor in enumerate(params['attractors']):
        marker = 'o' if i == params['target_attractor_idx'] else 's'
        color = '#4169E1' if i == params['target_attractor_idx'] else '#DC143C'  # Royal blue vs crimson
        size = 120 if i == params['target_attractor_idx'] else 100
        plt.scatter(attractor[0], attractor[1], 
                   c=color, marker=marker, s=size, 
                   edgecolors='black', linewidth=3,
                   label=f'Attractor {i}: {attractor}' + (' (Target)' if i == params['target_attractor_idx'] else ''),
                   zorder=10)  # Ensure attractors are on top
    
    plt.xlabel('θ (angle)', fontsize=12)
    plt.ylabel('θ̇ (angular velocity)', fontsize=12)
    plt.title(f'Three-Class ROA Classification Results\\n' + 
              f'Threshold: {params["prob_threshold"]}, Radius: {params["radius_threshold"]}', 
              fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(-np.pi - 0.5, np.pi + 0.5)
    plt.ylim(-8, 8)
    
    # Add π labels on x-axis
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
               ['-π', '-π/2', '0', 'π/2', 'π'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'three_class_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎨 Three-class scatter plot saved: three_class_scatter_plot.png")


def create_classification_visualizations(results, output_dir):
    """Create comprehensive visualizations of classification results"""
    
    # Extract data
    cm = results['confusion_matrix']
    y_true = results['y_true']
    y_pred = results['y_pred'] 
    y_probs = results['y_probs']
    params = results['evaluation_params']
    three_class_data = results['three_class']
    
    # Create three-class scatter plot first
    create_three_class_scatter_plot(results, output_dir)
    
    # Create main analysis figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))  # Made taller to accommodate more plots
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 4, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'])
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # 2. ROC Curve
    ax2 = plt.subplot(2, 4, 2)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        ax2.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {results["roc_auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], 'r--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    except Exception as e:
        ax2.text(0.5, 0.5, f'ROC curve error:\n{str(e)}', ha='center', va='center')
        ax2.set_title('ROC Curve (Error)')
    
    # 3. Probability Distribution
    ax3 = plt.subplot(2, 4, 3)
    bins = np.linspace(0, 1, 21)
    ax3.hist(y_probs[y_true == 0], bins=bins, alpha=0.7, label='True Negative', color='red')
    ax3.hist(y_probs[y_true == 1], bins=bins, alpha=0.7, label='True Positive', color='blue')
    ax3.axvline(params['prob_threshold'], color='black', linestyle='--', 
                label=f'Threshold = {params["prob_threshold"]}')
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Count')
    ax3.set_title('Probability Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics Bar Chart
    ax4 = plt.subplot(2, 4, 4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'TPR', 'TNR']
    values = [results['accuracy'], results['precision'], results['recall'], 
              results['f1_score'], results['tpr'], results['tnr']]
    bars = ax4.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'gold', 'lightcoral'])
    ax4.set_ylabel('Score')
    ax4.set_title('Classification Metrics')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    
    # 5. Error Analysis - False Positives
    ax5 = plt.subplot(2, 4, 5)
    fp_mask = (y_true == 0) & (y_pred == 1)
    if np.sum(fp_mask) > 0:
        ax5.scatter(y_probs[fp_mask], np.random.normal(0, 0.1, np.sum(fp_mask)), 
                   c='red', alpha=0.6, s=20)
        ax5.set_xlabel('Predicted Probability')
        ax5.set_title(f'False Positives (n={np.sum(fp_mask)})')
        ax5.axvline(params['prob_threshold'], color='black', linestyle='--')
    else:
        ax5.text(0.5, 0.5, 'No False Positives', ha='center', va='center')
        ax5.set_title('False Positives (n=0)')
    
    # 6. Error Analysis - False Negatives
    ax6 = plt.subplot(2, 4, 6)
    fn_mask = (y_true == 1) & (y_pred == 0)
    if np.sum(fn_mask) > 0:
        ax6.scatter(y_probs[fn_mask], np.random.normal(0, 0.1, np.sum(fn_mask)), 
                   c='orange', alpha=0.6, s=20)
        ax6.set_xlabel('Predicted Probability')
        ax6.set_title(f'False Negatives (n={np.sum(fn_mask)})')
        ax6.axvline(params['prob_threshold'], color='black', linestyle='--')
    else:
        ax6.text(0.5, 0.5, 'No False Negatives', ha='center', va='center')
        ax6.set_title('False Negatives (n=0)')
    
    # 7. Class Distribution
    ax7 = plt.subplot(2, 4, 7)
    true_counts = np.bincount(y_true)
    pred_counts = np.bincount(y_pred)
    x = ['Negative (0)', 'Positive (1)']
    width = 0.35
    ax7.bar([i - width/2 for i in range(2)], true_counts, width, 
            label='Ground Truth', color='lightblue')
    ax7.bar([i + width/2 for i in range(2)], pred_counts, width, 
            label='Predicted', color='lightcoral')
    ax7.set_ylabel('Count')
    ax7.set_title('Class Distribution')
    ax7.set_xticks(range(2))
    ax7.set_xticklabels(x)
    ax7.legend()
    
    # 8. Three-Class Summary
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Get three-class statistics
    stats = three_class_data['stats']
    total = params['n_points']
    
    summary_text = f"""Evaluation Parameters:
Samples per state: {params['num_samples']}
Probability threshold: {params['prob_threshold']}
Radius threshold: {params['radius_threshold']}
Total points: {total:,}

Three-Class Distribution:
Target (0,0): {stats['n_target_attractor']:6,} ({stats['n_target_attractor']/total*100:4.1f}%)
Other attractors: {stats['n_other_attractors']:6,} ({stats['n_other_attractors']/total*100:4.1f}%)
No convincing: {stats['n_no_convincing']:6,} ({stats['n_no_convincing']/total*100:4.1f}%)

Binary Metrics vs Ground Truth:
Accuracy: {results['accuracy']:.3f}
Precision: {results['precision']:.3f}
Recall: {results['recall']:.3f}
F1-Score: {results['f1_score']:.3f}
"""
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roa_classification_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Classification visualization saved: roa_classification_analysis.png")


def save_classification_results(results, output_dir):
    """Save classification results to files"""
    
    # Extract three-class data
    three_class_data = results['three_class']
    
    # Save numerical results including three-class data
    np.savez(
        output_dir / 'roa_classification_results.npz',
        # Binary classification data (for ground truth comparison)
        y_true=results['y_true'],
        y_pred=results['y_pred'],
        y_probs=results['y_probs'],
        confusion_matrix=results['confusion_matrix'],
        # Three-class classification data
        predicted_labels_3class=three_class_data['predicted_labels_3class'],
        predicted_probs_all=three_class_data['predicted_probs_all'],
        **results['evaluation_params']
    )
    
    # Save detailed report
    with open(output_dir / 'roa_classification_report.txt', 'w') as f:
        f.write("ROA CLASSIFICATION EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Parameters
        params = results['evaluation_params']
        f.write("Evaluation Parameters:\n")
        f.write(f"  Samples per state: {params['num_samples']}\n")
        f.write(f"  Probability threshold: {params['prob_threshold']}\n")
        f.write(f"  Radius threshold: {params['radius_threshold']}\n")
        f.write(f"  Total data points: {params['n_points']:,}\n")
        f.write(f"  Target attractor: {params['attractors'][params['target_attractor_idx']]}\n\n")
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        f.write("Confusion Matrix:\n")
        f.write(f"                Predicted\n")
        f.write(f"           Negative  Positive\n")
        f.write(f"True Neg     {cm[0,0]:6d}    {cm[0,1]:6d}\n")
        f.write(f"True Pos     {cm[1,0]:6d}    {cm[1,1]:6d}\n\n")
        
        # Individual counts
        f.write("Classification Counts:\n")
        f.write(f"  True Positives (TP):  {results['tp']:6d}\n")
        f.write(f"  True Negatives (TN):  {results['tn']:6d}\n")
        f.write(f"  False Positives (FP): {results['fp']:6d}\n")
        f.write(f"  False Negatives (FN): {results['fn']:6d}\n\n")
        
        # Performance metrics
        f.write("Performance Metrics:\n")
        f.write(f"  Accuracy:   {results['accuracy']:.6f}\n")
        f.write(f"  Precision:  {results['precision']:.6f}\n")
        f.write(f"  Recall:     {results['recall']:.6f}\n")
        f.write(f"  F1-Score:   {results['f1_score']:.6f}\n")
        f.write(f"  ROC AUC:    {results['roc_auc']:.6f}\n\n")
        
        # Rates
        f.write("Classification Rates:\n")
        f.write(f"  True Positive Rate (TPR):  {results['tpr']:.6f}\n")
        f.write(f"  True Negative Rate (TNR):  {results['tnr']:.6f}\n")
        f.write(f"  False Positive Rate (FPR): {results['fpr']:.6f}\n")
        f.write(f"  False Negative Rate (FNR): {results['fnr']:.6f}\n\n")
        
        # Three-class statistics
        stats = three_class_data['stats']
        total = params['n_points']
        f.write("THREE-CLASS CLASSIFICATION DISTRIBUTION:\n")
        f.write(f"  Target Attractor (1):     {stats['n_target_attractor']:6d} ({stats['n_target_attractor']/total*100:5.1f}%)\n")
        f.write(f"  Other Attractors (-1):    {stats['n_other_attractors']:6d} ({stats['n_other_attractors']/total*100:5.1f}%)\n")
        f.write(f"  No Convincing Attr (0):   {stats['n_no_convincing']:6d} ({stats['n_no_convincing']/total*100:5.1f}%)\n")
        f.write(f"  Total:                    {total:6d} (100.0%)\n\n")
    
    print(f"\n💾 Classification results saved:")
    print(f"  - roa_classification_results.npz (raw data)")
    print(f"  - roa_classification_report.txt (detailed report)")


def main():
    parser = argparse.ArgumentParser(description="ROA Classification Evaluation for Conditional Flow Matching")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to conditional flow matching checkpoint")
    parser.add_argument("--ground_truth", type=str, 
                       default="/common/users/shared/pracsys/genMoPlan/data_trajectories/pendulum_lqr_50k/roa_labels.txt",
                       help="Path to ground truth ROA labels file")
    parser.add_argument("--output_dir", type=str, default="roa_classification_eval", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per start state (default: 50)")
    parser.add_argument("--prob_threshold", type=float, default=0.6, help="Probability threshold for positive classification (default: 0.6)")
    parser.add_argument("--radius_threshold", type=float, default=0.075, help="Radius threshold for attractor convergence (default: 0.075)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing (default: 100)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (for testing)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use")
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"🎯 Set CUDA_VISIBLE_DEVICES={args.gpu}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.device_count()} devices")
        print(f"📍 Current device: {torch.cuda.current_device()}")
    else:
        print("💻 Running on CPU")
    
    # Load model
    print(f"Loading conditional flow matching model from: {args.checkpoint}")
    try:
        inferencer = ConditionalFlowMatchingInference(args.checkpoint)
        print("✅ Model loaded successfully!")
        print(f"Model info: {inferencer.get_model_info()}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load ground truth data
    ground_truth_data = load_ground_truth_data(args.ground_truth)
    
    # Limit samples if specified (for testing)
    if args.max_samples is not None:
        print(f"🔄 Limiting evaluation to {args.max_samples} samples for testing")
        indices = np.random.choice(len(ground_truth_data['start_states']), 
                                 min(args.max_samples, len(ground_truth_data['start_states'])), 
                                 replace=False)
        ground_truth_data['start_states'] = ground_truth_data['start_states'][indices]
        ground_truth_data['labels'] = ground_truth_data['labels'][indices]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 Starting ROA classification evaluation:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Ground truth: {args.ground_truth}")
    print(f"  Output directory: {output_dir}")
    print(f"  Samples per state: {args.num_samples}")
    print(f"  Probability threshold: {args.prob_threshold}")
    print(f"  Radius threshold: {args.radius_threshold}")
    print(f"  Batch size: {args.batch_size}")
    
    # Run evaluation
    results = evaluate_roa_classification(
        inferencer, 
        ground_truth_data,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        radius_threshold=args.radius_threshold,
        batch_size=args.batch_size
    )
    
    # Create visualizations
    create_classification_visualizations(results, output_dir)
    
    # Save results
    save_classification_results(results, output_dir)
    
    # Print summary
    print(f"\n🎉 ROA classification evaluation complete!")
    print(f"📁 All results saved to: {output_dir}")
    print(f"\n📈 Classification Performance Summary:")
    print(f"  Accuracy:  {results['accuracy']:.3f}")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall:    {results['recall']:.3f}")
    print(f"  F1-Score:  {results['f1_score']:.3f}")
    print(f"  ROC AUC:   {results['roc_auc']:.3f}")
    
    cm = results['confusion_matrix']
    print(f"\n📊 Confusion Matrix:")
    print(f"  TP: {results['tp']:6d}  |  FN: {results['fn']:6d}")
    print(f"  FP: {results['fp']:6d}  |  TN: {results['tn']:6d}")


if __name__ == "__main__":
    main()
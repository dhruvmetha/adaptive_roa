#!/usr/bin/env python3
"""
FIXED evaluation using the working legacy inference system.
"""

import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Use the WORKING legacy inference system
from src.inference_circular_flow_matching import CircularFlowMatchingInference
from src.systems.pendulum_config import PendulumConfig

def load_ground_truth_data(file_path: str, sampling_interval: int = 25):
    """Load ground truth data, sampling every N lines."""
    print(f"Loading ground truth data from {file_path}")
    print(f"Sampling every {sampling_interval} lines")
    
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i % sampling_interval == 0:  # Sample every N lines
                parts = line.strip().split()
                if len(parts) >= 4:
                    serial = int(parts[0])
                    q = float(parts[1])
                    q_dot = float(parts[2])
                    attractor_label = int(parts[3])
                    data.append([serial, q, q_dot, attractor_label])
    
    data = np.array(data)
    print(f"Loaded {len(data)} samples")
    print(f"Label distribution: {np.bincount(data[:, 3].astype(int))}")
    
    return data

def predict_and_classify_attractors(inferencer, states, config):
    """
    Predict final states and classify convergence to attractors.
    Uses comprehensive classification:
    - 1: Converges to center attractor (0,0)
    - 0: Converges to any other attractor (left/right) 
    - -1: Separatrix (doesn't converge to any attractor)
    """
    print(f"Predicting attractors for {len(states)} states using LEGACY inference")
    
    # Convert to tensor (states are in raw space, no normalization needed for legacy)
    states_tensor = torch.tensor(states, dtype=torch.float32)
    
    # Get final states by running flow matching
    with torch.no_grad():
        final_states = inferencer.predict_endpoint(states_tensor)
    
    # Convert back to numpy 
    final_states = final_states.cpu().numpy()
    
    print(f"Final state statistics:")
    print(f"  Angle range: [{final_states[:, 0].min():.4f}, {final_states[:, 0].max():.4f}]")
    print(f"  Velocity range: [{final_states[:, 1].min():.4f}, {final_states[:, 1].max():.4f}]")
    
    # Check convergence to ALL attractors
    attractor_convergence = config.is_in_attractor(final_states)  # [N, 3] array
    
    # Initialize predictions with separatrix label (-1)
    predictions = np.full(len(final_states), -1, dtype=int)
    
    # Check each attractor
    center_converged = attractor_convergence[:, 0]  # Center (0,0)
    right_converged = attractor_convergence[:, 1]   # Right (2.1,0) 
    left_converged = attractor_convergence[:, 2]    # Left (-2.1,0)
    
    # Classify based on convergence
    predictions[center_converged] = 1  # Center attractor
    predictions[right_converged | left_converged] = 0  # Other attractors
    # Remaining states stay as -1 (separatrix)
    
    print(f"Comprehensive attractor classification:")
    print(f"  Center attractor (1): {np.sum(predictions == 1)} states")
    print(f"  Other attractors (0): {np.sum(predictions == 0)} states") 
    print(f"  Separatrix (-1): {np.sum(predictions == -1)} states")
    
    # Get distances to center attractor for analysis
    distances_to_center = np.linalg.norm(final_states - config.ATTRACTORS[0], axis=1)
    
    print(f"\nDistance to center attractor statistics:")
    print(f"  Min: {distances_to_center.min():.4f}")
    print(f"  Max: {distances_to_center.max():.4f}")
    print(f"  Mean: {distances_to_center.mean():.4f}")
    print(f"  Median: {np.median(distances_to_center):.4f}")
    
    # Also get closest attractor info for reference
    closest_attractors, closest_distances = config.get_closest_attractor(final_states)
    print(f"\nClosest attractor analysis:")
    unique, counts = np.unique(closest_attractors, return_counts=True)
    for attractor_idx, count in zip(unique, counts):
        print(f"  Closest to {config.ATTRACTOR_NAMES[attractor_idx]}: {count} states")
    
    return predictions, final_states, distances_to_center, closest_attractors

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics for center vs non-center classification."""
    # Convert to binary classification (1 vs not-1) 
    # Note: For evaluation purposes, we treat separatrix (-1) as non-center (0)
    y_true_binary = (y_true == 1).astype(int)
    y_pred_binary = (y_pred == 1).astype(int)  # Only label 1 is positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calculate rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'tpr': tpr,  # True Positive Rate
        'fpr': fpr,  # False Positive Rate  
        'tnr': tnr,  # True Negative Rate
        'fnr': fnr,  # False Negative Rate
        'tp': tp,    # True Positives
        'tn': tn,    # True Negatives
        'fp': fp,    # False Positives
        'fn': fn,    # False Negatives
        'confusion_matrix': cm
    }
    
    return metrics

def plot_comprehensive_analysis(states, final_states, true_labels, predicted_labels, 
                               distances, closest_attractors, config, output_dir):
    """Create smart comprehensive analysis plots focused on ground truth vs predictions."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # TOP ROW: Ground Truth vs Model Predictions (Initial States)
    
    # Plot 1: Initial states colored by GROUND TRUTH
    ax1 = axes[0, 0]
    true_binary = (true_labels == 1)
    ax1.scatter(states[~true_binary, 0], states[~true_binary, 1], 
               c='red', alpha=0.6, s=8, label='Non-center (truth)')
    ax1.scatter(states[true_binary, 0], states[true_binary, 1], 
               c='blue', alpha=0.6, s=8, label='Center (truth)')
    
    # Plot attractors
    for i, attr in enumerate(config.ATTRACTORS):
        circle = plt.Circle(attr, config.ATTRACTOR_RADIUS, 
                          color='gray', alpha=0.5, fill=False, linewidth=2)
        ax1.add_patch(circle)
        ax1.scatter(attr[0], attr[1], color='black', s=100, marker='x')
    
    ax1.set_xlabel('Angle (θ)')
    ax1.set_ylabel('Angular Velocity (θ̇)')
    ax1.set_title('Ground Truth Labels')
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(-2*np.pi, 2*np.pi)
    ax1.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Initial states colored by MODEL PREDICTIONS (3-class)
    ax2 = axes[0, 1]
    center_pred = (predicted_labels == 1)
    other_pred = (predicted_labels == 0) 
    separatrix_pred = (predicted_labels == -1)
    
    if np.any(other_pred):
        ax2.scatter(states[other_pred, 0], states[other_pred, 1], 
                   c='red', alpha=0.6, s=8, label=f'Other attractors ({np.sum(other_pred)})')
    if np.any(center_pred):
        ax2.scatter(states[center_pred, 0], states[center_pred, 1], 
                   c='blue', alpha=0.6, s=8, label=f'Center ({np.sum(center_pred)})')
    if np.any(separatrix_pred):
        ax2.scatter(states[separatrix_pred, 0], states[separatrix_pred, 1], 
                   c='orange', alpha=0.8, s=8, label=f'Separatrix ({np.sum(separatrix_pred)})')
    
    # Plot attractors
    for i, attr in enumerate(config.ATTRACTORS):
        circle = plt.Circle(attr, config.ATTRACTOR_RADIUS, 
                          color='gray', alpha=0.5, fill=False, linewidth=2)
        ax2.add_patch(circle)
        ax2.scatter(attr[0], attr[1], color='black', s=100, marker='x')
    
    ax2.set_xlabel('Angle (θ)')
    ax2.set_ylabel('Angular Velocity (θ̇)')
    ax2.set_title('Model Predictions')
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-2*np.pi, 2*np.pi)
    ax2.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction ACCURACY MAP (Correct vs Incorrect)
    ax3 = axes[0, 2]
    correct_mask = (true_labels == predicted_labels)
    incorrect_mask = ~correct_mask
    
    ax3.scatter(states[correct_mask, 0], states[correct_mask, 1], 
               c='green', alpha=0.6, s=8, label=f'Correct ({np.sum(correct_mask)})')
    ax3.scatter(states[incorrect_mask, 0], states[incorrect_mask, 1], 
               c='red', alpha=0.8, s=12, label=f'Incorrect ({np.sum(incorrect_mask)})')
    
    # Plot attractors
    for i, attr in enumerate(config.ATTRACTORS):
        circle = plt.Circle(attr, config.ATTRACTOR_RADIUS, 
                          color='gray', alpha=0.5, fill=False, linewidth=2)
        ax3.add_patch(circle)
        ax3.scatter(attr[0], attr[1], color='black', s=100, marker='x')
    
    ax3.set_xlabel('Angle (θ)')
    ax3.set_ylabel('Angular Velocity (θ̇)')
    ax3.set_title('Prediction Accuracy Map')
    ax3.set_xlim(-np.pi, np.pi)
    ax3.set_ylim(-2*np.pi, 2*np.pi)
    ax3.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # BOTTOM ROW: Analysis and Metrics
    
    # Plot 4: COMPREHENSIVE PREDICTION ANALYSIS (3-class)
    ax4 = axes[1, 0]
    
    # Show all 3 prediction classes with different visualization
    center_pred = (predicted_labels == 1)
    other_pred = (predicted_labels == 0)
    separatrix_pred = (predicted_labels == -1)
    
    # Also overlay ground truth info with shapes
    true_center = (true_labels == 1)
    true_other = (true_labels != 1)
    
    # Plot predicted classes with ground truth overlay
    if np.any(other_pred):
        # Other attractor predictions
        other_correct = other_pred & true_other
        other_incorrect = other_pred & true_center
        if np.any(other_correct):
            ax4.scatter(states[other_correct, 0], states[other_correct, 1], 
                       c='red', alpha=0.6, s=8, marker='o', label=f'Other (correct: {np.sum(other_correct)})')
        if np.any(other_incorrect):
            ax4.scatter(states[other_incorrect, 0], states[other_incorrect, 1], 
                       c='red', alpha=0.9, s=15, marker='x', label=f'Other (wrong: {np.sum(other_incorrect)})')
    
    if np.any(center_pred):
        # Center attractor predictions
        center_correct = center_pred & true_center
        center_incorrect = center_pred & true_other
        if np.any(center_correct):
            ax4.scatter(states[center_correct, 0], states[center_correct, 1], 
                       c='blue', alpha=0.6, s=8, marker='o', label=f'Center (correct: {np.sum(center_correct)})')
        if np.any(center_incorrect):
            ax4.scatter(states[center_incorrect, 0], states[center_incorrect, 1], 
                       c='blue', alpha=0.9, s=15, marker='x', label=f'Center (wrong: {np.sum(center_incorrect)})')
    
    if np.any(separatrix_pred):
        # Separatrix predictions
        sep_from_center = separatrix_pred & true_center
        sep_from_other = separatrix_pred & true_other
        if np.any(sep_from_center):
            ax4.scatter(states[sep_from_center, 0], states[sep_from_center, 1], 
                       c='orange', alpha=0.8, s=12, marker='^', label=f'Separatrix (from center: {np.sum(sep_from_center)})')
        if np.any(sep_from_other):
            ax4.scatter(states[sep_from_other, 0], states[sep_from_other, 1], 
                       c='orange', alpha=0.6, s=8, marker='s', label=f'Separatrix (from other: {np.sum(sep_from_other)})')
    
    # Plot attractors
    for i, attr in enumerate(config.ATTRACTORS):
        circle = plt.Circle(attr, config.ATTRACTOR_RADIUS, 
                          color='gray', alpha=0.5, fill=False, linewidth=2)
        ax4.add_patch(circle)
        ax4.scatter(attr[0], attr[1], color='black', s=100, marker='x')
    
    ax4.set_xlabel('Angle (θ)')
    ax4.set_ylabel('Angular Velocity (θ̇)')
    ax4.set_title('Comprehensive Prediction Analysis')
    ax4.set_xlim(-np.pi, np.pi)
    ax4.set_ylim(-2*np.pi, 2*np.pi)
    ax4.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: CONFUSION MATRIX
    ax5 = axes[1, 1]
    metrics = calculate_metrics(true_labels, predicted_labels)
    cm = metrics['confusion_matrix']
    
    im = ax5.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax5.figure.colorbar(im, ax=ax5)
    
    classes = ['Non-center', 'Center']
    tick_marks = np.arange(len(classes))
    ax5.set_xticks(tick_marks)
    ax5.set_yticks(tick_marks)
    ax5.set_xticklabels(classes)
    ax5.set_yticklabels(classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax5.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax5.set_ylabel('True Label')
    ax5.set_xlabel('Predicted Label')
    ax5.set_title('Confusion Matrix')
    
    # Plot 6: COMPREHENSIVE DISTANCE ANALYSIS
    ax6 = axes[1, 2]
    
    # Distance histograms by true label
    dist_true_non_center = distances[true_labels != 1]
    dist_true_center = distances[true_labels == 1]
    
    # Distance histograms by predicted label (3-class)
    dist_pred_center = distances[predicted_labels == 1]
    dist_pred_other = distances[predicted_labels == 0]
    dist_pred_separatrix = distances[predicted_labels == -1]
    
    # Plot ground truth distributions
    ax6.hist(dist_true_non_center, bins=25, alpha=0.4, label='Non-center (truth)', 
             color='red', density=True)
    ax6.hist(dist_true_center, bins=25, alpha=0.4, label='Center (truth)', 
             color='blue', density=True)
    
    # Plot prediction distributions with outlines
    if len(dist_pred_center) > 0:
        ax6.hist(dist_pred_center, bins=25, alpha=0.3, label=f'Center pred ({len(dist_pred_center)})', 
                 color='cyan', histtype='step', linewidth=2, density=True)
    if len(dist_pred_other) > 0:
        ax6.hist(dist_pred_other, bins=25, alpha=0.3, label=f'Other pred ({len(dist_pred_other)})', 
                 color='pink', histtype='step', linewidth=2, density=True)
    if len(dist_pred_separatrix) > 0:
        ax6.hist(dist_pred_separatrix, bins=25, alpha=0.3, label=f'Separatrix pred ({len(dist_pred_separatrix)})', 
                 color='orange', histtype='step', linewidth=2, density=True)
    
    ax6.axvline(config.ATTRACTOR_RADIUS, color='black', linestyle='--', linewidth=2,
               label=f'Radius threshold ({config.ATTRACTOR_RADIUS})')
    
    ax6.set_xlabel('Distance to Center Attractor')
    ax6.set_ylabel('Density')
    ax6.set_title('Comprehensive Distance Analysis')
    ax6.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legends below plots
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/comprehensive_attractor_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

def print_metrics(metrics):
    """Print classification metrics in a formatted way."""
    print("\n" + "="*60)
    print("FIXED CLASSIFICATION METRICS (Using Legacy Inference)")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"TPR:       {metrics['tpr']:.4f}")
    print(f"FPR:       {metrics['fpr']:.4f}")
    print(f"TNR:       {metrics['tnr']:.4f}")
    print(f"FNR:       {metrics['fnr']:.4f}")
    print()
    print("Confusion Matrix Components:")
    print(f"TP (True Positives):  {metrics['tp']}")
    print(f"TN (True Negatives):  {metrics['tn']}")
    print(f"FP (False Positives): {metrics['fp']}")
    print(f"FN (False Negatives): {metrics['fn']}")
    print()
    print("Confusion Matrix:")
    print("         Predicted")
    print("       Non-C  Center")
    print(f"True Non-C  {metrics['confusion_matrix'][0,0]:4d}    {metrics['confusion_matrix'][0,1]:4d}")
    print(f"   Center   {metrics['confusion_matrix'][1,0]:4d}    {metrics['confusion_matrix'][1,1]:4d}")

def save_results(metrics, config, output_dir="evaluation_results"):
    """Save evaluation results."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save metrics to text file
    with open(f"{output_dir}/attractor_evaluation_FIXED.txt", 'w') as f:
        f.write("FIXED ATTRACTOR PREDICTION EVALUATION RESULTS\n")
        f.write("(Using Working Legacy Inference System)\n")
        f.write("="*60 + "\n")
        f.write(f"Attractor Radius Used: {config.ATTRACTOR_RADIUS}\n")
        f.write(f"Attractors: {config.ATTRACTORS.tolist()}\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"TPR:       {metrics['tpr']:.4f}\n")
        f.write(f"FPR:       {metrics['fpr']:.4f}\n")
        f.write(f"TNR:       {metrics['tnr']:.4f}\n")
        f.write(f"FNR:       {metrics['fnr']:.4f}\n")
        f.write(f"\nTrue Positives:  {metrics['tp']}\n")
        f.write(f"True Negatives:  {metrics['tn']}\n")
        f.write(f"False Positives: {metrics['fp']}\n")
        f.write(f"False Negatives: {metrics['fn']}\n")
        f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
    
    print(f"Results saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='FIXED evaluation using legacy inference')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to trained circular flow matching model checkpoint')
    parser.add_argument('--data_path', type=str, 
                        default='/common/users/dm1487/arcmg_datasets/pendulum_lqr/pendulum_roa.txt',
                        help='Path to ground truth data file')
    parser.add_argument('--sampling_interval', type=int, default=25,
                        help='Sample every N lines from the data file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    print("Starting FIXED attractor prediction evaluation...")
    print("Using WORKING legacy inference system")
    
    # Initialize config
    config = PendulumConfig()
    print(f"Attractor radius: {config.ATTRACTOR_RADIUS}")
    print(f"Attractors: {config.ATTRACTORS}")
    
    # Load ground truth data
    print("\nLoading ground truth data...")
    ground_truth_data = load_ground_truth_data(args.data_path, args.sampling_interval)
    
    # Extract states and labels
    states = ground_truth_data[:, 1:3]  # q, q_dot (in original space)
    true_labels = ground_truth_data[:, 3].astype(int)  # attractor labels
    
    print(f"Evaluating on {len(states)} state samples")
    print(f"State range: angle [{states[:, 0].min():.3f}, {states[:, 0].max():.3f}]")
    print(f"State range: velocity [{states[:, 1].min():.3f}, {states[:, 1].max():.3f}]")
    
    # Load trained model (using WORKING legacy inference)
    print(f"\nLoading model from {args.model_path} using LEGACY inference")
    inferencer = CircularFlowMatchingInference(args.model_path)
    
    # Predict and classify attractors
    print("\nRunning model predictions and classification...")
    predicted_labels, final_states, distances, closest_attractors = predict_and_classify_attractors(
        inferencer, states, config)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(true_labels, predicted_labels)
    
    # Print results
    print_metrics(metrics)
    
    # Create comprehensive visualization
    print("\nCreating comprehensive visualizations...")
    plot_comprehensive_analysis(states, final_states, true_labels, predicted_labels, 
                               distances, closest_attractors, config, args.output_dir)
    
    # Save results
    save_results(metrics, config, args.output_dir)
    
    print("\nEvaluation complete!")
    print("KEY FIX: Used working legacy inference instead of buggy unified system")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Corrected evaluation of circular flow matching model on attractor prediction task.
Uses proper normalization and attractor detection from PendulumConfig.
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

from src.flow_matching.circular import CircularFlowMatchingInference
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
    Predict final states and classify which attractor they converge to.
    Uses proper normalization and PendulumConfig attractor detection.
    """
    print(f"Predicting attractors for {len(states)} states")
    
    # Convert to tensor (states are in raw [-π, π] × [-2π, 2π] space)
    states_tensor = torch.tensor(states, dtype=torch.float32)
    
    # Get final states by running flow matching (handles normalization internally)
    with torch.no_grad():
        final_states = inferencer.predict_endpoint(states_tensor)
    
    # Convert back to numpy (final_states are in original space)
    final_states = final_states.cpu().numpy()
    
    print(f"Final state statistics:")
    print(f"  Angle range: [{final_states[:, 0].min():.4f}, {final_states[:, 0].max():.4f}]")
    print(f"  Velocity range: [{final_states[:, 1].min():.4f}, {final_states[:, 1].max():.4f}]")
    
    # Classify based on PendulumConfig attractors and radius
    # Check if states are in center attractor (index 0 = (0,0))
    in_center_attractor = config.is_in_attractor(final_states, attractor_idx=0)
    
    # Convert to binary labels: 1 if in center attractor, 0 otherwise
    predictions = in_center_attractor.astype(int)
    
    # Get distances to center attractor for analysis
    distances_to_center = np.linalg.norm(final_states - config.ATTRACTORS[0], axis=1)
    
    print(f"Distance to center attractor statistics:")
    print(f"  Min: {distances_to_center.min():.4f}")
    print(f"  Max: {distances_to_center.max():.4f}")
    print(f"  Mean: {distances_to_center.mean():.4f}")
    print(f"  Median: {np.median(distances_to_center):.4f}")
    print(f"  States within radius {config.ATTRACTOR_RADIUS}: {np.sum(in_center_attractor)}")
    
    return predictions, final_states, distances_to_center

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    # Convert to binary classification (1 vs not-1)
    y_true_binary = (y_true == 1).astype(int)
    y_pred_binary = y_pred.astype(int)
    
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

def plot_attractor_analysis(states, final_states, true_labels, predicted_labels, distances, config, output_dir):
    """Create comprehensive attractor analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Initial states colored by true labels
    ax1 = axes[0, 0]
    true_binary = (true_labels == 1)
    ax1.scatter(states[~true_binary, 0], states[~true_binary, 1], 
               c='red', alpha=0.6, s=10, label='Non-center (ground truth)')
    ax1.scatter(states[true_binary, 0], states[true_binary, 1], 
               c='blue', alpha=0.6, s=10, label='Center (ground truth)')
    
    # Plot attractors
    for i, attr in enumerate(config.ATTRACTORS):
        circle = plt.Circle(attr, config.ATTRACTOR_RADIUS, 
                          color='gray', alpha=0.5, fill=False, linewidth=2)
        ax1.add_patch(circle)
        ax1.scatter(attr[0], attr[1], color='black', s=100, marker='x')
    
    ax1.set_xlabel('Angle (θ)')
    ax1.set_ylabel('Angular Velocity (θ̇)')
    ax1.set_title('Initial States (Ground Truth Labels)')
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(-2*np.pi, 2*np.pi)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final states colored by predictions
    ax2 = axes[0, 1]
    pred_binary = (predicted_labels == 1)
    ax2.scatter(final_states[~pred_binary, 0], final_states[~pred_binary, 1], 
               c='red', alpha=0.6, s=10, label='Non-center (predicted)')
    ax2.scatter(final_states[pred_binary, 0], final_states[pred_binary, 1], 
               c='blue', alpha=0.6, s=10, label='Center (predicted)')
    
    # Plot attractors
    for i, attr in enumerate(config.ATTRACTORS):
        circle = plt.Circle(attr, config.ATTRACTOR_RADIUS, 
                          color='gray', alpha=0.5, fill=False, linewidth=2)
        ax2.add_patch(circle)
        ax2.scatter(attr[0], attr[1], color='black', s=100, marker='x')
    
    ax2.set_xlabel('Angle (θ)')
    ax2.set_ylabel('Angular Velocity (θ̇)')
    ax2.set_title('Final States (Model Predictions)')
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-2*np.pi, 2*np.pi)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance distribution
    ax3 = axes[1, 0]
    dist_true_0 = distances[true_labels != 1]
    dist_true_1 = distances[true_labels == 1]
    
    ax3.hist(dist_true_0, bins=50, alpha=0.7, label='Non-center (ground truth)', color='red')
    ax3.hist(dist_true_1, bins=50, alpha=0.7, label='Center (ground truth)', color='blue')
    ax3.axvline(config.ATTRACTOR_RADIUS, color='black', linestyle='--', 
               label=f'Attractor radius ({config.ATTRACTOR_RADIUS})')
    ax3.set_xlabel('Distance to Center Attractor')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Distances to Center Attractor')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confusion matrix visualization
    ax4 = axes[1, 1]
    metrics = calculate_metrics(true_labels, predicted_labels)
    cm = metrics['confusion_matrix']
    
    im = ax4.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax4.figure.colorbar(im, ax=ax4)
    
    classes = ['Non-center', 'Center']
    tick_marks = np.arange(len(classes))
    ax4.set_xticks(tick_marks)
    ax4.set_yticks(tick_marks)
    ax4.set_xticklabels(classes)
    ax4.set_yticklabels(classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')
    ax4.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attractor_analysis_corrected.png", dpi=150, bbox_inches='tight')
    plt.close()

def print_metrics(metrics):
    """Print classification metrics in a formatted way."""
    print("\n" + "="*60)
    print("CORRECTED CLASSIFICATION METRICS")
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
    with open(f"{output_dir}/attractor_evaluation_corrected.txt", 'w') as f:
        f.write("CORRECTED ATTRACTOR PREDICTION EVALUATION RESULTS\n")
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
    parser = argparse.ArgumentParser(description='Corrected evaluation of circular flow model')
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
    
    print("Starting CORRECTED attractor prediction evaluation...")
    print("Using proper normalization and PendulumConfig attractor detection")
    
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
    
    # Load trained model
    print(f"\nLoading model from {args.model_path}")
    inferencer = CircularFlowMatchingInference(args.model_path)
    
    # Predict and classify attractors
    print("\nRunning model predictions and classification...")
    predicted_labels, final_states, distances = predict_and_classify_attractors(
        inferencer, states, config)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(true_labels, predicted_labels)
    
    # Print results
    print_metrics(metrics)
    
    # Create comprehensive visualization
    print("\nCreating visualizations...")
    plot_attractor_analysis(states, final_states, true_labels, predicted_labels, 
                           distances, config, args.output_dir)
    
    # Save results
    save_results(metrics, config, args.output_dir)
    
    print("\nEvaluation complete!")
    print("Key improvements in this corrected version:")
    print("- Uses proper normalization via FlowMatchingConfig")
    print("- Uses PendulumConfig.ATTRACTOR_RADIUS for classification")
    print("- Uses PendulumConfig.is_in_attractor() for proper detection")
    print("- Handles states in original [-π, π] × [-2π, 2π] space")

if __name__ == "__main__":
    main()
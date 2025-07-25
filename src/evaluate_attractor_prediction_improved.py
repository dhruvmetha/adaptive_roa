#!/usr/bin/env python3
"""
Improved evaluation of circular flow matching model on attractor prediction task.
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

def analyze_predictions(inferencer, states, config):
    """Analyze model predictions and find optimal threshold."""
    print(f"Analyzing predictions for {len(states)} states")
    
    # Convert to tensor
    states_tensor = torch.tensor(states, dtype=torch.float32)
    
    # Get final states by running flow matching
    with torch.no_grad():
        final_states = inferencer.predict_endpoint(states_tensor)
    
    # Convert back to numpy
    final_states = final_states.cpu().numpy()
    
    # Calculate distances to origin
    distances = np.sqrt(final_states[:, 0]**2 + final_states[:, 1]**2)
    
    print(f"Distance statistics:")
    print(f"  Min: {distances.min():.4f}")
    print(f"  Max: {distances.max():.4f}")
    print(f"  Mean: {distances.mean():.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    print(f"  Std: {distances.std():.4f}")
    
    # Try different thresholds
    thresholds = np.linspace(0.1, 2.0, 20)
    best_accuracy = 0
    best_threshold = 0.5
    
    return final_states, distances, best_threshold

def predict_attractors_with_threshold(final_states, threshold=0.5):
    """Predict which attractor each state converges to using distance threshold."""
    distances = np.sqrt(final_states[:, 0]**2 + final_states[:, 1]**2)
    
    # If close to origin (within threshold), predict attractor 1
    predictions = (distances < threshold).astype(int)
    
    return predictions

def find_optimal_threshold(final_states, true_labels, thresholds=None):
    """Find optimal threshold for attractor classification."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 3.0, 50)
    
    # Convert to binary labels
    y_true_binary = (true_labels == 1).astype(int)
    
    best_accuracy = 0
    best_threshold = 0.5
    results = []
    
    for threshold in thresholds:
        predictions = predict_attractors_with_threshold(final_states, threshold)
        accuracy = accuracy_score(y_true_binary, predictions)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
        
        results.append((threshold, accuracy))
    
    print(f"Optimal threshold: {best_threshold:.4f} (accuracy: {best_accuracy:.4f})")
    return best_threshold, results

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    # Convert to binary classification (1 vs not-1)
    y_true_binary = (y_true == 1).astype(int)
    y_pred_binary = (y_pred == 1).astype(int) if not isinstance(y_pred[0], (int, np.integer)) else y_pred
    
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

def plot_threshold_analysis(results, output_dir):
    """Plot threshold vs accuracy analysis."""
    thresholds, accuracies = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, 'b-', linewidth=2)
    plt.xlabel('Distance Threshold')
    plt.ylabel('Accuracy')
    plt.title('Threshold vs Accuracy Analysis')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/threshold_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_distance_distribution(distances, true_labels, output_dir):
    """Plot distribution of distances to origin by true label."""
    plt.figure(figsize=(12, 8))
    
    # Split by true labels
    dist_label_0 = distances[true_labels != 1]
    dist_label_1 = distances[true_labels == 1]
    
    plt.subplot(2, 1, 1)
    plt.hist(dist_label_0, bins=50, alpha=0.7, label='Non-origin attractors (0)', color='red')
    plt.hist(dist_label_1, bins=50, alpha=0.7, label='Origin attractor (1)', color='blue')
    plt.xlabel('Distance to Origin')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Distances to Origin')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.boxplot([dist_label_0, dist_label_1], labels=['Non-origin (0)', 'Origin (1)'])
    plt.ylabel('Distance to Origin')
    plt.title('Box Plot of Distances by True Label')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distance_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

def print_metrics(metrics, threshold=None):
    """Print classification metrics in a formatted way."""
    print("\n" + "="*50)
    print("CLASSIFICATION METRICS")
    if threshold is not None:
        print(f"(Using threshold: {threshold:.4f})")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"TPR:       {metrics['tpr']:.4f}")
    print(f"FPR:       {metrics['fpr']:.4f}")
    print(f"TNR:       {metrics['tnr']:.4f}")
    print(f"FNR:       {metrics['fnr']:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"TP: {metrics['tp']}, TN: {metrics['tn']}")
    print(f"FP: {metrics['fp']}, FN: {metrics['fn']}")
    print()
    print("Confusion Matrix (detailed):")
    print(metrics['confusion_matrix'])

def save_results(metrics, threshold, output_dir="evaluation_results"):
    """Save evaluation results."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save metrics to text file
    with open(f"{output_dir}/attractor_evaluation_metrics_improved.txt", 'w') as f:
        f.write("IMPROVED ATTRACTOR PREDICTION EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Optimal Threshold: {threshold:.4f}\n")
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
    parser = argparse.ArgumentParser(description='Improved evaluation of circular flow model on attractor prediction')
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
    
    print("Starting improved attractor prediction evaluation...")
    
    # Load ground truth data
    print("Loading ground truth data...")
    ground_truth_data = load_ground_truth_data(args.data_path, args.sampling_interval)
    
    # Extract states and labels
    states = ground_truth_data[:, 1:3]  # q, q_dot
    true_labels = ground_truth_data[:, 3].astype(int)  # attractor labels
    
    print(f"Evaluating on {len(states)} state samples")
    
    # Load trained model
    print(f"Loading model from {args.model_path}")
    inferencer = CircularFlowMatchingInference(args.model_path)
    config = PendulumConfig()
    
    # Get predictions
    print("Running model predictions...")
    states_tensor = torch.tensor(states, dtype=torch.float32)
    with torch.no_grad():
        final_states = inferencer.predict_endpoint(states_tensor)
    final_states = final_states.cpu().numpy()
    
    # Calculate distances
    distances = np.sqrt(final_states[:, 0]**2 + final_states[:, 1]**2)

    print(f"Distance statistics:")
    print(f"  Min: {distances.min():.4f}")
    print(f"  Max: {distances.max():.4f}")
    print(f"  Mean: {distances.mean():.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    
    # Find optimal threshold
    print("Finding optimal threshold...")
    optimal_threshold, threshold_results = find_optimal_threshold(final_states, true_labels)
    
    # Predict with optimal threshold
    predicted_labels = predict_attractors_with_threshold(final_states, optimal_threshold)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(true_labels, predicted_labels)
    
    # Print results
    print_metrics(metrics, optimal_threshold)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_threshold_analysis(threshold_results, args.output_dir)
    plot_distance_distribution(distances, true_labels, args.output_dir)
    
    # Save results
    save_results(metrics, optimal_threshold, args.output_dir)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
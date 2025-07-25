#!/usr/bin/env python3
"""
Evaluate circular flow matching model on attractor prediction task using ground truth data.
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

def predict_attractors(inferencer, states, config, num_steps=100, dt=0.01):
    """Predict which attractor each state converges to using the flow model."""
    print(f"Predicting attractors for {len(states)} states")
    
    # Convert to tensor
    states_tensor = torch.tensor(states, dtype=torch.float32)
    
    # Get final states by running flow matching
    with torch.no_grad():
        final_states = inferencer.predict_endpoint(states_tensor)
    
    # Convert back to numpy
    final_states = final_states.cpu().numpy()
    
    # Classify based on distance to attractors
    predictions = []
    for final_state in final_states:
        q_final, q_dot_final = final_state[0], final_state[1]
        
        # Check distance to (0, 0) attractor
        dist_to_origin = np.sqrt(q_final**2 + q_dot_final**2)
        
        # If close to origin (within some tolerance), predict attractor 1
        # Otherwise predict attractor 0 or -1
        if dist_to_origin < 0.5:  # Tolerance for being at origin
            predictions.append(1)
        else:
            predictions.append(0)  # Could be 0 or -1, but we'll use 0 for non-origin
    
    return np.array(predictions)

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    # Convert to binary classification (1 vs not-1)
    y_true_binary = (y_true == 1).astype(int)
    y_pred_binary = (y_pred == 1).astype(int)
    
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

def print_metrics(metrics):
    """Print classification metrics in a formatted way."""
    print("\n" + "="*50)
    print("CLASSIFICATION METRICS")
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

def save_results(metrics, output_dir="evaluation_results"):
    """Save evaluation results."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save metrics to text file
    with open(f"{output_dir}/attractor_evaluation_metrics.txt", 'w') as f:
        f.write("ATTRACTOR PREDICTION EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
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
    parser = argparse.ArgumentParser(description='Evaluate circular flow model on attractor prediction')
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
    
    # Update todo status
    print("Starting attractor prediction evaluation...")
    
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
    
    # Predict attractors
    print("Running model predictions...")
    predicted_labels = predict_attractors(inferencer, states, config)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(true_labels, predicted_labels)
    
    # Print results
    print_metrics(metrics)
    
    # Save results
    save_results(metrics, args.output_dir)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
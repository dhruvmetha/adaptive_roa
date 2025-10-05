#!/usr/bin/env python3
"""
Enhanced ROA Classification Evaluation for LCFM - Excluding Separatrix Points

Copied exactly from evaluate_conditional_roa_classification_no_separatrix.py
and adapted for Latent Conditional Flow Matching models.
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
import matplotlib.cm as cm

from src.flow_matching.latent_conditional.inference import LatentConditionalFlowMatchingInference
from src.systems.pendulum_lcfm import PendulumSystemLCFM


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


def classify_endpoint_to_attractor(endpoints, system, radius_threshold=0.075):
    """
    Classify endpoints to nearest attractor within radius threshold
    
    Args:
        endpoints: Array of endpoint states [N, 2]
        system: PendulumSystemLCFM with attractor locations
        radius_threshold: Maximum distance to consider convergence
        
    Returns:
        attractor_indices: Index of nearest attractor (-1 if no convergence)
        distances: Distance to nearest attractor for each endpoint
    """
    n_endpoints = len(endpoints)
    attractor_indices = np.full(n_endpoints, -1, dtype=int)  # -1 = no convergence
    min_distances = np.full(n_endpoints, np.inf)
    
    # Get attractors from system
    attractors = np.array(system.attractors())
    
    # Check distance to each attractor
    for i, attractor in enumerate(attractors):
        # Calculate distances to this attractor
        distances = np.linalg.norm(endpoints - attractor, axis=1)
        
        # Update closest attractor for points within threshold
        mask = (distances < radius_threshold) & (distances < min_distances)
        attractor_indices[mask] = i
        min_distances[mask] = distances[mask]
    
    return attractor_indices, min_distances


def identify_separatrix_points_from_3class(predicted_labels_3class, start_states):
    """
    Identify separatrix points using three-class predictions:
    - Class 1: Target attractor (0,0) - BASIN
    - Class -1: Other attractors - BASIN  
    - Class 0: No convincing attractor - SEPARATRIX
    
    Args:
        predicted_labels_3class: Three-class predictions [N] with values {-1, 0, 1}
        start_states: Initial states [N, 2]
        
    Returns:
        separatrix_mask: Boolean array [N] indicating separatrix points (class 0 only)
        separatrix_info: Dictionary with detailed separatrix analysis
    """
    n_points = len(predicted_labels_3class)
    
    # Separatrix = only points with no convincing attractor (class 0)
    separatrix_mask = (predicted_labels_3class == 0)
    
    # Calculate statistics
    n_target = np.sum(predicted_labels_3class == 1)    # Target attractor (basin)
    n_other = np.sum(predicted_labels_3class == -1)    # Other attractors (basin)  
    n_unconvincing = np.sum(predicted_labels_3class == 0)  # No convincing attractor (separatrix)
    n_separatrix = n_unconvincing
    n_basin = n_target + n_other
    separatrix_percentage = (n_separatrix / n_points) * 100
    
    print(f"\n🔍 Identifying separatrix points from three-class predictions:")
    print(f"  Classification logic:")
    print(f"    Class  1 (Target attractor): BASIN")
    print(f"    Class -1 (Other attractors):  BASIN") 
    print(f"    Class  0 (No convincing):     SEPARATRIX")
    
    print(f"\n📊 Separatrix identification results:")
    print(f"  Total points: {n_points:,}")
    print(f"  Basin points (class 1,-1): {n_basin:,} ({n_basin/n_points*100:.1f}%)")
    print(f"    - Target attractor (class 1): {n_target:,} ({n_target/n_points*100:.1f}%)")
    print(f"    - Other attractors (class -1): {n_other:,} ({n_other/n_points*100:.1f}%)")
    print(f"  Separatrix points (class 0): {n_separatrix:,} ({separatrix_percentage:.1f}%)")
    
    # Detailed separatrix analysis
    separatrix_info = {
        'n_separatrix': n_separatrix,
        'n_basin': n_basin,
        'n_target_attractor': n_target,
        'n_other_attractors': n_other,
        'n_no_convincing': n_unconvincing, 
        'separatrix_percentage': separatrix_percentage,
        'separatrix_states': start_states[separatrix_mask],
        'basin_states': start_states[~separatrix_mask],
        'classification_counts': {
            'target_attractor': n_target,
            'other_attractors': n_other,
            'no_convincing': n_unconvincing
        }
    }
    
    return separatrix_mask, separatrix_info


def evaluate_roa_classification_no_separatrix(inferencer, ground_truth_data, num_samples=50, 
                                            prob_threshold=0.6, radius_threshold=0.075, 
                                            batch_size=100):
    """
    Enhanced ROA classification that identifies and excludes separatrix points
    
    Separatrix points are defined as class 0 (no convincing attractor).
    Basin points are class 1 (target attractor) and class -1 (other attractors).
    
    Args:
        inferencer: LCFM inference object
        ground_truth_data: Dictionary with start_states and labels
        num_samples: Number of samples per start state
        prob_threshold: Probability threshold for classification
        radius_threshold: Distance threshold for attractor convergence
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with comprehensive evaluation results including separatrix analysis
    """
    start_states = ground_truth_data['start_states']
    true_labels = ground_truth_data['labels']
    n_points = len(start_states)
    
    # Get system and attractors
    system = inferencer.system
    n_attractors = len(system.attractors())
    target_attractor_idx = 0  # (0,0) attractor is at index 0
    
    print(f"\n🎯 Enhanced ROA classification with separatrix exclusion:")
    print(f"  Start states: {n_points:,}")
    print(f"  Samples per state: {num_samples}")
    print(f"  Probability threshold: {prob_threshold}")
    print(f"  Radius threshold: {radius_threshold}")
    print(f"  Separatrix definition: Class 0 (no convincing attractor)")
    print(f"  Attractors: {system.attractors()}")
    print(f"  Target attractor (class 1): {system.attractors()[target_attractor_idx]}")
    print(f"  Total predictions: {n_points * num_samples:,}")
    
    # Storage for all endpoints (needed for separatrix analysis)
    all_endpoints = np.zeros((n_points, num_samples, 2))
    
    # Storage for results
    predicted_probs_all = np.zeros((n_points, n_attractors))
    predicted_labels_3class = np.zeros(n_points, dtype=int)
    predicted_labels_binary = np.zeros(n_points, dtype=int)
    predicted_probs_target = np.zeros(n_points)
    
    # Process in batches and collect all endpoints
    for i in tqdm(range(0, n_points, batch_size), desc="Generating endpoint predictions"):
        end_idx = min(i + batch_size, n_points)
        batch_states = start_states[i:end_idx]
        batch_size_actual = len(batch_states)
        
        # Generate multiple samples for each start state (mega-batch approach)
        mega_batch = np.repeat(batch_states, num_samples, axis=0)
        mega_batch_tensor = torch.tensor(mega_batch, dtype=torch.float32, device=inferencer.device)
        
        # Predict endpoints - LCFM specific parameters
        mega_endpoints = inferencer.predict_endpoint(
            mega_batch_tensor,
            num_steps=50  # Integration steps
        )
        
        # Convert to numpy if needed
        if hasattr(mega_endpoints, 'cpu'):
            mega_endpoints = mega_endpoints.cpu().numpy()
        
        # Reshape to [batch_size_actual, num_samples, 2]
        batch_endpoints = mega_endpoints.reshape(batch_size_actual, num_samples, 2)
        
        # Debug: Check if we're getting different samples for the first few states
        if i == 0:  # Only for first batch
            for debug_idx in range(min(3, batch_size_actual)):
                endpoints_for_state = batch_endpoints[debug_idx]  # [num_samples, 2]
                unique_endpoints = np.unique(endpoints_for_state, axis=0)
                print(f"🔍 Debug - State {debug_idx}: {unique_endpoints.shape[0]}/{num_samples} unique endpoints")
                if unique_endpoints.shape[0] == 1:
                    print(f"⚠️  WARNING: All samples identical for state {debug_idx}!")
                    print(f"   Start state: {batch_states[debug_idx]}")
                    print(f"   All endpoints: {endpoints_for_state}")
        
        # Store all endpoints for separatrix analysis
        all_endpoints[i:end_idx] = batch_endpoints
        
        # Calculate probabilities for this batch
        for j in range(batch_size_actual):
            state_endpoints = batch_endpoints[j]  # [num_samples, 2]
            
            # Classify endpoints to attractors
            attractor_indices, distances = classify_endpoint_to_attractor(
                state_endpoints, system, radius_threshold
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
    
    print(f"\n🧭 All endpoint predictions completed. Analyzing separatrix regions...")
    
    # Identify separatrix points using three-class predictions
    separatrix_mask, separatrix_info = identify_separatrix_points_from_3class(
        predicted_labels_3class, start_states
    )
    
    # Create basin-only datasets (excluding separatrix)
    basin_mask = ~separatrix_mask
    n_basin = np.sum(basin_mask)
    
    if n_basin == 0:
        print("⚠️  WARNING: No basin points found! All points classified as separatrix.")
        basin_results = None
    else:
        print(f"\n📊 Computing metrics for basin points only (n={n_basin:,}):")
        
        # Extract basin-only data
        basin_true_labels = true_labels[basin_mask]
        basin_pred_labels_binary = predicted_labels_binary[basin_mask] 
        basin_pred_probs_target = predicted_probs_target[basin_mask]
        basin_start_states = start_states[basin_mask]
        
        # Calculate metrics for basin points only
        basin_results = calculate_classification_metrics(
            basin_true_labels, basin_pred_labels_binary, basin_pred_probs_target
        )
        
        # Add basin-specific information
        basin_results['basin_mask'] = basin_mask
        basin_results['basin_start_states'] = basin_start_states
        basin_results['n_basin_points'] = n_basin
        
        print(f"  Basin-only accuracy: {basin_results['accuracy']:.4f}")
        print(f"  Basin-only precision: {basin_results['precision']:.4f}")
        print(f"  Basin-only recall: {basin_results['recall']:.4f}")
        print(f"  Basin-only F1-score: {basin_results['f1_score']:.4f}")
        print(f"  Basin-only TPR: {basin_results['tpr']:.4f}")
        print(f"  Basin-only TNR: {basin_results['tnr']:.4f}")
        print(f"  Basin-only FPR: {basin_results['fpr']:.4f}")
        print(f"  Basin-only FNR: {basin_results['fnr']:.4f}")
    
    # Calculate full dataset metrics (for comparison)
    print(f"\n📊 Computing metrics for full dataset (including separatrix):")
    full_results = calculate_classification_metrics(true_labels, predicted_labels_binary, predicted_probs_target)
    print(f"  Full dataset accuracy: {full_results['accuracy']:.4f}")
    print(f"  Full dataset precision: {full_results['precision']:.4f}")
    print(f"  Full dataset recall: {full_results['recall']:.4f}")  
    print(f"  Full dataset F1-score: {full_results['f1_score']:.4f}")
    print(f"  Full dataset TPR: {full_results['tpr']:.4f}")
    print(f"  Full dataset TNR: {full_results['tnr']:.4f}")
    print(f"  Full dataset FPR: {full_results['fpr']:.4f}")
    print(f"  Full dataset FNR: {full_results['fnr']:.4f}")
    
    # Three-class statistics for full dataset
    three_class_stats = calculate_three_class_stats(predicted_labels_3class, start_states)
    
    # Combine all results
    results = {
        'full_dataset': full_results,
        'basin_only': basin_results,
        'separatrix_info': separatrix_info,
        'three_class': {
            'predicted_labels_3class': predicted_labels_3class,
            'predicted_probs_all': predicted_probs_all,
            'stats': three_class_stats
        },
        'all_endpoints': all_endpoints,
        'separatrix_mask': separatrix_mask,
        'evaluation_params': {
            'num_samples': num_samples,
            'prob_threshold': prob_threshold,
            'radius_threshold': radius_threshold,
            'n_points': n_points,
            'attractors': system.attractors(),
            'target_attractor_idx': target_attractor_idx,
            'separatrix_definition': 'class_0_no_convincing_attractor'
        }
    }
    
    return results


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


def create_classification_performance_plot(results, output_dir):
    """Create a detailed state-space plot showing 5-category classification performance"""
    
    print("🎨 Creating 5-category classification performance plot...")
    
    # Get data from results (same as separatrix vs basin plot)
    separatrix_info = results['separatrix_info']
    separatrix_mask = results['separatrix_mask'] 
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
    
    # Plot separatrix points first (these are always category 4)
    if len(separatrix_info['separatrix_states']) > 0:
        ax.scatter(separatrix_info['separatrix_states'][:, 0], separatrix_info['separatrix_states'][:, 1],
                  c=SEPARATRIX_COLOR, alpha=0.8, s=2, label=f'SEPARATRIX ({separatrix_info["n_separatrix"]:,})')
    
    # For basin points, we need to categorize them based on ground truth vs predictions
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
    
    # Add attractors (same as other plots)
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


def plot_sample_trajectories(inferencer, ground_truth_data, output_dir, num_examples=4, num_samples_per_example=10):
    """
    Create trajectory visualization showing start states, random samples, and endpoints
    
    Args:
        inferencer: LCFM inference model
        ground_truth_data: Ground truth data with start states and labels  
        output_dir: Directory to save plots
        num_examples: Number of example start states to visualize
        num_samples_per_example: Number of random samples per start state
    """
    print(f"\n🎨 Creating sample trajectory visualizations...")
    
    # Get system for attractor positions
    from src.systems.pendulum_lcfm import PendulumSystemLCFM
    system = PendulumSystemLCFM()
    attractors = np.array(system.attractors())
    
    # Select diverse start states from different regions
    np.random.seed(42)  # Reproducible examples
    
    # Get a mix of different ground truth labels
    unique_labels = np.unique(ground_truth_data['labels'])
    selected_indices = []
    
    for label in unique_labels:
        if len(selected_indices) >= num_examples:
            break
        label_indices = np.where(ground_truth_data['labels'] == label)[0]
        if len(label_indices) > 0:
            # Pick a random state from this label class
            selected_indices.append(np.random.choice(label_indices))
    
    # Fill remaining examples if needed
    while len(selected_indices) < num_examples:
        remaining_indices = set(range(len(ground_truth_data['start_states']))) - set(selected_indices)
        if remaining_indices:
            selected_indices.append(np.random.choice(list(remaining_indices)))
        else:
            break
    
    selected_indices = selected_indices[:num_examples]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red']
    
    # Collect all selected start states for batch processing
    selected_start_states = []
    selected_labels = []
    for idx in selected_indices:
        selected_start_states.append(ground_truth_data['start_states'][idx])
        selected_labels.append(ground_truth_data['labels'][idx])
    
    # Create mega-batch for ALL examples at once (more efficient)
    all_start_states = np.array(selected_start_states)  # [num_examples, 2]
    mega_batch = np.repeat(all_start_states, num_samples_per_example, axis=0)  # [num_examples * num_samples_per_example, 2]
    
    print(f"  Processing mega-batch: {mega_batch.shape[0]} total predictions")
    
    # We need to capture both initial noise and endpoints, so we'll call the inference method that gives us both
    # For now, let's modify to get the initial noise points as well
    print(f"  Generating initial noise and endpoints...")
    
    # Collect initial noise points and endpoints for each example
    all_initial_noise = []
    all_endpoints = []
    
    for i in range(num_examples):
        example_start_states = mega_batch[i*num_samples_per_example:(i+1)*num_samples_per_example]
        example_tensor = torch.tensor(example_start_states, dtype=torch.float32, device=inferencer.device)
        
        # We need to access the noise generation from within the inference
        # Let's call integrate_trajectory directly to get both noise and endpoints
        final_states_list = []
        noise_points_list = []
        
        for j in range(num_samples_per_example):
            start_state_single = example_tensor[j:j+1]  # [1, 2]
            
            # Sample noise manually (same as in inference)
            noise_point = inferencer.sample_noisy_input(1)  # [1, 2] - θ ∈ [-π,π], θ̇ ∈ [-1,1]
            
            # Denormalize the angular velocity for plotting: [-1,1] → [-2π, 2π]  
            noise_denorm = noise_point.clone()
            noise_denorm[0, 1] = noise_denorm[0, 1] * 2 * np.pi  # θ̇: [-1,1] → [-2π, 2π]
            noise_points_list.append(noise_denorm.cpu().numpy())
            
            # Get final state by calling inference
            final_state, _ = inferencer.integrate_trajectory(start_state_single, num_steps=50)
            final_states_list.append(final_state.cpu().numpy())
        
        all_initial_noise.append(np.array(noise_points_list).squeeze())  # [num_samples_per_example, 2]
        all_endpoints.append(np.array(final_states_list).squeeze())      # [num_samples_per_example, 2]
    
    all_initial_noise = np.array(all_initial_noise)  # [num_examples, num_samples_per_example, 2]
    all_endpoints = np.array(all_endpoints)          # [num_examples, num_samples_per_example, 2]
    
    # Now plot each example
    for i, (start_state, true_label) in enumerate(zip(selected_start_states, selected_labels)):
        ax = axes[i]
        
        print(f"  Example {i+1}: Start state [{start_state[0]:.3f}, {start_state[1]:.3f}], True label: {true_label}")
        
        # Plot start state as red X
        ax.plot(start_state[0], start_state[1], 'rx', markersize=12, markeredgewidth=3, 
                label=f'Start State (Label: {true_label})')
        
        # Get endpoints and initial noise for this example
        endpoints = all_endpoints[i]        # [num_samples_per_example, 2]
        noise_points = all_initial_noise[i]  # [num_samples_per_example, 2]
        
        print(f"    Generated {len(endpoints)} endpoints and {len(noise_points)} noise points")
        print(f"    Endpoint range: θ=[{endpoints[:,0].min():.3f}, {endpoints[:,0].max():.3f}], "
              f"θ̇=[{endpoints[:,1].min():.3f}, {endpoints[:,1].max():.3f}]")
        print(f"    Noise range: θ=[{noise_points[:,0].min():.3f}, {noise_points[:,0].max():.3f}], "
              f"θ̇=[{noise_points[:,1].min():.3f}, {noise_points[:,1].max():.3f}]")
        
        # Plot each sample: noise point, endpoint, and connecting line
        for j in range(num_samples_per_example):
            noise_point = noise_points[j]
            endpoint = endpoints[j]
            color = colors[j % len(colors)]
            
            # Plot initial noise as black X
            ax.plot(noise_point[0], noise_point[1], 'kx', markersize=6, 
                   markeredgewidth=1, alpha=0.8)
            
            # Plot endpoint as colored X
            ax.plot(endpoint[0], endpoint[1], 'x', color=color, markersize=8, 
                   markeredgewidth=2, alpha=0.7)
            
            # Draw connecting line from noise to endpoint
            ax.plot([noise_point[0], endpoint[0]], [noise_point[1], endpoint[1]], 
                   '-', color=color, alpha=0.5, linewidth=1)
                
        # Add legend for first subplot only
        if i == 0:
            ax.plot([], [], 'kx', markersize=6, 
                   label='Initial Noise Points')
            ax.plot([], [], 'x', color='blue', markersize=8, 
                   label=f'{num_samples_per_example} Sample Endpoints')
            ax.plot([], [], '-', color='gray', alpha=0.5, 
                   label='Noise → Endpoint Trajectories')
        
        # Plot attractors
        for att_idx, attractor in enumerate(attractors):
            ax.plot(attractor[0], attractor[1], 'ko', markersize=10, 
                   markerfacecolor='yellow', markeredgewidth=2,
                   label=f'Attractor {att_idx}' if i == 0 else '')
        
        # Formatting - use fixed axis limits
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-2*np.pi, 2*np.pi)
            
        ax.set_xlabel('θ (angle)')
        ax.set_ylabel('θ̇ (angular velocity)')
        ax.set_title(f'Example {i+1}: Start State Sampling\n'
                    f'θ={start_state[0]:.3f}, θ̇={start_state[1]:.3f}, Label={true_label}')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    trajectory_plot_path = output_dir / 'sample_trajectories.png'
    plt.savefig(trajectory_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Sample trajectory visualization saved to: {trajectory_plot_path}")


def save_enhanced_results(results, output_dir):
    """Save comprehensive results including separatrix analysis"""
    
    separatrix_info = results['separatrix_info']
    params = results['evaluation_params']
    
    # Save numerical results
    np.savez(
        output_dir / 'roa_classification_results_no_separatrix.npz',
        # Full dataset results
        full_y_true=results['full_dataset']['y_true'],
        full_y_pred=results['full_dataset']['y_pred'],
        full_y_probs=results['full_dataset']['y_probs'],
        full_confusion_matrix=results['full_dataset']['confusion_matrix'],
        # Basin-only results (if available)
        basin_y_true=results['basin_only']['y_true'] if results['basin_only'] else np.array([]),
        basin_y_pred=results['basin_only']['y_pred'] if results['basin_only'] else np.array([]),
        basin_y_probs=results['basin_only']['y_probs'] if results['basin_only'] else np.array([]),
        basin_confusion_matrix=results['basin_only']['confusion_matrix'] if results['basin_only'] else np.array([]),
        # Separatrix analysis
        separatrix_mask=results['separatrix_mask'],
        separatrix_states=separatrix_info['separatrix_states'],
        basin_states=separatrix_info['basin_states'],
        # Three-class data
        predicted_labels_3class=results['three_class']['predicted_labels_3class'],
        predicted_probs_all=results['three_class']['predicted_probs_all'],
        # Parameters
        **{k: v for k, v in params.items() if k != 'attractors'},
        attractors=np.array(params['attractors'])
    )
    
    # Save detailed report
    with open(output_dir / 'roa_classification_report_no_separatrix.txt', 'w') as f:
        f.write("ENHANCED ROA CLASSIFICATION EVALUATION REPORT (LCFM)\n")
        f.write("Excludes Separatrix Points from Statistical Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        # Parameters
        f.write("Evaluation Parameters:\n")
        f.write(f"  Samples per state: {params['num_samples']}\n")
        f.write(f"  Probability threshold: {params['prob_threshold']}\n")
        f.write(f"  Radius threshold: {params['radius_threshold']}\n")
        f.write(f"  Total data points: {params['n_points']:,}\n")
        f.write(f"  Target attractor: {params['attractors'][params['target_attractor_idx']]}\n")
        f.write(f"  Integration steps: 50\n")
        f.write(f"  dt: 0.02\n\n")
        
        # Separatrix Analysis
        f.write("SEPARATRIX ANALYSIS:\n")
        f.write(f"  Total points: {separatrix_info['n_basin'] + separatrix_info['n_separatrix']:,}\n")
        f.write(f"  Basin points: {separatrix_info['n_basin']:,} ({100-separatrix_info['separatrix_percentage']:.1f}%)\n")
        f.write(f"  Separatrix points: {separatrix_info['n_separatrix']:,} ({separatrix_info['separatrix_percentage']:.1f}%)\n")
        f.write(f"  Separatrix definition: Class 0 (no convincing attractor)\n")
        f.write(f"  Basin definition: Class 1 (target) and Class -1 (other attractors)\n\n")
        
        # Full Dataset Performance
        f.write("FULL DATASET PERFORMANCE (including separatrix):\n")
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
        
        # Basin-Only Performance
        if results['basin_only'] is not None:
            f.write("BASIN-ONLY PERFORMANCE (excluding separatrix):\n")
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
            
            # Performance improvement
            f.write("PERFORMANCE IMPROVEMENT (Basin vs Full):\n")
            f.write(f"  Accuracy improvement:  {basin['accuracy'] - full['accuracy']:+.6f}\n")
            f.write(f"  Precision improvement: {basin['precision'] - full['precision']:+.6f}\n")
            f.write(f"  Recall improvement:    {basin['recall'] - full['recall']:+.6f}\n")
            f.write(f"  F1-Score improvement:  {basin['f1_score'] - full['f1_score']:+.6f}\n\n")
        else:
            f.write("BASIN-ONLY PERFORMANCE: N/A (no basin points found)\n\n")
        
        # Three-class distribution
        stats = results['three_class']['stats']
        total = params['n_points']
        f.write("THREE-CLASS DISTRIBUTION:\n")
        f.write(f"  Target Attractor (1):     {stats['n_target_attractor']:6d} ({stats['n_target_attractor']/total*100:5.1f}%)\n")
        f.write(f"  Other Attractors (-1):    {stats['n_other_attractors']:6d} ({stats['n_other_attractors']/total*100:5.1f}%)\n")
        f.write(f"  No Convincing Attr (0):   {stats['n_no_convincing']:6d} ({stats['n_no_convincing']/total*100:5.1f}%)\n")
        f.write(f"  Total:                    {total:6d} (100.0%)\n\n")
    
    print(f"💾 Enhanced results saved:")
    print(f"  - roa_classification_results_no_separatrix.npz")
    print(f"  - roa_classification_report_no_separatrix.txt")


def main():
    parser = argparse.ArgumentParser(description="Enhanced ROA Classification for LCFM - Excludes Separatrix Points")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to LCFM timestamped folder (e.g., outputs/name/2024-01-15_14-30-45)")
    parser.add_argument("--ground_truth", type=str, 
                       default="/common/users/shared/pracsys/genMoPlan/data_trajectories/pendulum_lqr_50k/roa_labels.txt",
                       help="Path to ground truth ROA labels file")
    parser.add_argument("--output_dir", type=str, default="roa_classification_lcfm_no_separatrix", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per start state (default: 50)")
    parser.add_argument("--prob_threshold", type=float, default=None, help="Probability threshold for positive classification (default: auto-calculated for majority vote)")
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
    
    # Auto-calculate probability threshold for majority vote (50% + 1 sample)
    if args.prob_threshold is None:
        args.prob_threshold = (args.num_samples // 2 + 1) / args.num_samples
        print(f"🎯 Auto-calculated prob_threshold: {args.prob_threshold:.3f} (majority vote: {args.num_samples//2+1}/{args.num_samples} samples)")
    
    # Load model using new simplified interface
    print(f"Loading LCFM model from: {args.model_path}")
    try:
        inferencer = LatentConditionalFlowMatchingInference(args.model_path, device="cuda")
        print("✅ Model loaded successfully!")
        print(f"Model info: {inferencer}")
        
        # Ensure proper random seeding for stochastic behavior
        import time
        random_seed = int(time.time() * 1000) % 2**32
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        print(f"🎲 Set random seed: {random_seed} for stochastic sampling")
        
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
    
    print(f"\n🎯 Starting enhanced ROA classification evaluation:")
    print(f"  Model path: {args.model_path}")
    print(f"  Ground truth: {args.ground_truth}")
    print(f"  Output directory: {output_dir}")
    print(f"  Samples per state: {args.num_samples}")
    print(f"  Probability threshold: {args.prob_threshold}")
    print(f"  Radius threshold: {args.radius_threshold}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Integration steps: 50")  # LCFM uses 50 steps
    print(f"  dt: 1/50 = 0.02")       # dt = 1.0 / num_steps
    
    # Run enhanced evaluation
    results = evaluate_roa_classification_no_separatrix(
        inferencer, 
        ground_truth_data,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        radius_threshold=args.radius_threshold,
        batch_size=args.batch_size
    )
    
    # Create enhanced visualizations
    create_classification_performance_plot(results, output_dir)
    create_separatrix_basin_plot(results, output_dir)
    
    # Save enhanced results
    save_enhanced_results(results, output_dir)
    
    # Print summary
    separatrix_info = results['separatrix_info']
    print(f"\n🎉 Enhanced ROA classification evaluation complete!")
    print(f"📁 All results saved to: {output_dir}")
    
    print(f"\n🧭 Separatrix Analysis Summary:")
    print(f"  Total points: {separatrix_info['n_basin'] + separatrix_info['n_separatrix']:,}")
    print(f"  Basin points: {separatrix_info['n_basin']:,} ({100-separatrix_info['separatrix_percentage']:.1f}%)")
    print(f"  Separatrix points: {separatrix_info['n_separatrix']:,} ({separatrix_info['separatrix_percentage']:.1f}%)")
    
    print(f"\n📈 Full Dataset Performance:")
    full = results['full_dataset']
    print(f"  Accuracy:  {full['accuracy']:.4f}")
    print(f"  Precision: {full['precision']:.4f}")
    print(f"  Recall:    {full['recall']:.4f}")
    print(f"  F1-Score:  {full['f1_score']:.4f}")
    print(f"  TPR:       {full['tpr']:.4f}")
    print(f"  TNR:       {full['tnr']:.4f}")
    print(f"  FPR:       {full['fpr']:.4f}")
    print(f"  FNR:       {full['fnr']:.4f}")
    
    if results['basin_only'] is not None:
        print(f"\n🎯 Basin-Only Performance (Excluding Separatrix):")
        basin = results['basin_only']
        print(f"  Accuracy:  {basin['accuracy']:.4f} ({basin['accuracy']-full['accuracy']:+.4f})")
        print(f"  Precision: {basin['precision']:.4f} ({basin['precision']-full['precision']:+.4f})")
        print(f"  Recall:    {basin['recall']:.4f} ({basin['recall']-full['recall']:+.4f})")
        print(f"  F1-Score:  {basin['f1_score']:.4f} ({basin['f1_score']-full['f1_score']:+.4f})")
        print(f"  TPR:       {basin['tpr']:.4f} ({basin['tpr']-full['tpr']:+.4f})")
        print(f"  TNR:       {basin['tnr']:.4f} ({basin['tnr']-full['tnr']:+.4f})")
        print(f"  FPR:       {basin['fpr']:.4f} ({basin['fpr']-full['fpr']:+.4f})")
        print(f"  FNR:       {basin['fnr']:.4f} ({basin['fnr']-full['fnr']:+.4f})")
        
        print(f"\n✨ Excluding separatrix points improved:")
        improvements = []
        if basin['accuracy'] > full['accuracy']: improvements.append(f"accuracy by {basin['accuracy']-full['accuracy']:.4f}")
        if basin['precision'] > full['precision']: improvements.append(f"precision by {basin['precision']-full['precision']:.4f}")
        if basin['recall'] > full['recall']: improvements.append(f"recall by {basin['recall']-full['recall']:.4f}")
        if basin['f1_score'] > full['f1_score']: improvements.append(f"F1-score by {basin['f1_score']-full['f1_score']:.4f}")
        
        if improvements:
            for improvement in improvements:
                print(f"  • {improvement}")
        else:
            print("  • No significant improvements (metrics within tolerance)")
    else:
        print(f"\n⚠️  Basin-Only Analysis: N/A (all points classified as separatrix)")
        print(f"   Consider adjusting prob_threshold or radius_threshold")


if __name__ == "__main__":
    main()
"""
Enhanced ROA Classification Evaluation for Universal Flow Matching - Excluding Separatrix Points

This script adapts the conditional flow matching ROA evaluation for the Universal Flow Matching Framework.
It identifies separatrix points (those that don't converge to any attractor) and excludes them from 
statistical calculations to focus on well-defined basin regions.

Key Features:
- Works with Universal Flow Matching Framework (pendulum system)
- Separatrix identification using attractor convergence analysis  
- Separate statistics for separatrix vs non-separatrix regions
- Clean basin analysis excluding uncertain regions
- Enhanced visualizations showing separatrix boundaries

Separatrix Definition:
- Points where endpoint predictions don't consistently converge to any single attractor
- High variance in endpoint predictions (model uncertainty)
- Points near basin boundaries where small perturbations change outcomes
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

# Universal Flow Matching imports  
from src.flow_matching.universal import UniversalFlowMatchingInference
from src.systems.pendulum_universal import PendulumSystem
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
        inferencer: Universal flow matching inference object
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
    
    # Get pendulum configuration for attractor locations
    config = PendulumConfig()
    n_attractors = len(config.ATTRACTORS)
    target_attractor_idx = 0  # (0,0) attractor is at index 0
    
    print(f"\n🎯 Enhanced ROA classification with separatrix exclusion:")
    print(f"  Start states: {n_points:,}")
    print(f"  Samples per state: {num_samples}")
    print(f"  Probability threshold: {prob_threshold}")
    print(f"  Radius threshold: {radius_threshold}")
    print(f"  Separatrix definition: Class 0 (no convincing attractor)")
    print(f"  Attractors: {[attr.tolist() for attr in config.ATTRACTORS]}")
    print(f"  Target attractor (class 1): {config.ATTRACTORS[target_attractor_idx].tolist()}")
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
        
        # Convert to tensor for universal flow matching
        mega_batch_tensor = torch.from_numpy(mega_batch).float()
        
        # Predict endpoints using universal flow matching
        mega_endpoints = inferencer.predict_endpoint(
            mega_batch_tensor,
            num_steps=50
        )
        
        # Convert to numpy if needed
        if hasattr(mega_endpoints, 'cpu'):
            mega_endpoints = mega_endpoints.cpu().numpy()
        elif hasattr(mega_endpoints, 'numpy'):
            mega_endpoints = mega_endpoints.numpy()
        
        # Reshape to [batch_size_actual, num_samples, 2]
        batch_endpoints = mega_endpoints.reshape(batch_size_actual, num_samples, 2)
        
        # Store all endpoints for separatrix analysis
        all_endpoints[i:end_idx] = batch_endpoints
        
        # Calculate probabilities for this batch
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
        print("    All points classified as separatrix (class 0).")
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
    
    # Calculate full dataset metrics (for comparison)
    print(f"\n📊 Computing metrics for full dataset (including separatrix):")
    full_results = calculate_classification_metrics(true_labels, predicted_labels_binary, predicted_probs_target)
    print(f"  Full dataset accuracy: {full_results['accuracy']:.4f}")
    print(f"  Full dataset precision: {full_results['precision']:.4f}")
    print(f"  Full dataset recall: {full_results['recall']:.4f}")  
    print(f"  Full dataset F1-score: {full_results['f1_score']:.4f}")
    
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
            'attractors': [attr.tolist() for attr in config.ATTRACTORS],
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


def create_separatrix_analysis_plots(results, output_dir):
    """Create comprehensive visualizations showing separatrix analysis"""
    
    separatrix_info = results['separatrix_info']
    separatrix_mask = results['separatrix_mask']
    params = results['evaluation_params']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Get all start states
    all_start_states = np.concatenate([
        separatrix_info['basin_states'],
        separatrix_info['separatrix_states']
    ]) if len(separatrix_info['separatrix_states']) > 0 else separatrix_info['basin_states']
    
    # 1. Separatrix vs Basin Classification
    ax1 = axes[0, 0]
    if len(separatrix_info['basin_states']) > 0:
        ax1.scatter(separatrix_info['basin_states'][:, 0], separatrix_info['basin_states'][:, 1],
                   c='lightblue', alpha=0.6, s=1, label=f'Basin ({separatrix_info["n_basin"]:,})')
    if len(separatrix_info['separatrix_states']) > 0:
        ax1.scatter(separatrix_info['separatrix_states'][:, 0], separatrix_info['separatrix_states'][:, 1],
                   c='red', alpha=0.8, s=2, label=f'Separatrix ({separatrix_info["n_separatrix"]:,})')
    
    # Add attractors
    for i, attractor in enumerate(params['attractors']):
        marker = 'o' if i == params['target_attractor_idx'] else 's'
        ax1.scatter(attractor[0], attractor[1], c='black', marker=marker, s=100, 
                   edgecolors='white', linewidth=2, zorder=10)
    
    ax1.set_xlabel('θ (angle)')
    ax1.set_ylabel('θ̇ (angular velocity)')
    ax1.set_title(f'Universal FM: Separatrix vs Basin Points\n{separatrix_info["separatrix_percentage"]:.1f}% Separatrix')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax1.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    # 2. Three-Class Classification Map
    ax2 = axes[0, 1]
    three_class_labels = results['three_class']['predicted_labels_3class']
    colors = {-1: 'orange', 0: 'red', 1: 'blue'}
    for class_val in [-1, 0, 1]:
        mask = three_class_labels == class_val
        if np.sum(mask) > 0:
            label_name = {-1: 'Other Attractors', 0: 'Separatrix', 1: 'Target Attractor'}[class_val]
            ax2.scatter(all_start_states[mask, 0], all_start_states[mask, 1], 
                       c=colors[class_val], s=1, alpha=0.7, label=f'{label_name} ({np.sum(mask)})')
    ax2.set_xlabel('θ (angle)')
    ax2.set_ylabel('θ̇ (angular velocity)')
    ax2.set_title('Universal FM: Three-Class Classification Results')
    ax2.legend()
    ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax2.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    # 3. Target Attractor Probability Map
    ax3 = axes[0, 2]
    target_probs = results['three_class']['predicted_probs_all'][:, 0]  # Probabilities for target attractor
    scatter = ax3.scatter(all_start_states[:, 0], all_start_states[:, 1], 
                         c=target_probs, cmap='viridis', s=1, alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax3, label='Target Attractor Probability')
    ax3.set_xlabel('θ (angle)')
    ax3.set_ylabel('θ̇ (angular velocity)')
    ax3.set_title(f'Universal FM: Target Attractor Probabilities\n(Threshold: {params["prob_threshold"]})')
    ax3.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax3.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    # 4. Performance Comparison
    ax4 = axes[1, 0]
    if results['basin_only'] is not None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        full_values = [results['full_dataset']['accuracy'], results['full_dataset']['precision'], 
                      results['full_dataset']['recall'], results['full_dataset']['f1_score']]
        basin_values = [results['basin_only']['accuracy'], results['basin_only']['precision'],
                       results['basin_only']['recall'], results['basin_only']['f1_score']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, full_values, width, label='Full Dataset', color='lightcoral', alpha=0.8)
        ax4.bar(x + width/2, basin_values, width, label='Basin Only', color='lightblue', alpha=0.8)
        
        ax4.set_ylabel('Score')
        ax4.set_title('Universal FM: Full Dataset vs Basin Only')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        # Add value labels
        for i, (full_val, basin_val) in enumerate(zip(full_values, basin_values)):
            ax4.text(i - width/2, full_val + 0.01, f'{full_val:.3f}', ha='center', va='bottom', fontsize=9)
            ax4.text(i + width/2, basin_val + 0.01, f'{basin_val:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'No basin points\nfor comparison', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance Comparison (N/A)')
    
    # 5. Probability Distribution by Class
    ax5 = axes[1, 1]
    target_probs = results['three_class']['predicted_probs_all'][:, 0]
    
    for class_val in [-1, 0, 1]:
        mask = three_class_labels == class_val
        if np.sum(mask) > 0:
            label_name = {-1: 'Other Attractors', 0: 'Separatrix', 1: 'Target Attractor'}[class_val]
            color = colors[class_val]
            ax5.hist(target_probs[mask], bins=20, alpha=0.7, 
                    label=f'{label_name} (n={np.sum(mask):,})', 
                    color=color, density=True)
    
    ax5.axvline(params['prob_threshold'], color='black', linestyle='--', 
                label=f'Threshold = {params["prob_threshold"]}')
    ax5.set_xlabel('Target Attractor Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Universal FM: Target Probability by Class')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Separatrix Statistics Summary
    ax6 = axes[1, 2]
    
    # Create a summary statistics plot
    stats_data = [
        ['Framework', 'Universal Flow Matching'],
        ['Total Points', f"{separatrix_info['n_basin'] + separatrix_info['n_separatrix']:,}"],
        ['Basin Points', f"{separatrix_info['n_basin']:,} ({100-separatrix_info['separatrix_percentage']:.1f}%)"],
        ['Separatrix Points', f"{separatrix_info['n_separatrix']:,} ({separatrix_info['separatrix_percentage']:.1f}%)"],
        ['Prob Threshold', f"{params['prob_threshold']:.3f}"],
        ['Radius Threshold', f"{params['radius_threshold']:.3f}"]
    ]
    
    # Create text-based summary
    for i, (label, value) in enumerate(stats_data):
        ax6.text(0.05, 0.9 - i*0.12, f"{label}:", fontweight='bold', transform=ax6.transAxes, fontsize=10)
        ax6.text(0.55, 0.9 - i*0.12, value, transform=ax6.transAxes, fontsize=10)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('Universal Flow Matching\nAnalysis Summary')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'universal_separatrix_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🎨 Universal flow matching separatrix analysis saved: universal_separatrix_analysis.png")


def save_enhanced_results(results, output_dir):
    """Save comprehensive results including separatrix analysis"""
    
    separatrix_info = results['separatrix_info']
    params = results['evaluation_params']
    
    # Save numerical results
    np.savez(
        output_dir / 'universal_roa_classification_results_no_separatrix.npz',
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
        **params
    )
    
    # Save detailed report
    with open(output_dir / 'universal_roa_classification_report_no_separatrix.txt', 'w') as f:
        f.write("UNIVERSAL FLOW MATCHING ROA CLASSIFICATION EVALUATION REPORT\n")
        f.write("Excludes Separatrix Points from Statistical Analysis\n")
        f.write("=" * 70 + "\n\n")
        
        # Parameters
        f.write("Evaluation Parameters:\n")
        f.write(f"  Framework: Universal Flow Matching (Pendulum System S¹ × ℝ)\n")
        f.write(f"  Samples per state: {params['num_samples']}\n")
        f.write(f"  Probability threshold: {params['prob_threshold']}\n")
        f.write(f"  Radius threshold: {params['radius_threshold']}\n")
        f.write(f"  Total data points: {params['n_points']:,}\n")
        f.write(f"  Target attractor: {params['attractors'][params['target_attractor_idx']]}\n\n")
        
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
        f.write(f"  ROC AUC:    {full['roc_auc']:.6f}\n\n")
        
        # Basin-Only Performance
        if results['basin_only'] is not None:
            f.write("BASIN-ONLY PERFORMANCE (excluding separatrix):\n")
            basin = results['basin_only']
            f.write(f"  Points analyzed: {basin['n_basin_points']:,}\n")
            f.write(f"  Accuracy:   {basin['accuracy']:.6f}\n")
            f.write(f"  Precision:  {basin['precision']:.6f}\n")
            f.write(f"  Recall:     {basin['recall']:.6f}\n")
            f.write(f"  F1-Score:   {basin['f1_score']:.6f}\n")
            f.write(f"  ROC AUC:    {basin['roc_auc']:.6f}\n\n")
            
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
    
    print(f"💾 Universal flow matching results saved:")
    print(f"  - universal_roa_classification_results_no_separatrix.npz")
    print(f"  - universal_roa_classification_report_no_separatrix.txt")


def main():
    parser = argparse.ArgumentParser(description="Universal Flow Matching ROA Classification - Excludes Separatrix Points")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to universal flow matching checkpoint")
    parser.add_argument("--ground_truth", type=str, 
                       default="/common/users/shared/pracsys/genMoPlan/data_trajectories/pendulum_lqr_50k/roa_labels.txt",
                       help="Path to ground truth ROA labels file")
    parser.add_argument("--output_dir", type=str, default="universal_roa_classification_no_separatrix", help="Output directory")
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
    
    # Load model using configuration saved in checkpoint
    print(f"Loading universal flow matching model from: {args.checkpoint}")
    try:
        # Load the checkpoint to get the saved system and config
        print("📦 Loading checkpoint to extract saved configuration...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        
        # Get the exact system and config used during training
        system = checkpoint['hyper_parameters']['system']
        config = checkpoint['hyper_parameters']['config']
        
        print(f"✅ Extracted configuration from checkpoint:")
        print(f"  System: {type(system).__name__}")
        print(f"  Model dims: {config.hidden_dims}")
        print(f"  Time emb dim: {config.time_emb_dim}")
        print(f"  Integration steps: {config.num_integration_steps}")
        print(f"  System dimensions: {config.state_dim}→{config.embedding_dim}→{config.tangent_dim}")
        
        # Create inference with the exact same system and config from training
        inferencer = UniversalFlowMatchingInference(
            checkpoint_path=args.checkpoint,
            system=system,
            config=config
        )
        
        print("✅ Universal flow matching model loaded successfully with checkpoint config!")
        print(f"Model info: {inferencer}")
        
    except Exception as e:
        print(f"❌ Error loading model from checkpoint config: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔄 Falling back to default system...")
        
        # Fallback to creating fresh system (less ideal)
        try:
            system = PendulumSystem()
            inferencer = UniversalFlowMatchingInference(
                checkpoint_path=args.checkpoint,
                system=system
            )
            print("✅ Model loaded with fresh system!")
        except Exception as fallback_e:
            print(f"❌ Fallback also failed: {fallback_e}")
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
    
    print(f"\n🎯 Starting universal flow matching ROA classification evaluation:")
    print(f"  Framework: Universal Flow Matching (S¹ × ℝ pendulum system)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Ground truth: {args.ground_truth}")
    print(f"  Output directory: {output_dir}")
    print(f"  Samples per state: {args.num_samples}")
    print(f"  Probability threshold: {args.prob_threshold}")
    print(f"  Radius threshold: {args.radius_threshold}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Separatrix definition: Class 0 (no convincing attractor)")
    
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
    create_separatrix_analysis_plots(results, output_dir)
    
    # Save enhanced results
    save_enhanced_results(results, output_dir)
    
    # Print summary
    separatrix_info = results['separatrix_info']
    print(f"\n🎉 Universal flow matching ROA classification evaluation complete!")
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
    
    if results['basin_only'] is not None:
        print(f"\n🎯 Basin-Only Performance (Excluding Separatrix):")
        basin = results['basin_only']
        print(f"  Accuracy:  {basin['accuracy']:.4f} ({basin['accuracy']-full['accuracy']:+.4f})")
        print(f"  Precision: {basin['precision']:.4f} ({basin['precision']-full['precision']:+.4f})")
        print(f"  Recall:    {basin['recall']:.4f} ({basin['recall']-full['recall']:+.4f})")
        print(f"  F1-Score:  {basin['f1_score']:.4f} ({basin['f1_score']-full['f1_score']:+.4f})")
        
        print(f"\n✨ Universal flow matching improvements by excluding separatrix:")
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
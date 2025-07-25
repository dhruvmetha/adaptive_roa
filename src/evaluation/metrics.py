"""
Centralized evaluation metrics for flow matching models
"""
import numpy as np
from typing import Dict, Tuple, Optional, Any
from ..systems.pendulum_config import PendulumConfig


class FlowMatchingMetrics:
    """Centralized metrics computation for flow matching evaluation"""
    
    def __init__(self, config: PendulumConfig = None):
        self.config = config or PendulumConfig()
    
    def compute_basic_metrics(self, 
                            predictions: np.ndarray, 
                            ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Compute basic MSE and MAE metrics
        
        Args:
            predictions: Array of shape [N, 2] with predicted endpoints
            ground_truth: Array of shape [N, 2] with true endpoints
            
        Returns:
            Dictionary with metric values
        """
        # Overall metrics
        mse = np.mean((predictions - ground_truth) ** 2)
        mae = np.mean(np.abs(predictions - ground_truth))
        
        # Per-dimension metrics
        mse_angle = np.mean((predictions[:, 0] - ground_truth[:, 0]) ** 2)
        mse_velocity = np.mean((predictions[:, 1] - ground_truth[:, 1]) ** 2)
        mae_angle = np.mean(np.abs(predictions[:, 0] - ground_truth[:, 0]))
        mae_velocity = np.mean(np.abs(predictions[:, 1] - ground_truth[:, 1]))
        
        return {
            'mse': mse,
            'mae': mae,
            'mse_angle': mse_angle,
            'mse_velocity': mse_velocity,
            'mae_angle': mae_angle,
            'mae_velocity': mae_velocity
        }
    
    def compute_attractor_metrics(self, 
                                predictions: np.ndarray, 
                                ground_truth: np.ndarray) -> Dict[str, Any]:
        """
        Compute attractor-based metrics
        
        Args:
            predictions: Array of shape [N, 2] with predicted endpoints
            ground_truth: Array of shape [N, 2] with true endpoints
            
        Returns:
            Dictionary with attractor metrics
        """
        # Check which predictions/ground truth are in any attractor
        pred_in_attractor = self.config.is_in_attractor(predictions).any(axis=1)
        true_in_attractor = self.config.is_in_attractor(ground_truth).any(axis=1)
        
        # Attractor prediction accuracy (both in attractor or both not in attractor)
        attractor_accuracy = np.mean(pred_in_attractor == true_in_attractor)
        
        # Get closest attractors
        pred_closest_idx, pred_distances = self.config.get_closest_attractor(predictions)
        true_closest_idx, true_distances = self.config.get_closest_attractor(ground_truth)
        
        # Closest attractor accuracy
        closest_attractor_accuracy = np.mean(pred_closest_idx == true_closest_idx)
        
        return {
            'predictions_in_attractor_count': np.sum(pred_in_attractor),
            'predictions_in_attractor_percent': 100 * np.mean(pred_in_attractor),
            'ground_truth_in_attractor_count': np.sum(true_in_attractor),
            'ground_truth_in_attractor_percent': 100 * np.mean(true_in_attractor),
            'attractor_prediction_accuracy': 100 * attractor_accuracy,
            'closest_attractor_accuracy': 100 * closest_attractor_accuracy,
            'avg_distance_to_closest_attractor_pred': np.mean(pred_distances),
            'avg_distance_to_closest_attractor_true': np.mean(true_distances)
        }
    
    def compute_all_metrics(self, 
                          predictions: np.ndarray, 
                          ground_truth: np.ndarray) -> Dict[str, Any]:
        """
        Compute all available metrics
        
        Args:
            predictions: Array of shape [N, 2] with predicted endpoints
            ground_truth: Array of shape [N, 2] with true endpoints
            
        Returns:
            Dictionary with all metrics
        """
        basic_metrics = self.compute_basic_metrics(predictions, ground_truth)
        attractor_metrics = self.compute_attractor_metrics(predictions, ground_truth)
        
        # Combine metrics
        all_metrics = {
            'n_samples': len(predictions),
            **basic_metrics,
            **attractor_metrics
        }
        
        return all_metrics
    
    def format_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics as a readable report string
        
        Args:
            metrics: Dictionary of computed metrics
            
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 50)
        report.append("FLOW MATCHING EVALUATION RESULTS")
        report.append("=" * 50)
        report.append(f"Test samples: {metrics['n_samples']}")
        
        report.append("\\nOverall Metrics:")
        report.append(f"  MSE: {metrics['mse']:.6f}")
        report.append(f"  MAE: {metrics['mae']:.6f}")
        
        report.append("\\nPer-dimension Metrics:")
        report.append(f"  Angle MSE: {metrics['mse_angle']:.6f}")
        report.append(f"  Angle MAE: {metrics['mae_angle']:.6f}")
        report.append(f"  Velocity MSE: {metrics['mse_velocity']:.6f}")
        report.append(f"  Velocity MAE: {metrics['mae_velocity']:.6f}")
        
        report.append("\\nAttractor Metrics:")
        report.append(f"  Predictions in attractor: {metrics['predictions_in_attractor_count']}/{metrics['n_samples']} ({metrics['predictions_in_attractor_percent']:.1f}%)")
        report.append(f"  Ground truth in attractor: {metrics['ground_truth_in_attractor_count']}/{metrics['n_samples']} ({metrics['ground_truth_in_attractor_percent']:.1f}%)")
        report.append(f"  Attractor prediction accuracy: {metrics['attractor_prediction_accuracy']:.1f}%")
        report.append(f"  Closest attractor accuracy: {metrics['closest_attractor_accuracy']:.1f}%")
        report.append(f"  Avg distance to closest attractor (pred): {metrics['avg_distance_to_closest_attractor_pred']:.4f}")
        report.append(f"  Avg distance to closest attractor (true): {metrics['avg_distance_to_closest_attractor_true']:.4f}")
        
        return "\\n".join(report)
    
    def save_metrics_report(self, 
                          metrics: Dict[str, Any], 
                          filepath: str,
                          model_name: str = "Flow Matching") -> None:
        """
        Save metrics report to file
        
        Args:
            metrics: Dictionary of computed metrics
            filepath: Path to save the report
            model_name: Name of the model for the report header
        """
        report = self.format_metrics_report(metrics)
        report = report.replace("FLOW MATCHING EVALUATION RESULTS", 
                               f"{model_name.upper()} EVALUATION RESULTS")
        
        with open(filepath, 'w') as f:
            f.write(report)


class CircularFlowMatchingMetrics(FlowMatchingMetrics):
    """Specialized metrics for circular flow matching with circular distance"""
    
    def circular_distance(self, angle1: float, angle2: float) -> float:
        """Compute circular distance between two angles"""
        diff = angle1 - angle2
        return abs(np.arctan2(np.sin(diff), np.cos(diff)))
    
    def compute_basic_metrics(self, 
                            predictions: np.ndarray, 
                            ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Compute basic metrics with circular distance for angles
        
        Args:
            predictions: Array of shape [N, 2] with predicted endpoints
            ground_truth: Array of shape [N, 2] with true endpoints
            
        Returns:
            Dictionary with metric values using circular distance for angles
        """
        # Standard velocity metrics
        velocity_mse = np.mean((predictions[:, 1] - ground_truth[:, 1]) ** 2)
        velocity_mae = np.mean(np.abs(predictions[:, 1] - ground_truth[:, 1]))
        
        # Circular angle metrics
        angle_errors_circular = np.array([
            self.circular_distance(pred, true) 
            for pred, true in zip(predictions[:, 0], ground_truth[:, 0])
        ])
        angle_mse_circular = np.mean(angle_errors_circular ** 2)
        angle_mae_circular = np.mean(angle_errors_circular)
        
        # Overall metrics (combine circular angle + regular velocity)
        # For MSE/MAE, we can't simply average angle and velocity due to different scales
        # So we compute the norm of the error vector where angle error is circular
        combined_errors = np.sqrt(angle_errors_circular**2 + (predictions[:, 1] - ground_truth[:, 1])**2)
        overall_mae = np.mean(combined_errors)
        overall_mse = np.mean(combined_errors**2)
        
        return {
            'mse': overall_mse,
            'mae': overall_mae,
            'mse_angle': angle_mse_circular,
            'mse_velocity': velocity_mse,
            'mae_angle': angle_mae_circular,
            'mae_velocity': velocity_mae
        }
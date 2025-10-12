"""
Unified evaluation pipeline for flow matching models
"""
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from tqdm import tqdm

from .metrics import FlowMatchingMetrics, CircularFlowMatchingMetrics
from src.visualization.phase_space_plots import PhaseSpacePlotter
from src.visualization.flow_visualizer import FlowVisualizer
from src.systems.pendulum_config import PendulumConfig

# Import both flow matching variants
from src.flow_matching.standard.inference import StandardFlowMatchingInference
from src.flow_matching.circular.inference import CircularFlowMatchingInference


class FlowMatchingEvaluator:
    """Unified evaluation pipeline for flow matching models"""
    
    def __init__(self, 
                 model_name: str = "Flow Matching",
                 config: PendulumConfig = None,
                 use_circular_metrics: bool = False,
                 auto_detect_variant: bool = True):
        self.model_name = model_name
        self.config = config or PendulumConfig()
        self.auto_detect_variant = auto_detect_variant
        self.use_circular_metrics = use_circular_metrics
        
        # Initialize metrics calculator
        if use_circular_metrics:
            self.metrics_calculator = CircularFlowMatchingMetrics(config)
        else:
            self.metrics_calculator = FlowMatchingMetrics(config)
        
        # Initialize visualizers
        self.plotter = PhaseSpacePlotter(config)
        self.flow_visualizer = FlowVisualizer(config)
        
        # Storage for results
        self.results = {}
    
    def detect_flow_matching_variant(self, inferencer) -> str:
        """
        Automatically detect which flow matching variant is being used
        
        Args:
            inferencer: Model inference object
            
        Returns:
            Variant name: 'standard' or 'circular'
        """
        if isinstance(inferencer, StandardFlowMatchingInference):
            return 'standard'
        elif isinstance(inferencer, CircularFlowMatchingInference):
            return 'circular'
        else:
            # Fallback: try to detect based on class name
            class_name = inferencer.__class__.__name__.lower()
            if 'circular' in class_name:
                return 'circular'
            else:
                return 'standard'
    
    def evaluate_on_dataloader(self, 
                             inferencer,
                             test_loader,
                             data_module,
                             max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate model on a PyTorch DataLoader
        
        Args:
            inferencer: Model inference object with predict_endpoint method
            test_loader: PyTorch DataLoader for test data
            data_module: Data module with denormalization methods
            max_samples: Maximum number of samples to evaluate (None for all)
            
        Returns:
            Dictionary with evaluation results
        """
        # Auto-detect variant if enabled
        if self.auto_detect_variant:
            variant = self.detect_flow_matching_variant(inferencer)
            print(f"Detected flow matching variant: {variant}")
            
            # Update metrics calculator if needed
            if variant == 'circular' and not self.use_circular_metrics:
                print("Switching to circular metrics for detected circular flow matching")
                self.metrics_calculator = CircularFlowMatchingMetrics(self.config)
                self.use_circular_metrics = True
        
        print(f"Evaluating {self.model_name} on test set...")
        
        all_predictions = []
        all_ground_truth = []
        all_start_states = []
        
        total_samples = 0
        
        # Evaluate on test set
        for batch in tqdm(test_loader, desc="Evaluating"):
            if max_samples and total_samples >= max_samples:
                break
                
            start_states = batch["start_state"]
            true_endpoints = batch["end_state"]
            
            # Determine batch size for this iteration
            current_batch_size = len(start_states)
            if max_samples and total_samples + current_batch_size > max_samples:
                # Truncate batch to not exceed max_samples
                remaining = max_samples - total_samples
                start_states = start_states[:remaining]
                true_endpoints = true_endpoints[:remaining]
                current_batch_size = remaining
            
            # Denormalize for inference (inference handles normalization internally)
            start_states_denorm = torch.tensor([
                data_module.test_dataset.denormalize_state(s.numpy()) 
                for s in start_states
            ])
            true_endpoints_denorm = torch.tensor([
                data_module.test_dataset.denormalize_state(s.numpy()) 
                for s in true_endpoints
            ])
            
            # Predict endpoints
            predicted_endpoints = inferencer.predict_endpoint(start_states_denorm)
            
            # Store results
            all_start_states.append(start_states_denorm.cpu().numpy())
            all_predictions.append(predicted_endpoints.cpu().numpy())
            all_ground_truth.append(true_endpoints_denorm.cpu().numpy())
            
            total_samples += current_batch_size
        
        # Concatenate all results
        start_states = np.concatenate(all_start_states, axis=0)
        predictions = np.concatenate(all_predictions, axis=0)
        ground_truth = np.concatenate(all_ground_truth, axis=0)
        
        print(f"Evaluated on {len(predictions)} samples")
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_all_metrics(predictions, ground_truth)
        
        # Store results
        self.results = {
            'start_states': start_states,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'metrics': metrics
        }
        
        return self.results
    
    def evaluate_on_arrays(self,
                          start_states: np.ndarray,
                          predictions: np.ndarray,
                          ground_truth: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on numpy arrays
        
        Args:
            start_states: Array of shape [N, 2] with start states
            predictions: Array of shape [N, 2] with predicted endpoints
            ground_truth: Array of shape [N, 2] with true endpoints
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating {self.model_name} on {len(predictions)} samples...")
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_all_metrics(predictions, ground_truth)
        
        # Store results
        self.results = {
            'start_states': start_states,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'metrics': metrics
        }
        
        return self.results
    
    def print_results(self) -> None:
        """Print evaluation results to console"""
        if not self.results:
            print("No evaluation results available. Run evaluate_* method first.")
            return
        
        report = self.metrics_calculator.format_metrics_report(self.results['metrics'])
        print(report)
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save evaluation results and visualizations
        
        Args:
            output_dir: Directory to save results
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_* method first.")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics report
        metrics_file = output_dir / "evaluation_results.txt"
        self.metrics_calculator.save_metrics_report(
            self.results['metrics'], 
            metrics_file, 
            self.model_name
        )
        print(f"Metrics saved to: {metrics_file}")
        
        # Create visualizations
        self._create_evaluation_plots(output_dir)
        
        print(f"Evaluation complete! Results saved to {output_dir}")
    
    def _create_evaluation_plots(self, output_dir: Path) -> None:
        """Create and save evaluation plots"""
        start_states = self.results['start_states']
        predictions = self.results['predictions']
        ground_truth = self.results['ground_truth']
        
        # 1. Prediction vs Ground Truth Scatter
        self.plotter.plot_scatter_comparison(
            ground_truth, predictions,
            save_path=output_dir / "prediction_scatter.png"
        )
        
        # 2. Error distribution
        self.plotter.plot_error_distribution(
            ground_truth, predictions,
            save_path=output_dir / "error_distribution.png"
        )
        
        # 3. Phase space comparison
        self.plotter.plot_phase_space_comparison(
            ground_truth, predictions, start_states,
            save_path=output_dir / "phase_space_comparison.png"
        )
        
        print(f"Evaluation plots saved to {output_dir}")
    
    def create_sample_flow_paths(self,
                                inferencer,
                                n_samples: int = 5,
                                output_dir: Optional[Union[str, Path]] = None) -> List:
        """
        Create sample flow path visualizations
        
        Args:
            inferencer: Model inference object with predict_endpoint method
            n_samples: Number of sample paths to create
            output_dir: Directory to save plots (optional)
            
        Returns:
            List of matplotlib figures
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_* method first.")
            return []
        
        start_states = self.results['start_states']
        ground_truth = self.results['ground_truth']
        
        # Select random samples
        indices = np.random.choice(len(start_states), n_samples, replace=False)
        
        # Collect flow paths
        paths = []
        endpoints = []
        
        print(f"Generating {n_samples} sample flow paths...")
        for idx in tqdm(indices):
            start = start_states[idx]
            
            # Get flow path
            pred_end, path = inferencer.predict_endpoint(
                torch.tensor(start), return_path=True
            )
            
            paths.append(path.cpu().numpy())
            endpoints.append(pred_end.cpu().numpy())
        
        # Create visualizations
        selected_starts = start_states[indices]
        selected_trues = ground_truth[indices]
        
        figures = self.flow_visualizer.plot_sample_flow_paths(
            selected_starts, paths, np.array(endpoints), selected_trues,
            n_samples, output_dir
        )
        
        print(f"Sample flow paths created")
        return figures
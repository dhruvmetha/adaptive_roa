"""
Simple validation inference callback using unified model inference methods
"""
from re import X
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback
from typing import Dict
import numpy as np


class ValidationInferenceCallback(Callback):
    """
    Simple callback that uses the model's unified inference methods
    to evaluate endpoint prediction on validation set every N epochs.
    
    **KEY INSIGHT**: This callback just calls model.predict_endpoint() - 
    the same method used by standalone inference scripts!
    """
    
    def __init__(self, 
                 inference_frequency: int = 10,
                 num_integration_steps: int = 100):
        """
        Initialize validation inference callback
        
        Args:
            inference_frequency: Perform inference every N epochs
            num_integration_steps: Number of integration steps
        """
        super().__init__()
        self.inference_frequency = inference_frequency
        self.num_integration_steps = num_integration_steps
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each validation epoch.
        Performs inference evaluation using model's unified methods.
        """
        current_epoch = trainer.current_epoch
        
        # Only run inference on specified intervals
        if (current_epoch + 1) % self.inference_frequency != 0:
            return
            
        print(f"\n🔍 Running unified validation inference at epoch {current_epoch + 1}...")
        
        # Get test dataloader for inference evaluation (separate from validation loss)
        test_dataloader = trainer.datamodule.test_dataloader()
        
        all_mse_losses = []
        all_mae_losses = []
        all_component_losses = []
        total_samples = 0
        
        for batch_idx, batch in enumerate(test_dataloader):
            # Extract start and end states
            start_states = batch["raw_start_state"]  # [B, 4]
            gt_endpoints = batch["raw_end_state"]    # [B, 4]
            
            batch_size = start_states.shape[0]
            
            # Ensure tensors are on the same device as the model
            start_states = start_states.to(pl_module.device)
            gt_endpoints = gt_endpoints.to(pl_module.device)
            
            # we need to wrap the angles in the end states
            gt_endpoints[:, 1] = gt_endpoints[:, 1] % (2 * np.pi)
            gt_endpoints[:, 1] = torch.where(gt_endpoints[:, 1] > np.pi, gt_endpoints[:, 1] - 2 * np.pi, gt_endpoints[:, 1])
            
            
            gt_endpoints_normalized = pl_module.normalize_state(gt_endpoints.clone())
            # **UNIFIED INFERENCE**: Use model's predict_endpoint method
            # This is the SAME method used by generate_lcfm_endpoints.py!
            
            predicted_endpoints_normalized, predicted_endpoints = pl_module.predict_endpoint(
                start_states, 
                num_steps=self.num_integration_steps
            )
            
            component_losses = self._calculate_component_losses(predicted_endpoints_normalized, gt_endpoints_normalized)
            
            all_mse_losses.append(component_losses['mse_total'])
            all_mae_losses.append(component_losses['mae_total'])
            all_component_losses.append(component_losses)
            total_samples += batch_size
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(test_dataloader)}: MSE={mse_loss.item():.6f}")
        
        # Aggregate results
        avg_mse = np.mean(all_mse_losses)
        avg_mae = np.mean(all_mae_losses)
        avg_component_losses = self._aggregate_component_losses(all_component_losses)
        
        # Log to TensorBoard with test_inference namespace
        self._log_metrics(pl_module, avg_mse, avg_mae, avg_component_losses, total_samples)
        
        print(f"✅ Test inference evaluation completed:")
        print(f"   Test samples: {total_samples}")
        print(f"   MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}")
        print(f"   Components - x: {avg_component_losses.get('x_mse', 0):.6f}, θ: {avg_component_losses.get('theta_mse', 0):.6f}")
        
        print(f"   Components - x: {avg_component_losses.get('x_mae', 0):.6f}, x_dot: {avg_component_losses.get('x_dot_mae', 0):.6f}, θ: {avg_component_losses.get('theta_mae', 0):.6f}, θ_dot: {avg_component_losses.get('theta_dot_mae', 0):.6f}")
    
    def _calculate_component_losses(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        """Calculate component-wise losses"""
        with torch.no_grad():
            # Overall losses
            mse_total = nn.functional.mse_loss(predicted, ground_truth).item()
            mae_total = nn.functional.l1_loss(predicted[:, [0, 2, 3]], ground_truth[:, [0, 2, 3]]).item()
            
            
            x_mae = nn.functional.l1_loss(predicted[:, 0], ground_truth[:, 0]).item()
            x_dot_mae = nn.functional.l1_loss(predicted[:, 2], ground_truth[:, 2]).item()
            theta_dot_mae = nn.functional.l1_loss(predicted[:, 3], ground_truth[:, 3]).item()
            
            # Component-wise MSE
            x_mse = nn.functional.mse_loss(predicted[:, 0], ground_truth[:, 0]).item()
            x_dot_mse = nn.functional.mse_loss(predicted[:, 2], ground_truth[:, 2]).item()
            
            # Circular distance for angle
            theta_pred = predicted[:, 1]
            theta_gt = ground_truth[:, 1]
            theta_diff = torch.atan2(torch.sin(theta_pred - theta_gt), torch.cos(theta_pred - theta_gt))
            theta_mae = torch.abs(theta_diff).mean().item()
            mae_mean = torch.cat([torch.abs(predicted[:, [0, 2, 3]] - ground_truth[:, [0, 2, 3]]), theta_diff.unsqueeze(-1)], dim=-1).mean().item()
            
            theta_mse = torch.mean(theta_diff ** 2).item()
            theta_dot_mse = nn.functional.mse_loss(predicted[:, 3], ground_truth[:, 3]).item()
        
            
            return {
                'mse_total': mse_total,
                'mae_total': mae_mean,
                'x_mae': x_mae,
                'x_dot_mae': x_dot_mae,
                'theta_dot_mae': theta_dot_mae,
                'theta_mae': theta_mae,
                'x_mse': x_mse,
                'x_dot_mse': x_dot_mse,
                'theta_mse': theta_mse,
                'theta_dot_mse': theta_dot_mse
            }
    
    def _aggregate_component_losses(self, all_losses: list) -> Dict[str, float]:
        """Aggregate losses across batches"""
        if not all_losses:
            return {}
        
        keys = all_losses[0].keys()
        return {key: np.mean([losses[key] for losses in all_losses]) for key in keys}
    
    def _log_metrics(self, pl_module, avg_mse: float, avg_mae: float, 
                    component_losses: Dict[str, float], total_samples: int):
        """Log metrics to TensorBoard"""
        # Main metrics (shown in progress bar)
        pl_module.log('test_inference/mse', avg_mse, on_epoch=True, prog_bar=True)
        pl_module.log('test_inference/mae', avg_mae, on_epoch=True, prog_bar=True)
        pl_module.log('test_inference/samples', float(total_samples), on_epoch=True, prog_bar=False)
        
        # Component-wise metrics
        for key, value in component_losses.items():
            pl_module.log(f'test_inference/{key}', value, on_epoch=True, prog_bar=False)
        
        # Relative contributions
        if component_losses.get('mse_total', 0) > 0:
            total = component_losses['mse_total']
            pl_module.log('test_inference/position_contrib', 
                         (component_losses.get('x_mse', 0) + component_losses.get('x_dot_mse', 0)) / total, 
                         on_epoch=True, prog_bar=False)
            pl_module.log('test_inference/angle_contrib',
                         (component_losses.get('theta_mse', 0) + component_losses.get('theta_dot_mse', 0)) / total,
                         on_epoch=True, prog_bar=False)
        
        print(f"📊 Logged to TensorBoard: test_inference/*") 
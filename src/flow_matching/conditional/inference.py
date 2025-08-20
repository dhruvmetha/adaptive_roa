"""
Inference utilities for conditional flow matching
"""
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
import numpy as np

from ..base.inference import BaseFlowMatchingInference
from ..utils.state_transformations import embed_circular_state, extract_circular_state
from .flow_matcher import ConditionalFlowMatcher


class ConditionalFlowMatchingInference(BaseFlowMatchingInference):
    """
    Inference class for conditional flow matching models
    
    Provides methods for loading models and generating endpoint predictions
    from noise conditioned on start states.
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize inference with trained conditional flow matching model
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to load model on ('cuda', 'cpu', or None for auto)
        """
        from ..base.config import FlowMatchingConfig
        
        config = FlowMatchingConfig()
        super().__init__(checkpoint_path, config)
    
    def _load_model(self) -> ConditionalFlowMatcher:
        """Load conditional flow matching model from checkpoint"""
        import functools
        
        # Load with weights_only=False since we trust this checkpoint
        try:
            model = ConditionalFlowMatcher.load_from_checkpoint(
                self.checkpoint_path, 
                map_location=self.device,
                # Pass empty model to avoid __init__ issues during loading
            )
            model.eval()
            return model
            
        except Exception as e:
            print(f"Standard loading failed: {e}")
            print("Trying manual loading approach...")
            
            # Manual approach: Load checkpoint and reconstruct model
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Get the config from checkpoint
            if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
                config = checkpoint['hyper_parameters']['config']
            else:
                # Use default config
                from ..base.config import FlowMatchingConfig
                config = FlowMatchingConfig()
            
            # Create the model architecture first
            from ..conditional.flow_matcher import ConditionalFlowMatcher
            from ...model.conditional_unet1d import ConditionalUNet1D
            
            # Initialize the UNet model
            unet_model = ConditionalUNet1D(
                input_dim=3,
                condition_dim=3,
                output_dim=3,
                hidden_dims=[64, 128, 256],
                time_emb_dim=128
            )
            
            # Initialize the flow matcher
            model = ConditionalFlowMatcher(
                model=unet_model,
                optimizer=None,
                scheduler=None,
                config=config
            )
            
            # Load the state dict
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            return model
    
    def _get_model_input_dim(self) -> int:
        """Get model input dimension for conditional flow matching"""
        return 3  # Embedded dimension (sin θ, cos θ, θ̇)
    
    def _prepare_state_for_integration(self, state: torch.Tensor) -> torch.Tensor:
        """Prepare state for ODE integration (already in embedded form)"""
        return state  # State is already embedded
    
    def _extract_state_from_integration(self, integrated_state: torch.Tensor) -> torch.Tensor:
        """Extract and denormalize final state from integration result (vectorized)"""
        # integrated_state can be [..., 3] with [sin θ, cos θ, θ̇_normalized]
        # Use the vectorized conversion method for consistency and efficiency
        return self._convert_embedded_to_original(integrated_state)
    
    def predict_endpoint(self, 
                        start_state: Union[torch.Tensor, np.ndarray],
                        num_steps: int = 100,
                        method: str = 'rk4',
                        return_trajectory: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict endpoint from start state using conditional flow matching
        
        Args:
            start_state: Start state as (θ, θ̇) [2] or [batch_size, 2]
            num_steps: Number of ODE integration steps
            method: Integration method ('euler' or 'rk4')
            return_trajectory: If True, return full trajectory
            
        Returns:
            If return_trajectory=False: endpoint [batch_size, 2] in original coordinates
            If return_trajectory=True: (endpoint, trajectory) where trajectory is [num_steps+1, batch_size, 2]
        """
        # Convert to tensor if needed
        if isinstance(start_state, np.ndarray):
            start_state = torch.tensor(start_state, dtype=torch.float32)
        
        # Handle single sample
        if start_state.dim() == 1:
            start_state = start_state.unsqueeze(0)  # [1, 2]
        
        start_state = start_state.to(self.device)
        batch_size = start_state.shape[0]
        
        # Vectorized embedding: Convert (θ, θ̇) to (sin θ, cos θ, θ̇_normalized)
        theta = start_state[:, 0]  # [batch_size]
        theta_dot = start_state[:, 1]  # [batch_size]
        
        # Vectorized θ̇ normalization to [-1, 1] range
        theta_dot_min, theta_dot_max = -6.28, 6.28
        theta_dot_normalized = 2 * (theta_dot - theta_dot_min) / (theta_dot_max - theta_dot_min) - 1
        
        # Vectorized embedding to S¹ × ℝ
        start_embedded = torch.stack([
            torch.sin(theta),
            torch.cos(theta), 
            theta_dot_normalized
        ], dim=-1)  # [batch_size, 3]
        
        # Generate trajectories using batch-capable methods
        self.model.eval()
        with torch.no_grad():
            if return_trajectory:
                # Batch trajectory generation
                trajectories_embedded = self.model.sample_trajectory(
                    start_embedded, num_steps, method
                )  # [num_steps+1, batch_size, 3]
                endpoints_embedded = trajectories_embedded[-1]  # [batch_size, 3]
            else:
                # Batch endpoint generation
                endpoints_embedded = self.model.generate_endpoint(
                    start_embedded, num_steps, method
                )  # [batch_size, 3]
        
        # Vectorized conversion back to original coordinates
        endpoints_original = self._convert_embedded_to_original(endpoints_embedded)
        
        if return_trajectory:
            # Vectorized trajectory conversion
            # trajectories_embedded is already [num_steps+1, batch_size, 3]
            trajectories_original = self._convert_embedded_trajectory_to_original(trajectories_embedded)
            return endpoints_original, trajectories_original
        
        return endpoints_original
    
    def _convert_embedded_to_original(self, embedded_states: torch.Tensor) -> torch.Tensor:
        """
        Vectorized conversion from embedded states to original coordinates
        
        Args:
            embedded_states: [batch_size, 3] tensor with (sin θ, cos θ, θ̇_normalized)
            
        Returns:
            original_states: [batch_size, 2] tensor with (θ, θ̇)
        """
        sin_theta = embedded_states[:, 0]  # [batch_size]
        cos_theta = embedded_states[:, 1]  # [batch_size]
        theta_dot_normalized = embedded_states[:, 2]  # [batch_size]
        
        # Vectorized angle conversion
        theta = torch.atan2(sin_theta, cos_theta)  # [batch_size]
        
        # Vectorized θ̇ denormalization
        theta_dot_min, theta_dot_max = -6.28, 6.28
        theta_dot = (theta_dot_normalized + 1) * (theta_dot_max - theta_dot_min) / 2 + theta_dot_min
        
        return torch.stack([theta, theta_dot], dim=-1)  # [batch_size, 2]
    
    def _convert_embedded_trajectory_to_original(self, embedded_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Vectorized conversion from embedded trajectory to original coordinates
        
        Args:
            embedded_trajectory: [num_steps+1, batch_size, 3] tensor with (sin θ, cos θ, θ̇_normalized)
            
        Returns:
            original_trajectory: [num_steps+1, batch_size, 2] tensor with (θ, θ̇)
        """
        num_steps_plus_1, batch_size, _ = embedded_trajectory.shape
        
        # Reshape to [num_steps_plus_1 * batch_size, 3] for vectorized processing
        embedded_flat = embedded_trajectory.view(-1, 3)
        
        # Use existing vectorized conversion method
        original_flat = self._convert_embedded_to_original(embedded_flat)  # [num_steps_plus_1 * batch_size, 2]
        
        # Reshape back to [num_steps+1, batch_size, 2]
        original_trajectory = original_flat.view(num_steps_plus_1, batch_size, 2)
        
        return original_trajectory
    
    def predict_multiple_samples(self,
                                start_state: Union[torch.Tensor, np.ndarray],
                                num_samples: int = 10,
                                num_steps: int = 100,
                                method: str = 'rk4') -> torch.Tensor:
        """
        Generate multiple endpoint samples from the same start state
        
        This demonstrates the stochastic nature of the conditional flow matching model.
        
        Args:
            start_state: Single start state as (θ, θ̇) [2]
            num_samples: Number of samples to generate
            num_steps: Number of ODE integration steps
            method: Integration method
            
        Returns:
            samples: Multiple endpoint samples [num_samples, 2]
        """
        # Convert to tensor if needed
        if isinstance(start_state, np.ndarray):
            start_state = torch.tensor(start_state, dtype=torch.float32)
        
        if start_state.dim() != 1 or start_state.shape[0] != 2:
            raise ValueError("start_state must be a single 2D state [2]")
        
        # Repeat start state for multiple samples
        start_states = start_state.unsqueeze(0).repeat(num_samples, 1)  # [num_samples, 2]
        
        # Generate samples
        samples = self.predict_endpoint(start_states, num_steps, method)
        
        return samples
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            'model_type': 'ConditionalFlowMatcher',
            'variant': 'conditional',
            'state_dim': 3,  # Embedded dimension
            'original_dim': 2,  # Original (θ, θ̇) dimension
            'device': str(self.device),
            'supports_stochastic_generation': True,
            'flow_direction': 'noise_to_endpoint'
        }
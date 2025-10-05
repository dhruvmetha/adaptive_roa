"""
Latent Conditional Flow Matching implementation
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import lightning.pytorch as pl

from ..base.flow_matcher import BaseFlowMatcher
from ...systems.base import DynamicalSystem


class LatentConditionalFlowMatcher(BaseFlowMatcher):
    """
    Latent Conditional Flow Matching:
    - Flows from noisy input to data endpoint
    - Neural net takes embedded x_t, time t, latent z, and start state condition
    - Predicts velocity in S¹ × ℝ tangent space
    """
    
    def __init__(self, 
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 config: Optional[dict] = None,
                 latent_dim: int = 2):
        """
        Initialize latent conditional flow matcher
        
        Args:
            system: DynamicalSystem (should be pendulum with S¹ × ℝ structure)
            model: LatentConditionalUNet1D model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            config: Configuration dict
            latent_dim: Dimension of latent space
        """
        self.system = system
        self.latent_dim = latent_dim
        super().__init__(model, optimizer, scheduler, config)
    
    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noisy input in S¹ × ℝ space
        
        Args:
            batch_size: Number of samples
            device: Device to create tensors on
            
        Returns:
            Noisy states [batch_size, 2] as (θ, θ̇)
        """
        # θ ~ Uniform[-π, π] 
        theta = torch.rand(batch_size, 1, device=device) * 2 * torch.pi - torch.pi
        
        # θ̇ ~ Uniform[-1, 1] (already normalized)
        theta_dot = torch.rand(batch_size, 1, device=device) * 2 - 1
        
        return torch.cat([theta, theta_dot], dim=1)
    
    def sample_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample Gaussian latent vector
        
        Args:
            batch_size: Number of samples
            device: Device to create tensors on
            
        Returns:
            Latent vectors [batch_size, latent_dim]
        """
        return torch.randn(batch_size, self.latent_dim, device=device)
    
    def interpolate_s1_x_r(self, 
                          x_noise: torch.Tensor,    # [B, 2] (θ, θ̇)
                          x_data: torch.Tensor,     # [B, 2] (θ, θ̇) 
                          t: torch.Tensor           # [B]
                          ) -> torch.Tensor:
        """
        Interpolate in S¹ × ℝ space
        - S¹ component: geodesic interpolation on circle
        - ℝ component: linear interpolation
        
        Returns:
            Interpolated states [B, 2] in S¹ × ℝ
        """
        # Handle scalar t
        if t.dim() == 0:
            t = t.expand(x_noise.shape[0])
        t = t.unsqueeze(-1)  # [B, 1]
        
        # Extract components
        theta_noise = x_noise[:, 0]  # [B]
        theta_data = x_data[:, 0]    # [B]
        
        # Geodesic interpolation on S¹ for θ
        angular_diff = theta_data - theta_noise
        angular_diff = torch.atan2(torch.sin(angular_diff), torch.cos(angular_diff))
        theta_t = theta_noise + t.squeeze(-1) * angular_diff
        
        # Linear interpolation on ℝ for θ̇
        theta_dot_t = (1 - t) * x_noise[:, 1:2] + t * x_data[:, 1:2]
        
        return torch.cat([theta_t.unsqueeze(-1), theta_dot_t], dim=1)
    
    def compute_target_velocity_s1_x_r(self,
                                      x_noise: torch.Tensor,  # [B, 2]
                                      x_data: torch.Tensor,   # [B, 2] 
                                      t: torch.Tensor         # [B]
                                      ) -> torch.Tensor:
        """
        Compute target velocity in S¹ × ℝ tangent space using Theseus
        
        Returns:
            Target velocity [B, 2] as (dθ/dt, dθ̇/dt) in tangent space
        """
        try:
            import theseus as th
            
            batch_size = x_noise.shape[0]
            device = x_noise.device
            dtype = x_noise.dtype
            
            # Extract angle components
            theta_noise = x_noise[:, 0]  # [B]
            theta_data = x_data[:, 0]    # [B]
            
            # For SO2 component: use Theseus SE2 with zero translation
            zero_translation = torch.zeros(batch_size, 2, device=device, dtype=dtype)
            
            noise_poses = torch.cat([zero_translation, theta_noise.unsqueeze(-1)], dim=-1)  # [B, 3]
            data_poses = torch.cat([zero_translation, theta_data.unsqueeze(-1)], dim=-1)     # [B, 3]
            
            noise_se2 = th.SE2(x_y_theta=noise_poses)
            data_se2 = th.SE2(x_y_theta=data_poses)
            
            # Compute relative transformation: noise^{-1} * data
            relative_se2 = noise_se2.inverse().compose(data_se2)
            
            # Get tangent vector using log map
            log_map_result = relative_se2.log_map()  # [B, 3]
            
            # Extract angular velocity (last component)
            angular_velocity = log_map_result[:, 2:3]  # [B, 1]
            
            # For ℝ component: simple difference
            theta_dot_velocity = x_data[:, 1:2] - x_noise[:, 1:2]  # [B, 1]
            
            return torch.cat([angular_velocity, theta_dot_velocity], dim=1)
            
        except Exception as e:
            print(f"Warning: Theseus computation failed ({e}), using fallback")
            
            # Fallback: manual computation
            theta_noise = x_noise[:, 0]
            theta_data = x_data[:, 0]
            
            # Angular velocity with wrap-around
            angular_diff = theta_data - theta_noise
            angular_diff = torch.atan2(torch.sin(angular_diff), torch.cos(angular_diff))
            
            # Linear velocity for θ̇
            theta_dot_velocity = x_data[:, 1:2] - x_noise[:, 1:2]
            
            return torch.cat([angular_diff.unsqueeze(-1), theta_dot_velocity], dim=1)
    
    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute latent conditional flow matching loss
        
        Args:
            batch: Dictionary containing 'start_state_original' and 'end_state_original'
            
        Returns:
            Flow matching loss
        """
        # Extract data endpoints and start states (raw S¹ × ℝ format)
        start_states = batch["start_state_original"]  # [B, 2] (θ, θ̇)
        data_endpoints = batch["end_state_original"]  # [B, 2] (θ, θ̇)
        
        batch_size = start_states.shape[0]
        device = self.device
        
        # Sample noisy inputs in S¹ × ℝ
        x_noise = self.sample_noisy_input(batch_size, device)
        
        # Sample random times
        t = torch.rand(batch_size, device=device)
        
        # Sample latent vectors
        z = self.sample_latent(batch_size, device)
        
        # Interpolate in S¹ × ℝ between noise and data
        x_t_s1_x_r = self.interpolate_s1_x_r(x_noise, data_endpoints, t)
        
        # Embed interpolated state for neural network input
        x_t_embedded = self.system.embed_state(x_t_s1_x_r)
        
        # Embed start state for conditioning  
        start_embedded = self.system.embed_state(start_states)
        
        # Predict velocity using the model
        predicted_velocity = self.forward(x_t_embedded, t, z, condition=start_embedded)
        
        # Compute target velocity in tangent space
        target_velocity = self.compute_target_velocity_s1_x_r(x_noise, data_endpoints, t)
        
        # Compute MSE loss between predicted and target velocities
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        
        return loss
    
    def forward(self, 
                x_t: torch.Tensor, 
                t: torch.Tensor, 
                z: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x_t: Embedded interpolated state [B, 3]
            t: Time parameter [B]
            z: Latent vector [B, 2]
            condition: Embedded start state [B, 3]
            
        Returns:
            Predicted velocity [B, 2] in tangent space
        """
        return self.model(x_t, t, z, condition)
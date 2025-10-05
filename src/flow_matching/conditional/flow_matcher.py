"""
Conditional flow matching implementation with noise-to-endpoint generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..base.flow_matcher import BaseFlowMatcher
from ..base.config import FlowMatchingConfig
from ..utils.geometry import geodesic_interpolation, compute_circular_velocity


class ConditionalFlowMatcher(BaseFlowMatcher):
    """
    Conditional flow matching with noise-to-endpoint generation
    
    This implementation uses gaussian noise as input and flows to endpoints
    conditioned on start states using FiLM conditioning in the UNet.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 config: Optional[FlowMatchingConfig] = None,
                 latent_dim: Optional[int] = None):
        super().__init__(model, optimizer, scheduler, config)
        
        # Latent variable configuration
        self.latent_dim = latent_dim
        self.use_latent = latent_dim is not None
        
        # Get noise parameters from config
        if config:
            self.noise_scale = getattr(config, 'noise_scale', 1.0)
            self.noise_distribution = getattr(config, 'noise_distribution', 'uniform')
            # Handle noise bounds - use custom or default
            if hasattr(config, 'noise_bounds') and config.noise_bounds is not None:
                self.noise_bounds = config.noise_bounds
            else:
                # Default bounds for embedded pendulum space (sin θ, cos θ, θ̇_normalized)
                # All dimensions now normalized to [-1, 1] for consistent scaling
                self.noise_bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        else:
            self.noise_scale = 1.0
            self.noise_distribution = 'uniform'
            self.noise_bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        
        print(f"Noise bounds: {self.noise_bounds}")
        if self.use_latent:
            print(f"Latent dimension: {self.latent_dim}")
    
    def prepare_states(self, start_states: torch.Tensor, end_states: torch.Tensor) -> tuple:
        """
        Prepare states for conditional flow matching
        
        Note: Data already comes pre-embedded from CircularEndpointDataset as (sin θ, cos θ, θ̇)
        so we don't need to embed again.
        
        Args:
            start_states: Initial embedded states [batch_size, 3] as (sin θ, cos θ, θ̇)
            end_states: Target embedded states [batch_size, 3] as (sin θ, cos θ, θ̇)
            
        Returns:
            Tuple of (start_states, end_states) unchanged [batch_size, 3]
        """
        return start_states, end_states
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model with optional latent conditioning
        
        Args:
            x: Current state in flow [batch_size, state_dim]
            t: Time [batch_size]
            condition: Start state conditioning [batch_size, condition_dim]
            latent: Optional latent variable [batch_size, latent_dim]
            
        Returns:
            Predicted velocity [batch_size, state_dim]
        """
        if latent is not None:
            # Concatenate latent with condition for enhanced conditioning
            combined_condition = torch.cat([condition, latent], dim=-1)
        else:
            combined_condition = condition
        
        return self.model(x, t, combined_condition)
    
    def sample_noise(self, batch_size: int, state_dim: int, device: torch.device) -> torch.Tensor:
        """
        Sample noise for conditional flow matching
        
        Supports both uniform and gaussian distributions. Uniform is default for better
        state space coverage and natural boundary respect.
        
        Args:
            batch_size: Number of samples
            state_dim: Dimensionality of state space (3 for embedded pendulum)
            device: Device to create tensor on
            
        Returns:
            Noise tensor [batch_size, state_dim]
        """
        if self.noise_distribution == 'uniform':
            # Sample uniform noise and scale to appropriate bounds
            noise = torch.rand(batch_size, state_dim, device=device)  # [0, 1]
            
            # Vectorized scaling to bounds
            # Convert bounds to tensors for vectorized operations
            min_bounds = torch.tensor([self.noise_bounds[2*i] for i in range(state_dim)], device=device)
            max_bounds = torch.tensor([self.noise_bounds[2*i + 1] for i in range(state_dim)], device=device)
            
            # Vectorized scaling: noise * (max - min) + min
            noise = noise * (max_bounds - min_bounds) + min_bounds
                
            return noise
            
        elif self.noise_distribution == 'gaussian':
            # Original gaussian sampling
            return torch.randn(batch_size, state_dim, device=device) * self.noise_scale
            
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_distribution}. "
                           f"Supported: 'uniform', 'gaussian'")
    
    def sample_latent(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Sample latent variable for conditional flow matching
        
        Args:
            batch_size: Number of samples
            device: Device to create tensor on
            
        Returns:
            Latent tensor [batch_size, latent_dim] or None if not using latent
        """
        if not self.use_latent:
            return None
        
        return torch.randn(batch_size, self.latent_dim, device=device)
    
    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute conditional flow matching loss using noise-to-endpoint flow
        
        Training procedure:
        1. Sample gaussian noise x_T
        2. Sample random time t ∈ [0,1]  
        3. Interpolate x_t = (1-t) * x_T + t * x_1 (endpoint)
        4. Compute target velocity u_t = x_1 - x_T (straight line in embedded space)
        5. Predict velocity v_t = model(x_t, t, condition=x_0)
        6. Loss = MSE(v_t, u_t)
        
        Args:
            batch: Dictionary containing 'start_state' and 'end_state' tensors
            
        Returns:
            Flow matching loss
        """
        # Extract states from batch
        start_states = batch["start_state"]  # [batch_size, 3] - conditioning
        end_states = batch["end_state"]      # [batch_size, 3] - target
        
        # Prepare states (already embedded)
        x0_embedded, x1_embedded = self.prepare_states(start_states, end_states)
        
        batch_size, state_dim = x1_embedded.shape
        device = self.device
        
        # Sample gaussian noise as starting point
        x_T = self.sample_noise(batch_size, state_dim, device)
        
        # Sample latent variable if using latent
        latent = self.sample_latent(batch_size, device)
        
        # Sample random times
        t = torch.rand(batch_size, device=device)
        
        # Linear interpolation from noise to endpoint
        # x_t = (1-t) * noise + t * endpoint
        t_expanded = t.unsqueeze(-1)  # [batch_size, 1] for broadcasting
        x_t = (1 - t_expanded) * x_T + t_expanded * x1_embedded
        
        # Target velocity is constant: endpoint - noise
        u_t = x1_embedded - x_T  # [batch_size, 3]
        
        # Predict velocity using the model
        # Input: current state x_t + time t + start state as condition + optional latent
        v_t = self.forward(x_t, t, condition=x0_embedded, latent=latent)
        
        # Compute MSE loss between predicted and target velocities
        loss = F.mse_loss(v_t, u_t)
        
        return loss
    
    def sample_trajectory(self, 
                         start_state: torch.Tensor, 
                         num_steps: int = 100,
                         method: str = 'euler',
                         latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample a trajectory from noise to endpoint given start state
        
        Args:
            start_state: Conditioning start state [batch_size, 3] or [3] for single sample
            num_steps: Number of integration steps
            method: Integration method ('euler' or 'rk4')
            latent: Optional latent variable [batch_size, latent_dim] or [latent_dim] for single sample
                   If None, samples random latent if using latent, or no latent if not using latent
            
        Returns:
            trajectory: Sampled trajectory [num_steps+1, batch_size, 3]
                       For single input [3], returns [num_steps+1, 1, 3]
        """
        # Handle single sample input for backward compatibility
        if start_state.dim() == 1:
            start_state = start_state.unsqueeze(0)  # [1, 3]
        elif start_state.dim() != 2:
            raise ValueError(f"start_state must be 1D or 2D tensor, got {start_state.dim()}D")
        
        device = start_state.device
        batch_size = start_state.shape[0]  # Support actual batch size
        state_dim = start_state.shape[-1]
        
        # Handle latent variable
        if latent is None:
            # Sample latent if using latent mode, otherwise None
            latent = self.sample_latent(batch_size, device)
        else:
            # Handle single latent input for backward compatibility
            if latent.dim() == 1 and batch_size > 1:
                latent = latent.unsqueeze(0).repeat(batch_size, 1)
        
        # Start from noise - sample for full batch
        x = self.sample_noise(batch_size, state_dim, device)  # [batch_size, 3]
        
        # Time steps
        dt = 1.0 / num_steps
        trajectory = [x.clone()]
        
        self.eval()
        with torch.no_grad():
            for step in range(num_steps):
                # Create time tensor matching batch size
                t = torch.full((batch_size,), step * dt, device=device)
                
                if method == 'euler':
                    # Euler integration - batched
                    v = self.forward(x, t, condition=start_state, latent=latent)
                    x = x + dt * v
                elif method == 'rk4':
                    # 4th order Runge-Kutta - batched
                    k1 = self.forward(x, t, condition=start_state, latent=latent)
                    
                    t_half = torch.full((batch_size,), step * dt + dt/2, device=device)
                    k2 = self.forward(x + dt * k1 / 2, t_half, condition=start_state, latent=latent)
                    k3 = self.forward(x + dt * k2 / 2, t_half, condition=start_state, latent=latent)
                    
                    t_full = torch.full((batch_size,), step * dt + dt, device=device)
                    k4 = self.forward(x + dt * k3, t_full, condition=start_state, latent=latent)
                    
                    x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                else:
                    raise ValueError(f"Unknown integration method: {method}")
                
                trajectory.append(x.clone())
        
        # Stack trajectory: [num_steps+1, batch_size, 3]
        return torch.stack(trajectory, dim=0)
    
    def generate_endpoint(self, 
                         start_state: torch.Tensor,
                         num_steps: int = 100,
                         method: str = 'euler',
                         latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate endpoint from noise conditioned on start state
        
        Args:
            start_state: Conditioning start state [batch_size, 3] or [3] for single sample
            num_steps: Number of integration steps
            method: Integration method
            latent: Optional latent variable [batch_size, latent_dim] or [latent_dim] for single sample
            
        Returns:
            endpoint: Generated endpoint [batch_size, 3]
                     For single input [3], returns [1, 3]
        """
        trajectory = self.sample_trajectory(start_state, num_steps, method, latent=latent)
        return trajectory[-1]  # Return final state [batch_size, 3]
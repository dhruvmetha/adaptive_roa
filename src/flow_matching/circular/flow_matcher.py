"""
Circular flow matching implementation with geodesic interpolation
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from ..base.flow_matcher import BaseFlowMatcher
from ..base.config import FlowMatchingConfig
from ..utils.state_transformations import embed_circular_state, extract_circular_state
from ..utils.geometry import geodesic_interpolation, compute_circular_velocity


class CircularFlowMatcher(BaseFlowMatcher):
    """
    Circular flow matching with geodesic interpolation on S¹ × ℝ
    
    This implementation handles the circular topology of pendulum angles
    by embedding states in (sin(θ), cos(θ), θ̇) space and using geodesic
    interpolation for proper handling of circular boundaries.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 config: Optional[FlowMatchingConfig] = None):
        super().__init__(model, optimizer, scheduler, config)
    
    def prepare_states(self, start_states: torch.Tensor, end_states: torch.Tensor) -> tuple:
        """
        Prepare states for circular flow matching by embedding in S¹ × ℝ
        
        Args:
            start_states: Initial states [batch_size, 2] as (θ, θ̇)
            end_states: Target states [batch_size, 2] as (θ, θ̇)
            
        Returns:
            Tuple of embedded (start_states, end_states) [batch_size, 3]
        """
        start_embedded = embed_circular_state(start_states)  # [batch_size, 3]
        end_embedded = embed_circular_state(end_states)      # [batch_size, 3]
        
        return start_embedded, end_embedded
    
    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute circular flow matching loss using geodesic interpolation
        
        Args:
            batch: Dictionary containing 'start_state' and 'end_state' tensors
            
        Returns:
            Flow matching loss
        """
        # Extract states from batch
        start_states = batch["start_state"]  # [batch_size, 2]
        end_states = batch["end_state"]      # [batch_size, 2]
        
        # Embed states in S¹ × ℝ space
        x0_embedded, x1_embedded = self.prepare_states(start_states, end_states)
        
        # Sample random times
        batch_size = start_states.shape[0]
        t = torch.rand(batch_size, device=self.device)
        
        # Geodesic interpolation to get x_t
        x_t = geodesic_interpolation(x0_embedded, x1_embedded, t)
        
        # Compute target velocity using circular geometry
        ut = compute_circular_velocity(x0_embedded, x1_embedded, t)
        
        # Predict velocity using the model
        # Input: current embedded state x_t + time t + start embedded state as condition
        vt = self.forward(x_t, t, condition=x0_embedded)
        
        # Compute MSE loss between predicted and target velocities
        loss = torch.mean((vt - ut) ** 2)
        
        return loss
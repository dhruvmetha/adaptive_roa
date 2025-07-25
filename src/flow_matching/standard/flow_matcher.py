"""
Standard flow matching implementation using torchcfm
"""
import torch
import torch.nn as nn
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from typing import Dict, Optional

from ..base.flow_matcher import BaseFlowMatcher
from ..base.config import FlowMatchingConfig


class StandardFlowMatcher(BaseFlowMatcher):
    """
    Standard flow matching using torchcfm ConditionalFlowMatcher
    
    This implementation uses the standard conditional flow matching approach
    for Euclidean state spaces without topology considerations.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 config: Optional[FlowMatchingConfig] = None):
        super().__init__(model, optimizer, scheduler, config)
        
        # Initialize standard flow matcher from torchcfm
        self.flow_matcher = ConditionalFlowMatcher(sigma=self.config.sigma)
    
    def prepare_states(self, start_states: torch.Tensor, end_states: torch.Tensor) -> tuple:
        """
        Prepare states for standard flow matching (no transformation needed)
        
        Args:
            start_states: Initial states [batch_size, 2] 
            end_states: Target states [batch_size, 2]
            
        Returns:
            Tuple of (start_states, end_states) unchanged
        """
        return start_states, end_states
    
    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute standard flow matching loss using torchcfm
        
        Args:
            batch: Dictionary containing 'start_state' and 'end_state' tensors
            
        Returns:
            Flow matching loss
        """
        # Extract states from batch
        start_states = batch["start_state"]  # [batch_size, 2]
        end_states = batch["end_state"]      # [batch_size, 2]
        
        # Prepare states (no-op for standard)
        x0, x1 = self.prepare_states(start_states, end_states)
        
        # Sample random times
        batch_size = start_states.shape[0]
        t = torch.rand(batch_size, device=self.device)
        
        # Get conditional flow matching samples and targets
        t_sampled, x_t, ut = self.flow_matcher.sample_location_and_conditional_flow(
            x0=x0, x1=x1, t=t
        )
        
        # Predict velocity using the model
        # Input: current state x_t + time t + start state as condition
        vt = self.forward(x_t, t_sampled, condition=x0)
        
        # Compute MSE loss between predicted and target velocities
        loss = torch.mean((vt - ut) ** 2)
        
        return loss
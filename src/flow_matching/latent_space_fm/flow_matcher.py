import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..base.flow_matcher import BaseFlowMatcher
from ..base.config import FlowMatchingConfig

class LatentCircularFlowMatcher(BaseFlowMatcher):
    """
    Latent CFM with straight bridge on S¹×ℝ data:
      - data comes embedded as x = (sinθ, cosθ, θ̇)
      - encode to z, do linear bridge in latent space
    """
    def __init__(self, 
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 config: Optional[FlowMatchingConfig] = None):
        super().__init__(model, optimizer, scheduler, config)

    def prepare_states(self, start_states: torch.Tensor, end_states: torch.Tensor) -> tuple:
        """Dataset already provides embedded states; return as-is to avoid double-embedding"""
        return start_states, end_states

    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute latent flow matching loss with straight bridge
        
        Args:
            batch: Dictionary containing 'start_state' and 'end_state' tensors
            
        Returns:
            Flow matching loss
        """
        # x0/x1 are already embedded (sinθ, cosθ, θ̇)
        x0 = batch["start_state"]
        x1 = batch["end_state"]

        # Encode to latent
        z0 = self.model.encode(x0)
        z1 = self.model.encode(x1)

        B = x0.shape[0]
        t = torch.rand(B, device=self.device)  # [B]

        # straight bridge in latent space
        z_t = (1.0 - t[:, None]) * z0 + t[:, None] * z1
        u_t = (z1 - z0)  # constant velocity target

        # predict latent velocity conditioned on start latent
        v_t = self.forward(z_t, t, condition=z0)

        return F.mse_loss(v_t, u_t)
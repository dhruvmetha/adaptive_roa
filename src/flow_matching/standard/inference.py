"""
Standard flow matching inference implementation
"""
import torch
from typing import Optional

from ..base.inference import BaseFlowMatchingInference
from ..base.config import FlowMatchingConfig
from .flow_matcher import StandardFlowMatcher
from ...model.unet1d import UNet1D


class StandardFlowMatchingInference(BaseFlowMatchingInference):
    """
    Inference for standard flow matching models
    
    Handles loading standard flow matching checkpoints and provides
    inference functionality for Euclidean state spaces.
    """
    
    def __init__(self, checkpoint_path: str, config: Optional[FlowMatchingConfig] = None):
        """
        Initialize standard flow matching inference
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config: Configuration object
        """
        super().__init__(checkpoint_path, config)
    
    def _load_model(self):
        """Load standard flow matching model from checkpoint"""
        # Create model architecture matching training setup
        model_net = UNet1D(
            input_dim=4,  # Current state (2D) + condition (2D)
            output_dim=2,  # Velocity prediction (2D)
            hidden_dims=list(self.config.hidden_dims),
            time_emb_dim=self.config.time_emb_dim
        )
        
        # Load model from checkpoint with the architecture
        model = StandardFlowMatcher.load_from_checkpoint(
            self.checkpoint_path,
            model=model_net,
            config=self.config,
            strict=False  # In case of minor mismatches
        )
        
        return model
    
    def _prepare_state_for_integration(self, state: torch.Tensor) -> torch.Tensor:
        """
        Prepare state for integration (no transformation for standard)
        
        Args:
            state: Normalized state tensor [batch_size, 2]
            
        Returns:
            Same state tensor (no transformation needed)
        """
        return state
    
    def _extract_state_from_integration(self, integrated_state: torch.Tensor) -> torch.Tensor:
        """
        Extract state from integration (no transformation for standard)
        
        Args:
            integrated_state: State after integration [batch_size, 2]
            
        Returns:
            Same state tensor (no extraction needed)
        """
        return integrated_state
    
    def _get_model_input_dim(self) -> int:
        """Get the input dimension for the standard model"""
        return 4  # Current state (2D) + condition (2D)
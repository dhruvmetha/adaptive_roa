"""
Common configuration for flow matching models
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class FlowMatchingConfig:
    """Base configuration for flow matching models"""
    
    # State space bounds
    angle_min: float = -np.pi
    angle_max: float = np.pi
    velocity_min: float = -2 * np.pi
    velocity_max: float = 2 * np.pi
    
    # Integration parameters
    num_integration_steps: int = 100
    
    # Model architecture
    hidden_dims: Tuple[int, ...] = (64, 128, 256)
    time_emb_dim: int = 128
    
    # Training parameters (legacy, not actively used)
    sigma: float = 0.0  # Noise level for flow matching (not used in current implementation)
    
    @property
    def state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state space bounds as (min_bounds, max_bounds)"""
        min_bounds = np.array([self.angle_min, self.velocity_min])
        max_bounds = np.array([self.angle_max, self.velocity_max])
        return min_bounds, max_bounds
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [0, 1] range"""
        min_bounds, max_bounds = self.state_bounds
        return (state - min_bounds) / (max_bounds - min_bounds)
    
    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        """Denormalize state from [0, 1] back to original range"""
        min_bounds, max_bounds = self.state_bounds
        return state * (max_bounds - min_bounds) + min_bounds
    

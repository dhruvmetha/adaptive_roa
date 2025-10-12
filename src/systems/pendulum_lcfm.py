"""
Pendulum system for Latent Conditional Flow Matching
"""
import torch
from src.systems.base import DynamicalSystem, ManifoldComponent
from typing import List, Dict, Tuple


class PendulumSystemLCFM(DynamicalSystem):
    """
    Pendulum system with S¹ × ℝ manifold structure for LCFM
    
    State representation: (θ, θ̇) where:
    - θ ∈ S¹ (circular angle) 
    - θ̇ ∈ ℝ (angular velocity, normalized to [-1, 1])
    """
    
    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define S¹ × ℝ manifold structure:
        - SO2 component for angle θ
        - Real component for angular velocity θ̇
        """
        return [
            ManifoldComponent("SO2", 1, "angle"),             # θ ∈ S¹
            ManifoldComponent("Real", 1, "angular_velocity")  # θ̇ ∈ ℝ (normalized)
        ]
    
    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define state bounds for normalization:
        - angle: [-π, π] (natural S¹ range)
        - angular_velocity: [-2π, 2π] (raw data range, normalization happens in data loader)
        """
        return {
            "angle": (-3.14159, 3.14159),
            "angular_velocity": (-6.28, 6.28)  # Raw data range [-2π, 2π]
        }
    
    def attractors(self) -> List[List[float]]:
        """
        Pendulum attractor positions in S¹ × ℝ space
        
        Returns:
            List of [θ, θ̇] attractor positions
        """
        return [
            [0.0, 0.0],      # Bottom equilibrium (stable)
            [2.1, 0.0],      # Top-right equilibrium  
            [-2.1, 0.0],     # Top-left equilibrium
        ]
    
    def is_in_attractor(self, state: torch.Tensor, radius: float = 0.1) -> torch.Tensor:
        """
        Check if states are within attractor basins
        
        Args:
            state: States [B, 2] as (θ, θ̇)
            radius: Attractor radius
            
        Returns:
            Boolean tensor [B] indicating attractor membership
        """
        attractors = torch.tensor(self.attractors(), device=state.device, dtype=state.dtype)
        
        # Compute distances to all attractors
        distances = torch.norm(state.unsqueeze(1) - attractors.unsqueeze(0), dim=2)  # [B, 3]
        
        # Check if any attractor is within radius
        return (distances < radius).any(dim=1)  # [B]

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        No normalization for pendulum - return state as-is

        Pendulum uses raw states directly:
        - θ ∈ [-π, π] (natural circular range)
        - θ̇ ∈ [-2π, 2π] (raw velocity range)

        Args:
            state: [B, 2] raw state (θ, θ̇)

        Returns:
            [B, 2] unchanged state
        """
        
        state[:, 0] = state[:, 0]
        state[:, 1] = torch.clamp(state[:, 1] / (self.state_bounds["angular_velocity"][1]), -1, 1)
        return state

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        No denormalization for pendulum - return state as-is

        Pendulum doesn't normalize, so denormalization is identity

        Args:
            normalized_state: [B, 2] state (θ, θ̇)

        Returns:
            [B, 2] unchanged state
        """
        normalized_state[:, 0] = normalized_state[:, 0]
        normalized_state[:, 1] = normalized_state[:, 1] * (self.state_bounds["angular_velocity"][1])
        return normalized_state

    def embed_state_for_model(self, state: torch.Tensor) -> torch.Tensor:
        """
        Embed pendulum state for model input using base class implementation

        Converts angle to sin/cos representation via ManifoldComponent.

        Args:
            state: [B, 2] (θ, θ̇)

        Returns:
            [B, 3] (sin θ, cos θ, θ̇)
        """
        return self.embed_state(state)
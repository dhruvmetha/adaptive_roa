"""
Pendulum system for Latent Conditional Flow Matching
"""
import torch
from .base import DynamicalSystem, ManifoldComponent
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
    
    def get_attractor_labels(self, state: torch.Tensor, radius: float = 0.1) -> torch.Tensor:
        """
        Get attractor labels for states
        
        Args:
            state: States [B, 2] as (θ, θ̇)
            radius: Attractor radius
            
        Returns:
            Integer labels [B]: 0,1,2 for attractors, -1 for separatrix/other
        """
        attractors = torch.tensor(self.attractors(), device=state.device, dtype=state.dtype)
        
        # Compute distances to all attractors
        distances = torch.norm(state.unsqueeze(1) - attractors.unsqueeze(0), dim=2)  # [B, 3]
        
        # Find closest attractor and check if within radius
        closest_distances, closest_indices = distances.min(dim=1)  # [B]
        
        # Assign labels: attractor index if within radius, -1 otherwise
        labels = torch.where(closest_distances < radius, closest_indices, -1)
        
        return labels
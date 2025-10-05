"""
Pendulum system implementation using universal Lie group framework
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
from .base import DynamicalSystem, ManifoldComponent


class PendulumSystem(DynamicalSystem):
    """
    Pendulum system with S¹ × ℝ state space
    
    State: [θ, θ̇] where θ ∈ [-π, π], θ̇ ∈ [-2π, 2π]
    Manifold: S¹ × ℝ (circle × real line)
    """
    
    def __init__(self, attractor_radius: float = 0.1):
        """
        Initialize pendulum system
        
        Args:
            attractor_radius: Radius for attractor detection
        """
        self.attractor_radius = attractor_radius
        super().__init__()
    
    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """Define S¹ × ℝ manifold structure"""
        return [
            ManifoldComponent("SO2", 1, "angle"),           # θ ∈ S¹
            ManifoldComponent("Real", 1, "angular_velocity") # θ̇ ∈ ℝ
        ]
    
    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Define normalization bounds"""
        return {
            "angle": (-np.pi, np.pi),
            "angular_velocity": (-2 * np.pi, 2 * np.pi)
        }
    
    def attractors(self) -> List[List[float]]:
        """Get attractor positions in state space"""
        return [
            [0.0, 0.0],      # Bottom equilibrium
            [2.1, 0.0],      # Top-right equilibrium (matches training data)
            [-2.1, 0.0],     # Top-left equilibrium (matches training data)
        ]
    
    def stable_attractors(self) -> List[List[float]]:
        """Get only stable attractor positions"""
        return [
            [0.0, 0.0],      # Bottom equilibrium (stable)
        ]
    
    def is_in_attractor(self, state: torch.Tensor, stable_only: bool = True) -> torch.Tensor:
        """
        Check if states are in attractor regions
        
        Args:
            state: State tensor [..., 2] as (θ, θ̇)
            stable_only: Only check stable attractors
            
        Returns:
            Boolean tensor [...] indicating attractor membership
        """
        attractors = self.stable_attractors() if stable_only else self.attractors()
        
        # Handle circular distance for angle component
        theta = state[..., 0]
        theta_dot = state[..., 1]
        
        in_attractor = torch.zeros(state.shape[:-1], dtype=torch.bool, device=state.device)
        
        for attractor in attractors:
            attr_theta, attr_theta_dot = attractor
            
            # Circular distance for angle
            angle_diff = theta - attr_theta
            angle_dist = torch.abs(torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)))
            
            # Euclidean distance for velocity
            vel_dist = torch.abs(theta_dot - attr_theta_dot)
            
            # Combined distance (could use proper metric)
            total_dist = torch.sqrt(angle_dist**2 + vel_dist**2)
            
            in_attractor |= (total_dist < self.attractor_radius)
            
        return in_attractor
    
    def get_attractor_classification(self, state: torch.Tensor) -> torch.Tensor:
        """
        Classify which attractor each state belongs to
        
        Args:
            state: State tensor [..., 2] as (θ, θ̇)
            
        Returns:
            Classification tensor [...] with attractor indices (-1 if none)
        """
        attractors = self.attractors()
        theta = state[..., 0]
        theta_dot = state[..., 1]
        
        classification = torch.full(state.shape[:-1], -1, dtype=torch.long, device=state.device)
        min_distances = torch.full(state.shape[:-1], float('inf'), device=state.device)
        
        for i, attractor in enumerate(attractors):
            attr_theta, attr_theta_dot = attractor
            
            # Circular distance for angle
            angle_diff = theta - attr_theta
            angle_dist = torch.abs(torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)))
            
            # Euclidean distance for velocity
            vel_dist = torch.abs(theta_dot - attr_theta_dot)
            
            # Combined distance
            total_dist = torch.sqrt(angle_dist**2 + vel_dist**2)
            
            # Update classification for closer attractors
            closer_mask = (total_dist < min_distances) & (total_dist < self.attractor_radius)
            classification[closer_mask] = i
            min_distances[closer_mask] = total_dist[closer_mask]
            
        return classification
    
    def wrap_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Wrap angle to [-π, π]"""
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    
    def __repr__(self) -> str:
        return f"PendulumSystem(S¹ × ℝ, attractors={len(self.attractors())})"


# Legacy compatibility
class Pendulum(PendulumSystem):
    """Legacy pendulum class for backward compatibility"""
    
    def __init__(self, name: str, attractor_radius: float = 0.1):
        super().__init__(attractor_radius)
        self.name = name
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Legacy normalization method"""
        data_tensor = torch.tensor(data, dtype=torch.float32)
        normalized = self.normalize_state(data_tensor)
        return normalized.numpy()
    
    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """Legacy denormalization method"""
        data_tensor = torch.tensor(data, dtype=torch.float32) 
        denormalized = self.denormalize_state(data_tensor)
        return denormalized.numpy()
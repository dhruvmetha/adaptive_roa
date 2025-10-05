"""
CartPole system implementation using universal Lie group framework
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
from .base import DynamicalSystem, ManifoldComponent


class CartPoleSystem(DynamicalSystem):
    """
    CartPole system with ℝ² × S¹ × ℝ state space
    
    State: [x, θ, ẋ, θ̇] where:
    - x: Cart position ∈ ℝ
    - θ: Pole angle ∈ [-π, π]
    - ẋ: Cart velocity ∈ ℝ
    - θ̇: Pole angular velocity ∈ ℝ
    
    Manifold: ℝ² × S¹ × ℝ
    """
    
    def __init__(self, 
                 cart_limit: float = 2.4, 
                 velocity_limit: float = 10.0,
                 angle_limit: float = np.pi,
                 angular_velocity_limit: float = 10.0):
        """
        Initialize CartPole system
        
        Args:
            cart_limit: Maximum cart position
            velocity_limit: Maximum cart velocity
            angle_limit: Maximum pole angle (typically π)
            angular_velocity_limit: Maximum pole angular velocity
        """
        self.cart_limit = cart_limit
        self.velocity_limit = velocity_limit
        self.angle_limit = angle_limit
        self.angular_velocity_limit = angular_velocity_limit
        super().__init__()
    
    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """Define ℝ² × S¹ × ℝ manifold structure"""
        return [
            ManifoldComponent("Real", 1, "cart_position"),      # x ∈ ℝ
            ManifoldComponent("Real", 1, "cart_velocity"),      # ẋ ∈ ℝ
            ManifoldComponent("SO2", 1, "pole_angle"),          # θ ∈ S¹  
            ManifoldComponent("Real", 1, "pole_angular_velocity") # θ̇ ∈ ℝ
        ]
    
    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Define normalization bounds"""
        return {
            "cart_position": (-self.cart_limit, self.cart_limit),
            "cart_velocity": (-self.velocity_limit, self.velocity_limit),
            "pole_angle": (-self.angle_limit, self.angle_limit),
            "pole_angular_velocity": (-self.angular_velocity_limit, self.angular_velocity_limit)
        }
    
    def target_states(self) -> List[List[float]]:
        """Get target states (upright pole, cart at center)"""
        return [
            [0.0, 0.0, 0.0, 0.0],    # Cart centered, pole upright, no velocities
            [0.0, 0.0, np.pi, 0.0],  # Cart centered, pole upright (wrapped), no velocities
            [0.0, 0.0, -np.pi, 0.0], # Cart centered, pole upright (wrapped), no velocities
        ]
    
    def stable_target_states(self) -> List[List[float]]:
        """Get stable target states"""
        return [
            [0.0, 0.0, 0.0, 0.0],    # Cart centered, pole upright
        ]
    
    def is_upright(self, state: torch.Tensor, angle_threshold: float = 0.2) -> torch.Tensor:
        """
        Check if pole is upright (within angle threshold of vertical)
        
        Args:
            state: State tensor [..., 4] as (x, θ, ẋ, θ̇)
            angle_threshold: Maximum angle deviation from vertical (radians)
            
        Returns:
            Boolean tensor [...] indicating upright status
        """
        pole_angle = state[..., 1]  # θ component
        
        # Distance from vertical (0 or ±π)
        dist_from_zero = torch.abs(pole_angle)
        dist_from_pi = torch.abs(torch.abs(pole_angle) - np.pi)
        
        # Minimum distance to any upright position
        min_dist = torch.min(dist_from_zero, dist_from_pi)
        
        return min_dist < angle_threshold
    
    def is_balanced(self, state: torch.Tensor, 
                   position_threshold: float = 2.0,
                   angle_threshold: float = 0.2,
                   velocity_threshold: float = 1.0) -> torch.Tensor:
        """
        Check if system is in balanced state
        
        Args:
            state: State tensor [..., 4] as (x, θ, ẋ, θ̇)
            position_threshold: Maximum cart position
            angle_threshold: Maximum pole angle deviation
            velocity_threshold: Maximum velocity magnitudes
            
        Returns:
            Boolean tensor [...] indicating balanced status
        """
        x, theta, x_dot, theta_dot = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
        
        # Check individual constraints
        position_ok = torch.abs(x) < position_threshold
        velocity_ok = torch.abs(x_dot) < velocity_threshold
        upright_ok = self.is_upright(state, angle_threshold)
        angular_velocity_ok = torch.abs(theta_dot) < velocity_threshold
        
        return position_ok & velocity_ok & upright_ok & angular_velocity_ok
    
    def get_failure_mode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Classify failure modes
        
        Args:
            state: State tensor [..., 4] as (x, θ, ẋ, θ̇)
            
        Returns:
            Failure mode tensor [...] with codes:
            0: No failure (balanced)
            1: Cart position limit exceeded  
            2: Pole fell over
            3: Both position and angle failures
        """
        x, theta, _, _ = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
        
        position_failed = torch.abs(x) >= self.cart_limit
        angle_failed = ~self.is_upright(state, angle_threshold=np.pi/4)  # More lenient for failure
        
        failure_mode = torch.zeros(state.shape[:-1], dtype=torch.long, device=state.device)
        failure_mode[position_failed & ~angle_failed] = 1  # Position only
        failure_mode[~position_failed & angle_failed] = 2  # Angle only  
        failure_mode[position_failed & angle_failed] = 3   # Both
        
        return failure_mode
    
    def wrap_pole_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Wrap pole angle to [-π, π]"""
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    
    def __repr__(self) -> str:
        return f"CartPoleSystem(ℝ² × S¹ × ℝ, limits=[{self.cart_limit}, {self.velocity_limit}, π, {self.angular_velocity_limit}])"
"""
CartPole system for Latent Conditional Flow Matching
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from .base import DynamicalSystem, ManifoldComponent
from typing import List, Dict, Tuple


class CartPoleSystemLCFM(DynamicalSystem):
    """
    CartPole system with ℝ² × S¹ × ℝ manifold structure for LCFM
    
    State representation: (x, θ, ẋ, θ̇) where:
    - x ∈ ℝ (cart position, normalized to [-1, 1])
    - θ ∈ S¹ (pole angle, circular)
    - ẋ ∈ ℝ (cart velocity, normalized to [-1, 1])
    - θ̇ ∈ ℝ (pole angular velocity, normalized to [-1, 1])
    """
    
    def __init__(self, 
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/cartpole/cartpole_data_bounds.pkl",
                 use_dynamic_bounds: bool = True):
        """
        Initialize CartPole system for LCFM
        
        Args:
            bounds_file: Path to pickle file containing actual data bounds
            use_dynamic_bounds: If True, load bounds from file; if False, use fallback defaults
        """
        if use_dynamic_bounds and Path(bounds_file).exists():
            self._load_bounds_from_file(bounds_file)
            print(f"Loaded CartPole bounds from: {bounds_file}")
        else:
            # Fallback to default bounds if file not found
            self._use_default_bounds()
            if use_dynamic_bounds:
                print(f"Warning: Bounds file not found at {bounds_file}, using defaults")
        
        super().__init__()
    
    def _load_bounds_from_file(self, bounds_file: str):
        """Load actual data bounds from pickle file"""
        with open(bounds_file, 'rb') as f:
            bounds_data = pickle.load(f)
        
        bounds = bounds_data['bounds']
        self.cart_limit = max(abs(bounds['x']['min']), abs(bounds['x']['max']))
        self.velocity_limit = max(abs(bounds['x_dot']['min']), abs(bounds['x_dot']['max']))
        # For angle, we'll wrap to [-π, π] in preprocessing, so use π as limit
        self.angle_limit = np.pi  # Always [-π, π] after wrapping  
        self.angular_velocity_limit = max(abs(bounds['theta_dot']['min']), abs(bounds['theta_dot']['max']))
        
        # Store the actual bounds for reference
        self.actual_bounds = bounds_data
        
        print(f"  Cart position: [{bounds['x']['min']:.3f}, {bounds['x']['max']:.3f}] -> limit: ±{self.cart_limit:.3f}")
        print(f"  Cart velocity: [{bounds['x_dot']['min']:.3f}, {bounds['x_dot']['max']:.3f}] -> limit: ±{self.velocity_limit:.3f}")
        print(f"  Pole angle: [{bounds['theta']['min']:.3f}, {bounds['theta']['max']:.3f}] -> WRAPPED to ±π")
        print(f"  Angular velocity: [{bounds['theta_dot']['min']:.3f}, {bounds['theta_dot']['max']:.3f}] -> limit: ±{self.angular_velocity_limit:.3f}")
    
    def _use_default_bounds(self):
        """Use default fallback bounds"""
        self.cart_limit = 2.4
        self.velocity_limit = 10.0
        self.angle_limit = np.pi
        self.angular_velocity_limit = 10.0
        self.actual_bounds = None
        print("Using default CartPole bounds (fallback mode)")
    
    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define ℝ² × S¹ × ℝ manifold structure:
        - Real components for cart position and velocity
        - SO2 component for pole angle θ
        - Real component for pole angular velocity θ̇
        """
        return [
            ManifoldComponent("Real", 1, "cart_position"),        # x ∈ ℝ
            ManifoldComponent("SO2", 1, "pole_angle"),            # θ ∈ S¹
            ManifoldComponent("Real", 1, "cart_velocity"),        # ẋ ∈ ℝ
            ManifoldComponent("Real", 1, "pole_angular_velocity") # θ̇ ∈ ℝ
        ]
    
    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define state bounds for normalization using actual data bounds
        """
        return {
            "cart_position": (-self.cart_limit, self.cart_limit),
            "cart_velocity": (-self.velocity_limit, self.velocity_limit),
            "pole_angle": (-self.angle_limit, self.angle_limit),  # Now uses actual data range
            "pole_angular_velocity": (-self.angular_velocity_limit, self.angular_velocity_limit)
        }
    
    def attractors(self) -> List[List[float]]:
        """
        CartPole attractor positions in ℝ² × S¹ × ℝ space
        Target state: cart centered, pole upright at 0°, no velocities
        
        Returns:
            List of [x, θ, ẋ, θ̇] attractor positions
        """
        return [
            [0.0, 0.0, 0.0, 0.0],      # Cart centered, pole upright (0° only)
        ]
    
    def is_in_attractor(self, state, radius: float = 1.0):
        """
        Check if states are within attractor basins (balanced CartPole)
        
        Args:
            state: States [B, 4] as (x, θ, ẋ, θ̇) - numpy array or torch tensor
            radius: Attractor radius for position/velocity tolerances
            
        Returns:
            Boolean tensor [B] indicating attractor membership
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x, theta, x_dot, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        
        # Position and velocity constraints (using radius = 0.1 consistently)
        position_ok = torch.abs(x) < radius  
        velocity_ok = torch.abs(x_dot) < radius  
        angular_velocity_ok = torch.abs(theta_dot) < radius
        
        # Angular constraint: check if close to upright (0° ONLY)
        # Distance from 0° (upright position)
        dist_from_zero = torch.abs(theta)
        angle_ok = dist_from_zero < radius
        
        result = position_ok & velocity_ok & angle_ok & angular_velocity_ok
        # Convert back to numpy if input was numpy
        if len(result) == 1:
            return result.item()
        return result
    
    def get_attractor_labels(self, state, radius: float = 0.1):
        """
        Get attractor labels for states
        
        Args:
            state: States [B, 4] as (x, θ, ẋ, θ̇) - numpy array or torch tensor
            radius: Attractor radius
            
        Returns:
            Integer labels [B]: 0 for upright attractor, -1 for separatrix/other
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Check if in any attractor basin
        in_attractor = self.is_in_attractor(state, radius)
        
        # For CartPole, all attractors are essentially the same (upright pole)
        # So we assign label 0 to all successful states, -1 to failures
        labels = torch.where(in_attractor, 0, -1)
        
        return labels
    
    def is_balanced(self, state: torch.Tensor, 
                   position_threshold: float = 2.0,
                   angle_threshold: float = 0.2,
                   velocity_threshold: float = 1.0) -> torch.Tensor:
        """
        Check if CartPole is in balanced state (more lenient than attractors)
        
        Args:
            state: State tensor [B, 4] as (x, θ, ẋ, θ̇)
            position_threshold: Maximum cart position
            angle_threshold: Maximum pole angle deviation from vertical
            velocity_threshold: Maximum velocity magnitudes
            
        Returns:
            Boolean tensor [B] indicating balanced status
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x, theta, x_dot, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        
        # Check individual constraints
        position_ok = torch.abs(x) < position_threshold
        velocity_ok = torch.abs(x_dot) < velocity_threshold
        angular_velocity_ok = torch.abs(theta_dot) < velocity_threshold
        
        
        # Check if pole is upright (within angle threshold of vertical)
        dist_from_zero = torch.abs(theta)
        dist_from_pi = torch.abs(torch.abs(theta) - np.pi)
        min_dist = torch.min(dist_from_zero, dist_from_pi)
        upright_ok = min_dist < angle_threshold
        
        return position_ok & velocity_ok & upright_ok & angular_velocity_ok
    
    def __repr__(self) -> str:
        return f"CartPoleSystemLCFM(ℝ² × S¹ × ℝ, limits=[{self.cart_limit}, {self.velocity_limit}, π, {self.angular_velocity_limit}])"
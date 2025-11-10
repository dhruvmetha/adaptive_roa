"""
CartPole system for Latent Conditional Flow Matching
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from src.systems.base import DynamicalSystem, ManifoldComponent
from typing import List, Dict, Tuple


class CartPoleSystem(DynamicalSystem):
    """
    CartPole system with ℝ² × S¹ × ℝ manifold structure

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
        Initialize CartPole system
        
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
        
        # Print in state vector order: [x, θ, ẋ, θ̇]
        print(f"  [0] Cart position (x): [{bounds['x']['min']:.3f}, {bounds['x']['max']:.3f}] -> limit: ±{self.cart_limit:.3f}")
        print(f"  [1] Pole angle (θ): [{bounds['theta']['min']:.3f}, {bounds['theta']['max']:.3f}] -> WRAPPED to ±π")
        print(f"  [2] Cart velocity (ẋ): [{bounds['x_dot']['min']:.3f}, {bounds['x_dot']['max']:.3f}] -> limit: ±{self.velocity_limit:.3f}")
        print(f"  [3] Angular velocity (θ̇): [{bounds['theta_dot']['min']:.3f}, {bounds['theta_dot']['max']:.3f}] -> limit: ±{self.angular_velocity_limit:.3f}")
    
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
    
    def is_in_attractor(self, state, radius: float = 0.1):
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

        # result = torch.norm(state, dim=1) < radius

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
        

    def classify_attractor(self, state: torch.Tensor, radius: float = 0.1) -> torch.Tensor:
        """
        Classify CartPole states into three categories based on termination conditions

        Three-way classification:
        1. SUCCESS (label=1): In upright balanced attractor [0,0,0,0]
        2. FAILURE (label=-1): Exceeded termination thresholds (system failed)
        3. SEPARATRIX (label=0): Between attractor and failure (uncertain region)

        Termination thresholds from PyBullet dataset:
        - |x| > 6.0 m (cart position)
        - |ẋ| > 5.0 m/s (cart velocity)
        - |θ̇| > 5.0 rad/s (angular velocity)
        - θ: no termination threshold (can flip fully)

        Args:
            state: States [B, 4] as (x, θ, ẋ, θ̇)
            radius: Attractor radius (default 0.1)

        Returns:
            Integer tensor [B] with:
                 1: State in upright balanced attractor (SUCCESS)
                -1: State exceeded termination thresholds (FAILURE)
                 0: State between attractor and failure (SEPARATRIX)
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        x, theta, x_dot, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

        # SUCCESS: Check all constraints for upright balanced attractor
        position_ok = torch.abs(x) < radius
        velocity_ok = torch.abs(x_dot) < radius
        angular_velocity_ok = torch.abs(theta_dot) < radius
        angle_ok = torch.abs(theta) < radius

        in_attractor = position_ok & velocity_ok & angle_ok & angular_velocity_ok

        # FAILURE: Check if exceeded termination thresholds
        # From dataset_description.json termination thresholds
        x_failed = torch.abs(x) > 6.0           # Cart hit boundary
        x_dot_failed = torch.abs(x_dot) > 5.0  # Cart velocity too high
        theta_dot_failed = torch.abs(theta_dot) > 5.0  # Angular velocity too high
        # Note: theta has no termination threshold (inf)

        exceeded_thresholds = x_failed | x_dot_failed | theta_dot_failed

        # Three-way classification:
        # - If in attractor → SUCCESS (1)
        # - Else if exceeded thresholds → FAILURE (-1)
        # - Else → SEPARATRIX (0) - between attractor and failure
        labels = torch.zeros_like(in_attractor, dtype=torch.long)  # Initialize as separatrix (0)
        labels[in_attractor] = 1                                   # Mark successes
        labels[exceeded_thresholds] = -1                          # Mark failures

        return labels

    # ===================================================================
    # NORMALIZATION & EMBEDDING FOR FLOW MATCHING
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw state coordinates (x, theta, x_dot, theta_dot) → (x_norm, theta, x_dot_norm, theta_dot_norm)

        Normalizes linear quantities to [-1, 1] using symmetric bounds.
        Angle remains unchanged (already in natural [-π, π] range).

        Args:
            state: [B, 4] raw cartpole state (theta already wrapped to [-π, π])

        Returns:
            [B, 4] normalized state (theta unchanged)
        """
        # Extract components
        x = state[:, 0]
        theta = state[:, 1]  # Keep as-is (already wrapped)
        x_dot = state[:, 2]
        theta_dot = state[:, 3]
        
        
        # print(self.cart_limit, self.velocity_limit, self.angular_velocity_limit)

        # Normalize linear quantities to [-1, 1] range using symmetric bounds
        x_norm = x / self.cart_limit
        x_dot_norm = x_dot / self.velocity_limit
        theta_dot_norm = theta_dot / self.angular_velocity_limit
        
        return torch.stack([x_norm, theta, x_dot_norm, theta_dot_norm], dim=1)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates.

        Args:
            normalized_state: [B, 4] normalized state (x_norm, theta, x_dot_norm, theta_dot_norm)

        Returns:
            [B, 4] raw state (x, theta, x_dot, theta_dot)
        """
        x_norm = normalized_state[:, 0]
        theta = normalized_state[:, 1]  # Already in natural coordinates [-π, π]
        x_dot_norm = normalized_state[:, 2]
        theta_dot_norm = normalized_state[:, 3]

        # Denormalize using system bounds
        x = x_norm * self.cart_limit
        x_dot = x_dot_norm * self.velocity_limit
        theta_dot = theta_dot_norm * self.angular_velocity_limit

        return torch.stack([x, theta, x_dot, theta_dot], dim=1)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Embed normalized state → (x_norm, sin(theta), cos(theta), x_dot_norm, theta_dot_norm)

        Converts circular angle to sin/cos representation for neural network input.

        Args:
            normalized_state: [B, 4] normalized state

        Returns:
            [B, 5] embedded state
        """
        x_norm = normalized_state[:, 0]
        theta = normalized_state[:, 1]
        x_dot_norm = normalized_state[:, 2]
        theta_dot_norm = normalized_state[:, 3]

        # Embed circular angle as sin/cos
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        return torch.stack([x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm], dim=1)

    def __repr__(self) -> str:
        return f"CartPoleSystemLCFM(ℝ² × S¹ × ℝ, limits=[{self.cart_limit}, {self.velocity_limit}, π, {self.angular_velocity_limit}])"
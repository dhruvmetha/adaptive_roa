"""
Centralized configuration for pendulum system parameters
"""
import numpy as np
from typing import List, Tuple, Dict, Any

class PendulumConfig:
    """Configuration class for pendulum system parameters"""
    
    # State space bounds
    ANGLE_MIN = -np.pi
    ANGLE_MAX = np.pi
    VELOCITY_MIN = -2 * np.pi
    VELOCITY_MAX = 2 * np.pi
    
    # Attractor positions [angle, velocity]
    ATTRACTORS = np.array([
        [0.0, 0.0],    # Center attractor
        [2.1, 0.0],    # Right attractor  
        [-2.1, 0.0]    # Left attractor
    ])
    
    # Attractor names for visualization
    ATTRACTOR_NAMES = [
        "Center (0,0)",
        "Right (2.1,0)", 
        "Left (-2.1,0)"
    ]
    
    # Attractor colors for visualization
    ATTRACTOR_COLORS = [
        'lightcoral',
        'lightblue', 
        'lightgreen'
    ]
    
    # Detection parameters
    ATTRACTOR_RADIUS = 0.1
    
    @classmethod
    def get_state_bounds(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Get state space bounds as (min_bounds, max_bounds)"""
        min_bounds = np.array([cls.ANGLE_MIN, cls.VELOCITY_MIN])
        max_bounds = np.array([cls.ANGLE_MAX, cls.VELOCITY_MAX])
        return min_bounds, max_bounds
    
    @classmethod
    def normalize_state(cls, state: np.ndarray) -> np.ndarray:
        """Normalize state to [0, 1] range"""
        min_bounds, max_bounds = cls.get_state_bounds()
        return (state - min_bounds) / (max_bounds - min_bounds)
    
    @classmethod
    def denormalize_state(cls, state: np.ndarray) -> np.ndarray:
        """Denormalize state from [0, 1] back to original range"""
        min_bounds, max_bounds = cls.get_state_bounds()
        return state * (max_bounds - min_bounds) + min_bounds
    
    @classmethod
    def is_in_attractor(cls, states: np.ndarray, attractor_idx: int = None) -> np.ndarray:
        """
        Check if states are within attractor radius
        
        Args:
            states: Array of shape [N, 2] with [angle, velocity] pairs
            attractor_idx: If specified, check only this attractor. Otherwise check all.
            
        Returns:
            Boolean array of shape [N] if attractor_idx specified, 
            or [N, 3] for all attractors
        """
        states = np.atleast_2d(states)
        
        if attractor_idx is not None:
            attractor = cls.ATTRACTORS[attractor_idx]
            distances = np.linalg.norm(states - attractor, axis=1)
            return distances < cls.ATTRACTOR_RADIUS
        else:
            # Check all attractors
            result = np.zeros((len(states), len(cls.ATTRACTORS)), dtype=bool)
            for i, attractor in enumerate(cls.ATTRACTORS):
                distances = np.linalg.norm(states - attractor, axis=1)
                result[:, i] = distances < cls.ATTRACTOR_RADIUS
            return result
    
    @classmethod
    def get_closest_attractor(cls, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find closest attractor for each state
        
        Args:
            states: Array of shape [N, 2]
            
        Returns:
            closest_idx: Array of shape [N] with attractor indices
            distances: Array of shape [N] with distances to closest attractor
        """
        states = np.atleast_2d(states)
        
        # Calculate distances to all attractors
        all_distances = np.zeros((len(states), len(cls.ATTRACTORS)))
        for i, attractor in enumerate(cls.ATTRACTORS):
            all_distances[:, i] = np.linalg.norm(states - attractor, axis=1)
        
        # Find closest
        closest_idx = np.argmin(all_distances, axis=1)
        distances = all_distances[np.arange(len(states)), closest_idx]
        
        return closest_idx, distances
    
    @classmethod
    def create_discretized_grid(cls, resolution: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create discretized grid of state space
        
        Args:
            resolution: Grid resolution
            
        Returns:
            grid_points: Array of shape [N, 2] with all grid points
            grid_shape: Tuple (n_angles, n_velocities) for reshaping
        """
        angle_points = np.arange(cls.ANGLE_MIN, cls.ANGLE_MAX + resolution, resolution)
        velocity_points = np.arange(cls.VELOCITY_MIN, cls.VELOCITY_MAX + resolution, resolution)
        
        angle_grid, velocity_grid = np.meshgrid(angle_points, velocity_points)
        grid_points = np.column_stack([angle_grid.ravel(), velocity_grid.ravel()])
        
        return grid_points, (len(velocity_points), len(angle_points))
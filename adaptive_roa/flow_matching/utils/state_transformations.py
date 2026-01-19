"""
State transformation utilities for flow matching
"""
import torch
import numpy as np
from typing import Union


def embed_circular_state(state: torch.Tensor) -> torch.Tensor:
    """
    Convert (θ, θ̇) → (sin(θ), cos(θ), θ̇) for circular topology
    
    Args:
        state: Tensor of shape [..., 2] with (angle, angular_velocity)
        
    Returns:
        embedded: Tensor of shape [..., 3] with (sin(θ), cos(θ), θ̇)
    """
    theta = state[..., 0]
    theta_dot = state[..., 1]
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    return torch.stack([sin_theta, cos_theta, theta_dot], dim=-1)


def extract_circular_state(embedded: torch.Tensor) -> torch.Tensor:
    """
    Convert (sin(θ), cos(θ), θ̇) → (θ, θ̇) from circular embedding
    
    Args:
        embedded: Tensor of shape [..., 3] with (sin(θ), cos(θ), θ̇)
        
    Returns:
        state: Tensor of shape [..., 2] with (angle, angular_velocity)
    """
    sin_theta = embedded[..., 0]
    cos_theta = embedded[..., 1] 
    theta_dot = embedded[..., 2]
    
    theta = torch.atan2(sin_theta, cos_theta)
    
    return torch.stack([theta, theta_dot], dim=-1)


def normalize_states(states: torch.Tensor, 
                    min_bounds: Union[torch.Tensor, np.ndarray],
                    max_bounds: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Normalize states to [0, 1] range
    
    Args:
        states: State tensor
        min_bounds: Minimum bounds for each dimension
        max_bounds: Maximum bounds for each dimension
        
    Returns:
        normalized: Normalized states in [0, 1] range
    """
    if isinstance(min_bounds, np.ndarray):
        min_bounds = torch.tensor(min_bounds, device=states.device, dtype=states.dtype)
    if isinstance(max_bounds, np.ndarray):
        max_bounds = torch.tensor(max_bounds, device=states.device, dtype=states.dtype)
    
    return (states - min_bounds) / (max_bounds - min_bounds)


def denormalize_states(states: torch.Tensor,
                      min_bounds: Union[torch.Tensor, np.ndarray], 
                      max_bounds: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Denormalize states from [0, 1] back to original range
    
    Args:
        states: Normalized state tensor
        min_bounds: Minimum bounds for each dimension  
        max_bounds: Maximum bounds for each dimension
        
    Returns:
        denormalized: States in original range
    """
    if isinstance(min_bounds, np.ndarray):
        min_bounds = torch.tensor(min_bounds, device=states.device, dtype=states.dtype)
    if isinstance(max_bounds, np.ndarray):
        max_bounds = torch.tensor(max_bounds, device=states.device, dtype=states.dtype)
    
    return states * (max_bounds - min_bounds) + min_bounds
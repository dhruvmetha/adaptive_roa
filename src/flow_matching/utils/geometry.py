"""
Geometric utilities for circular and standard flow matching
"""
import torch
import numpy as np
from typing import Union


def circular_distance(theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
    """
    Compute circular distance between angles
    
    Args:
        theta1: First angle tensor
        theta2: Second angle tensor
        
    Returns:
        distance: Circular distance between angles
    """
    diff = theta1 - theta2
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def geodesic_interpolation(x0_embedded: torch.Tensor, 
                          x1_embedded: torch.Tensor, 
                          t: torch.Tensor) -> torch.Tensor:
    """
    Geodesic interpolation on S¹ × ℝ
    
    For the circular component (sin(θ), cos(θ)), we use spherical interpolation.
    For the velocity component θ̇, we use linear interpolation.
    
    Args:
        x0_embedded: Start state (sin(θ₀), cos(θ₀), θ̇₀) [batch_size, 3]
        x1_embedded: End state (sin(θ₁), cos(θ₁), θ̇₁) [batch_size, 3]
        t: Interpolation parameter [batch_size] or scalar
        
    Returns:
        interpolated: Interpolated state [batch_size, 3]
    """
    # Handle scalar t
    if t.dim() == 0:
        t = t.expand(x0_embedded.shape[0])
    
    # Extract components
    sin_theta0, cos_theta0, theta_dot0 = x0_embedded[..., 0], x0_embedded[..., 1], x0_embedded[..., 2]
    sin_theta1, cos_theta1, theta_dot1 = x1_embedded[..., 0], x1_embedded[..., 1], x1_embedded[..., 2]
    
    # Spherical interpolation for (sin(θ), cos(θ))
    # First normalize to ensure unit circle
    norm0 = torch.sqrt(sin_theta0**2 + cos_theta0**2)
    norm1 = torch.sqrt(sin_theta1**2 + cos_theta1**2)
    
    sin_theta0_norm = sin_theta0 / (norm0 + 1e-8)
    cos_theta0_norm = cos_theta0 / (norm0 + 1e-8)
    sin_theta1_norm = sin_theta1 / (norm1 + 1e-8)
    cos_theta1_norm = cos_theta1 / (norm1 + 1e-8)
    
    # Compute angle between unit vectors
    dot_product = sin_theta0_norm * sin_theta1_norm + cos_theta0_norm * cos_theta1_norm
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Numerical stability
    
    omega = torch.acos(torch.abs(dot_product))
    
    # Handle near-parallel vectors
    sin_omega = torch.sin(omega)
    near_parallel = sin_omega < 1e-6
    
    # SLERP coefficients
    where_not_parallel = ~near_parallel
    coeff0 = torch.zeros_like(t)
    coeff1 = torch.zeros_like(t)
    
    coeff0[where_not_parallel] = torch.sin((1 - t[where_not_parallel]) * omega[where_not_parallel]) / sin_omega[where_not_parallel]
    coeff1[where_not_parallel] = torch.sin(t[where_not_parallel] * omega[where_not_parallel]) / sin_omega[where_not_parallel]
    
    # Linear interpolation for nearly parallel vectors
    coeff0[near_parallel] = 1 - t[near_parallel]
    coeff1[near_parallel] = t[near_parallel]
    
    # Handle sign for shortest path
    cross_product = sin_theta0_norm * cos_theta1_norm - cos_theta0_norm * sin_theta1_norm
    sign = torch.sign(cross_product)
    coeff1 = coeff1 * sign
    
    # Interpolated circular components
    sin_theta_interp = coeff0 * sin_theta0_norm + coeff1 * sin_theta1_norm
    cos_theta_interp = coeff0 * cos_theta0_norm + coeff1 * cos_theta1_norm
    
    # Renormalize to unit circle
    norm_interp = torch.sqrt(sin_theta_interp**2 + cos_theta_interp**2)
    sin_theta_interp = sin_theta_interp / (norm_interp + 1e-8)
    cos_theta_interp = cos_theta_interp / (norm_interp + 1e-8)
    
    # Linear interpolation for angular velocity
    theta_dot_interp = (1 - t) * theta_dot0 + t * theta_dot1
    
    return torch.stack([sin_theta_interp, cos_theta_interp, theta_dot_interp], dim=-1)


def compute_circular_velocity(x0_embedded: torch.Tensor,
                             x1_embedded: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
    """
    Compute velocity for circular geodesic flow
    
    This computes the time derivative of the geodesic interpolation.
    
    Args:
        x0_embedded: Start state [batch_size, 3]
        x1_embedded: End state [batch_size, 3]  
        t: Current time [batch_size]
        
    Returns:
        velocity: Velocity on manifold [batch_size, 3]
    """
    # Use finite differences to approximate derivative
    dt = 1e-4
    
    # Current interpolation
    xt = geodesic_interpolation(x0_embedded, x1_embedded, t)
    
    # Slightly forward in time
    t_forward = torch.clamp(t + dt, 0.0, 1.0)
    xt_forward = geodesic_interpolation(x0_embedded, x1_embedded, t_forward)
    
    # Compute finite difference
    velocity = (xt_forward - xt) / dt
    
    return velocity
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
    Geodesic interpolation on S¹ × ℝ using simple linear interpolation in angle space
    
    This matches the old implementation exactly: extract angles, interpolate linearly,
    then convert back to embedded form.
    
    Args:
        x0_embedded: Start state (sin(θ₀), cos(θ₀), θ̇₀) [batch_size, 3]
        x1_embedded: End state (sin(θ₁), cos(θ₁), θ̇₁) [batch_size, 3]
        t: Interpolation parameter [batch_size] or scalar
        
    Returns:
        interpolated: Interpolated state [batch_size, 3]
    """
    # Extract angles from embedded representation
    theta0 = torch.atan2(x0_embedded[..., 0], x0_embedded[..., 1])  # [batch_size]
    theta1 = torch.atan2(x1_embedded[..., 0], x1_embedded[..., 1])  # [batch_size]
    
    # Compute shortest angular path
    angular_diff = theta1 - theta0
    # Wrap to [-π, π] for shortest path
    angular_diff = torch.atan2(torch.sin(angular_diff), torch.cos(angular_diff))
    
    # Handle scalar t
    if t.dim() == 0:
        t = t.expand(x0_embedded.shape[0])
        
    # Ensure t has correct shape for broadcasting (this was causing issues)
    # The old implementation had this but it's redundant - t should already be [batch_size]
    
    # Interpolate angle along geodesic (shortest path on circle)
    theta_t = theta0 + t * angular_diff  # [batch_size]
    
    # Linear interpolation for angular velocity (on ℝ)
    theta_dot_0 = x0_embedded[..., 2]  # [batch_size]
    theta_dot_1 = x1_embedded[..., 2]  # [batch_size]
    theta_dot_t = (1 - t) * theta_dot_0 + t * theta_dot_1  # [batch_size]
    
    # Convert interpolated state back to embedded form
    x_t = torch.stack([torch.sin(theta_t), torch.cos(theta_t), theta_dot_t], dim=-1)
    
    return x_t


def compute_circular_velocity(x0_embedded: torch.Tensor,
                             x1_embedded: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D tangent velocity for circular geodesic flow
    
    Returns velocities in intrinsic manifold coordinates (dθ/dt, dθ̇/dt)
    rather than embedded 3D space derivatives.
    
    Args:
        x0_embedded: Start state [batch_size, 3] as (sin θ₀, cos θ₀, θ̇₀)
        x1_embedded: End state [batch_size, 3] as (sin θ₁, cos θ₁, θ̇₁)
        t: Current time [batch_size]
        
    Returns:
        velocity: 2D tangent velocity (dθ/dt, dθ̇/dt) [batch_size, 2]
    """
    # Extract angles from embedded representation
    theta0 = torch.atan2(x0_embedded[..., 0], x0_embedded[..., 1])  # [batch_size]
    theta1 = torch.atan2(x1_embedded[..., 0], x1_embedded[..., 1])  # [batch_size]
    
    # Compute shortest angular path using circular_distance function
    angular_diff = theta1 - theta0
    # Wrap to [-π, π] for shortest path (same as old circular_distance)
    angular_diff = torch.atan2(torch.sin(angular_diff), torch.cos(angular_diff))
    
    # Handle scalar t
    if t.dim() == 0:
        t = t.expand(x0_embedded.shape[0])
        
    # Ensure t has correct shape for broadcasting (this was causing issues)
    # The old implementation had this but it's redundant - t should already be [batch_size]
    
    # Current interpolated angle
    theta_t = theta0 + t * angular_diff  # [batch_size]
    
    # Extract angular velocities
    theta_dot_0 = x0_embedded[..., 2]  # [batch_size]
    theta_dot_1 = x1_embedded[..., 2]  # [batch_size]
    
    # Compute 2D tangent velocity in intrinsic coordinates
    dtheta_dt = angular_diff  # dθ/dt - angular velocity along geodesic
    dtheta_dot_dt = theta_dot_1 - theta_dot_0  # dθ̇/dt - angular acceleration
    
    # Return 2D tangent velocity (dθ/dt, dθ̇/dt)
    target_velocity = torch.stack([dtheta_dt, dtheta_dot_dt], dim=-1)
    
    return target_velocity
import torch

PI = torch.pi

def wrap_to_pi(theta: torch.Tensor) -> torch.Tensor:
    """Map angles to (-π, π]"""
    out = (theta + PI) % (2*PI) - PI
    # Ensure +π maps to +π (optional, keeps symmetry)
    return out

def shortest_arc(theta1: torch.Tensor, theta0: torch.Tensor) -> torch.Tensor:
    """Compute shortest arc Δθ in (-π, π]"""
    return wrap_to_pi(theta1 - theta0)
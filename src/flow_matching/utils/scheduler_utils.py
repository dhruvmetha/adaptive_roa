"""
Scheduler utility functions for flow matching training
"""
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def create_warmup_cosine_scheduler(optimizer, warmup_epochs: int = 20, total_epochs: int = 500, 
                                  eta_min: float = 1e-6, warmup_start_factor: float = 0.1):
    """
    Create a scheduler with linear warmup followed by cosine annealing
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup phase
        total_epochs: Total training epochs
        eta_min: Minimum learning rate for cosine annealing
        warmup_start_factor: Starting factor for warmup (fraction of initial LR)
    
    Returns:
        SequentialLR scheduler
    """
    # Linear warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # Cosine annealing scheduler for remaining epochs
    cosine_epochs = total_epochs - warmup_epochs
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=eta_min
    )
    
    # Combine schedulers
    sequential_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    return sequential_scheduler
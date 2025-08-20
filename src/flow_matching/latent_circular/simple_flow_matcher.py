"""
Simple Direct Latent Circular Flow Matching
No encoder needed - just random noise + flow network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional
import lightning.pytorch as pl
from torchmetrics import MeanMetric

from src.utils.angles import shortest_arc, wrap_to_pi


class SimpleLatentCircularFlowMatcher(pl.LightningModule):
    """
    Simple Latent Circular Flow Matching - direct noise conditioning
    
    Much simpler than VAE approach:
    1. Sample random noise z ~ N(0,I) 
    2. Flow network: v_θ(x_t, t; x₀, z)
    3. Different noise → different endpoints
    4. No encoder needed!
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: Any,
                 scheduler: Any,
                 latent_dim: int = 8):
        super().__init__()
        
        self.model = model
        self.latent_dim = latent_dim
        
        # Store optimizer and scheduler (partial functions from Hydra)
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        
        # Metrics tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        # Save hyperparameters (exclude model for checkpoint size)
        self.save_hyperparameters(ignore=['model'])

    def features(self, theta: Tensor, omega: Tensor) -> Tensor:
        """Convert state to circular features [sin θ, cos θ, ω]"""
        return torch.cat([theta.sin(), theta.cos(), omega], dim=-1)

    def cond_vec(self, theta0: Tensor, omega0: Tensor, z: Tensor) -> Tensor:
        """Create conditioning vector [sin θ₀, cos θ₀, ω₀, z]"""
        return torch.cat([theta0.sin(), theta0.cos(), omega0, z], dim=-1)

    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Simple flow matching loss with random noise conditioning
        
        Args:
            batch: Dictionary containing 'start_state' and 'end_state' tensors
            
        Returns:
            Flow matching loss
        """
        # Extract raw state coordinates [θ, ω] (not the embedded ones!)
        x0 = batch["start_state_original"]  # [B, 2] - raw [θ, ω]
        y = batch["end_state_original"]     # [B, 2] - raw [θ, ω]
        
        device = x0.device
        B = x0.size(0)

        # Sample random noise for each trajectory
        z = torch.randn(B, self.latent_dim, device=device)

        theta0, omega0 = x0[:, :1], x0[:, 1:2]
        theta1, omega1 = y[:, :1], y[:, 1:2]

        # Sample t ~ U(0,1)
        t = torch.rand(B, 1, device=device)

        # Circular straight bridge (same as before)
        dtheta = shortest_arc(theta1, theta0)          # shortest arc in (-π, π]
        theta_t = wrap_to_pi(theta0 + t * dtheta)      # interpolated angle
        omega_t = (1.0 - t) * omega0 + t * omega1      # linear interpolation for ω

        # Target velocity is constant in t
        u_star = torch.cat([dtheta, (omega1 - omega0)], dim=-1)

        # Pack features for model input
        phi_xt = self.features(theta_t, omega_t)       # [B, 3]
        t_feat = t                                     # [B, 1]
        cond = self.cond_vec(theta0, omega0, z)        # [B, 3 + latent_dim]

        net_in = torch.cat([phi_xt, t_feat, cond], dim=-1)  # [B, 7 + latent_dim]

        # Predict velocity
        u_hat = self.model(net_in)                     # [B, 2]

        # Simple MSE loss - no KL term needed!
        loss = F.mse_loss(u_hat, u_star)

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        loss = self.compute_flow_loss(batch)
        
        # Update metrics
        try:
            # TorchMetrics >= 1.2 style
            self.train_loss(loss)
        except Exception:
            # Fallback for older TorchMetrics versions
            self.train_loss.update(loss)
        
        # Log metrics
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        loss = self.compute_flow_loss(batch)
        
        # Update metrics
        try:
            # TorchMetrics >= 1.2 style
            self.val_loss(loss)
        except Exception:
            # Fallback for older TorchMetrics versions
            self.val_loss.update(loss)
        
        # Log metrics
        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Optimizer should be a partial function from Hydra
        optimizer = self.optimizer_partial(params=self.parameters())
        
        # Handle scheduler if present
        if self.scheduler_partial is not None:
            scheduler = self.scheduler_partial(optimizer=optimizer)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }
        
        # Return just the optimizer if no scheduler
        return optimizer
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        self.log('train_loss_epoch', self.train_loss.compute())
        self.train_loss.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        self.log('val_loss_epoch', self.val_loss.compute())
        self.val_loss.reset()
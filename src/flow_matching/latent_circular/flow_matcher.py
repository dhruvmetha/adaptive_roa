"""
Latent Circular Flow Matching implementation with VAE-style latent variables
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional
import lightning.pytorch as pl
from torchmetrics import MeanMetric

from src.utils.angles import shortest_arc, wrap_to_pi
from src.model.latent_encoder import LatentEncoder, reparameterize


class LatentCircularFlowMatcher(pl.LightningModule):
    """
    Latent Circular Flow Matching with straight bridge on S¹×R
    
    This implementation uses a VAE-style latent variable z to produce
    distributions of endpoints y from start state x₀. The model learns
    a flow field conditioned on both x₀ and latent z.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: Any,
                 scheduler: Any,
                 latent_dim: int = 8, 
                 kl_weight: float = 1e-4, 
                 endpoint_weight: float = 0.0,
                 posterior_std_min: float = 1e-5):
        super().__init__()
        
        self.model = model
        self.encoder = LatentEncoder(in_dim=4, latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.endpoint_weight = endpoint_weight
        self.posterior_std_min = posterior_std_min
        
        # Store optimizer and scheduler (partial functions from Hydra)
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        
        # Metrics tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_fm_loss = MeanMetric()
        self.val_fm_loss = MeanMetric()
        self.train_kl_loss = MeanMetric()
        self.val_kl_loss = MeanMetric()
        
        # Save hyperparameters (exclude model for checkpoint size)
        self.save_hyperparameters(ignore=['model'])

    def features(self, theta: Tensor, omega: Tensor) -> Tensor:
        """Convert state to circular features [sin θ, cos θ, ω]"""
        return torch.cat([theta.sin(), theta.cos(), omega], dim=-1)

    def cond_vec(self, theta0: Tensor, omega0: Tensor, z: Tensor) -> Tensor:
        """Create conditioning vector [sin θ₀, cos θ₀, ω₀, z]"""
        return torch.cat([theta0.sin(), theta0.cos(), omega0, z], dim=-1)

    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute latent circular flow matching loss with VAE terms
        
        Args:
            batch: Dictionary containing 'start_state' and 'end_state' tensors
            
        Returns:
            Dictionary with loss components
        """
        # Extract raw state coordinates [θ, ω] (not the embedded ones!)
        x0 = batch["start_state_original"]  # [B, 2] - raw [θ, ω] 
        y = batch["end_state_original"]     # [B, 2] - raw [θ, ω]
        
        device = x0.device
        B = x0.size(0)

        theta0, omega0 = x0[:, :1], x0[:, 1:2]
        theta1, omega1 = y[:, :1], y[:, 1:2]

        # Sample t ~ U(0,1)
        t = torch.rand(B, 1, device=device)

        # Circular straight bridge
        dtheta = shortest_arc(theta1, theta0)          # shortest arc in (-π, π]
        theta_t = wrap_to_pi(theta0 + t * dtheta)      # interpolated angle
        omega_t = (1.0 - t) * omega0 + t * omega1      # linear interpolation for ω

        # Target velocity is constant in t
        u_star = torch.cat([dtheta, (omega1 - omega0)], dim=-1)

        # Latent posterior q(z | x0, y)
        mu, logvar = self.encoder(x0, y)               # [B, latent_dim] each
        z = reparameterize(mu, logvar, self.posterior_std_min)

        # Pack features for model input
        phi_xt = self.features(theta_t, omega_t)       # [B, 3]
        t_feat = t                                     # [B, 1]
        cond = self.cond_vec(theta0, omega0, z)        # [B, 3 + latent_dim]

        net_in = torch.cat([phi_xt, t_feat, cond], dim=-1)  # [B, 7 + latent_dim]

        # Predict velocity
        u_hat = self.model(net_in)                     # [B, 2]

        # Flow matching loss
        fm_loss = F.mse_loss(u_hat, u_star)

        # KL regularization term
        kl_loss = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1))

        # Total loss
        total_loss = fm_loss + self.kl_weight * kl_loss
        
        # Optional endpoint integration loss (currently set to 0)
        if self.endpoint_weight > 0:
            # Could add trajectory integration loss here
            pass

        return {
            "loss": total_loss,
            "fm_loss": fm_loss,
            "kl_loss": kl_loss
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        losses = self.compute_flow_loss(batch)
        
        # Update metrics
        try:
            # TorchMetrics >= 1.2 style
            self.train_loss(losses["loss"])
            self.train_fm_loss(losses["fm_loss"])
            self.train_kl_loss(losses["kl_loss"])
        except Exception:
            # Fallback for older TorchMetrics versions
            self.train_loss.update(losses["loss"])
            self.train_fm_loss.update(losses["fm_loss"])
            self.train_kl_loss.update(losses["kl_loss"])
        
        # Log metrics
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_fm_loss', self.train_fm_loss, on_step=True, on_epoch=True)
        self.log('train_kl_loss', self.train_kl_loss, on_step=True, on_epoch=True)
        
        return losses["loss"]
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        losses = self.compute_flow_loss(batch)
        
        # Update metrics
        try:
            # TorchMetrics >= 1.2 style
            self.val_loss(losses["loss"])
            self.val_fm_loss(losses["fm_loss"])
            self.val_kl_loss(losses["kl_loss"])
        except Exception:
            # Fallback for older TorchMetrics versions
            self.val_loss.update(losses["loss"])
            self.val_fm_loss.update(losses["fm_loss"])
            self.val_kl_loss.update(losses["kl_loss"])
        
        # Log metrics
        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_fm_loss', self.val_fm_loss, on_step=False, on_epoch=True)
        self.log('val_kl_loss', self.val_kl_loss, on_step=False, on_epoch=True)
        
        return losses["loss"]
    
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
        self.log('train_fm_loss_epoch', self.train_fm_loss.compute())
        self.log('train_kl_loss_epoch', self.train_kl_loss.compute())
        
        self.train_loss.reset()
        self.train_fm_loss.reset()
        self.train_kl_loss.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        self.log('val_loss_epoch', self.val_loss.compute())
        self.log('val_fm_loss_epoch', self.val_fm_loss.compute())
        self.log('val_kl_loss_epoch', self.val_kl_loss.compute())
        
        self.val_loss.reset()
        self.val_fm_loss.reset()
        self.val_kl_loss.reset()
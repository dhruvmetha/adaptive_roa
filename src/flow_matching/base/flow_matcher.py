"""
Abstract base class for flow matching models
"""
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import MeanMetric
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import hydra

from .config import FlowMatchingConfig


class BaseFlowMatcher(pl.LightningModule, ABC):
    """
    Abstract base class for flow matching Lightning modules
    
    Provides common functionality for training, validation, and optimization
    while allowing subclasses to implement variant-specific flow dynamics.
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: Any,
                 scheduler: Any,
                 model_config: Optional[FlowMatchingConfig] = None):
        super().__init__()

        # Store model and configuration
        self.model = model
        self.config = model_config or FlowMatchingConfig()

        # Store optimizer and scheduler configs (will be instantiated in configure_optimizers)
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler
        
        # Metrics tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        # Save hyperparameters (exclude model and optimizer/scheduler to avoid pickle issues)
        self.save_hyperparameters(ignore=['model', 'optimizer', 'scheduler'])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Current state in flow [batch_size, state_dim]
            t: Time [batch_size]
            condition: Conditioning information [batch_size, condition_dim]
            
        Returns:
            Predicted velocity [batch_size, state_dim]
        """
        return self.model(x, t, condition)
    
    @abstractmethod
    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the flow matching loss for a batch
        
        This method must be implemented by subclasses to define the specific
        flow matching dynamics (standard vs circular, etc.)
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Loss tensor
        """
        pass
    
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - common across all variants"""
        loss = self.compute_flow_loss(batch)
        
        # Log metrics with compatibility for different TorchMetrics versions
        try:
            # TorchMetrics >= 1.2 style
            self.train_loss(loss)
        except Exception:
            # Fallback for older TorchMetrics versions
            self.train_loss.update(loss)
        
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step - common across all variants"""
        loss = self.compute_flow_loss(batch)
        
        # Log metrics with compatibility for different TorchMetrics versions
        try:
            # TorchMetrics >= 1.2 style
            self.val_loss(loss)
        except Exception:
            # Fallback for older TorchMetrics versions
            self.val_loss.update(loss)
        
        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Instantiate optimizer from config
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())

        # Handle scheduler if present
        if self.scheduler_config is not None:
            scheduler = hydra.utils.instantiate(self.scheduler_config, optimizer=optimizer)

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
        # Log epoch metrics
        self.log('train_loss_epoch', self.train_loss.compute())
        self.train_loss.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Log epoch metrics
        self.log('val_loss_epoch', self.val_loss.compute())
        self.val_loss.reset()
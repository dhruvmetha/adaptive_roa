"""
Abstract base class for flow matching models
"""
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import MeanMetric
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

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
                 config: Optional[FlowMatchingConfig] = None):
        super().__init__()
        
        # Store model and configuration
        self.model = model
        self.config = config or FlowMatchingConfig()
        
        # Store optimizer and scheduler (partial functions from Hydra)
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        
        # Metrics tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        # Save hyperparameters (exclude model for checkpoint size)
        self.save_hyperparameters(ignore=['model'])
    
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
    
    @abstractmethod
    def prepare_states(self, start_states: torch.Tensor, end_states: torch.Tensor) -> tuple:
        """
        Prepare states for flow matching computation
        
        This allows subclasses to transform states (e.g., circular embedding)
        before flow matching computation.
        
        Args:
            start_states: Initial states [batch_size, state_dim]
            end_states: Target states [batch_size, state_dim]
            
        Returns:
            Tuple of prepared (start_states, end_states)
        """
        pass
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - common across all variants"""
        loss = self.compute_flow_loss(batch)
        
        # Log metrics
        self.train_loss(loss)
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step - common across all variants"""
        loss = self.compute_flow_loss(batch)
        
        # Log metrics
        self.val_loss(loss)
        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        optimizer = self.optimizer_partial(params=self.parameters())
        
        if self.scheduler_partial is not None:
            scheduler = self.scheduler_partial(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
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
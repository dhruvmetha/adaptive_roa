"""
PyTorch Lightning training module for Trajectory MAE.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional
import math

from .trajectory_mae import TrajectoryMAE
from .masking import TrajectoryMasker


class TrajectoryMAELightningModule(pl.LightningModule):
    """Lightning module for training Simplified Trajectory MAE."""

    def __init__(
        self,
        # Model parameters
        state_dim: int = 4,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 1001,
        use_learned_embedding: bool = True,
        decoder_hidden_dim: Optional[int] = None,
        # Masking parameters
        mask_ratio: float = 0.75,
        mask_strategy: str = "random",
        block_size: Optional[int] = None,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        # Loss parameters
        reconstruction_loss: str = "mse",  # "mse" or "smooth_l1"
    ):
        """
        Initialize training module.

        Args:
            state_dim: Dimension of state (4 for CartPole)
            embed_dim: Embedding dimension
            encoder_depth: Number of encoder layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            use_learned_embedding: Use learned projection after manifold embedding
            decoder_hidden_dim: Hidden dim for decoder MLP (None = embed_dim * mlp_ratio)
            mask_ratio: Fraction of tokens to mask
            mask_strategy: "random" or "block" masking
            block_size: Block size for block masking
            learning_rate: Peak learning rate
            weight_decay: Weight decay for AdamW
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of epochs
            reconstruction_loss: Loss function ("mse" or "smooth_l1")
        """
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = TrajectoryMAE(
            state_dim=state_dim,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_learned_embedding=use_learned_embedding,
            decoder_hidden_dim=decoder_hidden_dim,
        )

        # Masker
        self.masker = TrajectoryMasker(
            mask_ratio=mask_ratio,
            mask_strategy=mask_strategy,
            block_size=block_size,
        )

        # Loss function
        if reconstruction_loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif reconstruction_loss == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss: {reconstruction_loss}")

    def _embed_states_for_targets(self, states: torch.Tensor) -> torch.Tensor:
        """
        Convert states to manifold features for loss computation.

        Args:
            states: (batch_size, seq_len, 4) - normalized states

        Returns:
            manifold_features: (batch_size, seq_len, 5) - (x, sin(theta), cos(theta), x_dot, theta_dot)
        """
        batch_size, seq_len, _ = states.shape

        x = states[..., 0:1]
        theta = states[..., 1:2]
        x_dot = states[..., 2:3]
        theta_dot = states[..., 3:4]

        # Convert theta to radians (same as in model)
        theta_rad = theta * 2 * math.pi

        sin_theta = torch.sin(theta_rad)
        cos_theta = torch.cos(theta_rad)

        manifold_features = torch.cat([x, sin_theta, cos_theta, x_dot, theta_dot], dim=-1)
        return manifold_features

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with masking and reconstruction.

        Args:
            batch: Batch dict with keys:
                - 'states': (batch_size, max_seq_len, 4)
                - 'padding_mask': (batch_size, max_seq_len)
                - 'seq_lengths': (batch_size,)

        Returns:
            Dict with 'loss', 'predictions', 'targets', 'mask'
        """
        states = batch['states']
        padding_mask = batch['padding_mask']
        batch_size, max_seq_len, state_dim = states.shape

        # Generate mask (True for MASKED positions)
        visible_mask, _ = self.masker.generate_mask(
            batch_size, max_seq_len, states.device
        )
        # visible_mask: True for visible, False for masked
        # We need: True for masked, False for visible
        mask = ~visible_mask

        # Don't mask padded positions (keep them as "not masked")
        mask = mask & padding_mask  # Only mask real tokens

        # Forward pass: reconstruct ALL positions
        predictions = self.model(
            states=states,
            mask=mask,
            padding_mask=padding_mask,
        )  # (batch_size, seq_len, 5)

        # Get ground truth manifold features for ALL positions
        targets = self._embed_states_for_targets(states)  # (batch_size, seq_len, 5)

        # Compute loss ONLY on masked positions
        # Create loss mask: only where mask=True AND padding_mask=True
        loss_mask = mask & padding_mask  # (batch_size, seq_len)

        if loss_mask.any():
            # Extract predictions and targets for masked positions
            predictions_masked = predictions[loss_mask]  # (num_masked_total, 5)
            targets_masked = targets[loss_mask]  # (num_masked_total, 5)

            loss = self.loss_fn(predictions_masked, targets_masked)
        else:
            loss = torch.tensor(0.0, device=states.device, requires_grad=True)

        return {
            'loss': loss,
            'predictions': predictions,
            'targets': targets,
            'mask': mask,
            'num_masked': loss_mask.sum(),
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self.forward(batch)
        loss = outputs['loss']

        # Log
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/num_masked', outputs['num_masked'].float().mean(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        outputs = self.forward(batch)
        loss = outputs['loss']

        # Log
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/num_masked', outputs['num_masked'].float().mean(), on_step=False, on_epoch=True)

        # Compute per-dimension errors on masked positions
        if outputs['num_masked'] > 0:
            predictions = outputs['predictions']
            targets = outputs['targets']
            mask = outputs['mask']
            padding_mask = batch['padding_mask']

            # Loss mask
            loss_mask = mask & padding_mask

            if loss_mask.any():
                pred_masked = predictions[loss_mask]  # (N, 5)
                target_masked = targets[loss_mask]  # (N, 5)

                # Per-dimension MAE
                per_dim_mae = (pred_masked - target_masked).abs().mean(dim=0)
                self.log('val/mae_x', per_dim_mae[0], on_step=False, on_epoch=True)
                self.log('val/mae_sin_theta', per_dim_mae[1], on_step=False, on_epoch=True)
                self.log('val/mae_cos_theta', per_dim_mae[2], on_step=False, on_epoch=True)
                self.log('val/mae_x_dot', per_dim_mae[3], on_step=False, on_epoch=True)
                self.log('val/mae_theta_dot', per_dim_mae[4], on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                # Linear warmup
                return epoch / self.hparams.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.hparams.warmup_epochs) / (self.hparams.max_epochs - self.hparams.warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }

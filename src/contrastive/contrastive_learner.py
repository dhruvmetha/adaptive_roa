"""
Contrastive Learning Lightning Module

Implements triplet and InfoNCE losses for self-supervised representation learning.
Follows existing Lightning module patterns from flow matching codebase.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics import MeanMetric
from typing import Dict, Any, Optional, Literal


class ContrastiveLearner(pl.LightningModule):
    """
    Lightning module for contrastive representation learning

    Supports:
    - Triplet loss with margin
    - InfoNCE loss (NT-Xent)
    - Metrics: cosine similarity gap, nearest-neighbor accuracy, embedding norm
    - Normalization via system.normalize_state()
    """

    def __init__(self,
                 system,
                 encoder: nn.Module,
                 optimizer: Any,
                 scheduler: Any,
                 loss_type: Literal["triplet", "infonce"] = "infonce",
                 margin: float = 0.5,
                 temperature: float = 0.07):
        """
        Initialize contrastive learner

        Args:
            system: DynamicalSystem instance (for normalization)
            encoder: Encoder network (ContrastiveEncoder)
            optimizer: Optimizer configuration
            scheduler: LR scheduler configuration
            loss_type: "triplet" or "infonce" (default: "infonce")
            margin: Margin for triplet loss (default: 0.5)
            temperature: Temperature for InfoNCE loss (default: 0.07)
        """
        super().__init__()

        self.system = system
        self.encoder = encoder
        self.loss_type = loss_type
        self.margin = margin
        self.temperature = temperature

        # Store optimizer and scheduler configs
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        # Metrics tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_similarity_gap = MeanMetric()
        self.val_similarity_gap = MeanMetric()
        self.train_nn_accuracy = MeanMetric()
        self.val_nn_accuracy = MeanMetric()
        self.train_embedding_norm = MeanMetric()
        self.val_embedding_norm = MeanMetric()

        # Save hyperparameters (exclude encoder, optimizer, scheduler to avoid pickle issues)
        self.save_hyperparameters(ignore=['encoder', 'optimizer', 'scheduler', 'system'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder

        Args:
            x: Normalized state [batch_size, state_dim]

        Returns:
            Embedding [batch_size, embedding_dim]
        """
        return self.encoder(x)

    def triplet_loss(self,
                     anchor_emb: torch.Tensor,
                     positive_emb: torch.Tensor,
                     negative_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss with margin

        Loss = max(0, d(anchor, pos) - d(anchor, neg) + margin)

        Args:
            anchor_emb: Anchor embeddings [B, embedding_dim]
            positive_emb: Positive embeddings [B, embedding_dim]
            negative_emb: Negative embeddings [B, embedding_dim]

        Returns:
            Triplet loss scalar
        """
        # Compute pairwise distances (using cosine distance = 1 - cosine similarity)
        pos_dist = 1 - F.cosine_similarity(anchor_emb, positive_emb, dim=-1)
        neg_dist = 1 - F.cosine_similarity(anchor_emb, negative_emb, dim=-1)

        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()

    def infonce_loss(self,
                     anchor_emb: torch.Tensor,
                     positive_emb: torch.Tensor,
                     negative_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss (NT-Xent) - Normalized Temperature-scaled Cross Entropy

        Treats positive pairs as targets and all negatives in batch as contrastive samples.

        Args:
            anchor_emb: Anchor embeddings [B, embedding_dim]
            positive_emb: Positive embeddings [B, embedding_dim]
            negative_emb: Negative embeddings [B, embedding_dim] or [B, num_neg, embedding_dim]

        Returns:
            InfoNCE loss scalar
        """
        batch_size = anchor_emb.shape[0]

        # Compute cosine similarity between anchor and positive
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=-1) / self.temperature  # [B]

        # Compute cosine similarity between anchor and all negatives
        if negative_emb.dim() == 3:  # Multiple negatives per anchor
            # negative_emb: [B, num_neg, embedding_dim]
            # Reshape for batch computation
            anchor_expanded = anchor_emb.unsqueeze(1)  # [B, 1, embedding_dim]
            neg_sim = F.cosine_similarity(anchor_expanded, negative_emb, dim=-1) / self.temperature  # [B, num_neg]

            # Also use other anchors in batch as additional negatives
            # Compute anchor-anchor similarities (exclude self)
            anchor_anchor_sim = torch.matmul(anchor_emb, anchor_emb.t()) / self.temperature  # [B, B]
            # Mask out self-similarities
            mask = torch.eye(batch_size, dtype=torch.bool, device=anchor_emb.device)
            anchor_anchor_sim = anchor_anchor_sim.masked_fill(mask, float('-inf'))

            # Concatenate all negative similarities
            all_neg_sim = torch.cat([neg_sim, anchor_anchor_sim], dim=1)  # [B, num_neg + B]
        else:
            # Single negative per anchor: [B, embedding_dim]
            neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=-1) / self.temperature  # [B]

            # Use other anchors in batch as additional negatives
            anchor_anchor_sim = torch.matmul(anchor_emb, anchor_emb.t()) / self.temperature  # [B, B]
            mask = torch.eye(batch_size, dtype=torch.bool, device=anchor_emb.device)
            anchor_anchor_sim = anchor_anchor_sim.masked_fill(mask, float('-inf'))

            # Concatenate
            all_neg_sim = torch.cat([neg_sim.unsqueeze(1), anchor_anchor_sim], dim=1)  # [B, 1 + B]

        # InfoNCE loss: -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sim))))
        # Numerically stable version using logsumexp
        pos_sim = pos_sim.unsqueeze(1)  # [B, 1]
        logits = torch.cat([pos_sim, all_neg_sim], dim=1)  # [B, 1 + num_negatives]

        # Targets: positive is always at index 0
        targets = torch.zeros(batch_size, dtype=torch.long, device=anchor_emb.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, targets)

        return loss

    def compute_metrics(self,
                       anchor_emb: torch.Tensor,
                       positive_emb: torch.Tensor,
                       negative_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics

        Args:
            anchor_emb: Anchor embeddings [B, embedding_dim]
            positive_emb: Positive embeddings [B, embedding_dim]
            negative_emb: Negative embeddings [B, embedding_dim]

        Returns:
            Dictionary of metrics
        """
        # Cosine similarities
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=-1)
        neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=-1)

        # Similarity gap: how much closer are positives than negatives?
        similarity_gap = (pos_sim - neg_sim).mean()

        # Nearest-neighbor accuracy: is positive closer than negative?
        nn_accuracy = (pos_sim > neg_sim).float().mean()

        # Embedding norm (should be ~1.0 with L2 normalization)
        embedding_norm = anchor_emb.norm(p=2, dim=-1).mean()

        return {
            'similarity_gap': similarity_gap,
            'nn_accuracy': nn_accuracy,
            'embedding_norm': embedding_norm
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step - normalize states and compute contrastive loss

        Args:
            batch: Dictionary with 'anchor', 'positive', 'negative' keys (RAW states)

        Returns:
            Loss scalar
        """
        # Extract RAW states from batch
        anchor_raw = batch['anchor']          # [B, state_dim]
        positive_raw = batch['positive']      # [B, state_dim]
        negative_raw = batch['negative']      # [B, state_dim] or [B, num_neg, state_dim]

        # Normalize states using system (following flow matching pattern)
        anchor_norm = self.system.normalize_state(anchor_raw)
        positive_norm = self.system.normalize_state(positive_raw)

        if negative_raw.dim() == 3:  # Multiple negatives
            # Flatten, normalize, then reshape
            batch_size, num_neg, state_dim = negative_raw.shape
            negative_flat = negative_raw.view(-1, state_dim)
            negative_norm_flat = self.system.normalize_state(negative_flat)
            negative_norm = negative_norm_flat.view(batch_size, num_neg, state_dim)
        else:
            negative_norm = self.system.normalize_state(negative_raw)

        # Encode to embeddings
        anchor_emb = self.forward(anchor_norm)
        positive_emb = self.forward(positive_norm)

        if negative_norm.dim() == 3:
            # Encode multiple negatives
            batch_size, num_neg, state_dim = negative_norm.shape
            negative_flat = negative_norm.view(-1, state_dim)
            negative_emb_flat = self.forward(negative_flat)
            negative_emb = negative_emb_flat.view(batch_size, num_neg, -1)
        else:
            negative_emb = self.forward(negative_norm)

        # Compute loss
        if self.loss_type == "triplet":
            if negative_emb.dim() == 3:
                # Use first negative for triplet loss
                loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb[:, 0, :])
            else:
                loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        elif self.loss_type == "infonce":
            loss = self.infonce_loss(anchor_emb, positive_emb, negative_emb)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Compute metrics
        if negative_emb.dim() == 3:
            metrics = self.compute_metrics(anchor_emb, positive_emb, negative_emb[:, 0, :])
        else:
            metrics = self.compute_metrics(anchor_emb, positive_emb, negative_emb)

        # Update metrics
        try:
            self.train_loss(loss)
            self.train_similarity_gap(metrics['similarity_gap'])
            self.train_nn_accuracy(metrics['nn_accuracy'])
            self.train_embedding_norm(metrics['embedding_norm'])
        except Exception:
            # Fallback for older TorchMetrics versions
            self.train_loss.update(loss)
            self.train_similarity_gap.update(metrics['similarity_gap'])
            self.train_nn_accuracy.update(metrics['nn_accuracy'])
            self.train_embedding_norm.update(metrics['embedding_norm'])

        # Log metrics
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_sim_gap', self.train_similarity_gap, on_step=False, on_epoch=True)
        self.log('train_nn_acc', self.train_nn_accuracy, on_step=False, on_epoch=True)
        self.log('train_emb_norm', self.train_embedding_norm, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        # Extract RAW states
        anchor_raw = batch['anchor']
        positive_raw = batch['positive']
        negative_raw = batch['negative']

        # Normalize
        anchor_norm = self.system.normalize_state(anchor_raw)
        positive_norm = self.system.normalize_state(positive_raw)

        if negative_raw.dim() == 3:
            batch_size, num_neg, state_dim = negative_raw.shape
            negative_flat = negative_raw.view(-1, state_dim)
            negative_norm_flat = self.system.normalize_state(negative_flat)
            negative_norm = negative_norm_flat.view(batch_size, num_neg, state_dim)
        else:
            negative_norm = self.system.normalize_state(negative_raw)

        # Encode
        anchor_emb = self.forward(anchor_norm)
        positive_emb = self.forward(positive_norm)

        if negative_norm.dim() == 3:
            batch_size, num_neg, state_dim = negative_norm.shape
            negative_flat = negative_norm.view(-1, state_dim)
            negative_emb_flat = self.forward(negative_flat)
            negative_emb = negative_emb_flat.view(batch_size, num_neg, -1)
        else:
            negative_emb = self.forward(negative_norm)

        # Compute loss
        if self.loss_type == "triplet":
            if negative_emb.dim() == 3:
                loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb[:, 0, :])
            else:
                loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        else:
            loss = self.infonce_loss(anchor_emb, positive_emb, negative_emb)

        # Compute metrics
        if negative_emb.dim() == 3:
            metrics = self.compute_metrics(anchor_emb, positive_emb, negative_emb[:, 0, :])
        else:
            metrics = self.compute_metrics(anchor_emb, positive_emb, negative_emb)

        # Update metrics
        try:
            self.val_loss(loss)
            self.val_similarity_gap(metrics['similarity_gap'])
            self.val_nn_accuracy(metrics['nn_accuracy'])
            self.val_embedding_norm(metrics['embedding_norm'])
        except Exception:
            self.val_loss.update(loss)
            self.val_similarity_gap.update(metrics['similarity_gap'])
            self.val_nn_accuracy.update(metrics['nn_accuracy'])
            self.val_embedding_norm.update(metrics['embedding_norm'])

        # Log metrics
        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_sim_gap', self.val_similarity_gap, on_step=False, on_epoch=True)
        self.log('val_nn_acc', self.val_nn_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_emb_norm', self.val_embedding_norm, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler (following flow matching pattern)"""
        import hydra

        # Instantiate optimizer
        optimizer = hydra.utils.instantiate(
            self.optimizer_config,
            params=self.parameters()
        )

        # Instantiate scheduler if provided
        if self.scheduler_config is not None:
            scheduler = hydra.utils.instantiate(
                self.scheduler_config,
                optimizer=optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return optimizer

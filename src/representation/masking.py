"""
Masking utilities for trajectory masked autoencoding.

Implements random and block masking strategies for time series data.
"""

import torch
import numpy as np
from typing import Tuple, Optional


class TrajectoryMasker:
    """Handles masking strategies for trajectory data."""

    def __init__(
        self,
        mask_ratio: float = 0.75,
        mask_strategy: str = "random",
        block_size: Optional[int] = None,
    ):
        """
        Initialize trajectory masker.

        Args:
            mask_ratio: Fraction of time steps to mask (0.0 to 1.0)
            mask_strategy: "random" for random masking, "block" for contiguous block masking
            block_size: For block masking, size of contiguous blocks to mask.
                       If None, uses adaptive block size based on sequence length.
        """
        assert 0.0 <= mask_ratio <= 1.0, "mask_ratio must be in [0, 1]"
        assert mask_strategy in ["random", "block"], "mask_strategy must be 'random' or 'block'"

        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.block_size = block_size

    def generate_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mask for a batch of sequences.

        Args:
            batch_size: Number of sequences in batch
            seq_len: Length of each sequence
            device: Device to place tensors on

        Returns:
            Tuple of:
                - visible_mask: Boolean tensor (batch_size, seq_len) - True for visible tokens
                - masked_indices: Tensor (batch_size, num_masked) - Indices of masked positions
        """
        if self.mask_strategy == "random":
            return self._random_mask(batch_size, seq_len, device)
        else:  # block
            return self._block_mask(batch_size, seq_len, device)

    def _random_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random masking (BERT-style)."""
        num_masked = int(seq_len * self.mask_ratio)

        # Create mask for each sequence independently
        visible_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        masked_indices = torch.zeros(batch_size, num_masked, dtype=torch.long, device=device)

        for i in range(batch_size):
            # Randomly select positions to mask
            perm = torch.randperm(seq_len, device=device)
            mask_idx = perm[:num_masked]
            visible_mask[i, mask_idx] = False
            masked_indices[i] = mask_idx.sort()[0]  # Sort for easier indexing later

        return visible_mask, masked_indices

    def _block_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate block masking (MAE-style with contiguous blocks)."""
        num_masked = int(seq_len * self.mask_ratio)

        # Determine block size
        if self.block_size is None:
            # Adaptive: ~sqrt(seq_len) for reasonable block sizes
            block_size = max(1, int(np.sqrt(seq_len)))
        else:
            block_size = self.block_size

        visible_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        masked_indices_list = []

        for i in range(batch_size):
            masked_positions = []
            remaining = num_masked

            # Keep trying to place blocks until we've masked enough
            attempts = 0
            max_attempts = seq_len * 2  # Prevent infinite loops

            while remaining > 0 and attempts < max_attempts:
                # Random starting position
                start = np.random.randint(0, seq_len)
                # Determine block size (can be smaller near the end or if we need fewer)
                curr_block_size = min(block_size, remaining, seq_len - start)

                # Add this block
                block_indices = list(range(start, start + curr_block_size))
                # Only add positions not already masked
                new_positions = [idx for idx in block_indices if idx not in masked_positions]
                masked_positions.extend(new_positions)

                remaining = num_masked - len(masked_positions)
                attempts += 1

            # If we couldn't reach exactly num_masked, just take what we have
            masked_positions = sorted(masked_positions)[:num_masked]

            # Pad if necessary
            if len(masked_positions) < num_masked:
                # Fill remaining with random unmasked positions
                unmasked = [j for j in range(seq_len) if j not in masked_positions]
                additional = np.random.choice(unmasked, num_masked - len(masked_positions), replace=False)
                masked_positions.extend(additional)
                masked_positions = sorted(masked_positions)

            masked_tensor = torch.tensor(masked_positions, dtype=torch.long, device=device)
            visible_mask[i, masked_tensor] = False
            masked_indices_list.append(masked_tensor)

        masked_indices = torch.stack(masked_indices_list, dim=0)
        return visible_mask, masked_indices

    def apply_mask(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mask to input sequences.

        Args:
            x: Input tensor (batch_size, seq_len, feature_dim)
            visible_mask: Boolean mask (batch_size, seq_len) - True for visible

        Returns:
            Tuple of:
                - visible_x: Visible tokens only (batch_size, num_visible, feature_dim)
                - visible_indices: Indices of visible tokens (batch_size, num_visible)
        """
        batch_size, seq_len, feature_dim = x.shape

        # Extract visible tokens
        visible_x_list = []
        visible_indices_list = []

        for i in range(batch_size):
            vis_mask = visible_mask[i]
            vis_x = x[i, vis_mask]  # (num_visible, feature_dim)
            vis_idx = torch.where(vis_mask)[0]  # (num_visible,)
            visible_x_list.append(vis_x)
            visible_indices_list.append(vis_idx)

        # Pad to same length for batching
        max_visible = max(len(v) for v in visible_x_list)

        visible_x = torch.zeros(batch_size, max_visible, feature_dim, device=x.device)
        visible_indices = torch.zeros(batch_size, max_visible, dtype=torch.long, device=x.device)

        for i in range(batch_size):
            n_vis = len(visible_x_list[i])
            visible_x[i, :n_vis] = visible_x_list[i]
            visible_indices[i, :n_vis] = visible_indices_list[i]

        return visible_x, visible_indices


def get_reconstruction_targets(
    x: torch.Tensor,
    masked_indices: torch.Tensor
) -> torch.Tensor:
    """
    Extract ground truth values for masked positions.

    Args:
        x: Original input tensor (batch_size, seq_len, feature_dim)
        masked_indices: Indices of masked positions (batch_size, num_masked)

    Returns:
        targets: Ground truth for masked positions (batch_size, num_masked, feature_dim)
    """
    batch_size, num_masked = masked_indices.shape
    feature_dim = x.shape[2]

    targets = torch.zeros(batch_size, num_masked, feature_dim, device=x.device)

    for i in range(batch_size):
        for j, idx in enumerate(masked_indices[i]):
            targets[i, j] = x[i, idx]

    return targets

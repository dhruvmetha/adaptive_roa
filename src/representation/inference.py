"""
Inference module for extracting trajectory representations from trained MAE.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
import pytorch_lightning as pl

from .train_module import TrajectoryMAELightningModule


class TrajectoryMAEInference:
    """
    Inference wrapper for extracting representations from trained Trajectory MAE.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize inference module.

        Args:
            checkpoint_path: Path to trained model checkpoint (.ckpt file)
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.lightning_module = TrajectoryMAELightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device
        )
        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        self.model = self.lightning_module.model

        print(f"Model loaded successfully!")
        print(f"  Embed dim: {self.model.embed_dim}")
        print(f"  Encoder depth: {self.lightning_module.hparams.encoder_depth}")
        print(f"  Device: {self.device}")

    @torch.no_grad()
    def extract_representation(
        self,
        states: Union[np.ndarray, torch.Tensor],
        aggregate: str = "mean",
        normalize: bool = True,
        state_bounds: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Extract representation from a single trajectory or batch of trajectories.

        Args:
            states: Trajectory states (seq_len, 4) or (batch_size, seq_len, 4)
            aggregate: Aggregation method ('mean', 'max', 'last')
            normalize: Whether states are already normalized
            state_bounds: Normalization bounds if normalize=False
                         {'min': [x_min, theta_min, ...], 'max': [...]}

        Returns:
            representation: (embed_dim,) for single trajectory or (batch_size, embed_dim) for batch
        """
        # Convert to tensor if needed
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        # Add batch dimension if single trajectory
        single_trajectory = False
        if states.ndim == 2:
            states = states.unsqueeze(0)  # (1, seq_len, 4)
            single_trajectory = True

        # Normalize if needed
        if not normalize and state_bounds is not None:
            state_min = torch.tensor(state_bounds['min'], device=states.device, dtype=states.dtype)
            state_max = torch.tensor(state_bounds['max'], device=states.device, dtype=states.dtype)
            range_vals = state_max - state_min
            range_vals = torch.where(range_vals > 0, range_vals, torch.ones_like(range_vals))
            states = (states - state_min) / range_vals

        # Move to device
        states = states.to(self.device)

        # Create padding mask (all True for single trajectory)
        batch_size, seq_len, _ = states.shape
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)

        # Extract representation
        representation = self.model.get_encoder_representation(
            states=states,
            padding_mask=padding_mask,
            aggregate=aggregate
        )  # (batch_size, embed_dim)

        # Convert to numpy
        representation = representation.cpu().numpy()

        # Remove batch dimension if single trajectory
        if single_trajectory:
            representation = representation[0]

        return representation

    @torch.no_grad()
    def extract_representations_from_files(
        self,
        trajectory_files: list,
        aggregate: str = "mean",
        state_bounds: Optional[Dict[str, np.ndarray]] = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract representations from multiple trajectory files.

        Args:
            trajectory_files: List of paths to trajectory files
            aggregate: Aggregation method
            state_bounds: Normalization bounds
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            representations: (num_trajectories, embed_dim)
        """
        representations = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(trajectory_files), batch_size), desc="Extracting representations")
            except ImportError:
                iterator = range(0, len(trajectory_files), batch_size)
                print(f"Processing {len(trajectory_files)} trajectories...")
        else:
            iterator = range(0, len(trajectory_files), batch_size)

        for i in iterator:
            batch_files = trajectory_files[i:i+batch_size]

            # Load trajectories
            batch_states = []
            max_len = 0

            for file_path in batch_files:
                traj = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
                if traj.ndim == 1:
                    traj = traj.reshape(1, -1)
                batch_states.append(traj)
                max_len = max(max_len, len(traj))

            # Pad trajectories to same length
            padded_batch = np.zeros((len(batch_states), max_len, 4), dtype=np.float32)
            padding_mask = np.zeros((len(batch_states), max_len), dtype=bool)

            for j, traj in enumerate(batch_states):
                seq_len = len(traj)
                padded_batch[j, :seq_len] = traj
                padding_mask[j, :seq_len] = True

            # Normalize
            if state_bounds is not None:
                state_min = np.array(state_bounds['min'])
                state_max = np.array(state_bounds['max'])
                range_vals = state_max - state_min
                range_vals = np.where(range_vals > 0, range_vals, 1.0)
                padded_batch = (padded_batch - state_min) / range_vals

            # Convert to tensors
            states_tensor = torch.from_numpy(padded_batch).float().to(self.device)
            mask_tensor = torch.from_numpy(padding_mask).bool().to(self.device)

            # Extract representations
            batch_repr = self.model.get_encoder_representation(
                states=states_tensor,
                padding_mask=mask_tensor,
                aggregate=aggregate
            )

            representations.append(batch_repr.cpu().numpy())

        # Concatenate all representations
        representations = np.concatenate(representations, axis=0)

        return representations

    @torch.no_grad()
    def reconstruct_trajectory(
        self,
        states: Union[np.ndarray, torch.Tensor],
        mask_ratio: float = 0.75,
        normalize: bool = True,
        state_bounds: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct a masked trajectory (for visualization/debugging).

        Args:
            states: Trajectory states (seq_len, 4)
            mask_ratio: Fraction of tokens to mask
            normalize: Whether states are already normalized
            state_bounds: Normalization bounds if normalize=False

        Returns:
            Tuple of:
                - original_states: Original states (seq_len, 4)
                - masked_indices: Indices that were masked
                - reconstructed_manifold: Reconstructed manifold features (num_masked, 5)
        """
        # Convert to tensor
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        states = states.unsqueeze(0).to(self.device)  # (1, seq_len, 4)

        # Normalize if needed
        if not normalize and state_bounds is not None:
            state_min = torch.tensor(state_bounds['min'], device=states.device, dtype=states.dtype)
            state_max = torch.tensor(state_bounds['max'], device=states.device, dtype=states.dtype)
            range_vals = state_max - state_min
            range_vals = torch.where(range_vals > 0, range_vals, torch.ones_like(range_vals))
            states = (states - state_min) / range_vals

        seq_len = states.shape[1]

        # Generate mask
        visible_mask, masked_indices = self.lightning_module.masker.generate_mask(
            1, seq_len, self.device
        )

        # Get visible positions
        visible_positions = torch.where(visible_mask[0])[0].unsqueeze(0)

        # Padding mask (all True)
        padding_mask = torch.ones(1, seq_len, dtype=torch.bool, device=self.device)

        # Forward pass
        predictions = self.model(
            states=states,
            visible_mask=visible_mask,
            masked_positions=masked_indices,
            visible_positions=visible_positions,
            padding_mask=padding_mask,
        )  # (1, num_masked, 5)

        # Convert to numpy
        original = states.cpu().numpy()[0]
        masked_idx = masked_indices.cpu().numpy()[0]
        reconstructed = predictions.cpu().numpy()[0]

        return original, masked_idx, reconstructed

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'checkpoint_path': str(self.checkpoint_path),
            'embed_dim': self.model.embed_dim,
            'encoder_depth': self.lightning_module.hparams.encoder_depth,
            'decoder_depth': self.lightning_module.hparams.decoder_depth,
            'num_heads': self.lightning_module.hparams.num_heads,
            'mask_ratio': self.lightning_module.hparams.mask_ratio,
            'mask_strategy': self.lightning_module.hparams.mask_strategy,
            'device': str(self.device),
        }

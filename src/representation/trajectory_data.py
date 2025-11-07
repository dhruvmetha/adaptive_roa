"""
Data module for loading trajectories for masked autoencoding.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json


class TrajectoryDataset(Dataset):
    """Dataset for loading variable-length trajectory data."""

    def __init__(
        self,
        trajectory_dir: str,
        labels_file: Optional[str] = None,
        indices_file: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        normalize: bool = True,
        state_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize trajectory dataset.

        Args:
            trajectory_dir: Directory containing trajectory files (sequence_*.txt)
            labels_file: Optional file with labels (initial_state, success_flag)
            indices_file: Optional file with specific indices to load
            max_seq_len: Maximum sequence length (longer sequences will be truncated)
            normalize: Whether to normalize states to [0, 1]
            state_bounds: Dict with 'min' and 'max' bounds for normalization.
                         If None, will compute from data.
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.max_seq_len = max_seq_len
        self.normalize = normalize

        # Load trajectory file paths
        self.trajectory_files = sorted(self.trajectory_dir.glob("sequence_*.txt"))
        print(f"Found {len(self.trajectory_files)} trajectory files")

        # Load specific indices if provided
        if indices_file is not None:
            indices = self._load_indices(indices_file)
            self.trajectory_files = [self.trajectory_files[i] for i in indices]
            print(f"Using {len(self.trajectory_files)} trajectories from indices file")

        # Load labels if provided
        self.labels = None
        if labels_file is not None:
            self.labels = self._load_labels(labels_file)

        # Set or compute normalization bounds
        if state_bounds is not None:
            self.state_min = torch.tensor(state_bounds["min"], dtype=torch.float32)
            self.state_max = torch.tensor(state_bounds["max"], dtype=torch.float32)
        else:
            # Will compute on first pass
            self.state_min = None
            self.state_max = None

        # Lazy load trajectories (will load on first epoch)
        self.trajectories = None
        self.seq_lengths = None

    def _load_indices(self, indices_file: str) -> List[int]:
        """Load trajectory indices from file."""
        with open(indices_file, 'r') as f:
            indices = [int(line.strip()) for line in f if line.strip()]
        return indices

    def _load_labels(self, labels_file: str) -> torch.Tensor:
        """Load labels from file."""
        labels_data = np.loadtxt(labels_file, delimiter=',')
        return torch.from_numpy(labels_data[:, -1]).long()  # Last column is success flag

    def _load_all_trajectories(self):
        """Load all trajectories into memory."""
        if self.trajectories is not None:
            return  # Already loaded

        print("Loading trajectories into memory...")
        self.trajectories = []
        self.seq_lengths = []

        all_states = []

        for traj_file in self.trajectory_files:
            # Load trajectory (CSV format: x,theta,x_dot,theta_dot)
            traj = np.loadtxt(traj_file, delimiter=',', dtype=np.float32)

            # Handle single time step case
            if len(traj.shape) == 1:
                traj = traj.reshape(1, -1)

            # Truncate if needed
            if self.max_seq_len is not None and len(traj) > self.max_seq_len:
                traj = traj[:self.max_seq_len]

            self.trajectories.append(torch.from_numpy(traj))
            self.seq_lengths.append(len(traj))

            # Collect for bounds computation
            if self.state_min is None:
                all_states.append(traj)

        # Compute bounds if not provided
        if self.state_min is None:
            all_states_concat = np.concatenate(all_states, axis=0)
            self.state_min = torch.from_numpy(all_states_concat.min(axis=0)).float()
            self.state_max = torch.from_numpy(all_states_concat.max(axis=0)).float()
            print(f"Computed state bounds:")
            print(f"  Min: {self.state_min}")
            print(f"  Max: {self.state_max}")

        print(f"Loaded {len(self.trajectories)} trajectories")
        print(f"Sequence lengths: min={min(self.seq_lengths)}, max={max(self.seq_lengths)}, "
              f"mean={np.mean(self.seq_lengths):.1f}")

    def __len__(self) -> int:
        return len(self.trajectory_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trajectory.

        Returns:
            Dict with keys:
                - 'states': Trajectory states (seq_len, 4)
                - 'seq_len': Actual sequence length
                - 'label': Success label (if labels provided)
                - 'traj_idx': Trajectory index
        """
        # Lazy load on first access
        if self.trajectories is None:
            self._load_all_trajectories()

        states = self.trajectories[idx]  # (seq_len, 4)
        seq_len = self.seq_lengths[idx]

        # Normalize if requested
        if self.normalize:
            # Avoid division by zero
            range_vals = self.state_max - self.state_min
            range_vals = torch.where(range_vals > 0, range_vals, torch.ones_like(range_vals))
            states = (states - self.state_min) / range_vals

        result = {
            'states': states,
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'traj_idx': torch.tensor(idx, dtype=torch.long),
        }

        if self.labels is not None:
            result['label'] = self.labels[idx]

        return result


def collate_trajectories(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate variable-length trajectories into a batch.

    Pads trajectories to the maximum length in the batch.

    Args:
        batch: List of trajectory dicts from dataset

    Returns:
        Batched dict with:
            - 'states': (batch_size, max_seq_len, 4)
            - 'seq_lengths': (batch_size,) - actual lengths
            - 'padding_mask': (batch_size, max_seq_len) - True for real tokens
            - 'labels': (batch_size,) - if labels present
            - 'traj_indices': (batch_size,)
    """
    batch_size = len(batch)
    max_len = max(item['seq_len'].item() for item in batch)
    state_dim = batch[0]['states'].shape[1]

    # Initialize padded tensors
    states = torch.zeros(batch_size, max_len, state_dim)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    traj_indices = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item['seq_len'].item()
        states[i, :seq_len] = item['states']
        seq_lengths[i] = seq_len
        padding_mask[i, :seq_len] = True
        traj_indices[i] = item['traj_idx']

    result = {
        'states': states,
        'seq_lengths': seq_lengths,
        'padding_mask': padding_mask,
        'traj_indices': traj_indices,
    }

    # Add labels if present
    if 'label' in batch[0]:
        labels = torch.stack([item['label'] for item in batch])
        result['labels'] = labels

    return result


class TrajectoryMAEDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for trajectory MAE."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        max_seq_len: Optional[int] = None,
        normalize: bool = True,
        state_bounds: Optional[Dict[str, List[float]]] = None,
        use_labels: bool = False,
    ):
        """
        Initialize data module.

        Args:
            data_dir: Base directory containing trajectories/ and labels
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            max_seq_len: Maximum sequence length
            normalize: Whether to normalize states
            state_bounds: Optional normalization bounds {'min': [...], 'max': [...]}
            use_labels: Whether to load and include success labels
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.use_labels = use_labels

        # Convert state_bounds to dict if provided
        if state_bounds is not None:
            self.state_bounds = {
                'min': state_bounds['min'],
                'max': state_bounds['max']
            }
        else:
            self.state_bounds = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        trajectory_dir = self.data_dir / "trajectories"
        labels_file = self.data_dir / "roa_labels.txt" if self.use_labels else None

        # Get all trajectory files to determine split
        all_files = sorted(trajectory_dir.glob("sequence_*.txt"))
        n_total = len(all_files)

        # Create indices for splits
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)

        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n_total))

        print(f"Dataset splits: train={len(train_indices)}, val={len(val_indices)}, "
              f"test={len(test_indices)}")

        # Create datasets
        if stage == "fit" or stage is None:
            # Create temporary index files
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write('\n'.join(map(str, train_indices)))
                train_idx_file = f.name

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write('\n'.join(map(str, val_indices)))
                val_idx_file = f.name

            self.train_dataset = TrajectoryDataset(
                trajectory_dir=str(trajectory_dir),
                labels_file=str(labels_file) if labels_file else None,
                indices_file=train_idx_file,
                max_seq_len=self.max_seq_len,
                normalize=self.normalize,
                state_bounds=self.state_bounds,
            )

            self.val_dataset = TrajectoryDataset(
                trajectory_dir=str(trajectory_dir),
                labels_file=str(labels_file) if labels_file else None,
                indices_file=val_idx_file,
                max_seq_len=self.max_seq_len,
                normalize=self.normalize,
                state_bounds=self.state_bounds,
            )

            # Clean up temp files
            Path(train_idx_file).unlink()
            Path(val_idx_file).unlink()

        if stage == "test" or stage is None:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write('\n'.join(map(str, test_indices)))
                test_idx_file = f.name

            self.test_dataset = TrajectoryDataset(
                trajectory_dir=str(trajectory_dir),
                labels_file=str(labels_file) if labels_file else None,
                indices_file=test_idx_file,
                max_seq_len=self.max_seq_len,
                normalize=self.normalize,
                state_bounds=self.state_bounds,
            )

            Path(test_idx_file).unlink()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_trajectories,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_trajectories,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_trajectories,
            persistent_workers=True if self.num_workers > 0 else False,
        )

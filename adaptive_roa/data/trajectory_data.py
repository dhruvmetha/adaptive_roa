"""
Trajectory Data Module for trajectory-level (local) flow matching.

Loads full trajectory files from disk and returns fixed-length subsampled sequences.

Data flow:
  AdaptiveDatasetBuilder writes trajectory index files (one filename per line)
  → TrajectoryDataModule reads index files
  → TrajectoryDataset loads and subsamples each trajectory
  → Returns {trajectory: [T, D], start_state: [D], end_state: [D]}
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, List
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class TrajectoryDataset(Dataset):
    """
    Dataset returning fixed-length trajectory subsequences.

    Loads trajectory files by path, subsamples to a fixed sequence_length,
    and returns trajectory tensor along with start/end states.

    Args:
        trajectories_dir: Directory containing trajectory files
        trajectory_files: List of trajectory file paths
        sequence_length: Fixed number of timesteps to subsample each trajectory to
        wrap_angles: If True, wrap angle component (index 0) to [-pi, pi]
    """

    def __init__(
        self,
        trajectories_dir: str,
        trajectory_files: List,
        sequence_length: int = 32,
        angle_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            trajectories_dir: Directory containing trajectory files
            trajectory_files: List of trajectory file paths
            sequence_length: Fixed number of timesteps to subsample each trajectory to
            angle_indices: List of state indices that are angles to wrap to [-pi, pi].
                          None = no wrapping. E.g., [0] for pendulum, [1] for cartpole,
                          [2] for quadrotor2d, [] for quadrotor3d (quaternion, no wrapping).
        """
        self.trajectories_dir = Path(trajectories_dir)
        self.trajectory_files = trajectory_files
        self.sequence_length = sequence_length
        self.angle_indices = angle_indices

        print(f"TrajectoryDataset: {len(self.trajectory_files)} trajectories, "
              f"sequence_length={sequence_length}")

    def __len__(self):
        return len(self.trajectory_files)

    def _load_trajectory(self, filepath) -> np.ndarray:
        """Load a single trajectory file (comma-separated, one timestep per line)."""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        trajectory = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split(',')))
                trajectory.append(values)

        return np.array(trajectory, dtype=np.float32)

    def _subsample(self, trajectory: np.ndarray) -> np.ndarray:
        """Evenly subsample trajectory to fixed sequence_length."""
        T = len(trajectory)
        if T <= self.sequence_length:
            # Pad by repeating last state
            pad_length = self.sequence_length - T
            padding = np.tile(trajectory[-1:], (pad_length, 1))
            return np.concatenate([trajectory, padding], axis=0)
        else:
            # Evenly spaced indices including first and last
            indices = np.linspace(0, T - 1, self.sequence_length, dtype=int)
            return trajectory[indices]

    def _wrap_angles(self, state: np.ndarray) -> np.ndarray:
        """Wrap angle components at specified indices to [-pi, pi]."""
        if not self.angle_indices:
            return state
        result = state.copy()
        for idx in self.angle_indices:
            result[..., idx] = np.arctan2(np.sin(state[..., idx]), np.cos(state[..., idx]))
        return result

    def __getitem__(self, idx):
        filepath = self.trajectory_files[idx]
        trajectory = self._load_trajectory(str(filepath))

        # Subsample to fixed length
        trajectory = self._subsample(trajectory)

        trajectory = self._wrap_angles(trajectory)

        start_state = trajectory[0]
        end_state = trajectory[-1]

        return {
            'trajectory': torch.tensor(trajectory, dtype=torch.float32),
            'start_state': torch.tensor(start_state, dtype=torch.float32),
            'end_state': torch.tensor(end_state, dtype=torch.float32),
        }


class TrajectoryDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for trajectory-level (local) flow matching.

    Reads trajectory index files written by AdaptiveDatasetBuilder.
    Each index file has one trajectory filename per line.

    Args:
        train_trajectory_file: Path to file listing train trajectory filenames
        val_trajectory_file: Path to file listing val trajectory filenames
        trajectories_dir: Directory containing the actual trajectory files
        sequence_length: Fixed trajectory subsequence length
        batch_size: Training batch size
        val_batch_size: Validation batch size
        num_workers: DataLoader workers
    """

    def __init__(
        self,
        train_trajectory_file: str,
        val_trajectory_file: str,
        trajectories_dir: str,
        sequence_length: int = 32,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        angle_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.train_trajectory_file = train_trajectory_file
        self.val_trajectory_file = val_trajectory_file
        self.trajectories_dir = Path(trajectories_dir)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.angle_indices = angle_indices

    def _read_trajectory_files(self, index_file: str) -> List[Path]:
        """Read trajectory index file → list of full paths."""
        with open(index_file, 'r') as f:
            filenames = [line.strip() for line in f.readlines() if line.strip()]
        return [self.trajectories_dir / fn for fn in filenames]

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_files = self._read_trajectory_files(self.train_trajectory_file)
            val_files = self._read_trajectory_files(self.val_trajectory_file)

            self.train_dataset = TrajectoryDataset(
                trajectories_dir=str(self.trajectories_dir),
                trajectory_files=train_files,
                sequence_length=self.sequence_length,
                angle_indices=self.angle_indices,
            )
            self.val_dataset = TrajectoryDataset(
                trajectories_dir=str(self.trajectories_dir),
                trajectory_files=val_files,
                sequence_length=self.sequence_length,
                angle_indices=self.angle_indices,
            )

        if stage == "test" or stage is None:
            # Test uses val set
            val_files = self._read_trajectory_files(self.val_trajectory_file)
            self.test_dataset = TrajectoryDataset(
                trajectories_dir=str(self.trajectories_dir),
                trajectory_files=val_files,
                sequence_length=self.sequence_length,
                angle_indices=self.angle_indices,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

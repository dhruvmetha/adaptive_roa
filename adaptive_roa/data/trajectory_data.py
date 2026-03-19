"""
Trajectory Data Module for trajectory-level flow matching.

Loads full trajectory files from disk and returns fixed-length subsampled sequences.
Designed to integrate with the adaptive sampling pipeline's TrajectoryPool.

Each trajectory file is a comma-separated text file with one timestep per line.
The trajectory index mapping comes from shuffled_indices files (same as endpoint FM).
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

    Loads trajectory files by index, subsamples to a fixed sequence_length,
    and returns trajectory tensor along with start/end states for validation.

    Args:
        trajectories_dir: Directory containing trajectory files (e.g., 0.txt, 1.txt, ...)
        trajectory_files: List of trajectory file paths (from shuffled indices)
        sequence_length: Fixed number of timesteps to subsample each trajectory to
        wrap_angles: If True, wrap angle component (index 0) to [-pi, pi]
    """

    def __init__(
        self,
        trajectories_dir: str,
        trajectory_files: List[str],
        sequence_length: int = 32,
        wrap_angles: bool = True,
    ):
        self.trajectories_dir = Path(trajectories_dir)
        self.trajectory_files = trajectory_files
        self.sequence_length = sequence_length
        self.wrap_angles = wrap_angles

        print(f"TrajectoryDataset: {len(self.trajectory_files)} trajectories, "
              f"sequence_length={sequence_length}")

    def __len__(self):
        return len(self.trajectory_files)

    def _load_trajectory(self, filepath: str) -> np.ndarray:
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

    def _wrap_angle(self, state: np.ndarray) -> np.ndarray:
        """Wrap angle component (index 0) to [-pi, pi]."""
        result = state.copy()
        result[..., 0] = np.arctan2(np.sin(state[..., 0]), np.cos(state[..., 0]))
        return result

    def __getitem__(self, idx):
        filepath = self.trajectory_files[idx]
        trajectory = self._load_trajectory(str(filepath))

        # Subsample to fixed length
        trajectory = self._subsample(trajectory)

        if self.wrap_angles:
            trajectory = self._wrap_angle(trajectory)

        start_state = trajectory[0]
        end_state = trajectory[-1]

        return {
            'trajectory': torch.tensor(trajectory, dtype=torch.float32),
            'start_state': torch.tensor(start_state, dtype=torch.float32),
            'end_state': torch.tensor(end_state, dtype=torch.float32),
        }


class TrajectoryEndpointDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for trajectory-level flow matching.

    Reads trajectory files via shuffled_indices, splits into train/val,
    and returns batches of fixed-length trajectory sequences.

    Integration with adaptive pipeline:
        The adaptive engine calls build_all_datasets() which writes endpoint files.
        For trajectory FM, we instead read the shuffled_indices file and use the
        same index splitting logic, but load full trajectories instead of endpoints.
    """

    def __init__(
        self,
        data_file: str,
        validation_file: str,
        test_file: str,
        trajectories_dir: str,
        shuffled_indices_file: str,
        sequence_length: int = 32,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        wrap_angles: bool = True,
    ):
        """
        Args:
            data_file: Path to training endpoint file (used to extract train indices)
            validation_file: Path to validation endpoint file (used to extract val indices)
            test_file: Path to test endpoint file (used to extract test indices)
            trajectories_dir: Directory containing trajectory files
            shuffled_indices_file: File mapping indices to trajectory filenames
            sequence_length: Fixed trajectory subsequence length
            batch_size: Training batch size
            val_batch_size: Validation batch size
            num_workers: DataLoader workers
            pin_memory: Pin memory for GPU transfer
            wrap_angles: Wrap angle components to [-pi, pi]
        """
        super().__init__()
        self.data_file = data_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.trajectories_dir = trajectories_dir
        self.shuffled_indices_file = shuffled_indices_file
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.wrap_angles = wrap_angles

        # Load shuffled indices mapping
        self.trajectories_dir_path = Path(trajectories_dir)
        with open(shuffled_indices_file, 'r') as f:
            self.all_filenames = [line.strip() for line in f.readlines()]
        self.all_trajectory_files = [
            self.trajectories_dir_path / fname for fname in self.all_filenames
        ]

    def _extract_indices_from_endpoint_file(self, endpoint_file: str) -> List[int]:
        """
        Extract trajectory indices by matching start states from the endpoint file
        back to trajectory files.

        Since the endpoint files contain start_state -> end_state pairs from
        specific trajectories, we match using start states. For each unique
        start state, we find the corresponding trajectory index.

        Simpler approach: the endpoint file has N lines, each from a trajectory.
        The dataset builder creates these from trajectory indices in order.
        We count the lines to get the dataset size and use the first N trajectory
        files from the pool. But this doesn't account for the specific indices
        chosen by the adaptive engine.

        Most robust approach: use the endpoint file for the endpoint-based dataloader,
        and create trajectory datasets from the same indices. Since we receive
        data_file/validation_file paths that were built by AdaptiveDatasetBuilder,
        we parse the files to count how many unique trajectories they represent.
        """
        # Count lines in endpoint file to determine dataset size
        with open(endpoint_file, 'r') as f:
            lines = [l for l in f.readlines() if l.strip()]
        return len(lines)

    def setup(self, stage: Optional[str] = None):
        """Create train/val/test datasets.

        For trajectory FM, we need to map the endpoint files back to trajectory
        indices. The simplest approach: read the endpoint files to get the
        start states, then find matching trajectory files.

        Alternative: accept trajectory index lists directly. For now, we create
        datasets from all available trajectories, split by the endpoint file sizes.
        """
        if stage == "fit" or stage is None:
            # Load endpoint files to get the start states
            # The train file has pairs from multiple trajectories (many-to-one mapping)
            # For trajectory FM, we need the underlying trajectory indices

            # Use endpoint files to create endpoint+trajectory hybrid datasets
            # Train: load trajectories corresponding to train endpoint pairs
            train_files = self._get_trajectory_files_from_endpoints(self.data_file)
            val_files = self._get_trajectory_files_from_endpoints(self.validation_file)

            self.train_dataset = TrajectoryDataset(
                trajectories_dir=str(self.trajectories_dir_path),
                trajectory_files=train_files,
                sequence_length=self.sequence_length,
                wrap_angles=self.wrap_angles,
            )
            self.val_dataset = TrajectoryDataset(
                trajectories_dir=str(self.trajectories_dir_path),
                trajectory_files=val_files,
                sequence_length=self.sequence_length,
                wrap_angles=self.wrap_angles,
            )

        if stage == "test" or stage is None:
            test_files = self._get_trajectory_files_from_endpoints(self.test_file)
            self.test_dataset = TrajectoryDataset(
                trajectories_dir=str(self.trajectories_dir_path),
                trajectory_files=test_files,
                sequence_length=self.sequence_length,
                wrap_angles=self.wrap_angles,
            )

    def _get_trajectory_files_from_endpoints(self, endpoint_file: str) -> List[Path]:
        """
        Match endpoint file entries to trajectory files.

        Strategy: each line in the endpoint file is a (start, end) pair.
        Multiple lines can come from the same trajectory (in "train" mode,
        every timestep of a trajectory becomes a start state).

        We match by comparing the first state of each trajectory file to the
        start states in the endpoint file. Since each trajectory contributes
        multiple pairs, we find unique trajectories.
        """
        # Load endpoint data
        with open(endpoint_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            return []

        # Parse start states from endpoint file
        endpoint_starts = []
        for line in lines:
            values = list(map(float, line.split()))
            state_dim = len(values) // 2
            endpoint_starts.append(values[:state_dim])
        endpoint_starts = np.array(endpoint_starts, dtype=np.float32)

        # Get unique start states (first state = trajectory start)
        # Match each unique trajectory start against all trajectory files
        matched_files = []
        matched_indices = set()

        for idx, traj_file in enumerate(self.all_trajectory_files):
            if idx in matched_indices:
                continue
            try:
                with open(traj_file, 'r') as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue
                    traj_start = np.array(list(map(float, first_line.split(','))), dtype=np.float32)

                # Check if this trajectory's start state appears in the endpoint file
                diffs = np.abs(endpoint_starts - traj_start[None, :]).sum(axis=1)
                if np.any(diffs < 1e-4):
                    matched_files.append(traj_file)
                    matched_indices.add(idx)
            except (FileNotFoundError, ValueError):
                continue

        print(f"Matched {len(matched_files)} trajectories from {endpoint_file}")
        return matched_files

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

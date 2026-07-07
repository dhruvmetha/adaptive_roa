import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Union
from tqdm import tqdm
import os
import lightning.pytorch as pl
import random
from adaptive_roa.utils.env_config import get_data_dir, get_noise_regime


class Quadrotor2DEndpointDataset(Dataset):
    """
    Dataset for Quadrotor 2D endpoint pairs (start_state, end_state)
    Handles 6D state with circular pitch angle

    State format: [x, z, theta, x_dot, z_dot, theta_dot]

    Supports two loading modes:
    1. From endpoint file (CSV with 12 or 13 columns)
    2. From shuffled_indices + trajectory files (loads start/end from each trajectory)
    """

    def __init__(self,
                 data_file: str = None,
                 dataset_dir: str = None,
                 shuffled_indices_file: str = None,
                 trajectories_dir: str = None,
                 max_samples: int = None):
        """
        Initialize Quadrotor 2D endpoint dataset.

        Args:
            data_file: Path to endpoint dataset file (CSV format with 12/13 columns)
                      If provided, loads directly from this file.
            dataset_dir: Path to dataset directory containing dataset_description.json.
                        If None, uses default path.
            shuffled_indices_file: Path to shuffled_indices_*.txt file.
                                  If provided (and data_file is None), loads from trajectories.
            trajectories_dir: Directory containing trajectory files (sequence_*.txt).
                             Required if shuffled_indices_file is provided.
            max_samples: Maximum number of samples to load (for limiting dataset size).
        """
        if dataset_dir is None:
            dataset_dir = f"{get_data_dir()}/{get_noise_regime()}/quadrotor2D_rl"
        self.dataset_dir = Path(dataset_dir)

        if data_file is not None:
            # Mode 1: Load from endpoint file
            self._load_from_endpoint_file(data_file, max_samples)
        elif shuffled_indices_file is not None:
            # Mode 2: Load from shuffled_indices + trajectories
            if trajectories_dir is None:
                trajectories_dir = self.dataset_dir / "trajectories"
            self._load_from_shuffled_indices(shuffled_indices_file, trajectories_dir, max_samples)
        else:
            raise ValueError("Must provide either data_file or shuffled_indices_file")

    def _load_from_endpoint_file(self, data_file: str, max_samples: int = None):
        """Load endpoint data directly from a file."""
        print(f"Loading Quadrotor2D endpoint data from {data_file}...")

        # Try comma-separated first, fall back to space-separated
        if data_file.endswith('.txt') or data_file.endswith('.csv'):
            try:
                data = np.loadtxt(data_file, delimiter=',', max_rows=max_samples)
            except ValueError:
                data = np.loadtxt(data_file, max_rows=max_samples)
        else:
            data = np.loadtxt(data_file, max_rows=max_samples)

        # Expected format: start_state (6) + end_state (6) = 12 columns
        # Or: start_state (6) + end_state (6) + label (1) = 13 columns (eval_states.txt)
        n_cols = data.shape[1]

        if n_cols == 12:
            self.start_states = data[:, :6].astype(np.float32)
            self.end_states = data[:, 6:12].astype(np.float32)
            self.labels = None
        elif n_cols == 13:
            self.start_states = data[:, :6].astype(np.float32)
            self.end_states = data[:, 6:12].astype(np.float32)
            self.labels = data[:, 12].astype(np.int64)
        else:
            raise ValueError(f"Expected 12 or 13 columns, got {n_cols}")

        print(f"Loaded {len(self.start_states)} samples for Quadrotor2D endpoint data")
        print(f"  Start states shape: {self.start_states.shape}")
        print(f"  End states shape: {self.end_states.shape}")

    def _load_from_shuffled_indices(self, shuffled_indices_file: str,
                                     trajectories_dir: Union[str, Path],
                                     max_samples: int = None):
        """Load endpoint data from shuffled_indices + trajectory files."""
        print(f"Loading Quadrotor2D endpoints from shuffled indices...")
        print(f"  Indices file: {shuffled_indices_file}")
        print(f"  Trajectories dir: {trajectories_dir}")

        trajectories_dir = Path(trajectories_dir)

        # Load trajectory filenames from shuffled indices
        with open(shuffled_indices_file, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]

        if max_samples is not None:
            filenames = filenames[:max_samples]

        n_samples = len(filenames)
        print(f"  Loading {n_samples} trajectories...")

        # Pre-allocate arrays
        self.start_states = np.zeros((n_samples, 6), dtype=np.float32)
        self.end_states = np.zeros((n_samples, 6), dtype=np.float32)
        self.labels = None

        # Load each trajectory file
        for i, fname in enumerate(tqdm(filenames, desc="Loading trajectories", disable=n_samples < 1000)):
            traj_path = trajectories_dir / fname
            try:
                traj = np.loadtxt(traj_path, delimiter=',')
                self.start_states[i] = traj[0, :6].astype(np.float32)
                self.end_states[i] = traj[-1, :6].astype(np.float32)
            except Exception as e:
                print(f"Warning: Could not load {traj_path}: {e}")
                self.start_states[i] = np.zeros(6, dtype=np.float32)
                self.end_states[i] = np.zeros(6, dtype=np.float32)

        print(f"Loaded {len(self.start_states)} endpoint pairs from trajectories")

    def __len__(self):
        return len(self.start_states)

    def wrap_angle(self, angle):
        """Wrap angle to [-pi, pi] for proper S^1 manifold representation"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def __getitem__(self, idx):
        start_state = self.start_states[idx].copy()
        end_state = self.end_states[idx].copy()

        # Wrap theta (index 2) to [-pi, pi]
        start_state[2] = self.wrap_angle(start_state[2])
        end_state[2] = self.wrap_angle(end_state[2])

        return {
            'start_state': torch.tensor(start_state, dtype=torch.float32),  # [6]
            'end_state': torch.tensor(end_state, dtype=torch.float32)       # [6]
        }


class Quadrotor2DEndpointDataModule(pl.LightningDataModule):
    """
    Quadrotor 2D Endpoint Data Module.

    Supports two modes:
    1. Direct file mode: Load from pre-built endpoint files (data_file, validation_file, test_file)
    2. Shuffled indices mode: Load from shuffled_indices_*.txt + trajectory files
    """

    def __init__(self,
                 # Direct file mode
                 data_file: str = None,
                 validation_file: str = None,
                 test_file: str = None,
                 # Shuffled indices mode
                 train_indices_file: str = None,
                 val_indices_file: str = None,
                 test_indices_file: str = None,
                 trajectories_dir: str = None,
                 # Common parameters
                 batch_size: int = 64,
                 val_batch_size: Optional[int] = None,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 dataset_dir: str = None,
                 max_train_samples: int = None,
                 max_val_samples: int = None):
        """
        Initialize Quadrotor 2D Endpoint Data Module.

        Direct file mode args:
            data_file: Path to training endpoint file
            validation_file: Path to validation endpoint file
            test_file: Path to test endpoint file

        Shuffled indices mode args:
            train_indices_file: Path to shuffled_indices_*.txt for training
            val_indices_file: Path to shuffled_indices_*.txt for validation
            test_indices_file: Path to shuffled_indices_*.txt for test
            trajectories_dir: Directory containing trajectory files

        Common args:
            batch_size: Batch size for training data loader
            val_batch_size: Batch size for validation/test (defaults to batch_size)
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory
            dataset_dir: Base dataset directory
            max_train_samples: Limit training samples (for debugging)
            max_val_samples: Limit validation samples
        """
        if dataset_dir is None:
            dataset_dir = f"{get_data_dir()}/{get_noise_regime()}/quadrotor2D_rl"
        super().__init__()

        self.dataset_dir = dataset_dir

        # Direct file mode
        self.data_file = data_file
        self.validation_file = validation_file
        self.test_file = test_file

        # Shuffled indices mode
        self.train_indices_file = train_indices_file
        self.val_indices_file = val_indices_file
        self.test_indices_file = test_indices_file
        self.trajectories_dir = trajectories_dir or f"{dataset_dir}/trajectories"

        # DataLoader params
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples

        # Quadrotor2D-specific dimensions
        self.state_dim = 6
        self.embedded_dim = 7  # (x, z, sin θ, cos θ, ẋ, ż, θ̇)

        # Determine mode
        self.use_shuffled_indices = train_indices_file is not None

    def setup(self, stage: Optional[str] = None):
        if self.use_shuffled_indices:
            if stage == "fit" or stage is None:
                self.train_dataset = Quadrotor2DEndpointDataset(
                    shuffled_indices_file=self.train_indices_file,
                    trajectories_dir=self.trajectories_dir,
                    dataset_dir=self.dataset_dir,
                    max_samples=self.max_train_samples
                )
                if self.val_indices_file:
                    self.val_dataset = Quadrotor2DEndpointDataset(
                        shuffled_indices_file=self.val_indices_file,
                        trajectories_dir=self.trajectories_dir,
                        dataset_dir=self.dataset_dir,
                        max_samples=self.max_val_samples
                    )
                else:
                    self.val_dataset = self.train_dataset

            if stage == "test" or stage is None:
                if self.test_indices_file:
                    self.test_dataset = Quadrotor2DEndpointDataset(
                        shuffled_indices_file=self.test_indices_file,
                        trajectories_dir=self.trajectories_dir,
                        dataset_dir=self.dataset_dir
                    )
        else:
            if stage == "fit" or stage is None:
                self.train_dataset = Quadrotor2DEndpointDataset(
                    data_file=self.data_file,
                    dataset_dir=self.dataset_dir,
                    max_samples=self.max_train_samples
                )
                self.val_dataset = Quadrotor2DEndpointDataset(
                    data_file=self.validation_file,
                    dataset_dir=self.dataset_dir,
                    max_samples=self.max_val_samples
                )

            if stage == "test" or stage is None:
                self.test_dataset = Quadrotor2DEndpointDataset(
                    data_file=self.test_file,
                    dataset_dir=self.dataset_dir
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def predict_dataloader(self):
        return self.test_dataloader()

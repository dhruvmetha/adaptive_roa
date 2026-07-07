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
from adaptive_roa.utils.env_config import get_shared_data_base, get_noise_regime


class Quadrotor3DEndpointDataset(Dataset):
    """
    Dataset for Quadrotor 3D endpoint pairs (start_state, end_state)
    Handles 13D state with quaternion orientation

    State format: [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r]

    Supports two loading modes:
    1. From endpoint file (CSV/space-separated with 26 or 27 columns)
    2. From shuffled_indices + trajectory files (loads start/end from each trajectory)
    """

    def __init__(self,
                 data_file: str = None,
                 dataset_dir: str = None,
                 shuffled_indices_file: str = None,
                 trajectories_dir: str = None,
                 max_samples: int = None):
        """
        Initialize Quadrotor 3D endpoint dataset.

        Args:
            data_file: Path to endpoint dataset file (CSV format with 26/27 columns)
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
            dataset_dir = f"{get_shared_data_base()}/{get_noise_regime()}/quadrotor3D_lqr"
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
        print(f"Loading Quadrotor3D endpoint data from {data_file}...")

        # Check file extension and format
        if data_file.endswith('.txt') or data_file.endswith('.csv'):
            # Try comma-separated first
            try:
                data = np.loadtxt(data_file, delimiter=',', max_rows=max_samples)
            except ValueError:
                # Fall back to space-separated
                data = np.loadtxt(data_file, max_rows=max_samples)
        else:
            data = np.loadtxt(data_file, max_rows=max_samples)

        # Expected format: start_state (13) + end_state (13) = 26 columns
        # Or: start_state (13) + end_state (13) + label (1) = 27 columns (eval_states.txt)
        n_cols = data.shape[1]

        if n_cols == 26:
            # Pure endpoint data
            self.start_states = data[:, :13].astype(np.float32)
            self.end_states = data[:, 13:26].astype(np.float32)
            self.labels = None
        elif n_cols == 27:
            # eval_states.txt format with label
            self.start_states = data[:, :13].astype(np.float32)
            self.end_states = data[:, 13:26].astype(np.float32)
            self.labels = data[:, 26].astype(np.int64)
        else:
            raise ValueError(f"Expected 26 or 27 columns, got {n_cols}")

        print(f"Loaded {len(self.start_states)} samples for Quadrotor3D endpoint data")
        print(f"  Start states shape: {self.start_states.shape}")
        print(f"  End states shape: {self.end_states.shape}")

    def _load_from_shuffled_indices(self, shuffled_indices_file: str,
                                     trajectories_dir: Union[str, Path],
                                     max_samples: int = None):
        """Load endpoint data from shuffled_indices + trajectory files."""
        print(f"Loading Quadrotor3D endpoints from shuffled indices...")
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
        self.start_states = np.zeros((n_samples, 13), dtype=np.float32)
        self.end_states = np.zeros((n_samples, 13), dtype=np.float32)
        self.labels = None  # Labels not available from trajectory files directly

        # Load each trajectory file
        for i, fname in enumerate(tqdm(filenames, desc="Loading trajectories", disable=n_samples < 1000)):
            traj_path = trajectories_dir / fname
            try:
                # Trajectory format: each row is a state at a timestep
                # Load first and last rows for start/end states
                traj = np.loadtxt(traj_path)
                self.start_states[i] = traj[0, :13].astype(np.float32)
                self.end_states[i] = traj[-1, :13].astype(np.float32)
            except Exception as e:
                print(f"Warning: Could not load {traj_path}: {e}")
                # Use zeros as fallback
                self.start_states[i] = np.zeros(13, dtype=np.float32)
                self.end_states[i] = np.zeros(13, dtype=np.float32)

        print(f"Loaded {len(self.start_states)} endpoint pairs from trajectories")

    def __len__(self):
        return len(self.start_states)

    def canonicalize_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """
        Canonicalize quaternion to ensure qw >= 0 (removes double-cover ambiguity)

        Args:
            quat: Quaternion [4] as (qw, qx, qy, qz)

        Returns:
            Canonicalized quaternion [4] with qw >= 0
        """
        if quat[0] < 0:
            return -quat
        return quat

    def normalize_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """
        Normalize quaternion to unit norm

        Args:
            quat: Quaternion [4]

        Returns:
            Unit quaternion [4]
        """
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return quat / norm

    def __getitem__(self, idx):
        start_state = self.start_states[idx].copy()
        end_state = self.end_states[idx].copy()

        # Normalize and canonicalize quaternions
        start_quat = self.normalize_quaternion(start_state[3:7])
        start_quat = self.canonicalize_quaternion(start_quat)
        start_state[3:7] = start_quat

        end_quat = self.normalize_quaternion(end_state[3:7])
        end_quat = self.canonicalize_quaternion(end_quat)
        end_state[3:7] = end_quat

        return {
            'start_state': torch.tensor(start_state, dtype=torch.float32),  # [13]
            'end_state': torch.tensor(end_state, dtype=torch.float32)       # [13]
        }


class Quadrotor3DEndpointDataModule(pl.LightningDataModule):
    """
    Quadrotor 3D Endpoint Data Module.

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
        Initialize Quadrotor 3D Endpoint Data Module.

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
            dataset_dir = f"{get_shared_data_base()}/{get_noise_regime()}/quadrotor3D_lqr"
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

        # Quadrotor3D-specific dimensions
        self.state_dim = 13
        self.embedded_dim = 13

        # Determine mode
        self.use_shuffled_indices = train_indices_file is not None

    def setup(self, stage: Optional[str] = None):
        if self.use_shuffled_indices:
            # Shuffled indices mode
            if stage == "fit" or stage is None:
                self.train_dataset = Quadrotor3DEndpointDataset(
                    shuffled_indices_file=self.train_indices_file,
                    trajectories_dir=self.trajectories_dir,
                    dataset_dir=self.dataset_dir,
                    max_samples=self.max_train_samples
                )
                if self.val_indices_file:
                    self.val_dataset = Quadrotor3DEndpointDataset(
                        shuffled_indices_file=self.val_indices_file,
                        trajectories_dir=self.trajectories_dir,
                        dataset_dir=self.dataset_dir,
                        max_samples=self.max_val_samples
                    )
                else:
                    # Use subset of training as validation
                    self.val_dataset = self.train_dataset

            if stage == "test" or stage is None:
                if self.test_indices_file:
                    self.test_dataset = Quadrotor3DEndpointDataset(
                        shuffled_indices_file=self.test_indices_file,
                        trajectories_dir=self.trajectories_dir,
                        dataset_dir=self.dataset_dir
                    )
        else:
            # Direct file mode
            if stage == "fit" or stage is None:
                self.train_dataset = Quadrotor3DEndpointDataset(
                    data_file=self.data_file,
                    dataset_dir=self.dataset_dir,
                    max_samples=self.max_train_samples
                )
                self.val_dataset = Quadrotor3DEndpointDataset(
                    data_file=self.validation_file,
                    dataset_dir=self.dataset_dir,
                    max_samples=self.max_val_samples
                )

            if stage == "test" or stage is None:
                self.test_dataset = Quadrotor3DEndpointDataset(
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


def create_endpoint_splits_from_eval_states(
    eval_states_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Create train/val/test splits from eval_states.txt file

    Args:
        eval_states_file: Path to eval_states.txt (27 columns: 13 start + 13 end + 1 label)
        output_dir: Directory to save split files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set (test = 1 - train - val)
        seed: Random seed for reproducibility
    """
    print(f"Loading eval_states from {eval_states_file}...")
    data = np.loadtxt(eval_states_file, delimiter=',')

    n_samples = len(data)
    print(f"Total samples: {n_samples}")

    # Shuffle indices
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)

    # Calculate split sizes
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    print(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    # Extract endpoint data (first 26 columns, excluding label)
    endpoint_data = data[:, :26]

    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as space-separated for consistency with other systems
    np.savetxt(output_dir / "train_endpoints.txt", endpoint_data[train_indices], fmt='%.6f')
    np.savetxt(output_dir / "val_endpoints.txt", endpoint_data[val_indices], fmt='%.6f')
    np.savetxt(output_dir / "test_endpoints.txt", endpoint_data[test_indices], fmt='%.6f')

    print(f"Saved endpoint splits to {output_dir}")
    print(f"  train_endpoints.txt: {len(train_indices)} samples")
    print(f"  val_endpoints.txt: {len(val_indices)} samples")
    print(f"  test_endpoints.txt: {len(test_indices)} samples")

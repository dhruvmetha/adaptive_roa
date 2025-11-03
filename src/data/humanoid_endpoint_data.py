import torch
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from tqdm import tqdm
import os
import lightning.pytorch as pl
import random


class HumanoidEndpointDataset(Dataset):
    def __init__(self, data_file: str,
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/humanoid_get_up/humanoid_data_bounds.pkl"):
        """
        Dataset for humanoid endpoint pairs (start_state, end_state)
        Handles 67D humanoid state with Sphere manifold for orientation

        Manifold: ℝ³⁴ × S² × ℝ³⁰
        State format: [euclidean1(34), sphere_x, sphere_y, sphere_z, euclidean2(30)]

        Args:
            data_file: Path to endpoint metadata file
            bounds_file: Path to pickle file with actual data bounds
        """
        self.bounds_file = bounds_file
        self._load_bounds()

        # Load the endpoint metadata
        with open(data_file, 'r') as f:
            lines = f.readlines()

        # Parse the metadata - each line has [file_path, start_idx, end_idx]
        self.metadata = []
        unique_files = set()

        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 3:
                    file_path = parts[0]
                    start_idx = int(parts[1])
                    end_idx = int(parts[2])
                    self.metadata.append((file_path, start_idx, end_idx))
                    unique_files.add(file_path)

        print(f"Loaded {len(self.metadata)} endpoint metadata entries")
        print(f"Loading {len(unique_files)} unique trajectory files...")

        # Load all unique trajectory files into dictionary
        self.trajectory_cache = {}
        for file_path in tqdm(unique_files, desc="Loading trajectories"):
            trajectory = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        values = list(map(float, line.strip().split(',')))
                        trajectory.append(values)
            self.trajectory_cache[file_path] = trajectory

        print(f"Cached {len(self.trajectory_cache)} trajectories in memory")

    def _load_bounds(self):
        """Load data bounds from pickle file (just for verification/logging)

        Note: Actual normalization is done by the HumanoidSystem class,
        which loads per-dimension bounds. This method just prints summary info.
        """
        if Path(self.bounds_file).exists():
            with open(self.bounds_file, 'rb') as f:
                bounds_data = pickle.load(f)

            statistics = bounds_data.get('statistics', {})

            print(f"✅ Bounds file found: {self.bounds_file}")
            print(f"   {statistics.get('euclidean_dimensions', 64)} Euclidean dims with PER-DIMENSION bounds")
            print(f"   {statistics.get('sphere_dimensions', 3)} Sphere dims (34-36): NO normalization")
            print(f"   (Normalization handled by HumanoidSystem with per-dimension limits)")
        else:
            print(f"⚠️  Bounds file not found: {self.bounds_file}")
            print(f"   HumanoidSystem will use default bounds (±20.0)")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        file_path, start_idx, end_idx = self.metadata[idx]

        # Look up trajectory in cache
        trajectory = self.trajectory_cache[file_path]

        # Extract start and end states
        start_state = trajectory[start_idx]
        end_state = trajectory[end_idx]

        return {
            'start_state': torch.tensor(start_state, dtype=torch.float32),  # [67] raw state
            'end_state': torch.tensor(end_state, dtype=torch.float32)       # [67] raw state
        }


class HumanoidEndpointDataModule(pl.LightningDataModule):
    def __init__(self, data_file: str, validation_file: str, test_file: str,
                 batch_size: int = 64, val_batch_size: Optional[int] = None,
                 num_workers: int = 4, pin_memory: bool = True,
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/humanoid_get_up/humanoid_data_bounds.pkl"):
        """
        Humanoid Endpoint Data Module with separate train/val/test files

        Args:
            data_file: Path to training dataset metadata file
            validation_file: Path to validation dataset metadata file
            test_file: Path to test dataset metadata file
            batch_size: Batch size for training data loader
            val_batch_size: Batch size for validation/test data loaders (defaults to batch_size if None)
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for data loaders
            bounds_file: Path to pickle file with actual data bounds
        """
        super().__init__()
        self.data_file = data_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.bounds_file = bounds_file

        # Humanoid-specific dimensions
        self.state_dim = 67  # Raw state: [euclidean1(34), sphere(3), euclidean2(30)]
        self.embedded_dim = 67  # Embedded: same as state (sphere already continuous)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = HumanoidEndpointDataset(
                self.data_file,
                bounds_file=self.bounds_file
            )
            self.val_dataset = HumanoidEndpointDataset(
                self.validation_file,
                bounds_file=self.bounds_file
            )

        if stage == "test" or stage is None:
            self.test_dataset = HumanoidEndpointDataset(
                self.test_file,
                bounds_file=self.bounds_file
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def predict_dataloader(self):
        return self.test_dataloader()

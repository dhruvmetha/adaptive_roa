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
            data_file: Path to endpoint dataset file
            bounds_file: Path to pickle file with actual data bounds
        """
        self.bounds_file = bounds_file
        self._load_bounds()

        # Load the endpoint data
        with open(data_file, 'r') as f:
            lines = f.readlines()

        # Parse the data - each line has [start_state(67D), end_state(67D)]
        data = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split()))
                if len(values) == 134:  # start_state (67D) + end_state (67D)
                    start_state = values[:67]
                    end_state = values[67:]
                    data.append((start_state, end_state))

        print(f"Loaded {len(data)} samples for humanoid endpoint data")
        self.data = data

    def _load_bounds(self):
        """Load data bounds from pickle file and compute symmetric limits"""
        if Path(self.bounds_file).exists():
            with open(self.bounds_file, 'rb') as f:
                bounds_data = pickle.load(f)

            limits = bounds_data.get('limits', {})
            self.euclidean_limit = limits.get('euclidean_limit', 20.0)

            summary = bounds_data.get('summary', {})
            print(f"Using bounds from {self.bounds_file}")
            print(f"  Euclidean (dims 0-33, 37-66): [{summary.get('euclidean_global_min', -20):.3f}, {summary.get('euclidean_global_max', 20):.3f}] -> symmetric: ±{self.euclidean_limit:.3f}")
            print(f"  Sphere (dims 34-36): unit norm (no normalization)")
        else:
            # Fallback to default bounds
            self.euclidean_limit = 20.0
            print(f"Warning: Bounds file not found, using default bounds")
            print(f"  Euclidean: ±{self.euclidean_limit}")
            print(f"  Sphere: unit norm (no normalization)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start_state, end_state = self.data[idx]

        # No processing needed - sphere components (dims 34-36) are already unit vectors
        # Data is already on the manifold (unit norm constraint satisfied)

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
            data_file: Path to training dataset file
            validation_file: Path to validation dataset file
            test_file: Path to test dataset file
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
            self.train_dataset = HumanoidEndpointDataset(self.data_file, bounds_file=self.bounds_file)
            self.val_dataset = HumanoidEndpointDataset(self.validation_file, bounds_file=self.bounds_file)

        if stage == "test" or stage is None:
            self.test_dataset = HumanoidEndpointDataset(self.test_file, bounds_file=self.bounds_file)

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

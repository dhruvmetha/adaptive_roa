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


class MountainCarEndpointDataset(Dataset):
    def __init__(self, data_file: str,
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/mountain_car/mountain_car_data_bounds.pkl"):
        """
        Dataset for Mountain Car endpoint pairs (start_state, end_state)
        Handles 2D Mountain Car state (pure Euclidean manifold)

        State format: [position, velocity]

        Args:
            data_file: Path to endpoint dataset file
            bounds_file: Path to pickle file with actual data bounds
        """
        self.bounds_file = bounds_file

        # Load the endpoint data
        with open(data_file, 'r') as f:
            lines = f.readlines()

        # Parse the data - each line has [start_position, start_velocity,
        #                                 end_position, end_velocity]
        data = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split()))
                if len(values) == 4:  # start_state (2D) + end_state (2D)
                    start_state = values[:2]
                    end_state = values[2:]
                    data.append((start_state, end_state))

        print(f"Loaded {len(data)} samples for Mountain Car endpoint data")
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start_state, end_state = self.data[idx]

        return {
            'start_state': torch.tensor(start_state, dtype=torch.float32),  # [2] raw state
            'end_state': torch.tensor(end_state, dtype=torch.float32)       # [2] raw state
        }


class MountainCarEndpointDataModule(pl.LightningDataModule):
    def __init__(self, data_file: str, validation_file: str, test_file: str,
                 batch_size: int = 64, val_batch_size: Optional[int] = None,
                 num_workers: int = 4, pin_memory: bool = True,
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/mountain_car/mountain_car_data_bounds.pkl"):
        """
        Mountain Car Endpoint Data Module with separate train/val/test files

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

        # Mountain Car dimensions (pure Euclidean)
        self.state_dim = 2  # Raw state: [position, velocity]
        self.embedded_dim = 2  # Embedded: same as raw (no embedding transformation needed)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MountainCarEndpointDataset(self.data_file, bounds_file=self.bounds_file)
            self.val_dataset = MountainCarEndpointDataset(self.validation_file, bounds_file=self.bounds_file)

        if stage == "test" or stage is None:
            self.test_dataset = MountainCarEndpointDataset(self.test_file, bounds_file=self.bounds_file)

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

from torch.utils.data import Dataset, DataLoader
from typing import Optional
from tqdm import tqdm
import os
import torch
import lightning.pytorch as pl
import numpy as np
import random

class EndpointDataset(Dataset):
    def __init__(self, data_file: str, split="train", train_split: float = 0.8, val_split: float = 0.1):
        """
        Dataset for endpoint pairs (start_state, end_state)
        Args:
            data_file: path to endpoint dataset file
            split: one of 'train', 'val', or 'test'
            train_split: Fraction of data for training (default: 0.8)
            val_split: Fraction of data for validation (default: 0.1)
        """
        self.split = split

        # Load the endpoint data
        with open(data_file, 'r') as f:
            lines = f.readlines()

        # Parse the data - each line has [start_x, start_y, end_x, end_y]
        data = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split()))
                if len(values) == 4:  # start_state (2D) + end_state (2D)
                    start_state = values[:2]
                    end_state = values[2:]
                    data.append((start_state, end_state))

        # Split the data using config parameters
        random.shuffle(data)
        n = len(data)
        train_end = int(train_split * n)
        val_end = int((train_split + val_split) * n)

        if self.split == "train":
            self.data = data[:train_end]
        elif self.split == "val":
            self.data = data[train_end:val_end]
        elif self.split == "test":
            self.data = data[val_end:]

        print(f"Loaded {len(self.data)} samples for pendulum endpoint data ({split} split)")
        print(f"  Total data: {n}, Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}")
        
        # Normalization bounds for pendulum data
        # Angle: [-π, π], Angular velocity: approximately [-2π, 2π]
        self.min_bounds = np.array([-np.pi, -2*np.pi])
        self.max_bounds = np.array([np.pi, 2*np.pi])
            
    def __len__(self):
        return len(self.data)
    
    def normalize_state(self, state):
        """Normalize state to [0, 1] range"""
        state = np.array(state)
        return (state - self.min_bounds) / (self.max_bounds - self.min_bounds)
    
    def denormalize_state(self, state):
        """Denormalize state from [0, 1] back to original range"""
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        return state * (self.max_bounds - self.min_bounds) + self.min_bounds
    
    def __getitem__(self, idx):
        start_state, end_state = self.data[idx]

        # Return raw (unnormalized) states for flow matching
        # The flow matcher handles normalization internally
        return {
            "start_state": torch.tensor(start_state, dtype=torch.float32),
            "end_state": torch.tensor(end_state, dtype=torch.float32)
        }

class EndpointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        shuffle: bool = True,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Check if data file exists"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Endpoint data file not found: {self.data_file}")

    def setup(self, stage: Optional[str] = None):
        """Load data and create datasets"""

        if stage == "fit" or stage is None:
            self.train_dataset = EndpointDataset(
                self.data_file, split="train",
                train_split=self.train_split, val_split=self.val_split
            )
            self.val_dataset = EndpointDataset(
                self.data_file, split="val",
                train_split=self.train_split, val_split=self.val_split
            )

        if stage == "test" or stage is None:
            self.test_dataset = EndpointDataset(
                self.data_file, split="test",
                train_split=self.train_split, val_split=self.val_split
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
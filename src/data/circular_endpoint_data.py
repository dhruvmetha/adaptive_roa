import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from tqdm import tqdm
import os
import lightning.pytorch as pl
import random

class CircularEndpointDataset(Dataset):
    def __init__(self, data_file: str, split="train"):
        """
        Dataset for circular endpoint pairs (start_state, end_state)
        Handles circular angle data properly
        """
        self.split = split
        
        # Load the endpoint data
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Parse the data - each line has [start_theta, start_theta_dot, end_theta, end_theta_dot]
        data = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split()))
                if len(values) == 4:  # start_state (2D) + end_state (2D)
                    start_state = values[:2]
                    end_state = values[2:]
                    data.append((start_state, end_state))
        
        # Split the data into train/val/test (80/10/10)
        random.shuffle(data)
        n = len(data)
        
        if self.split == "train":
            self.data = data[:int(0.8 * n)]
        elif self.split == "val":
            self.data = data[int(0.8 * n):int(0.9 * n)]
        elif self.split == "test":
            self.data = data[int(0.9 * n):]
        
        print(f"Loaded {len(self.data)} {split} samples for circular endpoint data")
            
    def __len__(self):
        return len(self.data)
    
    def embed_state(self, state):
        """Convert (θ, θ̇) → (sin(θ), cos(θ), θ̇_normalized)"""
        theta, theta_dot = state
        # Normalize θ̇ to [-1, 1] based on observed data range
        theta_dot_normalized = self.normalize_theta_dot(theta_dot)
        return [np.sin(theta), np.cos(theta), theta_dot_normalized]
    
    def normalize_theta_dot(self, theta_dot):
        """Normalize θ̇ to [-1, 1] range"""
        # Based on actual data analysis: θ̇ range is approximately [-6.28, +6.28]
        theta_dot_min, theta_dot_max = -6.28, 6.28
        return 2 * (theta_dot - theta_dot_min) / (theta_dot_max - theta_dot_min) - 1
    
    def denormalize_theta_dot(self, theta_dot_normalized):
        """Convert normalized θ̇ back to original range"""
        theta_dot_min, theta_dot_max = -6.28, 6.28
        return (theta_dot_normalized + 1) * (theta_dot_max - theta_dot_min) / 2 + theta_dot_min
    
    def __getitem__(self, idx):
        start_state, end_state = self.data[idx]
        
        # Embed states on S¹ × ℝ manifold
        start_embedded = self.embed_state(start_state)
        end_embedded = self.embed_state(end_state)
        
        return {
            "start_state": torch.tensor(start_embedded, dtype=torch.float32),
            "end_state": torch.tensor(end_embedded, dtype=torch.float32),
            "start_state_original": torch.tensor(start_state, dtype=torch.float32),
            "end_state_original": torch.tensor(end_state, dtype=torch.float32)
        }

class CircularEndpointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """Check if data file exists"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Circular endpoint data file not found: {self.data_file}")
        
    def setup(self, stage: Optional[str] = None):
        """Load data and create datasets"""
        
        if stage == "fit" or stage is None:
            self.train_dataset = CircularEndpointDataset(self.data_file, split="train")
            self.val_dataset = CircularEndpointDataset(self.data_file, split="val")
            
        if stage == "test" or stage is None:
            self.test_dataset = CircularEndpointDataset(self.data_file, split="test")
    
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
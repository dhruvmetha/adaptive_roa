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


class CartPoleEndpointDataset(Dataset):
    def __init__(self, data_file: str,
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/cartpole/cartpole_data_bounds.pkl"):
        """
        Dataset for cartpole endpoint pairs (start_state, end_state)
        Handles 4D cartpole state with proper embedding for circular angle

        State format: [x, theta, x_dot, theta_dot]

        Args:
            data_file: Path to endpoint dataset file
            bounds_file: Path to pickle file with actual data bounds
        """
        self.bounds_file = bounds_file
        self._load_bounds()

        # Load the endpoint data
        with open(data_file, 'r') as f:
            lines = f.readlines()

        # Parse the data - each line has [start_x, start_theta, start_x_dot, start_theta_dot,
        #                                 end_x, end_theta, end_x_dot, end_theta_dot]
        data = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split()))
                if len(values) == 8:  # start_state (4D) + end_state (4D)
                    start_state = values[:4]
                    end_state = values[4:]
                    data.append((start_state, end_state))

        print(f"Loaded {len(data)} samples for cartpole endpoint data")
        self.data = data
    
    def _load_bounds(self):
        """Load data bounds from pickle file and compute symmetric limits"""
        if Path(self.bounds_file).exists():
            with open(self.bounds_file, 'rb') as f:
                bounds_data = pickle.load(f)
            bounds = bounds_data['bounds']
            
            # Compute symmetric bounds (same as CartPoleSystemLCFM)
            self.cart_limit = max(abs(bounds['x']['min']), abs(bounds['x']['max']))
            self.velocity_limit = max(abs(bounds['x_dot']['min']), abs(bounds['x_dot']['max']))
            self.angular_velocity_limit = max(abs(bounds['theta_dot']['min']), abs(bounds['theta_dot']['max']))
            
            print(f"Using symmetric bounds from {self.bounds_file}")
            print(f"  Cart position: [{bounds['x']['min']:.3f}, {bounds['x']['max']:.3f}] -> symmetric: ±{self.cart_limit:.3f}")
            print(f"  Cart velocity: [{bounds['x_dot']['min']:.3f}, {bounds['x_dot']['max']:.3f}] -> symmetric: ±{self.velocity_limit:.3f}")
            print(f"  Angular velocity: [{bounds['theta_dot']['min']:.3f}, {bounds['theta_dot']['max']:.3f}] -> symmetric: ±{self.angular_velocity_limit:.3f}")
        else:
            # Fallback to default symmetric bounds
            self.cart_limit = 2.4
            self.velocity_limit = 10.0
            self.angular_velocity_limit = 10.0
            print(f"Warning: Bounds file not found, using default symmetric bounds")
            
    def __len__(self):
        return len(self.data)
    
    def wrap_angle(self, angle):
        """
        Wrap angle to [-π, π] for proper S¹ manifold representation
        
        Args:
            angle: Angle in radians (can be unwrapped)
            
        Returns:
            Wrapped angle in [-π, π]
        """
        # we need to wrap the angles in the end states
        angle = angle % (2 * np.pi)
        angle =  angle - 2 * np.pi if angle > np.pi else angle
        return angle
    
    def __getitem__(self, idx):
        start_state, end_state = self.data[idx]
        
        # Wrap angles in raw states for consistent interpolation
        start_state_wrapped = list(start_state)
        end_state_wrapped = list(end_state)
        start_state_wrapped[1] = self.wrap_angle(start_state[1])  # Wrap θ component
        end_state_wrapped[1] = self.wrap_angle(end_state[1])      # Wrap θ component
        
        return {
            'start_state': torch.tensor(start_state_wrapped, dtype=torch.float32),  # [4] raw with wrapped θ
            'end_state': torch.tensor(end_state_wrapped, dtype=torch.float32)       # [4] raw with wrapped θ
        }


class CartPoleEndpointDataModule(pl.LightningDataModule):
    def __init__(self, data_file: str, validation_file: str, test_file: str,
                 batch_size: int = 64, num_workers: int = 4, pin_memory: bool = True,
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/cartpole/cartpole_data_bounds.pkl"):
        """
        CartPole Endpoint Data Module with separate train/val/test files

        Args:
            data_file: Path to training dataset file
            validation_file: Path to validation dataset file
            test_file: Path to test dataset file
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for data loaders
            bounds_file: Path to pickle file with actual data bounds
        """
        super().__init__()
        self.data_file = data_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.bounds_file = bounds_file

        # Cartpole-specific dimensions
        self.state_dim = 4  # Raw state: [x, theta, x_dot, theta_dot]
        self.embedded_dim = 5  # Embedded: [x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm]

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = CartPoleEndpointDataset(self.data_file, bounds_file=self.bounds_file)
            self.val_dataset = CartPoleEndpointDataset(self.validation_file, bounds_file=self.bounds_file)

        if stage == "test" or stage is None:
            self.test_dataset = CartPoleEndpointDataset(self.test_file, bounds_file=self.bounds_file)
    
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
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def predict_dataloader(self):
        return self.test_dataloader()
import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from tqdm import tqdm
import os
import lightning.pytorch as pl
import random
from adaptive_roa.utils.env_config import get_data_dir, get_noise_regime


class CartPoleEndpointDataset(Dataset):
    def __init__(self, data_file: str,
                 dataset_dir: str = None):
        """
        Dataset for cartpole endpoint pairs (start_state, end_state)
        Handles 4D cartpole state with proper embedding for circular angle

        State format: [x, theta, x_dot, theta_dot]

        Args:
            data_file: Path to endpoint dataset file
            dataset_dir: Path to dataset directory containing dataset_description.json.
                        If None, uses default path.
        """
        if dataset_dir is None:
            dataset_dir = f"{get_data_dir()}/{get_noise_regime()}/cartpole_pybullet"
        self.dataset_dir = Path(dataset_dir)

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
        """Load data bounds from dataset_description.json and compute symmetric limits"""
        json_path = self.dataset_dir / "dataset_description.json"
        if json_path.exists():
            with open(json_path) as f:
                dataset_info = json.load(f)
            bounds = dataset_info['achieved_bounds']

            # Compute symmetric bounds (same as CartPoleSystem)
            self.cart_limit = max(abs(bounds['x']['min']), abs(bounds['x']['max']))
            self.velocity_limit = max(abs(bounds['x_dot']['min']), abs(bounds['x_dot']['max']))
            self.angular_velocity_limit = max(abs(bounds['theta_dot']['min']), abs(bounds['theta_dot']['max']))

            print(f"Using bounds from {json_path}")
            print(f"  Cart position: [{bounds['x']['min']:.3f}, {bounds['x']['max']:.3f}] -> symmetric: ±{self.cart_limit:.3f}")
            print(f"  Cart velocity: [{bounds['x_dot']['min']:.3f}, {bounds['x_dot']['max']:.3f}] -> symmetric: ±{self.velocity_limit:.3f}")
            print(f"  Angular velocity: [{bounds['theta_dot']['min']:.3f}, {bounds['theta_dot']['max']:.3f}] -> symmetric: ±{self.angular_velocity_limit:.3f}")
        else:
            # Fallback to default symmetric bounds
            self.cart_limit = 2.4
            self.velocity_limit = 10.0
            self.angular_velocity_limit = 10.0
            print(f"Warning: dataset_description.json not found, using default symmetric bounds")
            
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
        # Use atan2 for robust angle wrapping (handles all edge cases)
        return np.arctan2(np.sin(angle), np.cos(angle))
    
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
                 batch_size: int = 64, val_batch_size: Optional[int] = None,
                 num_workers: int = 4, pin_memory: bool = True,
                 dataset_dir: str = None):
        """
        CartPole Endpoint Data Module with separate train/val/test files

        Args:
            data_file: Path to training dataset file
            validation_file: Path to validation dataset file
            test_file: Path to test dataset file
            batch_size: Batch size for training data loader
            val_batch_size: Batch size for validation/test data loaders (defaults to batch_size if None)
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for data loaders
            dataset_dir: Path to dataset directory containing dataset_description.json.
                        If None, uses default path.
        """
        if dataset_dir is None:
            dataset_dir = f"{get_data_dir()}/{get_noise_regime()}/cartpole_pybullet"
        super().__init__()
        self.data_file = data_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_dir = dataset_dir

        # Cartpole-specific dimensions
        self.state_dim = 4  # Raw state: [x, theta, x_dot, theta_dot]
        self.embedded_dim = 5  # Embedded: [x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm]

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = CartPoleEndpointDataset(self.data_file, dataset_dir=self.dataset_dir)
            self.val_dataset = CartPoleEndpointDataset(self.validation_file, dataset_dir=self.dataset_dir)

        if stage == "test" or stage is None:
            self.test_dataset = CartPoleEndpointDataset(self.test_file, dataset_dir=self.dataset_dir)
    
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
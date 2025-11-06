"""CartPole Endpoint Dataset

Loads endpoint prediction data for CartPole using metadata format:
- Metadata file contains: [file_path, start_idx, end_idx, label]
- Trajectory files are loaded and cached in memory
- Start state: state at start_idx (x, θ, ẋ, θ̇)
- End state: state at end_idx (x, θ, ẋ, θ̇)
- Angle wrapping: θ is wrapped to [-π, π] for S¹ manifold
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning.pytorch as pl
from typing import Optional
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path


def _load_trajectory_file(file_path):
    """Helper function to load a single trajectory file (module-level for multiprocessing)"""
    return file_path, np.loadtxt(file_path, delimiter=",")


class CartPoleEndpointDataset(Dataset):
    """Dataset for CartPole endpoint prediction using metadata format.

    Each sample contains:
    - start_state: Initial state [x, θ, ẋ, θ̇]
    - end_state: Final state [x, θ, ẋ, θ̇]

    Metadata format: file_path start_idx end_idx label
    """

    def __init__(self, data_file: str, bounds_file: Optional[str] = None):
        """Initialize dataset.

        Args:
            data_file: Path to endpoint metadata file
                      Format: file_path start_idx end_idx label
            bounds_file: Path to bounds pickle file (optional)
        """
        self.data_file = data_file
        self.bounds_file = bounds_file

        # Load bounds if provided (for info/validation)
        if bounds_file and os.path.exists(bounds_file):
            with open(bounds_file, 'rb') as f:
                bounds_data = pickle.load(f)
            self._load_bounds(bounds_data)
        else:
            # Use default symmetric bounds
            self.cart_limit = 2.4
            self.velocity_limit = 10.0
            self.angular_velocity_limit = 10.0
            self.bounds_data = None

        # Load the endpoint metadata
        with open(data_file, 'r') as f:
            lines = f.readlines()

        # Parse the metadata - each line has [file_path, start_idx, end_idx, label]
        self.metadata = []
        self.labels = []
        unique_files = set()

        for line_num, line in enumerate(lines, start=1):
            if line.strip():
                parts = line.strip().split()
                if len(parts) != 4:
                    raise ValueError(
                        f"Invalid metadata format at line {line_num} in {data_file}. "
                        f"Expected 4 parts [file_path, start_idx, end_idx, label], got {len(parts)}. "
                        f"Line content: {line.strip()}"
                    )
                file_path = parts[0]
                start_idx = int(parts[1])
                end_idx = int(parts[2])
                label = int(parts[3])
                self.metadata.append((file_path, start_idx, end_idx))
                self.labels.append(label)
                unique_files.add(file_path)

        print(f"✅ Loaded {len(self.metadata)} endpoint metadata entries")
        print(f"   Loading {len(unique_files)} unique trajectory files...")

        # Load all unique trajectory files into dictionary (in parallel using multiprocessing)
        self.trajectory_cache = {}
        # Use ProcessPoolExecutor for parallel processing across multiple cores
        with ProcessPoolExecutor(max_workers=8) as executor:
            # Submit all loading tasks and wrap with tqdm for progress tracking
            results = list(tqdm(
                executor.map(_load_trajectory_file, unique_files),
                total=len(unique_files),
                desc="Loading trajectories"
            ))
            # Populate the cache dictionary
            for file_path, trajectory_data in results:
                self.trajectory_cache[file_path] = trajectory_data

        print(f"✅ Cached {len(self.trajectory_cache)} trajectories in memory")
        print(f"   State dim: 4 (x, θ, ẋ, θ̇)")

    def _load_bounds(self, bounds_data):
        """Load data bounds from pickle and compute symmetric limits"""
        bounds = bounds_data['bounds']

        # Compute symmetric bounds (same as CartPoleSystemLCFM)
        self.cart_limit = max(abs(bounds['x']['min']), abs(bounds['x']['max']))
        self.velocity_limit = max(abs(bounds['x_dot']['min']), abs(bounds['x_dot']['max']))
        self.angular_velocity_limit = max(abs(bounds['theta_dot']['min']), abs(bounds['theta_dot']['max']))

        print(f"   Using symmetric bounds from {self.bounds_file}")
        print(f"     Cart position: [{bounds['x']['min']:.3f}, {bounds['x']['max']:.3f}] -> symmetric: ±{self.cart_limit:.3f}")
        print(f"     Cart velocity: [{bounds['x_dot']['min']:.3f}, {bounds['x_dot']['max']:.3f}] -> symmetric: ±{self.velocity_limit:.3f}")
        print(f"     Angular velocity: [{bounds['theta_dot']['min']:.3f}, {bounds['theta_dot']['max']:.3f}] -> symmetric: ±{self.angular_velocity_limit:.3f}")

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

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx):
        file_path, start_idx, end_idx = self.metadata[idx]

        # Look up trajectory in cache
        trajectory = self.trajectory_cache[file_path]

        # Extract start and end states
        start_state = trajectory[start_idx]
        end_state = trajectory[end_idx]

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
    """Lightning DataModule for CartPole endpoint prediction.

    Supports:
    - Metadata-based loading
    - Separate train/validation/test files
    - Stratified sampling for class balance
    - Parallel trajectory loading
    - Angle wrapping for circular manifold
    """

    def __init__(
        self,
        train_file: str,
        validation_file: str,
        test_file: Optional[str] = None,
        bounds_file: Optional[str] = None,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        use_stratified_sampling: bool = False,
    ):
        """Initialize data module.

        Args:
            train_file: Path to training metadata file
            validation_file: Path to validation metadata file
            test_file: Path to test metadata file (optional)
            bounds_file: Path to bounds pickle file (optional)
            batch_size: Training batch size
            val_batch_size: Validation batch size (defaults to batch_size)
            num_workers: Number of dataloader workers
            shuffle: Whether to shuffle training data
            pin_memory: Whether to pin memory for faster GPU transfer
            use_stratified_sampling: Whether to use stratified sampling for training
        """
        super().__init__()

        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.bounds_file = bounds_file
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.use_stratified_sampling = use_stratified_sampling

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # For convenience - match other systems' interface
        self.state_dim = 4  # (x, θ, ẋ, θ̇)
        self.embedded_dim = 5  # (x_norm, sin(θ), cos(θ), ẋ_norm, θ̇_norm)

    def prepare_data(self):
        """Check if data files exist"""
        if not os.path.exists(self.train_file):
            raise FileNotFoundError(f"Training data file not found: {self.train_file}")
        if not os.path.exists(self.validation_file):
            raise FileNotFoundError(f"Validation data file not found: {self.validation_file}")
        if self.test_file and not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Test data file not found: {self.test_file}")

    def setup(self, stage: Optional[str] = None):
        """Load data and create datasets"""

        if stage == "fit" or stage is None:
            print("\n📊 Setting up CartPole datasets...")
            print(f"   Train file: {self.train_file}")
            print(f"   Validation file: {self.validation_file}")
            if self.use_stratified_sampling:
                print(f"   Using stratified sampling: True")
            print()

            self.train_dataset = CartPoleEndpointDataset(
                self.train_file,
                bounds_file=self.bounds_file
            )
            self.val_dataset = CartPoleEndpointDataset(
                self.validation_file,
                bounds_file=self.bounds_file
            )

        if stage == "test" or stage is None:
            if self.test_file:
                print(f"\n📊 Setting up test dataset: {self.test_file}")
                self.test_dataset = CartPoleEndpointDataset(
                    self.test_file,
                    bounds_file=self.bounds_file
                )

    def train_dataloader(self):
        """Create training dataloader with optional stratified sampling"""
        if self.use_stratified_sampling and hasattr(self.train_dataset, 'labels'):
            # Compute class weights for stratified sampling
            labels = np.array(self.train_dataset.labels)
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Compute inverse frequency weights
            class_weights = 1.0 / counts
            sample_weights = class_weights[labels]

            # Create weighted sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            print(f"✅ Using stratified sampling:")
            for label, count, weight in zip(unique_labels, counts, class_weights):
                print(f"   Label {label}: {count} samples, weight {weight:.4f}")

            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,  # Use sampler instead of shuffle
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=self.shuffle,
            )

    def val_dataloader(self):
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Create test dataloader"""
        if self.test_dataset is None:
            raise ValueError("Test dataset not initialized. Provide test_file in __init__")
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

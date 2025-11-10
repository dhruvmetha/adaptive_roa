"""Pendulum Cartesian Endpoint Dataset

Loads endpoint prediction data for Pendulum in Cartesian coordinates:
- Metadata file contains: [file_path, start_idx, end_idx, label]
- Trajectory files are loaded and cached in memory
- Start state: state at start_idx (x, y, ẋ, ẏ)
- End state: state at end_idx (x, y, ẋ, ẏ)
- All Euclidean coordinates (no angle wrapping needed)
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


class PendulumCartesianEndpointDataset(Dataset):
    """Dataset for Pendulum Cartesian endpoint prediction using metadata format.

    Each sample contains:
    - start_state: Initial state [x, y, ẋ, ẏ]
    - end_state: Final state [x, y, ẋ, ẏ]

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
            self.x_limit = 1.0
            self.y_limit = 1.0
            self.x_dot_limit = 2 * np.pi
            self.y_dot_limit = 2 * np.pi
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
        print(f"   State dim: 4 (x, y, ẋ, ẏ)")

    def _load_bounds(self, bounds_data):
        """Load data bounds from pickle and compute symmetric limits"""
        bounds = bounds_data['bounds']

        # Compute symmetric bounds (all Euclidean)
        self.x_limit = max(abs(bounds['x']['min']), abs(bounds['x']['max']))
        self.y_limit = max(abs(bounds['y']['min']), abs(bounds['y']['max']))
        self.x_dot_limit = max(abs(bounds['x_dot']['min']), abs(bounds['x_dot']['max']))
        self.y_dot_limit = max(abs(bounds['y_dot']['min']), abs(bounds['y_dot']['max']))

        print(f"   Using symmetric bounds from {self.bounds_file}")
        print(f"     Position x: [{bounds['x']['min']:.3f}, {bounds['x']['max']:.3f}] -> symmetric: ±{self.x_limit:.3f}")
        print(f"     Position y: [{bounds['y']['min']:.3f}, {bounds['y']['max']:.3f}] -> symmetric: ±{self.y_limit:.3f}")
        print(f"     Velocity ẋ: [{bounds['x_dot']['min']:.3f}, {bounds['x_dot']['max']:.3f}] -> symmetric: ±{self.x_dot_limit:.3f}")
        print(f"     Velocity ẏ: [{bounds['y_dot']['min']:.3f}, {bounds['y_dot']['max']:.3f}] -> symmetric: ±{self.y_dot_limit:.3f}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx):
        file_path, start_idx, end_idx = self.metadata[idx]

        # Look up trajectory in cache
        trajectory = self.trajectory_cache[file_path]

        # Extract start and end states
        start_state = trajectory[start_idx]
        end_state = trajectory[end_idx]

        # No angle wrapping needed - all Euclidean coordinates
        return {
            'start_state': torch.tensor(start_state, dtype=torch.float32),  # [4] raw (x, y, ẋ, ẏ)
            'end_state': torch.tensor(end_state, dtype=torch.float32)       # [4] raw (x, y, ẋ, ẏ)
        }


class PendulumCartesianEndpointDataModule(pl.LightningDataModule):
    """Lightning DataModule for Pendulum Cartesian endpoint prediction.

    Supports:
    - Metadata-based loading
    - Separate train/validation/test files
    - Stratified sampling for class balance
    - Parallel trajectory loading
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
        self.state_dim = 4         # (x, y, ẋ, ẏ)
        self.embedded_dim = 4      # Identity embedding (no sin/cos)

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
            print("\n📊 Setting up Pendulum Cartesian datasets...")
            print(f"   Train file: {self.train_file}")
            print(f"   Validation file: {self.validation_file}")
            if self.test_file:
                print(f"   Test file: {self.test_file}")
            if self.use_stratified_sampling:
                print(f"   Using stratified sampling: True")
            print()

            self.train_dataset = PendulumCartesianEndpointDataset(
                self.train_file,
                bounds_file=self.bounds_file
            )
            self.val_dataset = PendulumCartesianEndpointDataset(
                self.validation_file,
                bounds_file=self.bounds_file
            )
            # Also initialize test dataset during fit stage (needed for MAE computation)
            if self.test_file:
                self.test_dataset = PendulumCartesianEndpointDataset(
                    self.test_file,
                    bounds_file=self.bounds_file
                )

        if stage == "test":
            if self.test_file:
                print(f"\n📊 Setting up test dataset: {self.test_file}")
                self.test_dataset = PendulumCartesianEndpointDataset(
                    self.test_file,
                    bounds_file=self.bounds_file
                )

    def train_dataloader(self):
        """Create training dataloader with optional stratified sampling"""
        if self.use_stratified_sampling and hasattr(self.train_dataset, 'labels'):
            # Compute class weights for stratified sampling
            labels = np.array(self.train_dataset.labels)
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Compute balanced class weights: total / (n_classes * count_per_class)
            # This gives minority class weight > 1.0, majority class weight < 1.0
            n_samples = len(labels)
            n_classes = len(unique_labels)
            class_weights = n_samples / (n_classes * counts)

            # Map labels to weights
            weight_dict = dict(zip(unique_labels, class_weights))
            sample_weights = np.array([weight_dict[label] for label in labels])

            # Create weighted sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            print(f"✅ Using stratified sampling:")
            for label, count, weight in zip(unique_labels, counts, class_weights):
                print(f"   Label {label}: {count:,} samples, weight {weight:.6f} ({weight:.2e})")

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

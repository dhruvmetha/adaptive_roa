"""CartPole DeepMind Control Suite Endpoint Dataset

Loads endpoint prediction data for CartPole DM Control using metadata format:
- Metadata file contains: [file_path, start_idx, end_idx, label]
- Trajectory files are loaded and cached in memory
- Start state: state at start_idx (x, θ, ẋ, θ̇)
- End state: state at end_idx (x, θ, ẋ, θ̇)
- CRITICAL: Angle wrapping - θ is wrapped to [-π, π] for S¹ manifold

Key Difference from Regular CartPole:
- Raw data has UNWRAPPED theta (can exceed ±π, up to ±130 rad)
- Must wrap theta using arctan2(sin(θ), cos(θ)) during loading
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning.pytorch as pl
from typing import Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os


def _load_trajectory_file(file_path):
    """Helper function to load a single trajectory file (module-level for multiprocessing)"""
    return file_path, np.loadtxt(file_path, delimiter=",")


class CartPoleDMControlEndpointDataset(Dataset):
    """Dataset for CartPole DM Control endpoint prediction using metadata format.

    Each sample contains:
    - start_state: Initial state [x, θ, ẋ, θ̇] (theta WRAPPED to [-π, π])
    - end_state: Final state [x, θ, ẋ, θ̇] (theta WRAPPED to [-π, π])

    Metadata format: file_path start_idx end_idx label
    """

    def __init__(self, data_file: str):
        """Initialize dataset.

        Args:
            data_file: Path to endpoint metadata file
                      Format: file_path start_idx end_idx label
        """
        self.data_file = data_file

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
        print(f"   State dim: 4 (x, θ, ẋ, θ̇) - theta will be WRAPPED to [-π, π]")

    def wrap_angle(self, angle):
        """
        Wrap angle to [-π, π] for proper S¹ manifold representation

        CRITICAL: Raw data has unwrapped theta (up to ±130 rad)
        This wraps it to the natural [-π, π] range.

        Args:
            angle: Angle in radians (can be unwrapped, e.g., 130 rad)

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

        # Wrap angles in raw states for consistent S¹ manifold representation
        start_state_wrapped = list(start_state)
        end_state_wrapped = list(end_state)
        start_state_wrapped[1] = self.wrap_angle(start_state[1])  # Wrap θ component
        end_state_wrapped[1] = self.wrap_angle(end_state[1])      # Wrap θ component

        return {
            'start_state': torch.tensor(start_state_wrapped, dtype=torch.float32),  # [4] raw with wrapped θ
            'end_state': torch.tensor(end_state_wrapped, dtype=torch.float32)       # [4] raw with wrapped θ
        }


class CartPoleDMControlEndpointDataModule(pl.LightningDataModule):
    """Lightning DataModule for CartPole DM Control endpoint prediction.

    Supports:
    - Metadata-based loading
    - Separate train/validation/test files
    - Stratified sampling for class balance
    - Parallel trajectory loading
    - Angle wrapping for circular manifold (S¹)
    """

    def __init__(
        self,
        train_file: str,
        validation_file: str,
        test_file: Optional[str] = None,
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
            print("\n📊 Setting up CartPole DM Control datasets...")
            print(f"   Train file: {self.train_file}")
            print(f"   Validation file: {self.validation_file}")
            if self.test_file:
                print(f"   Test file: {self.test_file}")
            if self.use_stratified_sampling:
                print(f"   Using stratified sampling: True")
            print()

            self.train_dataset = CartPoleDMControlEndpointDataset(self.train_file)
            self.val_dataset = CartPoleDMControlEndpointDataset(self.validation_file)

            # Also initialize test dataset during fit stage (needed for MAE computation)
            if self.test_file:
                self.test_dataset = CartPoleDMControlEndpointDataset(self.test_file)

        if stage == "test":
            if self.test_file:
                print(f"\n📊 Setting up test dataset: {self.test_file}")
                self.test_dataset = CartPoleDMControlEndpointDataset(self.test_file)

    def train_dataloader(self):
        """Create training dataloader with optional stratified sampling"""
        if self.use_stratified_sampling and hasattr(self.train_dataset, 'labels'):
            # Create weighted sampler for balanced training
            labels = np.array(self.train_dataset.labels)

            # Count samples per class
            unique_labels, class_counts = np.unique(labels, return_counts=True)

            # Compute class weights (inverse frequency)
            class_weights = 1.0 / class_counts

            # Create sample weights
            label_to_weight = {label: weight for label, weight in zip(unique_labels, class_weights)}
            sample_weights = np.array([label_to_weight[label] for label in labels])

            # Create sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            print(f"\n🔀 Using stratified sampling for training:")
            print(f"  Class distribution:")
            for label, count in zip(unique_labels, class_counts):
                print(f"    Label {label}: {count} samples ({count/len(labels)*100:.1f}%)")

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

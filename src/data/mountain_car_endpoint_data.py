"""Mountain Car Endpoint Dataset

Loads endpoint prediction data for Mountain Car using metadata format:
- Metadata file contains: [file_path, start_idx, end_idx, label]
- Trajectory files are loaded and cached in memory
- Start state: state at start_idx
- End state: state at end_idx
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning.pytorch as pl
from typing import Optional
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def _load_trajectory_file(file_path):
    """Helper function to load a single trajectory file (module-level for multiprocessing)"""
    return file_path, np.loadtxt(file_path, delimiter=",")


class MountainCarEndpointDataset(Dataset):
    """Dataset for Mountain Car endpoint prediction using metadata format.

    Each sample contains:
    - start_state: Initial state [position, velocity]
    - end_state: Final state [position, velocity]

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
        if bounds_file:
            with open(bounds_file, 'rb') as f:
                self.bounds_data = pickle.load(f)
        else:
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
        print(f"   State dim: 2 (position, velocity)")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        """Get a single endpoint pair.

        Returns:
            Dictionary with 'start_state' and 'end_state' tensors
        """
        file_path, start_idx, end_idx = self.metadata[idx]

        # Look up trajectory in cache
        trajectory = self.trajectory_cache[file_path]

        # Extract start and end states
        start_state = trajectory[start_idx]
        end_state = trajectory[end_idx]

        return {
            'start_state': torch.tensor(start_state, dtype=torch.float32),  # [2] raw state
            'end_state': torch.tensor(end_state, dtype=torch.float32)       # [2] raw state
        }


class MountainCarEndpointDataModule(pl.LightningDataModule):
    """Lightning DataModule for Mountain Car endpoint data."""

    def __init__(
        self,
        data_file: str,
        validation_file: str,
        test_file: Optional[str] = None,
        batch_size: int = 256,
        val_batch_size: int = 1024,
        num_workers: int = 4,
        pin_memory: bool = True,
        bounds_file: Optional[str] = None,
        use_stratified_sampling: bool = False
    ):
        """Initialize DataModule.

        Args:
            data_file: Path to training metadata file
            validation_file: Path to validation metadata file
            test_file: Path to test metadata file (optional)
            batch_size: Training batch size
            val_batch_size: Validation/test batch size
            num_workers: Number of dataloader workers
            pin_memory: Whether to pin memory for GPU transfer
            bounds_file: Path to bounds pickle file
            use_stratified_sampling: Whether to use WeightedRandomSampler for balanced training
        """
        super().__init__()
        self.data_file = data_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.bounds_file = bounds_file
        self.use_stratified_sampling = use_stratified_sampling

        # Mountain Car dimensions
        self.state_dim = 2
        self.embedded_dim = 2  # No embedding for pure Euclidean

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""
        if stage == "fit" or stage is None:
            self.train_dataset = MountainCarEndpointDataset(
                self.data_file,
                self.bounds_file
            )
            self.val_dataset = MountainCarEndpointDataset(
                self.validation_file,
                self.bounds_file
            )

        if stage == "test" or stage is None:
            if self.test_file:
                self.test_dataset = MountainCarEndpointDataset(
                    self.test_file,
                    self.bounds_file
                )

    def train_dataloader(self):
        """Create training dataloader with optional stratified sampling."""
        if self.use_stratified_sampling:
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
                persistent_workers=True if self.num_workers > 0 else False
            )

        # Default: shuffle without stratification
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        """Create test dataloader."""
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=True if self.num_workers > 0 else False
            )
        return None

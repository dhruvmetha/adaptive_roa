import torch
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional
from tqdm import tqdm
import os
import lightning.pytorch as pl
import random
from concurrent.futures import ProcessPoolExecutor


def _load_trajectory_file(file_path):
    """Helper function to load a single trajectory file (module-level for multiprocessing)"""
    return file_path, np.loadtxt(file_path, delimiter=",")


class HumanoidEndpointDataset(Dataset):
    def __init__(self, data_file: str,
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/humanoid_get_up/humanoid_data_bounds.pkl"):
        """
        Dataset for humanoid endpoint pairs (start_state, end_state)
        Handles 67D humanoid state with Sphere manifold for orientation

        Manifold: ℝ³⁴ × S² × ℝ³⁰
        State format: [euclidean1(34), sphere_x, sphere_y, sphere_z, euclidean2(30)]

        Args:
            data_file: Path to endpoint metadata file
            bounds_file: Path to pickle file with actual data bounds
        """
        self.bounds_file = bounds_file

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
                
        self.metadata = self.metadata
        self.labels = self.labels
        print(f"Loaded {len(self.metadata)} endpoint metadata entries")
        print(f"Loading {len(unique_files)} unique trajectory files...")

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

        print(f"Cached {len(self.trajectory_cache)} trajectories in memory")


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        file_path, start_idx, end_idx = self.metadata[idx]

        # Look up trajectory in cache
        trajectory = self.trajectory_cache[file_path]

        # Extract start and end states
        start_state = trajectory[start_idx]
        end_state = trajectory[end_idx]

        return {
            'start_state': torch.tensor(start_state, dtype=torch.float32),  # [67] raw state
            'end_state': torch.tensor(end_state, dtype=torch.float32)       # [67] raw state
        }


class HumanoidEndpointDataModule(pl.LightningDataModule):
    def __init__(self, data_file: str, validation_file: str, test_file: str,
                 batch_size: int = 64, val_batch_size: Optional[int] = None,
                 num_workers: int = 4, pin_memory: bool = True,
                 bounds_file: str = "/common/users/dm1487/arcmg_datasets/humanoid_get_up/humanoid_data_bounds.pkl",
                 use_stratified_sampling: bool = False):
        """
        Humanoid Endpoint Data Module with separate train/val/test files

        Args:
            data_file: Path to training dataset metadata file
            validation_file: Path to validation dataset metadata file
            test_file: Path to test dataset metadata file
            batch_size: Batch size for training data loader
            val_batch_size: Batch size for validation/test data loaders (defaults to batch_size if None)
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for data loaders
            bounds_file: Path to pickle file with actual data bounds
            use_stratified_sampling: Whether to use WeightedRandomSampler for balanced training
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
        self.use_stratified_sampling = use_stratified_sampling

        # Humanoid-specific dimensions
        self.state_dim = 67  # Raw state: [euclidean1(34), sphere(3), euclidean2(30)]
        self.embedded_dim = 67  # Embedded: same as state (sphere already continuous)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = HumanoidEndpointDataset(
                self.data_file,
                bounds_file=self.bounds_file
            )
            self.val_dataset = HumanoidEndpointDataset(
                self.validation_file,
                bounds_file=self.bounds_file
            )
            # Also initialize test dataset during fit stage (needed for MAE computation)
            if self.test_file:
                self.test_dataset = HumanoidEndpointDataset(
                    self.test_file,
                    bounds_file=self.bounds_file
                )

        if stage == "test":
            if self.test_file:
                self.test_dataset = HumanoidEndpointDataset(
                    self.test_file,
                    bounds_file=self.bounds_file
                )

    def train_dataloader(self):
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
                pin_memory=self.pin_memory
            )

        # Default: shuffle without stratification
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

"""
CartPole Classification Data Module

Loads trajectories from shuffled indices files and extracts initial states
with corresponding success/failure labels for classification.

Uses proper manifold embedding:
- x: normalized to [-1, 1]
- theta: embedded as (sin(theta), cos(theta)) for S1 manifold
- x_dot, theta_dot: normalized to [-1, 1]

Input dimension: 5 (x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm)
"""

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List
from pathlib import Path
import torch
import lightning.pytorch as pl
import numpy as np


class CartPoleClassificationDataset(Dataset):
    """Dataset for CartPole classification from trajectory files with manifold embedding."""
    
    def __init__(
        self,
        trajectory_files: List[str],
        labels: List[int],
        trajectories_dir: str,
    ):
        """
        Args:
            trajectory_files: List of trajectory filenames
            labels: List of corresponding labels (0=failure, 1=success)
            trajectories_dir: Path to directory containing trajectory files
        """
        self.trajectory_files = trajectory_files
        self.labels = labels
        self.trajectories_dir = Path(trajectories_dir)
        
        # CartPole state bounds from dataset_description.json
        self.x_limit = 6.0
        self.x_dot_limit = 5.0
        self.theta_dot_limit = 5.0
        
        # Cache for loaded initial states
        self._cache = {}
        
    def __len__(self):
        return len(self.trajectory_files)
    
    def _load_initial_state(self, filename: str) -> np.ndarray:
        """Load the initial state (first line) from a trajectory file."""
        if filename in self._cache:
            return self._cache[filename]
            
        filepath = self.trajectories_dir / filename
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        
        # Parse comma-separated values: x,theta,x_dot,theta_dot
        state = np.array([float(x) for x in first_line.split(',')], dtype=np.float32)
        self._cache[filename] = state
        return state
    
    def _embed_state(self, state: np.ndarray) -> np.ndarray:
        """
        Embed state with proper manifold representation.
        
        Input: [x, theta, x_dot, theta_dot] (4D)
        Output: [x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm] (5D)
        """
        x, theta, x_dot, theta_dot = state
        
        # Normalize to [-1, 1]
        x_norm = x / self.x_limit
        x_dot_norm = x_dot / self.x_dot_limit
        theta_dot_norm = theta_dot / self.theta_dot_limit
        
        # Embed theta on S1 manifold using sin/cos
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        return np.array([x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm], dtype=np.float32)
    
    def __getitem__(self, idx):
        filename = self.trajectory_files[idx]
        label = self.labels[idx]
        
        # Load initial state
        state = self._load_initial_state(filename)
        
        # Embed with manifold-aware representation
        embedded = self._embed_state(state)
        
        return {
            "inputs": torch.from_numpy(embedded).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }


class CartPoleEvalDataset(Dataset):
    """Dataset for CartPole classification evaluation from eval_states.txt with manifold embedding."""
    
    def __init__(self, eval_file: str):
        self.data = []  # Original states
        self.labels = []
        
        # CartPole state bounds
        self.x_limit = 6.0
        self.x_dot_limit = 5.0
        self.theta_dot_limit = 5.0
        
        # Load data
        with open(eval_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 9:
                    state = np.array([float(parts[i]) for i in range(4)], dtype=np.float32)
                    label = int(float(parts[8]))
                    self.data.append(state)
                    self.labels.append(label)
        
        print(f"Loaded {len(self.data)} eval samples")
        print(f"  Success (1): {sum(self.labels)}")
        print(f"  Failure (0): {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def _embed_state(self, state: np.ndarray) -> np.ndarray:
        x, theta, x_dot, theta_dot = state
        x_norm = x / self.x_limit
        x_dot_norm = x_dot / self.x_dot_limit
        theta_dot_norm = theta_dot / self.theta_dot_limit
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        return np.array([x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm], dtype=np.float32)
    
    def __getitem__(self, idx):
        state = self.data[idx]
        label = self.labels[idx]
        embedded = self._embed_state(state)
        
        return {
            "inputs": torch.from_numpy(embedded).float(),
            "original_state": torch.from_numpy(state).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }


class CartPoleClassificationDataModule(pl.LightningDataModule):
    """Lightning DataModule for CartPole classification with manifold embedding and sample balancing."""
    
    def __init__(
        self,
        shuffled_indices_file: str,
        shuffled_labels_file: str,
        trajectories_dir: str,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_split: float = 0.95,
        val_split: float = 0.05,
        max_samples: Optional[int] = None,
        balance_samples: bool = False,  # Whether to balance training samples
    ):
        super().__init__()
        self.shuffled_indices_file = shuffled_indices_file
        self.shuffled_labels_file = shuffled_labels_file
        self.trajectories_dir = trajectories_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split
        self.max_samples = max_samples
        self.balance_samples = balance_samples
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.trajectory_files = []
        self.labels = []
        
    def prepare_data(self):
        """Load indices and labels from files."""
        with open(self.shuffled_indices_file, 'r') as f:
            self.trajectory_files = [line.strip() for line in f.readlines()]
        
        with open(self.shuffled_labels_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f.readlines()]
        
        assert len(self.trajectory_files) == len(self.labels)
        
        # Limit samples if specified
        if self.max_samples is not None:
            self.trajectory_files = self.trajectory_files[:self.max_samples]
            self.labels = self.labels[:self.max_samples]
        
        print(f"Loaded {len(self.trajectory_files)} samples")
        print(f"  Success (1): {sum(self.labels)}")
        print(f"  Failure (0): {len(self.labels) - sum(self.labels)}")
        
    def setup(self, stage: Optional[str] = None):
        """Split data and create datasets."""
        n = len(self.trajectory_files)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))
        
        if stage == "fit" or stage is None:
            train_files = self.trajectory_files[:train_end]
            train_labels = self.labels[:train_end]
            
            self.train_dataset = CartPoleClassificationDataset(
                trajectory_files=train_files,
                labels=train_labels,
                trajectories_dir=self.trajectories_dir,
            )
            
            # Compute sample weights for balanced sampling
            if self.balance_samples:
                label_counts = np.bincount(train_labels)
                # Weight inversely proportional to class frequency
                class_weights = 1.0 / label_counts
                self.sample_weights = [class_weights[label] for label in train_labels]
                print(f"Sample balancing enabled:")
                print(f"  Class 0 (failure) weight: {class_weights[0]:.6f}")
                print(f"  Class 1 (success) weight: {class_weights[1]:.6f}")
            
            self.val_dataset = CartPoleClassificationDataset(
                trajectory_files=self.trajectory_files[train_end:val_end],
                labels=self.labels[train_end:val_end],
                trajectories_dir=self.trajectories_dir,
            )
            
            print(f"Train: {len(self.train_dataset)} samples")
            print(f"Val: {len(self.val_dataset)} samples")
            
        if stage == "test" or stage is None:
            self.test_dataset = CartPoleClassificationDataset(
                trajectory_files=self.trajectory_files[val_end:],
                labels=self.labels[val_end:],
                trajectories_dir=self.trajectories_dir,
            )
            print(f"Test: {len(self.test_dataset)} samples")
    
    def train_dataloader(self):
        if self.balance_samples and hasattr(self, 'sample_weights'):
            # Use WeightedRandomSampler for balanced batches
            sampler = WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
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

"""
Quadrotor 2D All-Points Classification Data Module

Loads pre-built classification datasets where every point on a trajectory
has the same success/failure label as the trajectory outcome.

State space (6D input, 7D output):
- Position: x, z (normalized to [-1, 1])
- Orientation: theta → sin(theta), cos(theta) (S¹ manifold embedding)
- Linear velocity: x_dot, z_dot (normalized to [-1, 1])
- Angular velocity: theta_dot (normalized to [-1, 1])

Input dimension: 6D raw state
Output dimension: 7D embedded state (with sin/cos for theta)
"""

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List
from pathlib import Path
import torch
import lightning.pytorch as pl
import numpy as np


class Quadrotor2DAllPointsDataset(Dataset):
    """Dataset for Quadrotor 2D classification from pre-built all-points files."""

    # State bounds from dataset_description.json
    X_LIMIT = 1.0
    Z_MIN = 0.1
    Z_MAX = 1.5

    X_DOT_LIMIT = 1.0
    Z_DOT_LIMIT = 1.0

    # Angular velocity limit (achieved bounds ~±12.7 rad/s)
    THETA_DOT_LIMIT = 12.0

    def __init__(self, data: List[str]):
        """
        Args:
            data: List of data lines (space-separated: 6 state values + 1 label)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def _embed_state(self, state: np.ndarray) -> np.ndarray:
        """
        Embed state with normalization and S¹ manifold embedding for theta.

        Input: [x, z, theta, x_dot, z_dot, theta_dot] (6D)
        Output: [x_norm, z_norm, sin(theta), cos(theta), x_dot_norm, z_dot_norm, theta_dot_norm] (7D)
        """
        x, z, theta, x_dot, z_dot, theta_dot = state

        # Normalize x to [-1, 1]
        x_norm = x / self.X_LIMIT

        # Normalize z from [0.1, 1.5] to [-1, 1]
        z_center = (self.Z_MAX + self.Z_MIN) / 2  # 0.8
        z_range = (self.Z_MAX - self.Z_MIN) / 2   # 0.7
        z_norm = (z - z_center) / z_range

        # S¹ manifold embedding for theta (pitch angle)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Normalize velocities to [-1, 1]
        x_dot_norm = x_dot / self.X_DOT_LIMIT
        z_dot_norm = z_dot / self.Z_DOT_LIMIT

        # Normalize angular velocity to [-1, 1]
        theta_dot_norm = theta_dot / self.THETA_DOT_LIMIT

        return np.array([
            x_norm, z_norm,
            sin_theta, cos_theta,
            x_dot_norm, z_dot_norm, theta_dot_norm
        ], dtype=np.float32)

    def __getitem__(self, idx):
        line = self.data[idx]

        # Parse space-separated values: 6 state values + 1 label
        values = [float(x) for x in line.strip().split() if x != ""]
        state = np.array(values[:6], dtype=np.float32)

        # Label: 1 = success, -1 = failure
        # Convert to 0/1 for binary cross-entropy (1 = success, 0 = failure)
        raw_label = int(values[6])
        label = 1.0 if raw_label == 1 else 0.0

        # Embed with normalization and sin/cos
        embedded = self._embed_state(state)

        return {
            "inputs": torch.from_numpy(embedded).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }


class Quadrotor2DAllPointsDataModule(pl.LightningDataModule):
    """Lightning DataModule for Quadrotor 2D all-points classification."""

    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: Optional[str] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_samples: Optional[int] = None,
        balance_samples: bool = False,
    ):
        """
        Args:
            train_file: Path to training dataset file
            val_file: Path to validation dataset file
            test_file: Path to test dataset file (optional)
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            pin_memory: Whether to pin memory
            max_samples: Maximum samples per split (None = use all)
            balance_samples: Whether to balance classes during training
        """
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_samples = max_samples
        self.balance_samples = balance_samples

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_data = []
        self.val_data = []
        self.test_data = []

    def _load_file(self, filepath: str, max_samples: Optional[int] = None) -> List[str]:
        """Load data from file."""
        with open(filepath, 'r') as f:
            data = f.readlines()

        if max_samples is not None:
            data = data[:max_samples]

        return data

    def _count_labels(self, data: List[str]) -> dict:
        """Count success/failure labels in data."""
        success = 0
        failure = 0
        for line in data:
            values = [float(x) for x in line.strip().split() if x != ""]
            label = int(values[6])
            if label == 1:
                success += 1
            else:
                failure += 1
        return {"success": success, "failure": failure}

    def prepare_data(self):
        """Load data from files."""
        print("Loading Quadrotor 2D all-points classification data...")

        self.train_data = self._load_file(self.train_file, self.max_samples)
        self.val_data = self._load_file(self.val_file, self.max_samples)

        if self.test_file:
            self.test_data = self._load_file(self.test_file, self.max_samples)

        # Print statistics
        train_counts = self._count_labels(self.train_data)
        val_counts = self._count_labels(self.val_data)

        print(f"  Train: {len(self.train_data)} samples "
              f"(success: {train_counts['success']}, failure: {train_counts['failure']})")
        print(f"  Val: {len(self.val_data)} samples "
              f"(success: {val_counts['success']}, failure: {val_counts['failure']})")

        if self.test_file:
            test_counts = self._count_labels(self.test_data)
            print(f"  Test: {len(self.test_data)} samples "
                  f"(success: {test_counts['success']}, failure: {test_counts['failure']})")

    def setup(self, stage: Optional[str] = None):
        """Create datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = Quadrotor2DAllPointsDataset(data=self.train_data)

            # Compute sample weights for balanced sampling
            if self.balance_samples:
                labels = []
                for line in self.train_data:
                    values = [float(x) for x in line.strip().split() if x != ""]
                    raw_label = int(values[6])
                    labels.append(1 if raw_label == 1 else 0)

                label_counts = np.bincount(labels)
                if len(label_counts) < 2:
                    print("WARNING: Only one class in training data, cannot balance!")
                    self.balance_samples = False
                else:
                    class_weights = 1.0 / label_counts
                    self.sample_weights = [class_weights[label] for label in labels]
                    print(f"Sample balancing enabled:")
                    print(f"  Class 0 (failure) count: {label_counts[0]}, weight: {class_weights[0]:.6f}")
                    print(f"  Class 1 (success) count: {label_counts[1]}, weight: {class_weights[1]:.6f}")

            self.val_dataset = Quadrotor2DAllPointsDataset(data=self.val_data)

        if (stage == "test" or stage is None) and self.test_file:
            self.test_dataset = Quadrotor2DAllPointsDataset(data=self.test_data)

    def train_dataloader(self):
        if self.balance_samples and hasattr(self, 'sample_weights'):
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
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

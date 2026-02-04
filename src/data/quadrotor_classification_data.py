"""
Quadrotor Classification Data Module.

Loads expanded classification datasets with (state, label) pairs.
Applies system-specific embedding (normalization + manifold embedding).

State representations:
- Quadrotor 2D: 6D raw -> 7D embedded (sin/cos for theta)
- Quadrotor 3D: 13D raw -> 13D (quaternion kept as-is, positions/velocities normalized)
"""

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List, Literal
from pathlib import Path
import torch
import lightning.pytorch as pl
import numpy as np


# =============================================================================
# Embedders
# =============================================================================

class Quadrotor2DEmbedder:
    """
    Embed Quadrotor 2D states with normalization and S^1 manifold embedding.

    Input:  [x, z, theta, x_dot, z_dot, theta_dot] (6D)
    Output: [x_norm, z_norm, sin(theta), cos(theta), x_dot_norm, z_dot_norm, theta_dot_norm] (7D)
    """
    # Normalization bounds (from dataset_description.json)
    X_LIMIT = 1.0
    Z_MIN = 0.1
    Z_MAX = 1.5
    X_DOT_LIMIT = 1.2  # Achieved bounds ~1.22
    Z_DOT_LIMIT = 1.5  # Achieved bounds ~1.31
    THETA_DOT_LIMIT = 13.0  # Achieved bounds ~12.76

    INPUT_DIM = 6
    OUTPUT_DIM = 7

    @classmethod
    def embed(cls, state: np.ndarray) -> np.ndarray:
        """Embed a single state."""
        x, z, theta, x_dot, z_dot, theta_dot = state

        # Normalize x to [-1, 1]
        x_norm = np.clip(x / cls.X_LIMIT, -1.0, 1.0)

        # Normalize z from [0.1, 1.5] to [-1, 1]
        z_center = (cls.Z_MAX + cls.Z_MIN) / 2  # 0.8
        z_range = (cls.Z_MAX - cls.Z_MIN) / 2   # 0.7
        z_norm = np.clip((z - z_center) / z_range, -1.0, 1.0)

        # S^1 manifold embedding for theta
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Normalize velocities to [-1, 1]
        x_dot_norm = np.clip(x_dot / cls.X_DOT_LIMIT, -1.0, 1.0)
        z_dot_norm = np.clip(z_dot / cls.Z_DOT_LIMIT, -1.0, 1.0)
        theta_dot_norm = np.clip(theta_dot / cls.THETA_DOT_LIMIT, -1.0, 1.0)

        return np.array([
            x_norm, z_norm,
            sin_theta, cos_theta,
            x_dot_norm, z_dot_norm, theta_dot_norm
        ], dtype=np.float32)

    @classmethod
    def embed_batch(cls, states: np.ndarray) -> np.ndarray:
        """Embed a batch of states."""
        return np.array([cls.embed(s) for s in states], dtype=np.float32)


class Quadrotor3DEmbedder:
    """
    Embed Quadrotor 3D states with normalization.

    Input:  [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r] (13D)
    Output: [x_n, y_n, z_n, qw, qx, qy, qz, xd_n, yd_n, zd_n, p_n, q_n, r_n] (13D)

    Quaternion is kept as unit quaternion (already normalized).
    Positions and velocities are normalized to [-1, 1].
    """
    # Normalization bounds (from dataset_description.json)
    POS_LIMIT = 2.5      # x, y, z bounds
    VEL_LIMIT = 3.0      # x_dot, y_dot, z_dot bounds
    ANGULAR_VEL_LIMIT = 25.0  # p, q, r bounds

    INPUT_DIM = 13
    OUTPUT_DIM = 13

    @classmethod
    def embed(cls, state: np.ndarray) -> np.ndarray:
        """Embed a single state."""
        x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r = state

        # Normalize positions
        x_norm = np.clip(x / cls.POS_LIMIT, -1.0, 1.0)
        y_norm = np.clip(y / cls.POS_LIMIT, -1.0, 1.0)
        z_norm = np.clip(z / cls.POS_LIMIT, -1.0, 1.0)

        # Quaternion: keep as-is (already unit quaternion)
        # Optionally normalize to ensure unit norm
        quat_norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if quat_norm > 1e-6:
            qw, qx, qy, qz = qw/quat_norm, qx/quat_norm, qy/quat_norm, qz/quat_norm

        # Normalize velocities
        x_dot_norm = np.clip(x_dot / cls.VEL_LIMIT, -1.0, 1.0)
        y_dot_norm = np.clip(y_dot / cls.VEL_LIMIT, -1.0, 1.0)
        z_dot_norm = np.clip(z_dot / cls.VEL_LIMIT, -1.0, 1.0)

        # Normalize angular velocities
        p_norm = np.clip(p / cls.ANGULAR_VEL_LIMIT, -1.0, 1.0)
        q_norm = np.clip(q / cls.ANGULAR_VEL_LIMIT, -1.0, 1.0)
        r_norm = np.clip(r / cls.ANGULAR_VEL_LIMIT, -1.0, 1.0)

        return np.array([
            x_norm, y_norm, z_norm,
            qw, qx, qy, qz,
            x_dot_norm, y_dot_norm, z_dot_norm,
            p_norm, q_norm, r_norm
        ], dtype=np.float32)

    @classmethod
    def embed_batch(cls, states: np.ndarray) -> np.ndarray:
        """Embed a batch of states."""
        return np.array([cls.embed(s) for s in states], dtype=np.float32)


EMBEDDERS = {
    'quadrotor2d': Quadrotor2DEmbedder,
    'quadrotor3d': Quadrotor3DEmbedder,
}


# =============================================================================
# Dataset
# =============================================================================

class QuadrotorClassificationDataset(Dataset):
    """Dataset for quadrotor classification from expanded (state, label) files."""

    def __init__(
        self,
        data: List[str],
        system: Literal['quadrotor2d', 'quadrotor3d'],
    ):
        """
        Args:
            data: List of data lines (space-separated: state values + label)
            system: System type for embedding
        """
        self.data = data
        self.system = system
        self.embedder = EMBEDDERS[system]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]

        # Parse space-separated values
        values = [float(x) for x in line.strip().split()]
        state = np.array(values[:-1], dtype=np.float32)
        label = float(values[-1])  # 0 or 1

        # Embed state
        embedded = self.embedder.embed(state)

        return {
            "inputs": torch.from_numpy(embedded).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }


# =============================================================================
# Data Module
# =============================================================================

class QuadrotorClassificationDataModule(pl.LightningDataModule):
    """Lightning DataModule for quadrotor classification."""

    def __init__(
        self,
        system: Literal['quadrotor2d', 'quadrotor3d'],
        train_file: str,
        val_file: str,
        test_file: Optional[str] = None,
        batch_size: int = 512,
        num_workers: int = 4,
        pin_memory: bool = True,
        balance_samples: bool = True,
    ):
        """
        Args:
            system: System type ('quadrotor2d' or 'quadrotor3d')
            train_file: Path to training dataset file
            val_file: Path to validation dataset file
            test_file: Path to test dataset file (optional)
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            pin_memory: Whether to pin memory
            balance_samples: Whether to use weighted sampling for class balance
        """
        super().__init__()
        self.system = system
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.balance_samples = balance_samples

        self.embedder = EMBEDDERS[system]
        self.input_dim = self.embedder.OUTPUT_DIM

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.sample_weights = None

    def _load_file(self, filepath: str) -> List[str]:
        """Load data from file."""
        with open(filepath, 'r') as f:
            return f.readlines()

    def _count_labels(self, data: List[str]) -> dict:
        """Count success/failure labels in data."""
        success = 0
        failure = 0
        for line in data:
            values = [float(x) for x in line.strip().split()]
            label = int(values[-1])
            if label == 1:
                success += 1
            else:
                failure += 1
        return {"success": success, "failure": failure}

    def prepare_data(self):
        """Load data from files."""
        print(f"\nLoading {self.system} classification data...")

        self.train_data = self._load_file(self.train_file)
        self.val_data = self._load_file(self.val_file)

        if self.test_file:
            self.test_data = self._load_file(self.test_file)

        # Print statistics
        train_counts = self._count_labels(self.train_data)
        val_counts = self._count_labels(self.val_data)

        print(f"  Train: {len(self.train_data):,} samples "
              f"(success: {train_counts['success']:,}, failure: {train_counts['failure']:,})")
        print(f"  Val: {len(self.val_data):,} samples "
              f"(success: {val_counts['success']:,}, failure: {val_counts['failure']:,})")

        if self.test_file:
            test_counts = self._count_labels(self.test_data)
            print(f"  Test: {len(self.test_data):,} samples "
                  f"(success: {test_counts['success']:,}, failure: {test_counts['failure']:,})")

        print(f"  Input dim (after embedding): {self.input_dim}")

    def setup(self, stage: Optional[str] = None):
        """Create datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = QuadrotorClassificationDataset(
                data=self.train_data,
                system=self.system,
            )

            # Compute sample weights for balanced sampling
            if self.balance_samples:
                labels = []
                for line in self.train_data:
                    values = [float(x) for x in line.strip().split()]
                    labels.append(int(values[-1]))

                label_counts = np.bincount(labels)
                if len(label_counts) >= 2 and label_counts[0] > 0 and label_counts[1] > 0:
                    class_weights = 1.0 / label_counts
                    self.sample_weights = [class_weights[label] for label in labels]
                    print(f"\nSample balancing enabled:")
                    print(f"  Class 0 (failure): {label_counts[0]:,}, weight: {class_weights[0]:.6f}")
                    print(f"  Class 1 (success): {label_counts[1]:,}, weight: {class_weights[1]:.6f}")
                else:
                    print("WARNING: Cannot balance - need both classes in training data")
                    self.balance_samples = False

            self.val_dataset = QuadrotorClassificationDataset(
                data=self.val_data,
                system=self.system,
            )

        if (stage == "test" or stage is None) and self.test_file:
            self.test_dataset = QuadrotorClassificationDataset(
                data=self.test_data,
                system=self.system,
            )

    def train_dataloader(self):
        if self.balance_samples and self.sample_weights is not None:
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
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

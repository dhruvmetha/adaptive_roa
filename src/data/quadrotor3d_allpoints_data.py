"""
Quadrotor 3D All-Points Classification Data Module

Loads pre-built classification datasets where every point on a trajectory
has the same success/failure label as the trajectory outcome.

State space (13D input, 13D output):
- Position: x, y, z (normalized to [-1, 1])
- Orientation: qw, qx, qy, qz (unit quaternion, kept as-is)
- Linear velocity: x_dot, y_dot, z_dot (normalized to [-1, 1])
- Angular velocity: p, q, r (normalized to [-1, 1])

Input/output dimension: 13 (quaternion kept as unit quaternion)
"""

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List
from pathlib import Path
import torch
import lightning.pytorch as pl
import numpy as np


class Quadrotor3DAllPointsDataset(Dataset):
    """Dataset for Quadrotor 3D classification from pre-built all-points files."""

    # State bounds from dataset_description.json
    X_LIMIT = 1.8
    Y_LIMIT = 1.8
    Z_MIN = 0.1
    Z_MAX = 3.0

    X_DOT_LIMIT = 3.0
    Y_DOT_LIMIT = 3.0
    Z_DOT_LIMIT = 3.0

    # Angular velocity limits (achieved bounds ~±40 rad/s)
    P_LIMIT = 40.0
    Q_LIMIT = 40.0
    R_LIMIT = 40.0

    def __init__(
        self,
        data: List[str],
        check_quaternion_norm: bool = True,
    ):
        """
        Args:
            data: List of data lines (space-separated: 13 state values + 1 label)
            check_quaternion_norm: Whether to verify quaternion is unit norm
        """
        self.data = data
        self.check_quaternion_norm = check_quaternion_norm

        # Track quaternion norm statistics
        self._quat_norm_min = float('inf')
        self._quat_norm_max = float('-inf')
        self._quat_norm_violations = 0

    def __len__(self):
        return len(self.data)

    def _check_quaternion(self, quat: np.ndarray) -> bool:
        """Check if quaternion is unit norm (within tolerance)."""
        norm = np.linalg.norm(quat)
        self._quat_norm_min = min(self._quat_norm_min, norm)
        self._quat_norm_max = max(self._quat_norm_max, norm)

        if abs(norm - 1.0) > 0.01:
            self._quat_norm_violations += 1
            return False
        return True

    def _embed_state(self, state: np.ndarray) -> np.ndarray:
        """
        Embed state with normalization for Euclidean components.
        Quaternion is kept as-is (already unit norm).

        Input: [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r] (13D)
        Output: [x_norm, y_norm, z_norm, qw, qx, qy, qz, x_dot_norm, y_dot_norm, z_dot_norm, p_norm, q_norm, r_norm] (13D)
        """
        x, y, z = state[0], state[1], state[2]
        qw, qx, qy, qz = state[3], state[4], state[5], state[6]
        x_dot, y_dot, z_dot = state[7], state[8], state[9]
        p, q, r = state[10], state[11], state[12]

        # Check quaternion if enabled
        if self.check_quaternion_norm:
            self._check_quaternion(np.array([qw, qx, qy, qz]))

        # Normalize position to [-1, 1]
        x_norm = x / self.X_LIMIT
        y_norm = y / self.Y_LIMIT
        # z has asymmetric bounds [0.1, 3.0], normalize to [-1, 1]
        z_center = (self.Z_MAX + self.Z_MIN) / 2  # 1.55
        z_range = (self.Z_MAX - self.Z_MIN) / 2   # 1.45
        z_norm = (z - z_center) / z_range

        # Normalize velocities to [-1, 1]
        x_dot_norm = x_dot / self.X_DOT_LIMIT
        y_dot_norm = y_dot / self.Y_DOT_LIMIT
        z_dot_norm = z_dot / self.Z_DOT_LIMIT

        # Normalize angular velocities to [-1, 1]
        p_norm = p / self.P_LIMIT
        q_norm = q / self.Q_LIMIT
        r_norm = r / self.R_LIMIT

        # Quaternion is NOT normalized - kept as unit quaternion
        return np.array([
            x_norm, y_norm, z_norm,
            qw, qx, qy, qz,
            x_dot_norm, y_dot_norm, z_dot_norm,
            p_norm, q_norm, r_norm
        ], dtype=np.float32)

    def __getitem__(self, idx):
        line = self.data[idx]

        # Parse space-separated values: 13 state values + 1 label
        values = [float(x) for x in line.strip().split() if x != ""]
        state = np.array(values[:13], dtype=np.float32)

        # Label: 1 = success, -1 = failure
        # Convert to 0/1 for binary cross-entropy (1 = success, 0 = failure)
        raw_label = int(values[13])
        label = 1.0 if raw_label == 1 else 0.0

        # Embed with normalization
        embedded = self._embed_state(state)

        return {
            "inputs": torch.from_numpy(embedded).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }

    def get_quaternion_stats(self) -> dict:
        """Return quaternion norm statistics."""
        return {
            "min_norm": self._quat_norm_min,
            "max_norm": self._quat_norm_max,
            "violations": self._quat_norm_violations,
        }


class Quadrotor3DAllPointsDataModule(pl.LightningDataModule):
    """Lightning DataModule for Quadrotor 3D all-points classification."""

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
        check_quaternion_norm: bool = True,
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
            check_quaternion_norm: Whether to verify quaternion norms
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
        self.check_quaternion_norm = check_quaternion_norm

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
            label = int(values[13])
            if label == 1:
                success += 1
            else:
                failure += 1
        return {"success": success, "failure": failure}

    def prepare_data(self):
        """Load data from files."""
        print("Loading Quadrotor 3D all-points classification data...")

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
            self.train_dataset = Quadrotor3DAllPointsDataset(
                data=self.train_data,
                check_quaternion_norm=self.check_quaternion_norm,
            )

            # Compute sample weights for balanced sampling
            if self.balance_samples:
                labels = []
                for line in self.train_data:
                    values = [float(x) for x in line.strip().split() if x != ""]
                    raw_label = int(values[13])
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

            self.val_dataset = Quadrotor3DAllPointsDataset(
                data=self.val_data,
                check_quaternion_norm=self.check_quaternion_norm,
            )

        if (stage == "test" or stage is None) and self.test_file:
            self.test_dataset = Quadrotor3DAllPointsDataset(
                data=self.test_data,
                check_quaternion_norm=self.check_quaternion_norm,
            )

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

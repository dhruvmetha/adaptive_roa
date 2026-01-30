"""
Quadrotor 3D Classification Data Module

Loads trajectories from shuffled indices files and extracts initial states
with corresponding success/failure labels for classification.

State space (13D):
- Position: x, y, z (normalized to [-1, 1] using bounds)
- Orientation: qw, qx, qy, qz (unit quaternion, NOT normalized - already unit norm)
- Linear velocity: x_dot, y_dot, z_dot (normalized to [-1, 1])
- Angular velocity: p, q, r (normalized to [-1, 1])

Input dimension: 13 (all components, quaternion kept as-is)
"""

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List
from pathlib import Path
import torch
import lightning.pytorch as pl
import numpy as np


class Quadrotor3DClassificationDataset(Dataset):
    """Dataset for Quadrotor 3D classification from trajectory files."""

    def __init__(
        self,
        trajectory_files: List[str],
        labels: List[int],
        trajectories_dir: str,
        check_quaternion_norm: bool = True,
    ):
        """
        Args:
            trajectory_files: List of trajectory filenames
            labels: List of corresponding labels (0=failure, 1=success)
            trajectories_dir: Path to directory containing trajectory files
            check_quaternion_norm: Whether to verify quaternion is unit norm
        """
        self.trajectory_files = trajectory_files
        self.labels = labels
        self.trajectories_dir = Path(trajectories_dir)
        self.check_quaternion_norm = check_quaternion_norm

        # Quadrotor 3D state bounds from dataset_description.json
        # Position bounds
        self.x_limit = 1.8
        self.y_limit = 1.8
        self.z_min = 0.1
        self.z_max = 3.0

        # Velocity bounds
        self.x_dot_limit = 3.0
        self.y_dot_limit = 3.0
        self.z_dot_limit = 3.0

        # Angular velocity bounds (can exceed during tumbling, use achieved bounds)
        self.p_limit = 40.0  # Achieved bounds show ~±40 rad/s
        self.q_limit = 40.0
        self.r_limit = 40.0

        # Cache for loaded initial states
        self._cache = {}

        # Track quaternion norm statistics
        self._quat_norm_min = float('inf')
        self._quat_norm_max = float('-inf')
        self._quat_norm_violations = 0

    def __len__(self):
        return len(self.trajectory_files)

    def _load_initial_state(self, filename: str) -> np.ndarray:
        """Load the initial state (first line) from a trajectory file."""
        if filename in self._cache:
            return self._cache[filename]

        filepath = self.trajectories_dir / filename
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()

        # Parse comma-separated values: x,y,z,qw,qx,qy,qz,x_dot,y_dot,z_dot,p,q,r
        state = np.array([float(x) for x in first_line.split(',')], dtype=np.float32)
        self._cache[filename] = state
        return state

    def _check_quaternion(self, quat: np.ndarray) -> bool:
        """Check if quaternion is unit norm (within tolerance)."""
        norm = np.linalg.norm(quat)
        self._quat_norm_min = min(self._quat_norm_min, norm)
        self._quat_norm_max = max(self._quat_norm_max, norm)

        # Allow small tolerance for numerical precision
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
        x_norm = x / self.x_limit
        y_norm = y / self.y_limit
        # z has asymmetric bounds [0.1, 3.0], normalize to [-1, 1]
        z_center = (self.z_max + self.z_min) / 2  # 1.55
        z_range = (self.z_max - self.z_min) / 2   # 1.45
        z_norm = (z - z_center) / z_range

        # Normalize velocities to [-1, 1]
        x_dot_norm = x_dot / self.x_dot_limit
        y_dot_norm = y_dot / self.y_dot_limit
        z_dot_norm = z_dot / self.z_dot_limit

        # Normalize angular velocities to [-1, 1]
        p_norm = p / self.p_limit
        q_norm = q / self.q_limit
        r_norm = r / self.r_limit

        # Quaternion is NOT normalized - kept as unit quaternion
        return np.array([
            x_norm, y_norm, z_norm,
            qw, qx, qy, qz,
            x_dot_norm, y_dot_norm, z_dot_norm,
            p_norm, q_norm, r_norm
        ], dtype=np.float32)

    def __getitem__(self, idx):
        filename = self.trajectory_files[idx]
        label = self.labels[idx]

        # Load initial state
        state = self._load_initial_state(filename)

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


class Quadrotor3DEvalDataset(Dataset):
    """Dataset for Quadrotor 3D classification evaluation from eval_states.txt."""

    def __init__(self, eval_file: str, check_quaternion_norm: bool = True):
        self.data = []  # Original states (13D)
        self.labels = []
        self.check_quaternion_norm = check_quaternion_norm

        # Quadrotor 3D state bounds
        self.x_limit = 1.8
        self.y_limit = 1.8
        self.z_min = 0.1
        self.z_max = 3.0
        self.x_dot_limit = 3.0
        self.y_dot_limit = 3.0
        self.z_dot_limit = 3.0
        self.p_limit = 40.0
        self.q_limit = 40.0
        self.r_limit = 40.0

        # Track quaternion stats
        self._quat_norm_min = float('inf')
        self._quat_norm_max = float('-inf')
        self._quat_norm_violations = 0

        # Load data
        # Format: x,y,z,qw,qx,qy,qz,x_dot,y_dot,z_dot,p,q,r, (13 values)
        #         x_f,y_f,z_f,qw_f,qx_f,qy_f,qz_f,x_dot_f,y_dot_f,z_dot_f,p_f,q_f,r_f, (13 values)
        #         success_flag (1 value)
        # Total: 27 values per line
        with open(eval_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 27:
                    # Extract initial state (first 13 values)
                    state = np.array([float(parts[i]) for i in range(13)], dtype=np.float32)
                    # Label is the last value
                    label = int(float(parts[26]))
                    self.data.append(state)
                    self.labels.append(label)

        # Check quaternion norms
        if self.check_quaternion_norm:
            for state in self.data:
                quat = state[3:7]
                norm = np.linalg.norm(quat)
                self._quat_norm_min = min(self._quat_norm_min, norm)
                self._quat_norm_max = max(self._quat_norm_max, norm)
                if abs(norm - 1.0) > 0.01:
                    self._quat_norm_violations += 1

        print(f"Loaded {len(self.data)} eval samples")
        print(f"  Success (1): {sum(self.labels)}")
        print(f"  Failure (0): {len(self.labels) - sum(self.labels)}")
        if self.check_quaternion_norm:
            print(f"  Quaternion norm range: [{self._quat_norm_min:.6f}, {self._quat_norm_max:.6f}]")
            if self._quat_norm_violations > 0:
                print(f"  WARNING: {self._quat_norm_violations} quaternion norm violations!")

    def __len__(self):
        return len(self.data)

    def _embed_state(self, state: np.ndarray) -> np.ndarray:
        """Embed state with normalization (same as training)."""
        x, y, z = state[0], state[1], state[2]
        qw, qx, qy, qz = state[3], state[4], state[5], state[6]
        x_dot, y_dot, z_dot = state[7], state[8], state[9]
        p, q, r = state[10], state[11], state[12]

        # Normalize position
        x_norm = x / self.x_limit
        y_norm = y / self.y_limit
        z_center = (self.z_max + self.z_min) / 2
        z_range = (self.z_max - self.z_min) / 2
        z_norm = (z - z_center) / z_range

        # Normalize velocities
        x_dot_norm = x_dot / self.x_dot_limit
        y_dot_norm = y_dot / self.y_dot_limit
        z_dot_norm = z_dot / self.z_dot_limit

        # Normalize angular velocities
        p_norm = p / self.p_limit
        q_norm = q / self.q_limit
        r_norm = r / self.r_limit

        return np.array([
            x_norm, y_norm, z_norm,
            qw, qx, qy, qz,
            x_dot_norm, y_dot_norm, z_dot_norm,
            p_norm, q_norm, r_norm
        ], dtype=np.float32)

    def __getitem__(self, idx):
        state = self.data[idx]
        label = self.labels[idx]
        embedded = self._embed_state(state)

        return {
            "inputs": torch.from_numpy(embedded).float(),
            "original_state": torch.from_numpy(state).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }


class Quadrotor3DClassificationDataModule(pl.LightningDataModule):
    """Lightning DataModule for Quadrotor 3D classification with sample balancing."""

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
        balance_samples: bool = False,
        check_quaternion_norm: bool = True,
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
        self.check_quaternion_norm = check_quaternion_norm

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

        assert len(self.trajectory_files) == len(self.labels), \
            f"Mismatch: {len(self.trajectory_files)} files vs {len(self.labels)} labels"

        # Limit samples if specified
        if self.max_samples is not None:
            self.trajectory_files = self.trajectory_files[:self.max_samples]
            self.labels = self.labels[:self.max_samples]

        n_success = sum(self.labels)
        n_failure = len(self.labels) - n_success
        print(f"Loaded {len(self.trajectory_files)} samples")
        print(f"  Success (1): {n_success} ({100*n_success/len(self.labels):.2f}%)")
        print(f"  Failure (0): {n_failure} ({100*n_failure/len(self.labels):.2f}%)")

    def setup(self, stage: Optional[str] = None):
        """Split data and create datasets."""
        n = len(self.trajectory_files)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))

        if stage == "fit" or stage is None:
            train_files = self.trajectory_files[:train_end]
            train_labels = self.labels[:train_end]

            self.train_dataset = Quadrotor3DClassificationDataset(
                trajectory_files=train_files,
                labels=train_labels,
                trajectories_dir=self.trajectories_dir,
                check_quaternion_norm=self.check_quaternion_norm,
            )

            # Compute sample weights for balanced sampling
            if self.balance_samples:
                label_counts = np.bincount(train_labels)
                if len(label_counts) < 2:
                    # Handle case where only one class exists in training data
                    print("WARNING: Only one class in training data, cannot balance!")
                    self.balance_samples = False
                else:
                    # Weight inversely proportional to class frequency
                    class_weights = 1.0 / label_counts
                    self.sample_weights = [class_weights[label] for label in train_labels]
                    print(f"Sample balancing enabled:")
                    print(f"  Class 0 (failure) count: {label_counts[0]}, weight: {class_weights[0]:.6f}")
                    print(f"  Class 1 (success) count: {label_counts[1]}, weight: {class_weights[1]:.6f}")

            self.val_dataset = Quadrotor3DClassificationDataset(
                trajectory_files=self.trajectory_files[train_end:val_end],
                labels=self.labels[train_end:val_end],
                trajectories_dir=self.trajectories_dir,
                check_quaternion_norm=self.check_quaternion_norm,
            )

            print(f"Train: {len(self.train_dataset)} samples")
            print(f"Val: {len(self.val_dataset)} samples")

        if stage == "test" or stage is None:
            self.test_dataset = Quadrotor3DClassificationDataset(
                trajectory_files=self.trajectory_files[val_end:],
                labels=self.labels[val_end:],
                trajectories_dir=self.trajectories_dir,
                check_quaternion_norm=self.check_quaternion_norm,
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

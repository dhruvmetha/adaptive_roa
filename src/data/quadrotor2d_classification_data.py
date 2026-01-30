"""
Quadrotor 2D Classification Data Module

Loads trajectories from shuffled indices files and extracts initial states
with corresponding success/failure labels for classification.

State space (6D raw, 7D embedded):
- Position: x (normalized to [-1, 1]), z (normalized from [0.1, 1.5] to [-1, 1])
- Orientation: theta (pitch angle) -> sin(theta), cos(theta) for S^1 manifold embedding
- Linear velocity: x_dot, z_dot (normalized to [-1, 1])
- Angular velocity: theta_dot (normalized to [-1, 1] using achieved bounds)

Input dimension: 7 (after sin/cos embedding of theta)

Labels are computed from trajectory files by checking if final state is within
0.05 Euclidean distance of goal state [0, 1, 0, 0, 0, 0].
"""

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List
from pathlib import Path
import torch
import lightning.pytorch as pl
import numpy as np


class Quadrotor2DClassificationDataset(Dataset):
    """Dataset for Quadrotor 2D classification from trajectory files."""

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

        # Quadrotor 2D state bounds from dataset_description.json
        # Position bounds
        self.x_limit = 1.0
        self.z_min = 0.1
        self.z_max = 1.5

        # Velocity bounds
        self.x_dot_limit = 1.0
        self.z_dot_limit = 1.0

        # Angular velocity bounds (use achieved bounds for better normalization)
        self.theta_dot_limit = 10.5  # Achieved bounds show ~±10.5 rad/s

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

        # Parse comma-separated values: x,z,theta,x_dot,z_dot,theta_dot
        state = np.array([float(x) for x in first_line.split(',')], dtype=np.float32)
        self._cache[filename] = state
        return state

    def _embed_state(self, state: np.ndarray) -> np.ndarray:
        """
        Embed state with normalization and S^1 manifold embedding for theta.

        Input: [x, z, theta, x_dot, z_dot, theta_dot] (6D)
        Output: [x_norm, z_norm, sin(theta), cos(theta), x_dot_norm, z_dot_norm, theta_dot_norm] (7D)
        """
        x, z, theta, x_dot, z_dot, theta_dot = state

        # Normalize x to [-1, 1]
        x_norm = x / self.x_limit

        # Normalize z from [0.1, 1.5] to [-1, 1]
        z_center = (self.z_max + self.z_min) / 2  # 0.8
        z_range = (self.z_max - self.z_min) / 2   # 0.7
        z_norm = (z - z_center) / z_range

        # S^1 manifold embedding for theta (pitch angle)
        # This properly handles the circular nature of the angle
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Normalize velocities to [-1, 1]
        x_dot_norm = x_dot / self.x_dot_limit
        z_dot_norm = z_dot / self.z_dot_limit

        # Normalize angular velocity to [-1, 1] using achieved bounds
        theta_dot_norm = theta_dot / self.theta_dot_limit

        return np.array([
            x_norm, z_norm,
            sin_theta, cos_theta,
            x_dot_norm, z_dot_norm, theta_dot_norm
        ], dtype=np.float32)

    def __getitem__(self, idx):
        filename = self.trajectory_files[idx]
        label = self.labels[idx]

        # Load initial state
        state = self._load_initial_state(filename)

        # Embed with normalization and S^1 embedding
        embedded = self._embed_state(state)

        return {
            "inputs": torch.from_numpy(embedded).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }


class Quadrotor2DEvalDataset(Dataset):
    """Dataset for Quadrotor 2D classification evaluation from eval_states.txt."""

    def __init__(self, eval_file: str):
        self.data = []  # Original states (6D)
        self.labels = []

        # Quadrotor 2D state bounds
        self.x_limit = 1.0
        self.z_min = 0.1
        self.z_max = 1.5
        self.x_dot_limit = 1.0
        self.z_dot_limit = 1.0
        self.theta_dot_limit = 10.5

        # Load data
        # Format: x,z,theta,x_dot,z_dot,theta_dot (6 values for init)
        #         x_f,z_f,theta_f,x_dot_f,z_dot_f,theta_dot_f (6 values for final)
        #         success_flag (1 value)
        # Total: 13 values per line
        with open(eval_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 13:
                    # Extract initial state (first 6 values)
                    state = np.array([float(parts[i]) for i in range(6)], dtype=np.float32)
                    # Label is the last value
                    label = int(float(parts[12]))
                    self.data.append(state)
                    self.labels.append(label)

        print(f"Loaded {len(self.data)} eval samples")
        print(f"  Success (1): {sum(self.labels)}")
        print(f"  Failure (0): {len(self.labels) - sum(self.labels)}")

    def __len__(self):
        return len(self.data)

    def _embed_state(self, state: np.ndarray) -> np.ndarray:
        """Embed state with normalization and S^1 embedding (same as training)."""
        x, z, theta, x_dot, z_dot, theta_dot = state

        # Normalize x to [-1, 1]
        x_norm = x / self.x_limit

        # Normalize z from [0.1, 1.5] to [-1, 1]
        z_center = (self.z_max + self.z_min) / 2
        z_range = (self.z_max - self.z_min) / 2
        z_norm = (z - z_center) / z_range

        # S^1 manifold embedding for theta
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Normalize velocities
        x_dot_norm = x_dot / self.x_dot_limit
        z_dot_norm = z_dot / self.z_dot_limit
        theta_dot_norm = theta_dot / self.theta_dot_limit

        return np.array([
            x_norm, z_norm,
            sin_theta, cos_theta,
            x_dot_norm, z_dot_norm, theta_dot_norm
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


class Quadrotor2DClassificationDataModule(pl.LightningDataModule):
    """Lightning DataModule for Quadrotor 2D classification with sample balancing.
    
    Labels are computed from trajectory files by checking if final state is within
    success_threshold of goal state.
    """

    # Success criteria from dataset_description.json
    GOAL_STATE = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # x, z, theta, x_dot, z_dot, theta_dot
    SUCCESS_THRESHOLD = 0.05  # Euclidean distance threshold

    def __init__(
        self,
        shuffled_indices_file: str,
        trajectories_dir: str,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_split: float = 0.95,
        val_split: float = 0.05,
        max_samples: Optional[int] = None,
        balance_samples: bool = False,
    ):
        super().__init__()
        self.shuffled_indices_file = shuffled_indices_file
        self.trajectories_dir = Path(trajectories_dir)
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

    def _compute_label_from_trajectory(self, filename: str) -> int:
        """
        Compute success/failure label from trajectory file.
        
        Reads the final state (last line) and checks if it's within
        SUCCESS_THRESHOLD Euclidean distance of GOAL_STATE.
        """
        filepath = self.trajectories_dir / filename
        
        # Read last line of file (final state)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if not lines:
                raise ValueError(f"Empty trajectory file: {filename}")
            last_line = lines[-1].strip()
        
        # Parse final state: x, z, theta, x_dot, z_dot, theta_dot
        final_state = np.array([float(x) for x in last_line.split(',')], dtype=np.float32)
        
        # Compute Euclidean distance to goal
        distance = np.linalg.norm(final_state - self.GOAL_STATE)
        
        # Return 1 if within threshold (success), 0 otherwise (failure)
        return 1 if distance < self.SUCCESS_THRESHOLD else 0

    def prepare_data(self):
        """Load indices and compute labels from trajectory files."""
        # Load trajectory filenames
        with open(self.shuffled_indices_file, 'r') as f:
            self.trajectory_files = [line.strip() for line in f.readlines()]

        # Limit samples if specified (do this BEFORE computing labels for efficiency)
        if self.max_samples is not None:
            self.trajectory_files = self.trajectory_files[:self.max_samples]

        # Compute labels from trajectory files
        print(f"Computing labels from {len(self.trajectory_files)} trajectory files...")
        self.labels = []
        for i, filename in enumerate(self.trajectory_files):
            label = self._compute_label_from_trajectory(filename)
            self.labels.append(label)
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(self.trajectory_files)} trajectories...")

        assert len(self.trajectory_files) == len(self.labels), \
            f"Mismatch: {len(self.trajectory_files)} files vs {len(self.labels)} labels"

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

            self.train_dataset = Quadrotor2DClassificationDataset(
                trajectory_files=train_files,
                labels=train_labels,
                trajectories_dir=self.trajectories_dir,
            )

            # Compute sample weights for balanced sampling
            if self.balance_samples:
                label_counts = np.bincount(train_labels)
                if len(label_counts) < 2:
                    print("WARNING: Only one class in training data, cannot balance!")
                    self.balance_samples = False
                else:
                    # Weight inversely proportional to class frequency
                    class_weights = 1.0 / label_counts
                    self.sample_weights = [class_weights[label] for label in train_labels]
                    print(f"Sample balancing enabled:")
                    print(f"  Class 0 (failure) count: {label_counts[0]}, weight: {class_weights[0]:.6f}")
                    print(f"  Class 1 (success) count: {label_counts[1]}, weight: {class_weights[1]:.6f}")

            self.val_dataset = Quadrotor2DClassificationDataset(
                trajectory_files=self.trajectory_files[train_end:val_end],
                labels=self.labels[train_end:val_end],
                trajectories_dir=self.trajectories_dir,
            )

            print(f"Train: {len(self.train_dataset)} samples")
            print(f"Val: {len(self.val_dataset)} samples")

        if stage == "test" or stage is None:
            self.test_dataset = Quadrotor2DClassificationDataset(
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


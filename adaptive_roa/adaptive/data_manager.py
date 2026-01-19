"""
Data Manager for Adaptive Sampling.

Tracks trajectory and classification data across epochs and provides
splits for flow matcher training and conformal prediction.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass, field


@dataclass
class DataManagerConfig:
    """Configuration for data manager."""
    test_split: float = 0.2      # Fraction of initial data reserved for testing
    seed: int = 42               # Random seed for splitting


class DataManager:
    """
    Manage trajectory and classification data for adaptive sampling.

    Tracks two types of data:
    1. Trajectory Data: (start_state, end_state) pairs for FM training
    2. Classification Data: (start_state, label) pairs for CP training

    Data accumulates across epochs. The test set is fixed from initial data.

    Attributes:
        state_dim: Dimension of state space
        config: DataManagerConfig
        trajectory_starts: All start states [N, state_dim]
        trajectory_ends: All end states [N, state_dim]
        labels: All labels [N] (1 for success, -1 for failure)
        test_starts: Fixed test set starts [N_test, state_dim]
        test_ends: Fixed test set ends [N_test, state_dim]
        test_labels: Fixed test labels [N_test]
    """

    def __init__(
        self,
        state_dim: int,
        config: Optional[DataManagerConfig] = None
    ):
        """
        Initialize data manager.

        Args:
            state_dim: Dimension of state space
            config: DataManagerConfig (uses defaults if None)
        """
        self.state_dim = state_dim
        self.config = config or DataManagerConfig()

        # Training/calibration data (accumulates)
        self.trajectory_starts: Optional[np.ndarray] = None
        self.trajectory_ends: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

        # Fixed test set
        self.test_starts: Optional[np.ndarray] = None
        self.test_ends: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None

        # Statistics
        self.n_simulations_total = 0
        self.n_simulations_saved = 0  # Skipped due to confidence
        self.epoch_history = []

    def initialize_from_file(
        self,
        trajectory_file: str,
        system,
        attractor_radius: float = 0.2
    ) -> Dict:
        """
        Initialize data manager from a trajectory file.

        Loads trajectory pairs, computes labels, and splits into train/test.

        Args:
            trajectory_file: Path to trajectory file (space-separated, 2*state_dim columns)
            system: Dynamical system with classify_attractor()
            attractor_radius: Radius for classification

        Returns:
            Dict with initialization statistics
        """
        # Load trajectory file
        data = np.loadtxt(trajectory_file)
        n_samples = len(data)

        # Split into start and end states
        starts = data[:, :self.state_dim]
        ends = data[:, self.state_dim:]

        # Compute labels
        ends_tensor = torch.from_numpy(ends).float()
        labels = system.classify_attractor(ends_tensor, radius=attractor_radius).numpy()

        # Split into train and test
        np.random.seed(self.config.seed)
        n_test = int(n_samples * self.config.test_split)
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Store test set (fixed)
        self.test_starts = starts[test_indices]
        self.test_ends = ends[test_indices]
        self.test_labels = labels[test_indices]

        # Store train set (will accumulate)
        self.trajectory_starts = starts[train_indices]
        self.trajectory_ends = ends[train_indices]
        self.labels = labels[train_indices]

        self.n_simulations_total = len(train_indices)

        stats = {
            'total_loaded': n_samples,
            'train_size': len(train_indices),
            'test_size': n_test,
            'success_rate_train': np.mean(self.labels == 1),
            'success_rate_test': np.mean(self.test_labels == 1),
        }

        return stats

    def add_trajectories(
        self,
        starts: Union[torch.Tensor, np.ndarray],
        ends: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray]
    ):
        """
        Add new trajectory data.

        Args:
            starts: [N, state_dim] start states
            ends: [N, state_dim] end states
            labels: [N] labels (-1 or 1)
        """
        # Convert to numpy
        if isinstance(starts, torch.Tensor):
            starts = starts.cpu().numpy()
        if isinstance(ends, torch.Tensor):
            ends = ends.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Append to existing data
        if self.trajectory_starts is None:
            self.trajectory_starts = starts
            self.trajectory_ends = ends
            self.labels = labels
        else:
            self.trajectory_starts = np.vstack([self.trajectory_starts, starts])
            self.trajectory_ends = np.vstack([self.trajectory_ends, ends])
            self.labels = np.concatenate([self.labels, labels])

        self.n_simulations_total += len(starts)

    def record_epoch(
        self,
        n_simulated: int,
        n_skipped: int,
        lambda_star: float,
        q_hat: float,
        test_metrics: Dict
    ):
        """
        Record statistics for an epoch.

        Args:
            n_simulated: Number of simulations run this epoch
            n_skipped: Number of points skipped (confident predictions)
            lambda_star: Optimal lambda from this epoch
            q_hat: Calibration threshold from this epoch
            test_metrics: Evaluation metrics on test set
        """
        self.n_simulations_saved += n_skipped

        self.epoch_history.append({
            'n_simulated': n_simulated,
            'n_skipped': n_skipped,
            'total_data': len(self.labels),
            'lambda_star': lambda_star,
            'q_hat': q_hat,
            'test_metrics': test_metrics,
        })

    def get_fm_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all trajectory data for flow matcher training.

        Returns:
            Tuple of (starts, ends) numpy arrays
        """
        return self.trajectory_starts, self.trajectory_ends

    def get_cp_data_split(
        self,
        cal_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/calibration split for conformal prediction.

        Args:
            cal_ratio: Fraction of data for calibration

        Returns:
            Tuple of (X_train, y_train, X_cal, y_cal)
        """
        n = len(self.labels)
        n_cal = int(n * cal_ratio)

        # Random split (different each time)
        indices = np.random.permutation(n)
        cal_indices = indices[:n_cal]
        train_indices = indices[n_cal:]

        X_train = self.trajectory_starts[train_indices]
        y_train = self.labels[train_indices]
        X_cal = self.trajectory_starts[cal_indices]
        y_cal = self.labels[cal_indices]

        return X_train, y_train, X_cal, y_cal

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get fixed test set.

        Returns:
            Tuple of (X_test, y_test, ends_test)
        """
        return self.test_starts, self.test_labels, self.test_ends

    def save_trajectory_file(self, path: str):
        """
        Save all trajectory data to a text file.

        Format: space-separated, one trajectory per line
        Columns: [start_0, ..., start_d, end_0, ..., end_d]

        Args:
            path: Output file path
        """
        data = np.hstack([self.trajectory_starts, self.trajectory_ends])
        np.savetxt(path, data, fmt='%.8f')

    def get_statistics(self) -> Dict:
        """
        Get current data statistics.

        Returns:
            Dict with statistics
        """
        return {
            'n_trajectories': len(self.labels) if self.labels is not None else 0,
            'n_test': len(self.test_labels) if self.test_labels is not None else 0,
            'n_simulations_total': self.n_simulations_total,
            'n_simulations_saved': self.n_simulations_saved,
            'savings_rate': self.n_simulations_saved / max(1, self.n_simulations_total + self.n_simulations_saved),
            'success_rate': np.mean(self.labels == 1) if self.labels is not None else 0,
            'n_epochs': len(self.epoch_history),
        }

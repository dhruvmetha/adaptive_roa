"""
Trajectory Data Source for Adaptive Sampling.

Manages the mapping between roa_labels, shuffled_indices, and trajectory files.
Provides efficient access to trajectory data by index.
"""
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass


@dataclass
class TrajectoryDataSourceConfig:
    """Configuration for trajectory data source."""
    trajectories_dir: str           # Base directory containing trajectory files
    shuffled_indices_file: str      # File mapping indices to trajectory filenames

    # New: shuffled_labels_file aligned with shuffled_indices (for training)
    shuffled_labels_file: Optional[str] = None

    # Legacy: roa_labels_file for full ROA evaluation only (not aligned with shuffled indices)
    roa_labels_file: Optional[str] = None

    # Label mapping (external format → internal format)
    # External: 0 = failure, 1 = success (from labels files)
    # Internal: -1 = failure, 0 = separatrix, 1 = success
    label_mapping: Dict[int, int] = None

    def __post_init__(self):
        if self.label_mapping is None:
            # Default: map 0 → -1 (failure), 1 → 1 (success)
            # Separatrix (0 internal) would need to be computed differently
            self.label_mapping = {0: -1, 1: 1}


class TrajectoryDataSource:
    """
    Manages trajectory data for adaptive sampling.

    Loads the index mappings once, then provides efficient access to:
    - Individual trajectories by index
    - Batches of trajectories
    - Pre-computed labels (if available)
    - Start states (on-demand from trajectory files)

    This is the "pool" of all available data that adaptive sampling draws from.

    Note: ROA evaluation should load roa_labels.txt directly, not through this class.
    This class focuses on training data management using shuffled_indices and
    shuffled_labels files.

    Attributes:
        config: TrajectoryDataSourceConfig
        trajectory_files: List of trajectory file paths (ordered by shuffled index)
        labels: Optional pre-computed labels from shuffled_labels.txt
        start_states: Optional pre-computed start states (None for on-demand loading)
        n_trajectories: Total number of trajectories available
    """

    def __init__(self, config: TrajectoryDataSourceConfig):
        """
        Initialize trajectory data source.

        Args:
            config: TrajectoryDataSourceConfig with file paths
        """
        self.config = config
        self.trajectories_dir = Path(config.trajectories_dir)

        # Load shuffled indices → trajectory filenames
        self._load_shuffled_indices(config.shuffled_indices_file)

        # Load labels from shuffled_labels file (aligned with shuffled_indices)
        if config.shuffled_labels_file:
            self._load_shuffled_labels(config.shuffled_labels_file)
        else:
            self.labels = None

        # Start states: always load on-demand from trajectory files
        self.start_states = None

        print(f"TrajectoryDataSource initialized:")
        print(f"  Trajectories: {self.n_trajectories}")
        print(f"  Directory: {self.trajectories_dir}")
        if self.labels is not None:
            n_success = np.sum(self.labels == 1)
            n_failure = np.sum(self.labels == -1)
            print(f"  Labels: {n_success} success, {n_failure} failure")

    def _load_shuffled_indices(self, filepath: str):
        """Load shuffled indices file."""
        with open(filepath, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]

        self.trajectory_files = [
            self.trajectories_dir / fname for fname in filenames
        ]
        self.n_trajectories = len(self.trajectory_files)

    def _load_shuffled_labels(self, filepath: str):
        """
        Load shuffled labels file (single column, aligned with shuffled_indices).

        Format: one label per line (0 or 1)
        """
        raw_labels = np.loadtxt(filepath, dtype=int)

        # Map external labels to internal format (0 → -1, 1 → 1)
        self.labels = np.array([
            self.config.label_mapping.get(l, 0) for l in raw_labels
        ], dtype=np.int64)

        assert len(self.labels) == self.n_trajectories, \
            f"Label count ({len(self.labels)}) != trajectory count ({self.n_trajectories})"

    def load_trajectory(self, idx: int) -> np.ndarray:
        """
        Load a single trajectory by index.

        Args:
            idx: Trajectory index (0 to n_trajectories-1)

        Returns:
            [T, state_dim] array of states over time
        """
        filepath = self.trajectory_files[idx]

        with open(filepath, 'r') as f:
            lines = f.readlines()

        trajectory = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split(',')))
                trajectory.append(values)

        return np.array(trajectory, dtype=np.float32)

    def load_trajectories(self, indices: List[int]) -> List[np.ndarray]:
        """
        Load multiple trajectories by indices.

        Args:
            indices: List of trajectory indices

        Returns:
            List of [T_i, state_dim] arrays
        """
        return [self.load_trajectory(idx) for idx in indices]

    def get_start_state(self, idx: int) -> np.ndarray:
        """Get start state for trajectory idx."""
        if self.start_states is not None:
            return self.start_states[idx]
        else:
            traj = self.load_trajectory(idx)
            return traj[0]

    def get_end_state(self, idx: int) -> np.ndarray:
        """Get end state for trajectory idx."""
        traj = self.load_trajectory(idx)
        return traj[-1]

    def get_label(self, idx: int) -> int:
        """Get label for trajectory idx."""
        if self.labels is not None:
            return self.labels[idx]
        else:
            raise ValueError("No labels loaded. Provide shuffled_labels_file in config.")

    def get_start_states(self, indices: List[int]) -> np.ndarray:
        """Get start states for multiple trajectories."""
        if self.start_states is not None:
            return self.start_states[indices]
        else:
            return np.array([self.get_start_state(i) for i in indices])

    def get_labels(self, indices: List[int]) -> np.ndarray:
        """Get labels for multiple trajectories."""
        if self.labels is not None:
            return self.labels[indices]
        else:
            raise ValueError("No labels loaded. Provide shuffled_labels_file in config.")

    def get_endpoint_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get (start_state, end_state) pair for trajectory idx.

        Args:
            idx: Trajectory index

        Returns:
            Tuple of (start_state, end_state) arrays
        """
        traj = self.load_trajectory(idx)
        return traj[0], traj[-1]

    def get_endpoint_pairs(
        self,
        indices: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get endpoint pairs for multiple trajectories.

        Args:
            indices: List of trajectory indices

        Returns:
            Tuple of (start_states [N, dim], end_states [N, dim])
        """
        pairs = [self.get_endpoint_pair(i) for i in indices]
        starts = np.array([p[0] for p in pairs])
        ends = np.array([p[1] for p in pairs])
        return starts, ends

    def get_all_endpoint_pairs_from_trajectory(
        self,
        idx: int,
        mode: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all endpoint pairs from a single trajectory.

        For training: every state → final state (many pairs per trajectory)
        For testing: only first state → final state (one pair per trajectory)

        Args:
            idx: Trajectory index
            mode: "train" (all states) or "test" (first state only)

        Returns:
            Tuple of (start_states [N, dim], end_states [N, dim])
        """
        traj = self.load_trajectory(idx)
        end_state = traj[-1]

        if mode == "test":
            # Only first state → end state
            return traj[0:1], np.array([end_state])
        else:
            # All states except last → end state
            n_states = len(traj) - 1
            starts = traj[:-1]
            ends = np.tile(end_state, (n_states, 1))
            return starts, ends

    def build_endpoint_dataset(
        self,
        indices: List[int],
        mode: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build endpoint dataset from trajectory indices.

        Args:
            indices: List of trajectory indices to include
            mode: "train" (all states per trajectory) or "test" (one per trajectory)

        Returns:
            Tuple of:
                start_states: [N, state_dim]
                end_states: [N, state_dim]
                labels: [N] (per-trajectory label, repeated for each pair)
        """
        all_starts = []
        all_ends = []
        all_labels = []

        for idx in indices:
            starts, ends = self.get_all_endpoint_pairs_from_trajectory(idx, mode)
            label = self.get_label(idx)

            all_starts.append(starts)
            all_ends.append(ends)
            all_labels.extend([label] * len(starts))

        return (
            np.vstack(all_starts),
            np.vstack(all_ends),
            np.array(all_labels, dtype=np.int64)
        )

    def save_endpoint_dataset(
        self,
        indices: List[int],
        output_file: str,
        mode: str = "train"
    ) -> int:
        """
        Build and save endpoint dataset to file.

        Format: space-separated, one line per pair
        [start_state... end_state...]

        Args:
            indices: Trajectory indices to include
            output_file: Output file path
            mode: "train" or "test"

        Returns:
            Number of endpoint pairs written
        """
        starts, ends, _ = self.build_endpoint_dataset(indices, mode)

        # Concatenate start and end states
        data = np.hstack([starts, ends])

        np.savetxt(output_file, data, fmt='%.8f')

        return len(data)

    def get_state_dim(self) -> int:
        """Get state dimension from first trajectory."""
        traj = self.load_trajectory(0)
        return traj.shape[1]

    def get_statistics(self, indices: Optional[List[int]] = None) -> Dict:
        """
        Get statistics for trajectory subset.

        Args:
            indices: Subset indices (all if None)

        Returns:
            Dict with statistics
        """
        if indices is None:
            indices = list(range(self.n_trajectories))

        labels = self.get_labels(indices)

        return {
            'n_trajectories': len(indices),
            'n_success': int(np.sum(labels == 1)),
            'n_failure': int(np.sum(labels == -1)),
            'n_separatrix': int(np.sum(labels == 0)),
            'success_rate': float(np.mean(labels == 1)),
        }

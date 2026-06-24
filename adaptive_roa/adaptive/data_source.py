"""
Trajectory Data Source for Adaptive Sampling.

Manages the mapping between eval_states, shuffled_indices, and trajectory files.
Provides efficient access to trajectory data by index.
"""
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass


def load_eval_states(
    filepath: str,
    label_mapping: Optional[Dict[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load eval_states.txt file containing start states, end states, and labels.

    File format (comma-separated):
    - Pendulum (2D state): θ_s, θ̇_s, θ_e, θ̇_e, label (5 columns)
    - CartPole (4D state): x_s, θ_s, ẋ_s, θ̇_s, x_e, θ_e, ẋ_e, θ̇_e, label (9 columns)
    - Mountain Car (2D state): pos_s, vel_s, pos_e, vel_e, label (5 columns)
    - Pendulum Cartesian (4D state): x_s, y_s, vx_s, vy_s, x_e, y_e, vx_e, vy_e, label (9 columns)

    The state dimension is automatically inferred from the number of columns:
    - 5 columns → 2D state (pendulum, mountain car)
    - 9 columns → 4D state (cartpole, pendulum cartesian)

    Args:
        filepath: Path to eval_states.txt file
        label_mapping: Optional mapping from external labels (0/1) to internal format.
                      Default: {0: -1, 1: 1} (0 → failure, 1 → success)

    Returns:
        Tuple of:
            start_states: [N, state_dim] array of start states
            end_states: [N, state_dim] array of end states
            labels: [N] array of labels in internal format (-1 = failure, 1 = success)
    """
    if label_mapping is None:
        label_mapping = {0: -1, 1: 1}

    # Load data (comma-separated)
    data = np.loadtxt(filepath, delimiter=',')

    n_cols = data.shape[1]

    # Infer state dimension from number of columns
    # Format: [start_state..., end_state..., label]
    # So: n_cols = 2 * state_dim + 1
    state_dim = (n_cols - 1) // 2

    if n_cols != 2 * state_dim + 1:
        raise ValueError(
            f"Invalid number of columns in eval_states file: {n_cols}. "
            f"Expected 2*state_dim + 1 (e.g., 5 for 2D state, 9 for 4D state)"
        )

    # Extract components
    start_states = data[:, :state_dim].astype(np.float32)
    end_states = data[:, state_dim:2*state_dim].astype(np.float32)
    raw_labels = data[:, -1].astype(int)

    # Map labels to internal format
    labels = np.array([label_mapping.get(l, 0) for l in raw_labels], dtype=np.int64)

    return start_states, end_states, labels


@dataclass
class TrajectoryDataSourceConfig:
    """Configuration for trajectory data source."""
    trajectories_dir: str           # Base directory containing trajectory files
    shuffled_indices_file: str      # File mapping indices to trajectory filenames

    # shuffled_labels_file aligned with shuffled_indices (for training)
    shuffled_labels_file: Optional[str] = None

    # eval_states_file for full ROA evaluation (contains start, end, and labels)
    eval_states_file: Optional[str] = None

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

    Note: For ROA evaluation, use the load_eval_states() function to load
    eval_states.txt directly. This class focuses on training data management
    using shuffled_indices and shuffled_labels files.

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

        # Cache for loaded trajectories (keyed by idx) to avoid repeated disk reads
        self._traj_cache: Dict[int, np.ndarray] = {}

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
        raw_labels = np.loadtxt(filepath, dtype=int, ndmin=1)

        # Map external labels to internal format (0 → -1, 1 → 1)
        self.labels = np.array([
            self.config.label_mapping.get(l, 0) for l in raw_labels
        ], dtype=np.int64)

        assert len(self.labels) == self.n_trajectories, \
            f"Label count ({len(self.labels)}) != trajectory count ({self.n_trajectories})"

    def load_trajectory(self, idx: int) -> np.ndarray:
        """
        Load a single trajectory by index.

        Results are cached internally to avoid repeated disk reads.

        Args:
            idx: Trajectory index (0 to n_trajectories-1)

        Returns:
            [T, state_dim] array of states over time
        """
        if idx in self._traj_cache:
            return self._traj_cache[idx]

        filepath = self.trajectory_files[idx]

        with open(filepath, 'r') as f:
            lines = f.readlines()

        trajectory = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split(',')))
                trajectory.append(values)

        result = np.array(trajectory, dtype=np.float32)
        self._traj_cache[idx] = result
        return result

    def get_trajectory_length(self, idx: int) -> int:
        """Return the number of rows (time steps) in trajectory idx."""
        return len(self.load_trajectory(idx))

    def get_state_at(self, idx: int, row: int) -> np.ndarray:
        """Return the state vector at a specific row in trajectory idx."""
        return self.load_trajectory(idx)[row]

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
        mode: str = "train",
        start_row: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all endpoint pairs from a single trajectory.

        For training: every state → final state (many pairs per trajectory)
        For testing: only first state → final state (one pair per trajectory)

        Args:
            idx: Trajectory index
            mode: "train" (all states) or "test" (first state only)
            start_row: First row to include in training mode (default 0 = current behaviour).
                       Ignored in "test" mode.

        Returns:
            Tuple of (start_states [N, dim], end_states [N, dim])
        """
        traj = self.load_trajectory(idx)
        end_state = traj[-1]

        if mode == "test":
            # Only first state → end state
            return traj[0:1], np.array([end_state])
        else:
            # All states from start_row (inclusive) up to but not including last → end state
            starts = traj[start_row:-1]
            n_states = len(starts)
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

    def save_classification_dataset(
        self,
        indices: List[int],
        output_file: str,
        mode: str = "train"
    ) -> int:
        """
        Build and save a (state, binary_label) classification dataset.

        Format: space-separated, one line per state:
            [start_state... label01]
        where label01 = 1.0 for success (internal label 1) and 0.0 otherwise
        (internal -1 failure). Uses the same per-trajectory state expansion as
        the endpoint dataset (every state along a trajectory inherits the
        trajectory's outcome), so the classifier and the flow matcher train on
        the same data footprint.

        SAFETY: refuses to write anywhere under DATA_DIR (source data is
        read-only / shared).

        Args:
            indices: Trajectory indices to include
            output_file: Output file path (must NOT be under DATA_DIR)
            mode: "train" (all states per trajectory) or "test" (first only)

        Returns:
            Number of (state, label) rows written
        """
        import os as _os
        from adaptive_roa.utils.env_config import get_data_dir

        abs_out = _os.path.abspath(output_file)
        data_dir = _os.path.abspath(get_data_dir())
        if abs_out == data_dir or abs_out.startswith(data_dir + _os.sep):
            raise ValueError(
                f"Refusing to write dataset under DATA_DIR ({data_dir}): {abs_out}"
            )

        starts, _ends, labels = self.build_endpoint_dataset(indices, mode)
        # internal labels {-1 failure, 1 success} -> binary {0.0, 1.0} for BCE
        label01 = (np.asarray(labels) == 1).astype(np.float64).reshape(-1, 1)
        data = np.hstack([starts, label01])

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

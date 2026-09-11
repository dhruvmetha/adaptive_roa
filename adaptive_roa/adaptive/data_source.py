"""
Trajectory Data Source for Adaptive Sampling.

Manages the mapping between eval_states, shuffled_indices, and trajectory files.
Provides efficient access to trajectory data by index.
"""
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass

# Maximum rows per trajectory for packed candidate-id scheme.
# Any trajectory longer than this will raise at get_trajectory_length time.
MAX_ROWS = 10000

# How training rows are built from a TIMEOUT trajectory, one the collector cut at
# its horizon T because time ran out (not by success, not at a wall).
#
# A trajectory that terminates at step L-1 <= T reaches that terminal state from
# every x_t within T steps, so (x_t, x_{L-1}) is a valid T-step pair. A timeout
# is not: the T-step future of x_t is x_{t+T}, which was never recorded, so
# (x_t, x_T) claims an outcome over only T - t steps.
#   "keep": every row paired with x_T (historical behaviour, the default so that
#           running and resumed runs are unchanged).
#   "drop": a timeout contributes only (x_0, x_T), the one pair whose future is
#           known. Row inclusion then depends on the outcome: measured on the
#           pendulum-med pool, rows at true p_success 0.1-0.2 carry mean label
#           0.93 under "drop" against 0.13 under "keep".
TIMEOUT_INTERMEDIATE_MODES = ("keep", "drop")


def _horizon_from_description(desc: dict) -> Optional[int]:
    """Collector horizon in steps from a dataset description json, if recorded.

    Pendulum writes a top-level `horizon_steps`; cartpole and the quadrotors
    write `horizon: {steps: ...}`.
    """
    h = desc.get("horizon_steps")
    if h is None and isinstance(desc.get("horizon"), dict):
        h = desc["horizon"].get("steps")
    return None if h is None else int(h)


def resolve_horizon_steps(config: "TrajectoryDataSourceConfig") -> int:
    """The collector horizon T for this pool, in control steps.

    Read from train_description.json / dataset_description.json next to the
    pool (the train.npz's directory, or the parent of a trajectories/ dir).
    An explicit `config.horizon_steps` must agree with them. Raises when no
    horizon is found or two sources disagree.
    """
    explicit = getattr(config, "horizon_steps", None)
    base = Path(config.trajectories_dir)
    roots = ([base] if base.is_dir() else []) + [base.parent]
    found: Dict[str, int] = {}
    for root in roots:
        for name in ("train_description.json", "dataset_description.json"):
            p = root / name
            if p.is_file():
                with open(p) as f:
                    h = _horizon_from_description(json.load(f))
                if h is not None:
                    found[str(p)] = h
        if found:
            break
    values = set(found.values())
    if explicit is not None:
        values.add(int(explicit))
    if not values:
        raise ValueError(
            f"timeout_intermediates='drop' needs the collector horizon, but no "
            f"horizon was found beside {base} and data_source.horizon_steps is unset.")
    if len(values) > 1:
        raise ValueError(
            f"Conflicting horizon values for {base}: {found}, explicit horizon_steps={explicit}.")
    return values.pop()


def load_eval_states(
    filepath: str,
    label_mapping: Optional[Dict[int, int]] = None,
    max_rows: Optional[int] = None,
    state_dim: Optional[int] = None,
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
        max_rows: Optional maximum number of rows to load (passed to np.loadtxt).
                  Default None loads all rows (unchanged behaviour).  Use to cap
                  large FPS files (e.g. the 1.3GB humanoid test set) without
                  reading the entire file into memory.

    Returns:
        Tuple of:
            start_states: [N, state_dim] array of start states
            end_states: [N, state_dim] array of end states
            labels: [N] array of labels in internal format (-1 = failure, 1 = success)
    """
    if label_mapping is None:
        label_mapping = {0: -1, 1: 1}

    # Load data (comma-separated); max_rows=None → numpy loads all rows (default).
    # ndmin=2 because np.loadtxt collapses a single-row file to 1-D, and the very
    # next line indexes shape[1] — a one-row eval/cal file raised IndexError here
    # regardless of state_dim. Multi-row files are unaffected.
    loadtxt_kwargs: dict = {"delimiter": ",", "ndmin": 2}
    if max_rows is not None:
        loadtxt_kwargs["max_rows"] = max_rows
    data = np.loadtxt(filepath, **loadtxt_kwargs)

    n_cols = data.shape[1]

    # ---- Disambiguated path: caller told us the state dimension --------------
    # Column count ALONE cannot identify the format. A 5-column file is
    # "2 start + 2 end + label" for a deterministic 2-D system and
    # "4 state + p_success" for a probabilistic 4-D system. Guessing picked the
    # first reading for stochastic cartpole and silently returned 2-D states for
    # a 4-D system, which surfaced far away as
    # `IndexError: index 2 is out of bounds` inside CartPoleSystem.normalize_state.
    # Pass state_dim whenever it is known; only the legacy branch below guesses.
    if state_dim is not None:
        if n_cols == state_dim + 1:
            start_states = data[:, :state_dim].astype(np.float32)
            p_success = data[:, state_dim]
            raw_labels = (p_success >= 0.5).astype(int)
            labels = np.array([label_mapping.get(l, 0) for l in raw_labels], dtype=np.int64)
            return start_states, None, labels
        if n_cols == 2 * state_dim + 1:
            start_states = data[:, :state_dim].astype(np.float32)
            end_states = data[:, state_dim:2 * state_dim].astype(np.float32)
            raw_labels = data[:, -1].astype(int)
            labels = np.array([label_mapping.get(l, 0) for l in raw_labels], dtype=np.int64)
            return start_states, end_states, labels
        raise ValueError(
            f"{filepath}: {n_cols} columns is neither the probabilistic layout "
            f"({state_dim} state + p_success = {state_dim + 1}) nor the deterministic "
            f"layout ({state_dim} start + {state_dim} end + label = {2 * state_dim + 1}) "
            f"for state_dim={state_dim}."
        )

    # ---- Legacy path: state_dim unknown, infer from column count ------------
    # Kept byte-identical so every existing caller behaves exactly as before.
    # Only correct while probabilistic files are 2-D (pendulum).
    #
    # Probabilistic format (stochastic pendulum): θ_s, θ̇_s, p_success — no end
    # states (each start has many stochastic rollouts). Labels are binarized at
    # p >= 0.5; end_states is returned as None and callers must guard on it.
    if n_cols == 3:
        start_states = data[:, :2].astype(np.float32)
        p_success = data[:, 2]
        raw_labels = (p_success >= 0.5).astype(int)
        labels = np.array([label_mapping.get(l, 0) for l in raw_labels], dtype=np.int64)
        return start_states, None, labels

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

    # Pairing of timeout trajectories' intermediate states; see
    # TIMEOUT_INTERMEDIATE_MODES. "drop" needs the horizon (from the dataset
    # description, or horizon_steps below).
    timeout_intermediates: str = "keep"
    horizon_steps: Optional[int] = None

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

        # Cache for trajectory lengths (keyed by idx); populated lazily via line-count
        # without a full array parse — used by the packed candidate-id scheme.
        self._length_cache: Dict[int, int] = {}

        print(f"TrajectoryDataSource initialized:")
        print(f"  Trajectories: {self.n_trajectories}")
        print(f"  Directory: {self.trajectories_dir}")
        if self.labels is not None:
            n_success = np.sum(self.labels == 1)
            n_failure = np.sum(self.labels == -1)
            print(f"  Labels: {n_success} success, {n_failure} failure")
        # Text pools: lengths are checked per trajectory as rows are built.
        self._init_timeout_policy()

    def _init_timeout_policy(
        self,
        lengths: Optional[np.ndarray] = None,
        timeout_flags: Optional[np.ndarray] = None,
    ) -> None:
        """Validate `timeout_intermediates` and, for "drop", fix the horizon.

        lengths: rows per pool trajectory, when the whole pool is known up front
            (npz). Checked against the horizon, and used for a log line.
        timeout_flags: the collector's own per-trajectory timeout flag, when the
            pool ships one (quadrotor3D ppo). Must agree with the length rule.
        """
        mode = getattr(self.config, "timeout_intermediates", None) or "keep"
        if mode not in TIMEOUT_INTERMEDIATE_MODES:
            raise ValueError(
                f"data_source.timeout_intermediates={mode!r}; expected one of "
                f"{TIMEOUT_INTERMEDIATE_MODES}")
        self.timeout_intermediates = mode
        self.horizon_steps = None
        if mode == "keep":
            return
        if self.labels is None:
            raise ValueError("timeout_intermediates='drop' needs shuffled_labels_file: a "
                             "trajectory cut at the horizon is a timeout only if it did not succeed.")
        self.horizon_steps = resolve_horizon_steps(self.config)
        if lengths is None:
            print(f"  Timeout intermediates: drop (horizon {self.horizon_steps} steps)")
            return
        lengths = np.asarray(lengths)
        too_long = lengths > self.horizon_steps + 1
        if too_long.any():
            raise ValueError(
                f"{int(too_long.sum())} pool trajectories have more than horizon+1 = "
                f"{self.horizon_steps + 1} rows (max {int(lengths.max())}); the horizon is wrong.")
        timeout = (lengths == self.horizon_steps + 1) & (self.labels != 1)
        if timeout_flags is not None:
            flagged = (np.asarray(timeout_flags) != 0) & (self.labels != 1)
            if not np.array_equal(flagged, timeout):
                raise ValueError(
                    f"The pool's timeout flag marks {int(flagged.sum())} unsuccessful "
                    f"trajectories, the horizon rule marks {int(timeout.sum())}; they must agree.")
        dropped = int((lengths[timeout] - 2).sum())
        print(f"  Timeout intermediates: drop (horizon {self.horizon_steps} steps): "
              f"{int(timeout.sum())} of {len(lengths)} pool trajectories time out; "
              f"{dropped} of {int((lengths - 1).sum())} pool rows would be dropped")

    def _drops_intermediates(self, idx: int, n_rows: int) -> bool:
        """True when trajectory idx is a timeout and its intermediate rows go."""
        # getattr: instances unpickled from before the attribute existed keep "keep".
        if getattr(self, "timeout_intermediates", "keep") != "drop":
            return False
        if n_rows > self.horizon_steps + 1:
            raise ValueError(f"Trajectory {idx} has {n_rows} rows, more than horizon+1 = "
                             f"{self.horizon_steps + 1}; the horizon is wrong.")
        return n_rows == self.horizon_steps + 1 and self.get_label(idx) != 1

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
        """Return the number of rows (time steps) in trajectory idx.

        Uses a cheap line-count (no full parse) when the trajectory is not
        already cached.  A non-empty line == one time step, matching the
        behaviour of ``load_trajectory`` exactly.

        Result is cached in ``_length_cache`` to avoid repeated file scans.
        Raises ``ValueError`` if the trajectory has more rows than ``MAX_ROWS``
        (packed candidate-id scheme invariant).
        """
        if idx in self._traj_cache:
            length = len(self._traj_cache[idx])
            self._length_cache[idx] = length
            return length
        if idx in self._length_cache:
            return self._length_cache[idx]
        filepath = self.trajectory_files[idx]
        count = 0
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        if count > MAX_ROWS:
            raise ValueError(
                f"Trajectory {idx} has {count} rows, which exceeds MAX_ROWS={MAX_ROWS}. "
                f"Increase MAX_ROWS or pre-process the data."
            )
        self._length_cache[idx] = count
        return count

    def get_state_at(self, idx: int, row: int) -> np.ndarray:
        """Return the state vector at a specific row in trajectory idx."""
        return self.load_trajectory(idx)[row]

    def trajectory_name(self, idx: int) -> str:
        """Stable name for trajectory idx (path relative to trajectories_dir).

        Written to the train/val trajectory index files consumed by local
        (trajectory) prediction mode.
        """
        return str(self.trajectory_files[idx].relative_to(self.trajectories_dir))

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

        For training: every state → final state (many pairs per trajectory),
        except that with timeout_intermediates="drop" a timeout trajectory gives
        only its first state (see TIMEOUT_INTERMEDIATE_MODES)
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
        if self._drops_intermediates(idx, len(traj)):
            # Only x_0's T-step future (x_T) is on record; none exists from row > 0.
            starts = traj[0:1] if start_row == 0 else traj[0:0]
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
        n_timeouts = n_dropped = 0

        for idx in indices:
            starts, ends = self.get_all_endpoint_pairs_from_trajectory(idx, mode)
            label = self.get_label(idx)

            all_starts.append(starts)
            all_ends.append(ends)
            all_labels.extend([label] * len(starts))
            if mode != "test":
                n_rows = self.get_trajectory_length(idx)
                if self._drops_intermediates(idx, n_rows):
                    n_timeouts += 1
                    n_dropped += n_rows - 2

        if getattr(self, "timeout_intermediates", "keep") == "drop" and mode != "test":
            print(f"  timeout_intermediates=drop: {n_timeouts} of {len(indices)} trajectories "
                  f"time out, {n_dropped} intermediate rows dropped")

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

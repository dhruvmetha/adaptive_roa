"""
Adaptive Dataset Builder for incremental training.

Manages trajectory indices and builds endpoint datasets for flow matching training.
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from src.adaptive.data_source import TrajectoryDataSource


class DatasetSplit:
    """Tracks indices for a dataset split."""

    def __init__(self, indices: Optional[List[int]] = None):
        self.indices: List[int] = list(indices) if indices else []

    def add(self, idx: int):
        if idx not in self.indices:
            self.indices.append(idx)

    def add_many(self, indices: List[int]):
        for idx in indices:
            self.add(idx)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


class AdaptiveDatasetBuilder:
    """
    Builds and manages datasets for adaptive sampling.

    Tracks which trajectory indices are assigned to train/val/test splits.
    Builds endpoint dataset files on demand for flow matcher training.

    Supports two sampling modes:
    1. Sequential (default): Uses pointer, indices 0, 1, 2, ... in sequence
    2. Balanced: Uses explicit "used" set, allows discarding without marking

    Key responsibilities:
    - Manage train/val/test index splits
    - Build endpoint datasets from selected indices
    - Track which indices have been used
    - Support incremental addition of new data

    Attributes:
        data_source: TrajectoryDataSource providing access to trajectory data
        train_split: Indices assigned to training
        output_dir: Directory for saving dataset files
        next_available_idx: Pointer to next unused index in sequential order
        used_indices: Set of indices that have been added to training
    """

    def __init__(
        self,
        data_source: TrajectoryDataSource,
        output_dir: str,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        """
        Initialize dataset builder.

        Args:
            data_source: TrajectoryDataSource for loading trajectories
            output_dir: Directory for saving built datasets
            val_ratio: Fraction of training set to use for validation (with overlap)
            test_ratio: Fraction of training set to use for testing (with overlap)
        """
        self.data_source = data_source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_split = DatasetSplit()
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Sequential pointer - tracks next available index (for sequential mode)
        self.next_available_idx = 0

        # Explicit used set - tracks indices added to training (for balanced mode)
        self.used_indices: set = set()

        # All trajectories available for training (no reserved sets)
        self.max_train_idx = self.data_source.n_trajectories

        print(f"Total trajectories available: {self.max_train_idx}")
        print(f"Val/test will be {val_ratio:.0%}/{test_ratio:.0%} of training set (with overlap)")

    def get_next_batch(self, n: int) -> List[int]:
        """
        Get next n trajectory indices in sequential order.

        This is the PRIMARY method for getting new data. It maintains
        the ordering from shuffled_indices.txt.

        Args:
            n: Number of trajectories to get

        Returns:
            List of trajectory indices (sequential from current pointer)
        """
        # Calculate how many we can actually get
        remaining = self.max_train_idx - self.next_available_idx
        n_actual = min(n, remaining)

        if n_actual == 0:
            print(f"WARNING: No more trajectories available for training!")
            return []

        if n_actual < n:
            print(f"WARNING: Only {n_actual} trajectories remaining (requested {n})")

        # Get sequential indices
        indices = list(range(self.next_available_idx, self.next_available_idx + n_actual))

        # Update pointer
        self.next_available_idx += n_actual

        return indices

    def add_to_training(self, indices: List[int]):
        """
        Add trajectory indices to training set AND mark as used.

        Args:
            indices: Trajectory indices to add
        """
        for idx in indices:
            if idx < self.max_train_idx:
                self.train_split.add(idx)
                self.used_indices.add(idx)  # Also mark as used
            else:
                print(f"WARNING: Index {idx} is reserved for val/test, skipping")

    # =========================================================================
    # Balanced Sampling Methods (for balanced_uncertain strategy)
    # =========================================================================

    def get_available_indices(self) -> List[int]:
        """
        Get all indices that have NOT been used (not in training).

        Returns:
            List of available trajectory indices
        """
        all_indices = set(range(self.max_train_idx))
        available = all_indices - self.used_indices
        return sorted(list(available))

    def get_n_available(self) -> int:
        """Get count of available (unused) indices."""
        return self.max_train_idx - len(self.used_indices)

    def sample_candidates_without_marking(
        self, 
        n: int, 
        exclude: set = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Sample n candidates from available pool WITHOUT marking as used.

        This allows evaluating candidates and discarding certain ones
        without permanently removing them from the pool.

        Args:
            n: Number of candidates to sample
            exclude: Optional set of indices to exclude (e.g., already sampled this epoch)

        Returns:
            Tuple of (start_states [n, dim], trajectory_indices)
        """
        # Get available indices as a set for efficient operations
        all_indices = set(range(self.max_train_idx))
        available_set = all_indices - self.used_indices
        
        # Exclude indices if provided (for within-epoch deduplication)
        if exclude:
            available_set = available_set - exclude
        
        # Convert to sorted list (maintains consistent ordering)
        available = sorted(available_set)

        if len(available) == 0:
            print("WARNING: No more trajectories available!")
            return np.array([]), []

        # Take first n from available (maintains shuffled order)
        n_actual = min(n, len(available))
        if n_actual < n:
            print(f"WARNING: Only {n_actual} trajectories remaining (requested {n})")

        indices = available[:n_actual]
        starts = self.data_source.get_start_states(indices)
        return starts, indices

    def mark_indices_as_used(self, indices: List[int]):
        """
        Mark indices as used (added to training).

        Call this ONLY when actually adding to training set.
        Discarded candidates should NOT be marked.

        Args:
            indices: Trajectory indices to mark as used
        """
        for idx in indices:
            self.used_indices.add(idx)

    def add_to_training_balanced(self, indices: List[int]):
        """
        Add indices to training set AND mark as used.

        Use this for balanced sampling strategy.

        Args:
            indices: Trajectory indices to add to training
        """
        for idx in indices:
            if idx < self.max_train_idx:
                self.train_split.add(idx)
                self.used_indices.add(idx)
            else:
                print(f"WARNING: Index {idx} out of range, skipping")

    def get_initial_training_set(self, n: int) -> List[int]:
        """
        Get initial training set of n trajectories (sequential from index 0).

        Args:
            n: Number of trajectories for initial training

        Returns:
            List of trajectory indices
        """
        indices = self.get_next_batch(n)
        self.add_to_training(indices)
        return indices

    def build_train_dataset(self, filename: str = "train_endpoint_dataset.txt") -> str:
        """
        Build training endpoint dataset file.

        Args:
            filename: Output filename

        Returns:
            Path to built dataset file
        """
        output_path = self.output_dir / filename

        n_pairs = self.data_source.save_endpoint_dataset(
            list(self.train_split),
            str(output_path),
            mode="train"
        )

        print(f"Built training dataset: {n_pairs} pairs from {len(self.train_split)} trajectories")
        return str(output_path)

    def build_val_dataset(self, filename: str = "val_endpoint_dataset.txt") -> str:
        """Build validation endpoint dataset file (subset of training with overlap)."""
        output_path = self.output_dir / filename

        # Take val_ratio of training indices
        train_indices = list(self.train_split)
        n_val = max(1, int(len(train_indices) * self.val_ratio))
        val_indices = train_indices[:n_val]  # First n_val from training

        n_pairs = self.data_source.save_endpoint_dataset(
            val_indices,
            str(output_path),
            mode="train"
        )

        print(f"Built validation dataset: {n_pairs} pairs from {n_val} trajectories (subset of training)")
        return str(output_path)

    def build_test_dataset(self, filename: str = "test_endpoint_dataset.txt") -> str:
        """Build test endpoint dataset file (subset of training with overlap)."""
        output_path = self.output_dir / filename

        # Take test_ratio of training indices
        train_indices = list(self.train_split)
        n_test = max(1, int(len(train_indices) * self.test_ratio))
        test_indices = train_indices[-n_test:]  # Last n_test from training

        n_pairs = self.data_source.save_endpoint_dataset(
            test_indices,
            str(output_path),
            mode="train"  # Use all states, not just first
        )

        print(f"Built test dataset: {n_pairs} pairs from {n_test} trajectories (subset of training)")
        return str(output_path)

    def build_all_datasets(self) -> Dict[str, str]:
        """
        Build all dataset files (train, val, test).

        Returns:
            Dict mapping split name to file path
        """
        return {
            'train': self.build_train_dataset(),
            'val': self.build_val_dataset(),
            'test': self.build_test_dataset(),
        }

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training data as numpy arrays (for conformal prediction).

        Returns:
            Tuple of (start_states, end_states, labels)
        """
        return self.data_source.build_endpoint_dataset(
            list(self.train_split),
            mode="train"
        )

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get test data as numpy arrays (subset of training with overlap).

        Returns:
            Tuple of (start_states, end_states, labels)
        """
        # Take test_ratio of training indices (last portion)
        train_indices = list(self.train_split)
        n_test = max(1, int(len(train_indices) * self.test_ratio))
        test_indices = train_indices[-n_test:]  # Last n_test from training
        
        return self.data_source.build_endpoint_dataset(
            test_indices,
            mode="test"
        )

    def get_train_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training start states and labels (for conformal prediction).

        Returns:
            Tuple of (start_states [N_traj, dim], labels [N_traj])
            Note: One per trajectory, not expanded
        """
        indices = list(self.train_split)
        starts = self.data_source.get_start_states(indices)
        labels = self.data_source.get_labels(indices)
        return starts, labels

    def get_test_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get test start states and labels (subset of training with overlap).

        Returns:
            Tuple of (start_states [N_traj, dim], labels [N_traj])
        """
        # Take test_ratio of training indices (last portion)
        train_indices = list(self.train_split)
        n_test = max(1, int(len(train_indices) * self.test_ratio))
        test_indices = train_indices[-n_test:]  # Last n_test from training

        starts = self.data_source.get_start_states(test_indices)
        labels = self.data_source.get_labels(test_indices)
        return starts, labels

    def get_candidate_states(self, n: int) -> Tuple[np.ndarray, List[int]]:
        """
        Get candidate start states from available pool for evaluation.

        Uses sequential sampling to maintain the ordering from shuffled_indices.txt.

        Args:
            n: Number of candidates to sample

        Returns:
            Tuple of (start_states [n, dim], trajectory_indices)
        """
        indices = self.get_next_batch(n)
        if len(indices) == 0:
            return np.array([]), []

        starts = self.data_source.get_start_states(indices)
        return starts, indices

    def add_selected_to_training(self, indices: List[int]):
        """
        Add selected trajectory indices to training set.

        Called after conformal prediction selects uncertain points.

        Args:
            indices: Trajectory indices to add
        """
        self.add_to_training(indices)

    def get_statistics(self) -> Dict:
        """Get current dataset statistics."""
        train_stats = self.data_source.get_statistics(list(self.train_split))

        # Available = indices not in used_indices (for balanced mode)
        available_balanced = self.get_n_available()
        # Available = indices from next_available_idx (for sequential mode)
        available_sequential = self.max_train_idx - self.next_available_idx

        return {
            'train_trajectories': len(self.train_split),
            'used_trajectories': len(self.used_indices),
            'available_trajectories': available_balanced,
            'available_sequential': available_sequential,
            'next_available_idx': self.next_available_idx,
            'max_train_idx': self.max_train_idx,
            'train_success_rate': train_stats['success_rate'],
        }

    def save_state(self, filepath: str):
        """Save current state (indices, pointer, and used set) to file."""
        import json
        state = {
            'train_indices': list(self.train_split),
            'used_indices': list(self.used_indices),
            'next_available_idx': self.next_available_idx,
            'max_train_idx': self.max_train_idx,
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load state (indices, pointer, and used set) from file."""
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.train_split = DatasetSplit(state['train_indices'])

        # Restore sequential pointer
        self.next_available_idx = state['next_available_idx']
        self.max_train_idx = state['max_train_idx']

        # Restore used indices set (with backward compatibility)
        if 'used_indices' in state:
            self.used_indices = set(state['used_indices'])
        else:
            # Backward compatibility: used_indices = train_indices
            self.used_indices = set(state['train_indices'])

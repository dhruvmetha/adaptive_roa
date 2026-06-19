"""
Adaptive Dataset Builder for incremental training.

Manages trajectory indices and builds endpoint datasets for flow matching training.
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from adaptive_roa.adaptive.data_source import TrajectoryDataSource


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

    Uses a set-based tracking approach (`used_indices`) that allows:
    - Sampling candidates without permanently marking them
    - Discarding "certain" points back to the pool
    - Keeping only "uncertain" points for training

    Key responsibilities:
    - Manage train/val/test index splits
    - Build endpoint datasets from selected indices
    - Track which indices have been used
    - Support incremental addition of new data

    Attributes:
        data_source: TrajectoryDataSource providing access to trajectory data
        train_split: Indices assigned to training
        output_dir: Directory for saving dataset files
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

        # Tracks indices added to training (used for sampling available candidates)
        self.used_indices: set = set()

        # All trajectories available for training (no reserved sets)
        self.max_train_idx = self.data_source.n_trajectories

        print(f"Total trajectories available: {self.max_train_idx}")
        print(f"Val will be {val_ratio:.0%} of training set (no overlap — FM trains on remaining {1-val_ratio:.0%})")

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
        # Take first n indices (0, 1, 2, ..., n-1)
        n_actual = min(n, self.max_train_idx)
        if n_actual < n:
            print(f"WARNING: Only {n_actual} trajectories available (requested {n})")

        indices = list(range(n_actual))
        self.add_to_training_balanced(indices)
        return indices

    def build_train_dataset(self, filename: str = "train_endpoint_dataset.txt") -> str:
        """
        Build training endpoint dataset file (excludes val indices).

        The first ``n_val`` indices of ``train_split`` are reserved for
        validation, so FM gradient updates only use the remaining portion.

        Args:
            filename: Output filename

        Returns:
            Path to built dataset file
        """
        output_path = self.output_dir / filename

        train_indices = list(self.train_split)
        n_val = max(1, int(len(train_indices) * self.val_ratio))
        train_only = train_indices[n_val:]  # Exclude first n_val (val portion)

        n_pairs = self.data_source.save_endpoint_dataset(
            train_only,
            str(output_path),
            mode="train"
        )

        print(f"Built training dataset: {n_pairs} pairs from {len(train_only)} trajectories (excl. {n_val} val)")
        return str(output_path)

    def build_val_dataset(self, filename: str = "val_endpoint_dataset.txt") -> str:
        """Build validation endpoint dataset file (no overlap with train)."""
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

    def build_train_classification_dataset(self, filename: str = "train_classification_dataset.txt") -> str:
        """Build training (state, binary_label) classification dataset (excludes val portion)."""
        output_path = self.output_dir / filename
        train_indices = list(self.train_split)
        n_val = max(1, int(len(train_indices) * self.val_ratio))
        train_only = train_indices[n_val:]
        n_rows = self.data_source.save_classification_dataset(train_only, str(output_path), mode="train")
        print(f"Built training classification dataset: {n_rows} (state,label) rows from {len(train_only)} trajectories (excl. {n_val} val)")
        return str(output_path)

    def build_val_classification_dataset(self, filename: str = "val_classification_dataset.txt") -> str:
        """Build validation (state, binary_label) classification dataset (no overlap with train)."""
        output_path = self.output_dir / filename
        train_indices = list(self.train_split)
        n_val = max(1, int(len(train_indices) * self.val_ratio))
        val_indices = train_indices[:n_val]
        n_rows = self.data_source.save_classification_dataset(val_indices, str(output_path), mode="train")
        print(f"Built validation classification dataset: {n_rows} (state,label) rows from {n_val} trajectories")
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

    def build_trajectory_index(self, indices: List[int], filename: str) -> str:
        """
        Write trajectory filenames for given indices to a file.

        Used by local (trajectory) prediction mode — the data module reads
        this file directly instead of reverse-engineering indices from
        endpoint files.

        Args:
            indices: Trajectory indices to include
            filename: Output filename

        Returns:
            Path to written file
        """
        output_path = self.output_dir / filename
        traj_dir = self.data_source.trajectories_dir
        with open(output_path, 'w') as f:
            for idx in indices:
                # Write path relative to trajectories_dir (handles subdirectories)
                rel_path = self.data_source.trajectory_files[idx].relative_to(traj_dir)
                f.write(f"{rel_path}\n")
        return str(output_path)

    def build_all_datasets(self, dataset_kind: str = "endpoint") -> Dict[str, str]:
        """
        Build all dataset files (train, val).

        dataset_kind="endpoint" (default): endpoint-pair files for flow matching
        plus trajectory index files for local mode, keyed as:
            'train', 'val' — endpoint pair files
            'train_trajectories', 'val_trajectories' — trajectory filename lists
        dataset_kind="classification": (state, binary_label) files for the
        discriminative classifier, keyed as 'train', 'val'.

        Returns:
            Dict mapping split name to file path
        """
        train_indices = list(self.train_split)
        n_val = max(1, int(len(train_indices) * self.val_ratio))
        train_only = train_indices[n_val:]
        val_indices = train_indices[:n_val]

        if dataset_kind == "classification":
            return {
                'train': self.build_train_classification_dataset(),
                'val': self.build_val_classification_dataset(),
            }

        return {
            'train': self.build_train_dataset(),
            'val': self.build_val_dataset(),
            'train_trajectories': self.build_trajectory_index(train_only, "train_trajectories.txt"),
            'val_trajectories': self.build_trajectory_index(val_indices, "val_trajectories.txt"),
        }

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train-only data as numpy arrays (excludes val portion).

        Returns the same indices used by build_train_dataset(), ensuring
        alignment between the train file and returned arrays.

        Returns:
            Tuple of (start_states, end_states, labels)
        """
        train_indices = list(self.train_split)
        n_val = max(1, int(len(train_indices) * self.val_ratio))
        train_only = train_indices[n_val:]
        return self.data_source.build_endpoint_dataset(train_only, mode="train")

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get validation endpoint data as numpy arrays.

        Returns the val portion (first ``n_val`` indices of ``train_split``),
        which the FM never trains on.

        Returns:
            Tuple of (start_states, end_states, labels)
        """
        train_indices = list(self.train_split)
        n_val = max(1, int(len(train_indices) * self.val_ratio))
        val_indices = train_indices[:n_val]
        return self.data_source.build_endpoint_dataset(val_indices, mode="train")

    def get_val_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get validation start states and labels (for threshold optimization).

        Returns the val portion (first ``n_val`` indices of ``train_split``),
        which the FM never trains on — one entry per trajectory.

        Returns:
            Tuple of (start_states [N_val, dim], labels [N_val])
        """
        train_indices = list(self.train_split)
        n_val = max(1, int(len(train_indices) * self.val_ratio))
        val_indices = train_indices[:n_val]
        starts = self.data_source.get_start_states(val_indices)
        labels = self.data_source.get_labels(val_indices)
        return starts, labels

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
        Get train-only start states and labels (excludes val portion).

        Returns the same indices used by build_train_dataset(), ensuring
        alignment between the train file and returned labels.

        Returns:
            Tuple of (start_states [N_traj, dim], labels [N_traj])
            Note: One per trajectory, not expanded
        """
        train_indices = list(self.train_split)
        n_val = max(1, int(len(train_indices) * self.val_ratio))
        train_only = train_indices[n_val:]
        starts = self.data_source.get_start_states(train_only)
        labels = self.data_source.get_labels(train_only)
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

        Takes the first n available (unused) indices, maintaining the
        ordering from shuffled_indices.txt.

        Note: This method does NOT mark indices as used. Call
        mark_indices_as_used() or add_to_training_balanced() after
        deciding which indices to keep.

        Args:
            n: Number of candidates to sample

        Returns:
            Tuple of (start_states [n, dim], trajectory_indices)
        """
        return self.sample_candidates_without_marking(n)

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

        return {
            'train_trajectories': len(self.train_split),
            'used_trajectories': len(self.used_indices),
            'available_trajectories': self.get_n_available(),
            'max_train_idx': self.max_train_idx,
            'train_success_rate': train_stats['success_rate'],
        }

    def save_state(self, filepath: str):
        """Save current state (indices and used set) to file."""
        import json
        state = {
            'train_indices': list(self.train_split),
            'used_indices': list(self.used_indices),
            'max_train_idx': self.max_train_idx,
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load state (indices and used set) from file."""
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.train_split = DatasetSplit(state['train_indices'])
        self.max_train_idx = state['max_train_idx']

        # Restore used indices set (with backward compatibility)
        if 'used_indices' in state:
            self.used_indices = set(state['used_indices'])
        else:
            # Backward compatibility: used_indices = train_indices
            self.used_indices = set(state['train_indices'])

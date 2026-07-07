"""
Adaptive Dataset Builder for incremental training.

Manages trajectory indices and builds endpoint datasets for flow matching training.
"""
import random
import warnings
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from adaptive_roa.adaptive.data_source import TrajectoryDataSource, MAX_ROWS


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
        candidate_mode: str = "start",
        seed: int = 42,
    ):
        """
        Initialize dataset builder.

        Args:
            data_source: TrajectoryDataSource for loading trajectories
            output_dir: Directory for saving built datasets
            val_ratio: Fraction of training set to use for validation (with overlap)
            test_ratio: Fraction of training set to use for testing (with overlap)
            candidate_mode: "start" (default, byte-identical to legacy behaviour) or
                            "intermediate" (maps opaque candidate-ids to (traj_idx, row)).
            seed: RNG seed for val/test split shuffle in intermediate mode (default 42).
        """
        self.data_source = data_source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_split = DatasetSplit()
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.candidate_mode = candidate_mode
        self.seed = seed

        # Tracks indices added to training (used for sampling available candidates)
        self.used_indices: set = set()

        # All trajectories available for training (no reserved sets)
        self.max_train_idx = self.data_source.n_trajectories

        print(f"Total trajectories available: {self.max_train_idx}")
        print(f"Val will be {val_ratio:.0%} of training set (no overlap — FM trains on remaining {1-val_ratio:.0%})")

        # ---- intermediate mode state ----------------------------------------
        if self.candidate_mode == "intermediate":
            # Packed candidate-id scheme: cid = traj_idx * MAX_ROWS + row
            # No flat registry is built; lengths are computed lazily via line-count.
            # See traj_row_to_candidate / candidate_to_traj_row for encoding.

            # Per-trajectory smallest marked row (marks row and all later rows
            # as used for that trajectory).  Absent key means nothing marked.
            self._marked_from_row: Dict[int, int] = {}

            # Per-trajectory minimum added row (populated by add_to_training_balanced)
            self._added_min_row: Dict[int, int] = {}

    # =========================================================================
    # Intermediate-mode helpers (candidate-id ↔ (traj_idx, row))
    # =========================================================================

    def traj_row_to_candidate(self, traj_idx: int, row: int) -> int:
        """Return the opaque candidate-id for (traj_idx, row).

        Packed encoding: ``cid = traj_idx * MAX_ROWS + row``.
        O(1), no registry required.

        Only valid in ``candidate_mode="intermediate"``.
        """
        return traj_idx * MAX_ROWS + row

    def candidate_to_traj_row(self, cid: int) -> Tuple[int, int]:
        """Return (traj_idx, row) for a candidate-id.

        Inverse of packed encoding: ``divmod(cid, MAX_ROWS)``.
        O(1), no registry required.

        Only valid in ``candidate_mode="intermediate"``.
        """
        return divmod(cid, MAX_ROWS)

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
            Tuple of (start_states [n, dim], trajectory_indices or candidate_ids)
        """
        if self.candidate_mode == "intermediate":
            # Guard n<=0: matches start mode (min(n,...) -> [:0] -> []) and the
            # old intermediate behaviour. Without this, the collect-until-n loop
            # never terminates early (len starts at 0) and returns ALL candidates.
            if n <= 0:
                return np.array([]), []
            # Packed candidate-id scheme: iterate trajectories in index order,
            # then rows in ascending order.  Available non-terminal rows for
            # trajectory i are [0, _marked_from_row.get(i, length-1)).
            # We skip cids in exclude_set and collect until n are found.
            exclude_set = set(exclude) if exclude else set()
            selected_ids = []
            selected_states = []
            for traj_idx in range(self.max_train_idx):
                # Upper bound on available rows (exclusive): smallest marked row,
                # or length-1 when nothing is marked (length-1 is the terminal row).
                if traj_idx in self._marked_from_row:
                    row_limit = self._marked_from_row[traj_idx]
                else:
                    traj_len = self.data_source.get_trajectory_length(traj_idx)
                    row_limit = traj_len - 1
                # Available rows are 0 .. row_limit-1 (non-terminal rows not marked)
                for row in range(row_limit):
                    cid = self.traj_row_to_candidate(traj_idx, row)
                    if cid in exclude_set:
                        continue
                    selected_ids.append(cid)
                    selected_states.append(self.data_source.get_state_at(traj_idx, row))
                    if len(selected_ids) == n:
                        break
                if len(selected_ids) == n:
                    break
            if len(selected_ids) == 0:
                print("WARNING: No more candidates available!")
                return np.array([]), []
            if len(selected_ids) < n:
                print(f"WARNING: Only {len(selected_ids)} candidates remaining (requested {n})")
            states = np.array(selected_states)
            return states, selected_ids

        # ---- start mode (unchanged) ----
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

        In ``intermediate`` mode: marking candidate-id (i, t) marks ALL
        candidate-ids with traj==i and row>=t (the tail of the trajectory).

        Args:
            indices: Trajectory indices (start mode) or candidate-ids (intermediate mode)
        """
        if self.candidate_mode == "intermediate":
            for cid in indices:
                traj_i, row_t = self.candidate_to_traj_row(cid)
                # Record the smallest marked row for this trajectory.
                # All rows >= that value are considered used.
                current = self._marked_from_row.get(
                    traj_i,
                    self.data_source.get_trajectory_length(traj_i) - 1
                )
                self._marked_from_row[traj_i] = min(current, row_t)
            return

        # ---- start mode (unchanged) ----
        for idx in indices:
            self.used_indices.add(idx)

    def add_to_training_balanced(self, indices: List[int]):
        """
        Add indices to training set AND mark as used.

        Use this for balanced sampling strategy.

        In ``intermediate`` mode: for each candidate-id (i, t), updates
        per-trajectory ``_added_min_row[i] = min(existing, t)`` and then
        marks the tail via ``mark_indices_as_used``.

        Args:
            indices: Trajectory indices (start mode) or candidate-ids (intermediate mode)
        """
        if self.candidate_mode == "intermediate":
            for cid in indices:
                traj_i, row_t = self.candidate_to_traj_row(cid)
                # Update per-trajectory minimum added row
                traj_len = self.data_source.get_trajectory_length(traj_i)
                current_added = self._added_min_row.get(traj_i, traj_len - 1)
                self._added_min_row[traj_i] = min(current_added, row_t)
                # Mark the tail (row_t and beyond) as used for this trajectory
                self.mark_indices_as_used([cid])
            return

        # ---- start mode (unchanged) ----
        for idx in indices:
            if idx < self.max_train_idx:
                self.train_split.add(idx)
                self.used_indices.add(idx)
            else:
                print(f"WARNING: Index {idx} out of range, skipping")

    def get_initial_training_set(self, n: int) -> List[int]:
        """
        Get initial training set of n trajectories (sequential from index 0).

        In ``start`` mode: adds trajectory indices 0..n-1 (byte-identical to
        legacy behaviour).

        In ``intermediate`` mode: seeds n DISTINCT trajectories, each at row 0
        (full tail), so that the initial dataset covers n different trajectories
        rather than n rows within trajectory 0.

        Args:
            n: Number of trajectories for initial training

        Returns:
            List of trajectory indices (start mode) or candidate-ids (intermediate mode)
        """
        n_actual = min(n, self.max_train_idx)
        if n_actual < n:
            print(f"WARNING: Only {n_actual} trajectories available (requested {n})")

        if self.candidate_mode == "intermediate":
            # Seed n distinct trajectories at row 0 (full tails).
            # Passing list(range(n)) would encode integers 0..n-1 as candidate-ids
            # (traj_idx * MAX_ROWS + row) which maps to very small traj/row values
            # — wrong. Explicitly convert trajectory index i to its row-0 candidate-id.
            cids = [self.traj_row_to_candidate(i, 0) for i in range(n_actual)]
            self.add_to_training_balanced(cids)
            return cids

        # ---- start mode (unchanged) ----
        indices = list(range(n_actual))
        self.add_to_training_balanced(indices)
        return indices

    def build_train_dataset(self, filename: str = "train_endpoint_dataset.txt") -> str:
        """
        Build training endpoint dataset file (excludes val indices).

        The first ``n_val`` indices of ``train_split`` are reserved for
        validation, so FM gradient updates only use the remaining portion.

        In ``intermediate`` mode: writes tail-expanded pairs per trajectory
        (from ``_added_min_row[i]`` to the penultimate row, all pointing to
        final state).  val_ratio is intentionally ignored in this mode.

        Args:
            filename: Output filename

        Returns:
            Path to built dataset file
        """
        output_path = self.output_dir / filename

        if self.candidate_mode == "intermediate":
            train, _val, _test = self._intermediate_split()
            if train:
                starts = np.vstack([p[0] for p in train])
                ends = np.vstack([p[1] for p in train])
                data = np.hstack([starts, ends])
            else:
                state_dim = self.data_source.get_state_dim()
                data = np.zeros((0, state_dim * 2), dtype=np.float32)
            np.savetxt(str(output_path), data, fmt='%.8f')
            n_pairs = len(data)
            print(f"Built intermediate training dataset: {n_pairs} pairs from {len(self._added_min_row)} trajectories")
            return str(output_path)

        # ---- start mode (unchanged) ----
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

        if self.candidate_mode == "intermediate":
            _train, val, _test = self._intermediate_split()
            if val:
                starts = np.vstack([p[0] for p in val])
                ends = np.vstack([p[1] for p in val])
                data = np.hstack([starts, ends])
            else:
                state_dim = self.data_source.get_state_dim()
                data = np.zeros((0, state_dim * 2), dtype=np.float32)
            np.savetxt(str(output_path), data, fmt='%.8f')
            n_pairs = len(data)
            print(f"Built intermediate validation dataset: {n_pairs} pairs")
            return str(output_path)

        # ---- start mode (unchanged) ----
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

    def _build_empty_val_dataset(self, filename: str) -> str:
        """Write an empty val file (used in start-mode when val is out of scope)."""
        output_path = self.output_dir / filename
        output_path.write_text("")
        return str(output_path)

    # =========================================================================
    # Intermediate-mode: single source of truth for train/val/test split
    # =========================================================================

    def _intermediate_split(self):
        """Return (train_pairs, val_pairs, test_pairs) for intermediate mode.

        Each element is a list of (start_state, end_state, label) tuples built
        from the ordered pair list P:
            P = [(rows[r], final_i, label_i)
                 for each trajectory i in _added_min_row,
                 for r in range(_added_min_row[i], len_i - 1)]

        Slicing (P ordered as built):
            n_val  = max(1, int(N * val_ratio))  if val_ratio > 0 and N > 0, else 0
            n_test = int(N * test_ratio)          (plain floor, no max-1 guard)
            val  = P[:n_val]
            test = P[n_val : n_val + n_test]
            train= P[n_val + n_test :]

        This is the single source of truth; all intermediate-mode get_*/build_*
        methods consume it.

        Returns:
            (train, val, test) — each a list of (start_arr, end_arr, label_int)
        """
        # Build P: list of (start_state, end_state, label)
        P: List[Tuple[np.ndarray, np.ndarray, int]] = []
        for traj_i, min_row in sorted(self._added_min_row.items()):
            starts, ends = self.data_source.get_all_endpoint_pairs_from_trajectory(
                traj_i, mode="train", start_row=min_row
            )
            label = self.data_source.get_label(traj_i)
            for s, e in zip(starts, ends):
                P.append((s, e, label))

        # Seeded shuffle so val/test draw is representative across all trajectories
        # rather than being biased toward the lowest-index trajectories' earliest rows.
        # A fresh Random instance per call ensures every caller (build_train,
        # build_val, get_val_labels, …) sees the identical shuffle, giving a
        # consistent split without a persistent RNG attribute.
        _rng = random.Random(self.seed if self.seed is not None else 42)
        _rng.shuffle(P)

        N = len(P)
        # val: non-empty when val_ratio > 0 and N > 0
        if self.val_ratio > 0 and N > 0:
            n_val = max(1, int(N * self.val_ratio))
        else:
            n_val = 0
        # test: plain floor, no max(1,...) — matches task spec
        n_test = int(N * self.test_ratio)

        val = P[:n_val]
        test = P[n_val:n_val + n_test]
        train = P[n_val + n_test:]
        return train, val, test

    def build_all_datasets(self, dataset_kind: str = "endpoint") -> Dict[str, str]:
        """
        Build all dataset files (train, val).

        dataset_kind="endpoint" (default): endpoint-pair files for flow matching
        plus trajectory index files for local mode, keyed as:
            'train', 'val' — endpoint pair files
            'train_trajectories', 'val_trajectories' — trajectory filename lists
        dataset_kind="classification": (state, binary_label) files for the
        discriminative classifier, keyed as 'train', 'val'.

        In ``intermediate`` mode: train and val files are built from the
        intermediate-state pair split (non-empty val when val_ratio>0 and
        data available); trajectory-index files are empty (local mode is out
        of scope for intermediate).

        Returns:
            Dict mapping split name to file path
        """
        if self.candidate_mode == "intermediate":
            if dataset_kind == "classification":
                # build_train/val_dataset emit endpoint pairs (start,end), NOT
                # (state,label) classification rows, so silently returning them
                # here would hand malformed data to the classifier trainer.
                raise NotImplementedError(
                    "candidate_mode='intermediate' does not support "
                    "dataset_kind='classification': the intermediate builders emit "
                    "endpoint pairs (start,end), not (state,label) classification rows. "
                    "Use dataset_kind='endpoint' (FM predictor), or candidate_mode='start' "
                    "for the classifier predictor."
                )
            if self.test_ratio > 0:
                # The intermediate test slice is not consumed by the full-ROA engine
                # (which evaluates on the FPS test_set_file), so with test_ratio>0
                # those pairs are excluded from training with nothing reading them.
                warnings.warn(
                    "candidate_mode='intermediate' with test_ratio>0: the intermediate "
                    "test slice is unused by the full-ROA engine, so those pairs are "
                    "silently dropped from training. Set test_ratio=0 for intermediate mode.",
                    stacklevel=2,
                )
            train_path = self.build_train_dataset()
            val_path = self.build_val_dataset()
            empty_traj_path = self._build_empty_val_dataset("val_trajectories.txt")
            return {
                'train': train_path,
                'val': val_path,
                'train_trajectories': self._build_empty_val_dataset("train_trajectories.txt"),
                'val_trajectories': empty_traj_path,
            }

        # ---- start mode (unchanged) ----
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

        In ``intermediate`` mode: returns the train slice from the
        intermediate-state pair split (pairs after the val and test slices
        are removed from the front).

        Returns:
            Tuple of (start_states, end_states, labels)
        """
        if self.candidate_mode == "intermediate":
            train, _val, _test = self._intermediate_split()
            if not train:
                state_dim = self.data_source.get_state_dim()
                return (
                    np.zeros((0, state_dim), dtype=np.float32),
                    np.zeros((0, state_dim), dtype=np.float32),
                    np.zeros(0, dtype=np.int64),
                )
            return (
                np.vstack([p[0] for p in train]),
                np.vstack([p[1] for p in train]),
                np.array([p[2] for p in train], dtype=np.int64),
            )

        # ---- start mode (unchanged) ----
        train_indices = list(self.train_split)
        n_val = max(1, int(len(train_indices) * self.val_ratio))
        train_only = train_indices[n_val:]
        return self.data_source.build_endpoint_dataset(train_only, mode="train")

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get validation endpoint data as numpy arrays.

        Returns the val portion (first ``n_val`` indices of ``train_split``),
        which the FM never trains on.

        In ``intermediate`` mode: returns intermediate-state pairs from the
        train-val split (non-empty when val_ratio > 0 and data available).

        Returns:
            Tuple of (start_states, end_states, labels)
        """
        if self.candidate_mode == "intermediate":
            _train, val, _test = self._intermediate_split()
            if not val:
                state_dim = self.data_source.get_state_dim()
                return (
                    np.zeros((0, state_dim), dtype=np.float32),
                    np.zeros((0, state_dim), dtype=np.float32),
                    np.zeros(0, dtype=np.int64),
                )
            return (
                np.vstack([p[0] for p in val]),
                np.vstack([p[1] for p in val]),
                np.array([p[2] for p in val], dtype=np.int64),
            )

        # ---- start mode (unchanged) ----
        train_indices = list(self.train_split)
        n_val = max(1, int(len(train_indices) * self.val_ratio))
        val_indices = train_indices[:n_val]
        return self.data_source.build_endpoint_dataset(val_indices, mode="train")

    def get_val_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get validation start states and labels (for threshold optimization).

        Returns the val portion (first ``n_val`` indices of ``train_split``),
        which the FM never trains on — one entry per trajectory.

        In ``intermediate`` mode: returns intermediate states (not necessarily
        row 0) paired with their trajectory's label.

        Returns:
            Tuple of (start_states [N_val, dim], labels [N_val])
        """
        if self.candidate_mode == "intermediate":
            _train, val, _test = self._intermediate_split()
            if not val:
                state_dim = self.data_source.get_state_dim()
                return (
                    np.zeros((0, state_dim), dtype=np.float32),
                    np.zeros(0, dtype=np.int64),
                )
            return (
                np.vstack([p[0] for p in val]),
                np.array([p[2] for p in val], dtype=np.int64),
            )

        # ---- start mode (unchanged) ----
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

    def get_labels(self, ids: List[int]) -> np.ndarray:
        """
        Get labels for a list of ids.

        In ``start`` mode: ids are trajectory indices; delegates directly to
        ``data_source.get_labels(ids)`` (byte-identical to previous behaviour
        of ``TrajectoryPool.get_labels``).

        In ``intermediate`` mode: ids are opaque candidate-ids; each is mapped
        to its trajectory index via ``candidate_to_traj_row`` and the
        trajectory label is returned.

        Args:
            ids: Trajectory indices (start mode) or candidate-ids (intermediate mode)

        Returns:
            labels [N] array
        """
        if self.candidate_mode == "intermediate":
            return np.array(
                [self.data_source.get_label(self.candidate_to_traj_row(cid)[0]) for cid in ids],
                dtype=np.int64,
            )
        # ---- start mode (unchanged) ----
        return self.data_source.get_labels(ids)

    def get_test_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get test start states and labels (subset of training with overlap).

        In ``intermediate`` mode: returns intermediate-state pairs from the
        test slice of the split; empty when test_ratio == 0 (the default).

        Returns:
            Tuple of (start_states [N, dim], labels [N])
        """
        if self.candidate_mode == "intermediate":
            _train, _val, test = self._intermediate_split()
            if not test:
                state_dim = self.data_source.get_state_dim()
                return (
                    np.zeros((0, state_dim), dtype=np.float32),
                    np.zeros(0, dtype=np.int64),
                )
            return (
                np.vstack([p[0] for p in test]),
                np.array([p[2] for p in test], dtype=np.int64),
            )

        # ---- start mode (unchanged) ----
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

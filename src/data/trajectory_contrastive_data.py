"""
Trajectory Contrastive Dataset for Self-Supervised Representation Learning

Implements temporal triplet sampling from trajectory data:
- Anchors: Any timestep in a trajectory
- Positives: States within ±k timesteps of anchor (temporal neighbors)
- Negatives: States outside the temporal window (different time or trajectory)

Follows codebase pattern:
- Dataset returns RAW states (no normalization)
- Normalization happens during training via system.normalize_state()
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
import lightning.pytorch as pl
from tqdm import tqdm
import random


class TrajectoryContrastiveDataset(Dataset):
    """
    Dataset for trajectory-based contrastive learning with triplet sampling.

    Supports temporal triplet mining:
    - Positives: States within ±k timesteps of anchor
    - Negatives: States outside temporal window (same or different trajectory)

    Returns RAW states (following flow matching pattern).
    Normalization is handled during training by Lightning module.
    """

    def __init__(self,
                 trajectory_dir: str,
                 state_dim: int,
                 temporal_window: int = 10,
                 num_negatives: int = 1,
                 max_trajectories: Optional[int] = None,
                 trajectory_subsample: int = 1,
                 cache_trajectories: bool = True,
                 split: Optional[str] = None,
                 val_split_ratio: float = 0.1,
                 seed: int = 42):
        """
        Initialize trajectory contrastive dataset

        Args:
            trajectory_dir: Directory containing sequence_*.txt files
            state_dim: Dimension of state vectors (e.g., 67 for humanoid, 4 for cartpole)
            temporal_window: ±k timesteps for positive sampling (default: 10)
            num_negatives: Number of negative samples per anchor (default: 1)
            max_trajectories: Maximum number of trajectories to load (None = all)
            trajectory_subsample: Load every Nth trajectory (default: 1 = all)
            cache_trajectories: Cache trajectories in memory (default: True)
            split: Which split to use - "train", "val", or None (uses all data)
            val_split_ratio: Fraction of data to use for validation (default: 0.1 = 10%)
            seed: Random seed for reproducible train/val split (default: 42)
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.state_dim = state_dim
        self.temporal_window = temporal_window
        self.num_negatives = num_negatives
        self.cache_trajectories = cache_trajectories

        # Find all trajectory files
        trajectory_files = sorted(self.trajectory_dir.glob("sequence_*.txt"))

        # Subsample trajectories if requested
        if trajectory_subsample > 1:
            trajectory_files = trajectory_files[::trajectory_subsample]

        # Apply train/val split if requested (before max_trajectories limit)
        if split is not None:
            # Use deterministic shuffle for reproducible splits
            rng = np.random.RandomState(seed)
            indices = np.arange(len(trajectory_files))
            rng.shuffle(indices)

            # Split indices
            n_val = int(len(indices) * val_split_ratio)
            val_indices = set(indices[:n_val])
            train_indices = set(indices[n_val:])

            if split == "train":
                trajectory_files = [trajectory_files[i] for i in sorted(train_indices)]
                print(f"Using TRAIN split: {len(trajectory_files)} trajectories ({100*(1-val_split_ratio):.0f}%)")
            elif split == "val":
                trajectory_files = [trajectory_files[i] for i in sorted(val_indices)]
                print(f"Using VAL split: {len(trajectory_files)} trajectories ({100*val_split_ratio:.0f}%)")
            else:
                raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or None")

        # Limit number of trajectories if requested
        if max_trajectories is not None:
            trajectory_files = trajectory_files[:max_trajectories]

        print(f"Loading {len(trajectory_files)} trajectories from {trajectory_dir}")

        # Load trajectory metadata
        self.trajectory_metadata = []  # List of (traj_id, num_timesteps)
        self.trajectories = [] if cache_trajectories else None
        self.trajectory_files = []  # Store file paths for on-the-fly loading

        for traj_id, traj_file in enumerate(tqdm(trajectory_files, desc="Loading trajectories")):
            # Load trajectory
            trajectory = self._load_trajectory(traj_file)

            if trajectory is not None and len(trajectory) > 0:
                num_timesteps = len(trajectory)
                self.trajectory_metadata.append((traj_id, num_timesteps))
                self.trajectory_files.append(traj_file)

                if cache_trajectories:
                    self.trajectories.append(trajectory)

        # Build flat index: list of (traj_id, timestep_idx)
        self.anchor_indices = []
        for traj_id, num_timesteps in self.trajectory_metadata:
            # Only use timesteps that have valid positive neighbors
            # Exclude first/last temporal_window timesteps to ensure valid positives
            for t in range(self.temporal_window, num_timesteps - self.temporal_window):
                self.anchor_indices.append((traj_id, t))

        print(f"Dataset initialized:")
        print(f"  Trajectories: {len(self.trajectory_metadata)}")
        print(f"  Total anchors: {len(self.anchor_indices)}")
        print(f"  Temporal window: ±{temporal_window} timesteps")
        print(f"  Negatives per anchor: {num_negatives}")
        print(f"  State dimension: {state_dim}")
        print(f"  Cached in memory: {cache_trajectories}")

    def _load_trajectory(self, trajectory_file: Path) -> Optional[np.ndarray]:
        """Load trajectory from file"""
        try:
            # Load as numpy array (each row is a state)
            trajectory = np.loadtxt(trajectory_file, delimiter=',')

            # Handle both 1D and 2D arrays
            if trajectory.ndim == 1:
                trajectory = trajectory.reshape(1, -1)

            # Verify dimensions match system
            if trajectory.shape[1] != self.state_dim:
                print(f"Warning: Trajectory {trajectory_file} has wrong dimensions "
                      f"({trajectory.shape[1]} != {self.state_dim}), skipping")
                return None

            return trajectory

        except Exception as e:
            print(f"Error loading {trajectory_file}: {e}")
            return None

    def _get_state(self, traj_id: int, timestep: int) -> np.ndarray:
        """Get state from trajectory (either cached or load from disk)"""
        if self.cache_trajectories:
            return self.trajectories[traj_id][timestep]
        else:
            # Load trajectory on-the-fly (slower but memory efficient)
            traj_file = self.trajectory_files[traj_id]
            trajectory = self._load_trajectory(traj_file)
            return trajectory[timestep] if trajectory is not None else None

    def __len__(self):
        return len(self.anchor_indices)

    def __getitem__(self, idx):
        """
        Sample triplet: (anchor, positive, negative)

        Returns RAW states (no normalization - follows flow matching pattern).
        Normalization happens during training via system.normalize_state().

        Returns:
            Dictionary with:
                - anchor: [state_dim] RAW anchor state
                - positive: [state_dim] RAW positive state (within ±k timesteps)
                - negative: [state_dim] RAW negative state (outside window)
                - metadata: Additional info (traj_id, timestep, etc.)
        """
        # Get anchor
        traj_id, anchor_t = self.anchor_indices[idx]
        anchor_state = self._get_state(traj_id, anchor_t)

        # Sample positive: within ±temporal_window timesteps
        positive_offset = random.randint(-self.temporal_window, self.temporal_window)
        # Ensure offset is non-zero (positive != anchor)
        while positive_offset == 0:
            positive_offset = random.randint(-self.temporal_window, self.temporal_window)

        positive_t = anchor_t + positive_offset
        positive_state = self._get_state(traj_id, positive_t)

        # Sample negative(s): outside temporal window
        # Strategy: randomly choose from other trajectories or far timesteps
        negatives = []
        for _ in range(self.num_negatives):
            # 50% chance: sample from different trajectory
            # 50% chance: sample from far timestep in same trajectory
            if random.random() < 0.5 or len(self.trajectory_metadata) == 1:
                # Sample from far timestep in same trajectory
                _, num_timesteps = self.trajectory_metadata[traj_id]

                # Define "far" as outside [anchor_t - 2*window, anchor_t + 2*window]
                far_range_min = max(self.temporal_window, anchor_t - 2 * self.temporal_window)
                far_range_max = min(num_timesteps - self.temporal_window,
                                   anchor_t + 2 * self.temporal_window)

                # Sample from valid range
                if far_range_min < self.temporal_window:
                    # Sample from end of trajectory
                    negative_t = random.randint(far_range_max, num_timesteps - self.temporal_window - 1)
                elif far_range_max >= num_timesteps - self.temporal_window:
                    # Sample from start of trajectory
                    negative_t = random.randint(self.temporal_window, far_range_min)
                else:
                    # Sample from either end
                    if random.random() < 0.5:
                        negative_t = random.randint(self.temporal_window, far_range_min)
                    else:
                        negative_t = random.randint(far_range_max, num_timesteps - self.temporal_window - 1)

                negative_state = self._get_state(traj_id, negative_t)
            else:
                # Sample from different trajectory
                neg_traj_id = random.randint(0, len(self.trajectory_metadata) - 1)
                while neg_traj_id == traj_id:
                    neg_traj_id = random.randint(0, len(self.trajectory_metadata) - 1)

                _, neg_num_timesteps = self.trajectory_metadata[neg_traj_id]
                negative_t = random.randint(self.temporal_window,
                                           neg_num_timesteps - self.temporal_window - 1)
                negative_state = self._get_state(neg_traj_id, negative_t)

            negatives.append(negative_state)

        # Convert to tensors (RAW states - no normalization!)
        anchor = torch.tensor(anchor_state, dtype=torch.float32)
        positive = torch.tensor(positive_state, dtype=torch.float32)

        # Handle multiple negatives
        if self.num_negatives == 1:
            negative = torch.tensor(negatives[0], dtype=torch.float32)
        else:
            negative = torch.stack([torch.tensor(neg, dtype=torch.float32) for neg in negatives])

        return {
            'anchor': anchor,           # [state_dim] RAW state
            'positive': positive,       # [state_dim] RAW state
            'negative': negative,       # [state_dim] or [num_negatives, state_dim] RAW states
            'metadata': {
                'traj_id': traj_id,
                'anchor_t': anchor_t,
                'positive_t': positive_t,
            }
        }


class TrajectoryContrastiveDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for trajectory contrastive learning

    Supports train/val/test splits with separate trajectory directories,
    or automatic splitting from a single directory using val_split_ratio.
    """

    def __init__(self,
                 train_trajectory_dir: str,
                 val_trajectory_dir: Optional[str] = None,
                 test_trajectory_dir: Optional[str] = None,
                 state_dim: int = 67,
                 temporal_window: int = 10,
                 num_negatives: int = 1,
                 max_trajectories_train: Optional[int] = None,
                 max_trajectories_val: Optional[int] = None,
                 max_trajectories_test: Optional[int] = None,
                 trajectory_subsample: int = 1,
                 batch_size: int = 256,
                 val_batch_size: Optional[int] = None,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 cache_trajectories: bool = True,
                 use_split: bool = False,
                 val_split_ratio: float = 0.1,
                 seed: int = 42):
        """
        Initialize trajectory contrastive data module

        Args:
            train_trajectory_dir: Directory with training trajectories
            val_trajectory_dir: Directory with validation trajectories (optional)
            test_trajectory_dir: Directory with test trajectories (optional)
            state_dim: Dimension of state vectors
            temporal_window: ±k timesteps for positive sampling
            num_negatives: Number of negative samples per anchor
            max_trajectories_train: Max training trajectories to load
            max_trajectories_val: Max validation trajectories to load
            max_trajectories_test: Max test trajectories to load
            trajectory_subsample: Load every Nth trajectory
            batch_size: Training batch size
            val_batch_size: Validation batch size (defaults to batch_size)
            num_workers: DataLoader workers
            pin_memory: Pin memory for DataLoader
            cache_trajectories: Cache trajectories in memory
            use_split: If True, split train_trajectory_dir into train/val sets
            val_split_ratio: Fraction of data for validation when use_split=True
            seed: Random seed for reproducible splits
        """
        super().__init__()
        self.train_trajectory_dir = train_trajectory_dir
        self.val_trajectory_dir = val_trajectory_dir or train_trajectory_dir
        self.test_trajectory_dir = test_trajectory_dir or val_trajectory_dir or train_trajectory_dir

        self.state_dim = state_dim
        self.temporal_window = temporal_window
        self.num_negatives = num_negatives
        self.max_trajectories_train = max_trajectories_train
        self.max_trajectories_val = max_trajectories_val
        self.max_trajectories_test = max_trajectories_test
        self.trajectory_subsample = trajectory_subsample
        self.cache_trajectories = cache_trajectories

        # Split settings
        self.use_split = use_split
        self.val_split_ratio = val_split_ratio
        self.seed = seed

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            print("\n=== Setting up training dataset ===")
            self.train_dataset = TrajectoryContrastiveDataset(
                trajectory_dir=self.train_trajectory_dir,
                state_dim=self.state_dim,
                temporal_window=self.temporal_window,
                num_negatives=self.num_negatives,
                max_trajectories=self.max_trajectories_train,
                trajectory_subsample=self.trajectory_subsample,
                cache_trajectories=self.cache_trajectories,
                split="train" if self.use_split else None,
                val_split_ratio=self.val_split_ratio,
                seed=self.seed
            )

            print("\n=== Setting up validation dataset ===")
            # If use_split is True, use same dir with "val" split
            # Otherwise use val_trajectory_dir (which may be same or different)
            val_dir = self.train_trajectory_dir if self.use_split else self.val_trajectory_dir
            self.val_dataset = TrajectoryContrastiveDataset(
                trajectory_dir=val_dir,
                state_dim=self.state_dim,
                temporal_window=self.temporal_window,
                num_negatives=self.num_negatives,
                max_trajectories=self.max_trajectories_val,
                trajectory_subsample=self.trajectory_subsample,
                cache_trajectories=self.cache_trajectories,
                split="val" if self.use_split else None,
                val_split_ratio=self.val_split_ratio,
                seed=self.seed
            )

        if stage == "test" or stage is None:
            print("\n=== Setting up test dataset ===")
            self.test_dataset = TrajectoryContrastiveDataset(
                trajectory_dir=self.test_trajectory_dir,
                state_dim=self.state_dim,
                temporal_window=self.temporal_window,
                num_negatives=self.num_negatives,
                max_trajectories=self.max_trajectories_test,
                trajectory_subsample=self.trajectory_subsample,
                cache_trajectories=self.cache_trajectories,
                split=None,  # Test uses all data from test dir
                val_split_ratio=self.val_split_ratio,
                seed=self.seed
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

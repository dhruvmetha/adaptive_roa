"""Horizon dataset for the partial-trajectory T-step dynamics model.

Reads ``train_splits/horizons.npy`` (one row per horizon) and
``train_splits/states_cache.npz`` (float32 states keyed by ``str(traj_id)``),
performs a trajectory-level train/val split (no horizon from a val trajectory
leaks into train), and yields ``(x_start, x_end, is_freeze, label, motion)``.

Sampling is uniform over horizons by default; motion-weighted / freeze-stratified
sampling is a documented future option (spec §4) not built here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class HorizonDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 0,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.split = split
        train_splits = Path(dataset_dir) / "train_splits"

        horizons = np.load(train_splits / "horizons.npy", mmap_mode="r")

        # Trajectory-level split: partition unique traj_ids, then keep this
        # split's horizons. Deterministic given seed.
        unique_ids = np.unique(horizons["traj_id"])
        perm = np.random.default_rng(seed).permutation(unique_ids)
        n_val = int(round(len(perm) * val_fraction))
        val_ids = perm[:n_val]
        keep_ids = val_ids if split == "val" else perm[n_val:]
        self.traj_ids = np.sort(keep_ids)

        mask = np.isin(horizons["traj_id"], self.traj_ids)
        self.horizons = np.array(horizons[mask])  # materialize this split

        # Load only this split's trajectories into memory.
        cache = np.load(train_splits / "states_cache.npz")
        self.states: Dict[str, np.ndarray] = {
            str(int(t)): cache[str(int(t))] for t in self.traj_ids
        }

        self._motion = self._compute_motion(self.horizons)

    @staticmethod
    def _compute_motion(horizons: np.ndarray) -> np.ndarray:
        """Normalize per-system motion fields to a single non-negative scalar."""
        fields = horizons.dtype.names
        if "jump_mag" in fields:
            return horizons["jump_mag"].astype(np.float32)
        # pendulum: S^1-wrapped |dtheta| and |dthetadot|
        return np.sqrt(
            horizons["abs_dtheta"].astype(np.float32) ** 2
            + horizons["abs_dthetadot"].astype(np.float32) ** 2
        )

    def __len__(self) -> int:
        return len(self.horizons)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.horizons[idx]
        traj = self.states[str(int(rec["traj_id"]))]
        return {
            "x_start": torch.as_tensor(traj[int(rec["start"])], dtype=torch.float32),
            "x_end": torch.as_tensor(traj[int(rec["end"])], dtype=torch.float32),
            "is_freeze": torch.tensor(int(rec["is_freeze"]), dtype=torch.long),
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
            "motion": torch.tensor(float(self._motion[idx]), dtype=torch.float32),
        }

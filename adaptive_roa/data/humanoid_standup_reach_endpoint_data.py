"""Endpoint data for HumanoidStandUpReach: (query_state -> final_state) pairs.

query_mode controls which trajectory row is the query:
  - "start": row 0 (Quad3D parity)
  - "random_intermediate": a uniformly sampled non-terminal row (default; FPS-style coverage)
  - "all_intermediate": every non-terminal row expanded into its own pair
The target is always the trajectory's final row. Sphere block (dims 34:37) is unit-normalized.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

SPHERE_START, SPHERE_END = 34, 37


def _load_trajectory(path: Path) -> np.ndarray:
    # Comma delimiter handles both "," and ", "; failures raise (never zero-fill).
    traj = np.loadtxt(path, delimiter=",")
    if traj.ndim == 1:
        traj = traj[None, :]
    if traj.shape[1] != 67:
        raise ValueError(f"{path}: expected 67 columns, got {traj.shape[1]}")
    return traj.astype(np.float32)


def _unit_sphere(vec: np.ndarray) -> np.ndarray:
    out = vec.copy()
    block = out[SPHERE_START:SPHERE_END]
    n = np.linalg.norm(block)
    out[SPHERE_START:SPHERE_END] = block / n if n > 1e-8 else block
    return out


class HumanoidStandUpReachEndpointDataset(Dataset):
    def __init__(self, shuffled_indices_file: str, trajectories_dir: str,
                 query_mode: str = "random_intermediate", max_samples: Optional[int] = None):
        if query_mode not in ("start", "random_intermediate", "all_intermediate"):
            raise ValueError(f"invalid query_mode: {query_mode}")
        self.query_mode = query_mode
        self.trajectories_dir = Path(trajectories_dir)
        with open(shuffled_indices_file) as f:
            filenames = [ln.strip() for ln in f if ln.strip()]
        if max_samples is not None:
            filenames = filenames[:max_samples]
        self._traj_cache = {}
        if query_mode == "all_intermediate":
            # Expand to (filename, row_idx) pairs. Reads each file once up front
            # and caches it so __getitem__ never re-reads from disk for this mode.
            self._index = []
            for fn in filenames:
                traj = _load_trajectory(self.trajectories_dir / fn)
                self._traj_cache[fn] = traj
                for r in range(traj.shape[0] - 1):  # non-terminal rows only
                    self._index.append((fn, r))
        else:
            self._index = [(fn, None) for fn in filenames]
        print(f"HumanoidStandUpReach dataset: {len(self._index)} samples (query_mode={query_mode})")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fname, fixed_row = self._index[idx]
        if fixed_row is not None:  # all_intermediate: reuse the cached array
            traj = self._traj_cache[fname]
        else:  # start / random_intermediate: lazy per-call read (intended pattern)
            traj = _load_trajectory(self.trajectories_dir / fname)
        n = traj.shape[0]
        end = traj[-1]
        if self.query_mode == "start":
            q = 0
        elif self.query_mode == "all_intermediate":
            q = fixed_row
        else:  # random_intermediate; uses global torch RNG (varies per epoch, seedable)
            q = int(torch.randint(0, max(n - 1, 1), (1,)).item())
        start = traj[q]
        return {
            "start_state": torch.from_numpy(_unit_sphere(start)),
            "end_state": torch.from_numpy(_unit_sphere(end)),
        }


class HumanoidStandUpReachDirectFileDataset(Dataset):
    def __init__(self, data_file: str):
        data = np.loadtxt(data_file)  # pool writes space-separated
        if data.ndim == 1:
            data = data[None, :]
        n = data.shape[1]
        if n not in (134, 135):
            raise ValueError(f"{data_file}: expected 134 or 135 columns, got {n}")
        self.starts = data[:, :67].astype(np.float32)
        self.ends = data[:, 67:134].astype(np.float32)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        return {
            "start_state": torch.from_numpy(_unit_sphere(self.starts[idx])),
            "end_state": torch.from_numpy(_unit_sphere(self.ends[idx])),
        }


class HumanoidStandUpReachEndpointDataModule(pl.LightningDataModule):
    def __init__(self, train_indices_file: str = None, val_indices_file: str = None,
                 test_indices_file: str = None, trajectories_dir: str = None,
                 data_file: str = None, validation_file: str = None, test_file: str = None,
                 query_mode: str = "random_intermediate",
                 batch_size: int = 256, val_batch_size: Optional[int] = None,
                 num_workers: int = 4, pin_memory: bool = True,
                 dataset_dir: str = None, max_train_samples: Optional[int] = None,
                 max_val_samples: Optional[int] = None):
        super().__init__()
        self.use_direct_file = data_file is not None
        self.train_indices_file = train_indices_file
        self.val_indices_file = val_indices_file
        self.test_indices_file = test_indices_file
        self.trajectories_dir = trajectories_dir
        self.data_file = data_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.query_mode = query_mode
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.state_dim = 67
        self.embedded_dim = 67

    def setup(self, stage: Optional[str] = None):
        if self.use_direct_file:
            if stage in ("fit", None):
                self.train_dataset = HumanoidStandUpReachDirectFileDataset(self.data_file)
                self.val_dataset = HumanoidStandUpReachDirectFileDataset(self.validation_file)
            if stage in ("test", None):
                self.test_dataset = HumanoidStandUpReachDirectFileDataset(self.test_file)
        else:
            if stage in ("fit", None):
                self.train_dataset = HumanoidStandUpReachEndpointDataset(
                    self.train_indices_file, self.trajectories_dir, self.query_mode, self.max_train_samples)
                self.val_dataset = HumanoidStandUpReachEndpointDataset(
                    self.val_indices_file, self.trajectories_dir, self.query_mode, self.max_val_samples)
            if stage in ("test", None):
                self.test_dataset = HumanoidStandUpReachEndpointDataset(
                    self.test_indices_file, self.trajectories_dir, self.query_mode)

    def _loader(self, ds, bs, shuffle):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_dataset, self.batch_size, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, self.val_batch_size, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, self.val_batch_size, False)

    def predict_dataloader(self):
        return self.test_dataloader()

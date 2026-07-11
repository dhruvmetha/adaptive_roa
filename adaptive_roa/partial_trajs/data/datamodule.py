"""LightningDataModule wrapping HorizonDataset for the partial-trajectory model."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from adaptive_roa.partial_trajs.data.horizon_dataset import HorizonDataset


class HorizonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        batch_size: int = 1024,
        val_fraction: float = 0.2,
        seed: int = 0,
        # Data is preloaded into in-memory tensors; worker processes add no value
        # and break on NFS (rmtree of `.nfs*` temp dirs). Mirrors the classifier DM.
        num_workers: int = 0,
        # Cap the trajectory pool to the first N (gold order); None -> full pool.
        max_trajectories: Optional[int] = None,
        # If True, use max_trajectories only as a horizon *budget*: sample that many
        # horizons IID from the full pool and split by horizon (see HorizonDataset).
        iid_horizons: bool = False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.seed = seed
        self.num_workers = num_workers
        self.max_trajectories = max_trajectories
        self.iid_horizons = iid_horizons
        self.train_ds: Optional[HorizonDataset] = None
        self.val_ds: Optional[HorizonDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = HorizonDataset(
            self.dataset_dir, split="train", val_fraction=self.val_fraction,
            seed=self.seed, max_trajectories=self.max_trajectories,
            iid_horizons=self.iid_horizons,
        )
        self.val_ds = HorizonDataset(
            self.dataset_dir, split="val", val_fraction=self.val_fraction,
            seed=self.seed, max_trajectories=self.max_trajectories,
            iid_horizons=self.iid_horizons,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

"""
Pendulum Endpoint Data Module for Adaptive Sampling Pipeline.

Similar to CartPoleEndpointDataModule, accepts separate train/val/test files
instead of splitting from a single file.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import lightning.pytorch as pl

from adaptive_roa.data.batched_indexing import BatchedIndexingMixin, collate_fn_for


class PendulumEndpointDataset(BatchedIndexingMixin, Dataset):
    """
    Dataset for pendulum endpoint pairs (start_state, end_state).

    State format: [theta, theta_dot] where:
    - theta: pole angle in [-pi, pi]
    - theta_dot: angular velocity
    """

    def __init__(self, data_file: str):
        """
        Args:
            data_file: Path to endpoint dataset file.
                      Each line: theta_start theta_dot_start theta_end theta_dot_end
        """
        # Load the endpoint data
        with open(data_file, 'r') as f:
            lines = f.readlines()

        # Parse the data - each line has [start_theta, start_theta_dot, end_theta, end_theta_dot]
        data = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split()))
                if len(values) == 4:  # start_state (2D) + end_state (2D)
                    start_state = values[:2]
                    end_state = values[2:]
                    data.append((start_state, end_state))

        print(f"Loaded {len(data)} samples for pendulum endpoint data from {Path(data_file).name}")
        self.data = data

    def wrap_angle(self, angle: float) -> float:
        """
        Wrap angle to [-pi, pi] for proper S^1 manifold representation.

        Args:
            angle: Angle in radians (can be unwrapped)

        Returns:
            Wrapped angle in [-pi, pi]
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start_state, end_state = self.data[idx]

        # Wrap angles for consistent S^1 representation
        start_state_wrapped = [self.wrap_angle(start_state[0]), start_state[1]]
        end_state_wrapped = [self.wrap_angle(end_state[0]), end_state[1]]

        return {
            'start_state': torch.tensor(start_state_wrapped, dtype=torch.float32),
            'end_state': torch.tensor(end_state_wrapped, dtype=torch.float32)
        }


class PendulumEndpointDataModule(pl.LightningDataModule):
    """
    Pendulum Endpoint Data Module with separate train/val/test files.

    Used by the adaptive sampling pipeline where datasets are built incrementally.
    """

    def __init__(
        self,
        data_file: str,
        validation_file: str,
        test_file: str,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        batched_indexing: bool = False,
    ):
        """
        Args:
            data_file: Path to training dataset file
            validation_file: Path to validation dataset file
            test_file: Path to test dataset file
            batch_size: Batch size for training data loader
            val_batch_size: Batch size for validation/test data loaders
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for data loaders
            batched_indexing: Materialize rows once and fetch each batch with one
                index operation (data/batched_indexing.py). Same batches, same order.
        """
        super().__init__()
        self.data_file = data_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batched_indexing = bool(batched_indexing)

        # Pendulum-specific dimensions
        self.state_dim = 2  # Raw state: [theta, theta_dot]
        self.embedded_dim = 3  # Embedded: [sin_theta, cos_theta, theta_dot_norm]

    def setup(self, stage: Optional[str] = None):
        # Lightning calls setup("fit") again inside trainer.fit; a trainer that
        # already called setup() would otherwise parse both files a second time.
        if (stage == "fit" or stage is None) and getattr(self, "train_dataset", None) is None:
            self.train_dataset = PendulumEndpointDataset(self.data_file)
            self.val_dataset = PendulumEndpointDataset(self.validation_file)
            if self.batched_indexing:
                self.train_dataset.enable_batched_indexing()
                self.val_dataset.enable_batched_indexing()

        if (stage == "test" or stage is None) and getattr(self, "test_dataset", None) is None:
            self.test_dataset = PendulumEndpointDataset(self.test_file)
            if self.batched_indexing:
                self.test_dataset.enable_batched_indexing()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_for(self.batched_indexing),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_for(self.batched_indexing),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_for(self.batched_indexing),
        )

    def predict_dataloader(self):
        return self.test_dataloader()

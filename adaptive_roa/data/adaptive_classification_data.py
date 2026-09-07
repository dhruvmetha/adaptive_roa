"""DataModule for the adaptive binary ROA classifier.

Reads the per-epoch ``(state..., label)`` files produced by
``AdaptiveDatasetBuilder.build_*_classification_dataset`` (space-separated,
last column is the binary label in {0.0, 1.0}). Yields batches of RAW states;
the ``ClassifierModule`` normalizes + embeds internally.

Computes ``pos_weight = n_neg / n_pos`` from the training labels to counter
class imbalance (quad2d is ~8% success).
"""
from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class _ClassificationDataset(Dataset):
    def __init__(self, states: np.ndarray, labels: np.ndarray):
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx):
        return {"inputs": self.states[idx], "label": self.labels[idx]}


class _SliceBatches:
    """Batches by slicing whole tensors instead of one ``__getitem__`` per row.

    With ``num_workers=0`` a map-style ``DataLoader`` spends its whole step on
    1,024 Python indexing calls plus a collate, ~10 ms on a 3M-row set, while
    the outcome-FM network needs under 2 ms. Measured 2026-09-07 on the quad3d
    PPO arms: 6x faster per step with the tensors resident on the GPU, and
    identical rows, batch size and loss. Only the shuffle's random stream
    differs from ``RandomSampler``'s.
    """

    def __init__(self, states: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool):
        self.states, self.labels = states, labels
        self.batch_size, self.shuffle = int(batch_size), bool(shuffle)

    def __len__(self) -> int:
        return (self.states.shape[0] + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.states.shape[0]
        # Consume the global RNG exactly as a DataLoader iterator does, so a run
        # is bit-identical to the DataLoader path under the same manual_seed:
        # _BaseDataLoaderIter draws one int64 for its _base_seed on creation
        # (train and val alike), then RandomSampler draws its own seed for a
        # fresh CPU generator on first next(). Skipping either shifts every
        # later torch.randn/rand in training_step by that many draws.
        torch.empty((), dtype=torch.int64).random_()
        order = None
        if self.shuffle:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            gen = torch.Generator(); gen.manual_seed(seed)
            order = torch.randperm(n, generator=gen).to(self.states.device)
        for start in range(0, n, self.batch_size):
            idx = slice(start, start + self.batch_size) if order is None else order[start:start + self.batch_size]
            yield {"inputs": self.states[idx], "label": self.labels[idx]}


class AdaptiveClassificationDataModule(pl.LightningDataModule):
    def __init__(self, train_file: str, val_file: str, batch_size: int = 1024, num_workers: int = 4,
                 device: str | None = None):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        # When set, the tensors live on this device and batches are sliced
        # there (see _SliceBatches). None keeps the DataLoader path.
        self.device = device
        self.pos_weight: float = 1.0
        self._train: _ClassificationDataset | None = None
        self._val: _ClassificationDataset | None = None

    @staticmethod
    def _load(path: str):
        # pandas parses the savetxt output ~10x faster than np.loadtxt; same
        # values. On the quad3d PPO arms loadtxt alone took minutes per round.
        import pandas as pd
        try:
            data = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64, comment="#").to_numpy()
        except pd.errors.EmptyDataError:
            data = np.empty((0, 0))
        if data.size == 0:
            raise ValueError(f"Classification dataset file is empty: {path}")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        states = data[:, :-1].astype(np.float32)
        labels = data[:, -1].astype(np.float32)
        return states, labels

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return  # already loaded (avoid reload when Lightning calls setup again)
        x_train, y_train = self._load(self.train_file)
        x_val, y_val = self._load(self.val_file)
        self._train = _ClassificationDataset(x_train, y_train)
        self._val = _ClassificationDataset(x_val, y_val)
        if self.device is not None:
            for ds in (self._train, self._val):
                ds.states = ds.states.to(self.device)
                ds.labels = ds.labels.to(self.device)

        n_pos = float((y_train == 1.0).sum())
        n_neg = float((y_train == 0.0).sum())
        self.pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    def train_dataloader(self):
        if self.device is not None:
            return _SliceBatches(self._train.states, self._train.labels, self.batch_size, shuffle=True)
        return DataLoader(
            self._train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=False,
        )

    def val_dataloader(self):
        if self.device is not None:
            return _SliceBatches(self._val.states, self._val.labels, self.batch_size, shuffle=False)
        return DataLoader(
            self._val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
        )

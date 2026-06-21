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


class AdaptiveClassificationDataModule(pl.LightningDataModule):
    def __init__(self, train_file: str, val_file: str, batch_size: int = 1024, num_workers: int = 4):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pos_weight: float = 1.0
        self._train: _ClassificationDataset | None = None
        self._val: _ClassificationDataset | None = None

    @staticmethod
    def _load(path: str):
        data = np.loadtxt(path)
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

        n_pos = float((y_train == 1.0).sum())
        n_neg = float((y_train == 0.0).sum())
        self.pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
        )

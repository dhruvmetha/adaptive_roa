"""One index operation per batch for the in-memory endpoint and outcome datasets.

The stock DataLoader path calls ``dataset[i]`` once per row and then
``default_collate`` stacks the rows. For the endpoint datasets each
``__getitem__`` does Python work per row (angle wrapping, quaternion
normalization, two ``torch.tensor`` constructions), which costs 20-40 us/row
and dominates a small-MLP training step.

Batched mode materializes every row ONCE, through the dataset's own
``__getitem__`` and ``default_collate`` (the exact functions the stock path
runs per batch), into stacked tensors. ``__getitems__`` then answers a batch
with one ``index_select`` per field and the DataLoader passes the result
through ``identity_collate`` unchanged. The sampler is untouched, so the same
RandomSampler draws the same permutation from the same RNG, and every batch
holds the same rows, in the same order, with the same float32 values.

Off by default. A datamodule turns it on only when constructed with
``batched_indexing=True``; the trainers read ``predictor.batched_indexing``.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import default_collate

# Rows per default_collate call while materializing. Bounds the transient list
# of per-row dicts; the stacked result does not depend on it.
_MATERIALIZE_CHUNK = 65536


def identity_collate(batch: Any) -> Any:
    """Collate for a dataset whose ``__getitems__`` already returns a batch."""
    return batch


def collate_fn_for(batched_indexing: bool):
    """The DataLoader ``collate_fn`` for a mode. None is DataLoader's default_collate."""
    return identity_collate if batched_indexing else None


def materialize_rows(dataset, chunk: int = _MATERIALIZE_CHUNK) -> dict[str, torch.Tensor] | None:
    """Every row of ``dataset`` as stacked tensors, built the way the stock path batches.

    Each chunk is ``default_collate([dataset[i] for i in chunk])``, i.e. the
    per-row ``__getitem__`` and the stacking the DataLoader would apply to a
    batch holding those rows, so a row's values here are the values it would
    have in any stock batch. Returns None for an empty dataset.
    """
    n = len(dataset)
    if n == 0:
        return None
    out: dict[str, torch.Tensor] | None = None
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        part = default_collate([dataset[i] for i in range(c0, c1)])
        if out is None:
            out = {k: torch.empty((n, *v.shape[1:]), dtype=v.dtype) for k, v in part.items()}
        for k, v in part.items():
            out[k][c0:c1] = v
    return out


class BatchedIndexingMixin:
    """``__getitems__`` for a map-style dataset: per-row by default, batched once enabled.

    Disabled, ``__getitems__`` returns ``[self[i] for i in indices]``, which is
    exactly what the DataLoader fetcher does for a dataset without the method.
    Enabled, it returns a dict of ``[B, ...]`` tensors and the DataLoader must
    use ``identity_collate``; ``default_collate`` would index the dict by 0.
    """

    _batched_rows: dict[str, torch.Tensor] | None = None

    def _materialize_for_batching(self) -> dict[str, torch.Tensor] | None:
        """Override to supply rows that are already known to equal ``__getitem__``'s."""
        return materialize_rows(self)

    def enable_batched_indexing(self) -> None:
        if self._batched_rows is None:
            self._batched_rows = self._materialize_for_batching()

    @property
    def batched_indexing(self) -> bool:
        return self._batched_rows is not None

    def batched_rows(self) -> dict[str, torch.Tensor] | None:
        """The materialized rows (shared, do not modify), or None when disabled."""
        return self._batched_rows

    def __getitems__(self, indices):
        rows = self._batched_rows
        if rows is None:
            return [self[i] for i in indices]
        idx = torch.as_tensor(indices, dtype=torch.long)
        return {k: v.index_select(0, idx) for k, v in rows.items()}

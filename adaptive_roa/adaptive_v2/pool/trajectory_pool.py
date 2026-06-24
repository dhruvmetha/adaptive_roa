"""Trajectory pool wrapper for adaptive v2."""

from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive.data_source import TrajectoryDataSource, TrajectoryDataSourceConfig
from adaptive_roa.adaptive.dataset_builder import AdaptiveDatasetBuilder


class TrajectoryPool:
    """Wraps the legacy dataset builder as a v2 dataset pool."""

    def __init__(
        self,
        data_source_cfg: Any,
        output_dir: str,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        candidate_mode: str = "start",
    ):
        cfg = TrajectoryDataSourceConfig(
            trajectories_dir=data_source_cfg.trajectories_dir,
            shuffled_indices_file=data_source_cfg.shuffled_indices_file,
            shuffled_labels_file=data_source_cfg.get("shuffled_labels_file", None),
            eval_states_file=data_source_cfg.get("eval_states_file", None),
        )
        data_source = TrajectoryDataSource(cfg)
        self.dataset_builder = AdaptiveDatasetBuilder(
            data_source=data_source,
            output_dir=output_dir,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            candidate_mode=candidate_mode,
        )

    def initialize(self, initial_train_size: int, dataset_kind: str = "endpoint") -> dict[str, str]:
        self.dataset_builder.get_initial_training_set(initial_train_size)
        return self.dataset_builder.build_all_datasets(dataset_kind=dataset_kind)

    def sample_candidates_without_marking(
        self,
        n: int,
        exclude: set[int] | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        return self.dataset_builder.sample_candidates_without_marking(n, exclude=exclude)

    def mark_indices_as_used(self, indices: list[int]) -> None:
        self.dataset_builder.mark_indices_as_used(indices)

    def add_to_training_balanced(self, indices: list[int]) -> None:
        self.dataset_builder.add_to_training_balanced(indices)

    def build_all_datasets(self, dataset_kind: str = "endpoint") -> dict[str, str]:
        return self.dataset_builder.build_all_datasets(dataset_kind=dataset_kind)

    def get_training_data(self):
        return self.dataset_builder.get_training_data()

    def get_val_labels(self):
        return self.dataset_builder.get_val_labels()

    def get_test_labels(self):
        return self.dataset_builder.get_test_labels()

    def get_labels(self, indices: list[int]):
        return self.dataset_builder.data_source.get_labels(indices)

    def get_statistics(self) -> dict[str, Any]:
        return self.dataset_builder.get_statistics()

    def save_state(self, filepath: str) -> None:
        self.dataset_builder.save_state(filepath)

    @property
    def train_size(self) -> int:
        return len(self.dataset_builder.train_split)

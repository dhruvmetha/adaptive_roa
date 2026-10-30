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
        seed: int = 42,
    ):
        cfg = TrajectoryDataSourceConfig(
            trajectories_dir=data_source_cfg.trajectories_dir,
            shuffled_indices_file=data_source_cfg.shuffled_indices_file,
            shuffled_labels_file=data_source_cfg.get("shuffled_labels_file", None),
            eval_states_file=data_source_cfg.get("eval_states_file", None),
            # Absent from every config written before the timeout fix, so a
            # resumed legacy run keeps its original rows.
            timeout_intermediates=str(data_source_cfg.get("timeout_intermediates", "keep")),
            horizon_steps=data_source_cfg.get("horizon_steps", None),
        )
        if str(data_source_cfg.get("pool_format", "text")) == "npz":
            from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource
            data_source = NpzTrajectoryDataSource(cfg)
        else:
            data_source = TrajectoryDataSource(cfg)
        self.dataset_builder = AdaptiveDatasetBuilder(
            data_source=data_source,
            output_dir=output_dir,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            candidate_mode=candidate_mode,
            # Only suffix mode is seeded from the run; intermediate keeps its
            # historical fixed split seed.
            **({"seed": int(seed)} if candidate_mode == "suffix" else {}),
        )
        self.candidate_mode = candidate_mode

    def initialize(self, initial_train_size: int, dataset_kind: str = "endpoint") -> dict[str, str]:
        self.dataset_builder.get_initial_training_set(initial_train_size)
        return self.dataset_builder.build_all_datasets(dataset_kind=dataset_kind)

    def sample_candidates_without_marking(
        self,
        n: int,
        exclude: set[int] | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        return self.dataset_builder.sample_candidates_without_marking(n, exclude=exclude)

    def sample_uniform(self, n: int) -> tuple[np.ndarray, list[int]]:
        """The uniform (1 - d2) share. Suffix mode takes trajectory starts in pool
        order; the other modes keep their historical draw."""
        if self.candidate_mode == "suffix":
            return self.dataset_builder.sample_uniform_starts(n)
        return self.dataset_builder.sample_candidates_without_marking(n)

    def mark_indices_as_used(self, indices: list[int]) -> None:
        self.dataset_builder.mark_indices_as_used(indices)

    def add_to_training_balanced(self, indices: list[int], source: str = "acquisition") -> None:
        self.dataset_builder.add_to_training_balanced(indices, source=source)

    def build_all_datasets(self, dataset_kind: str = "endpoint") -> dict[str, str]:
        return self.dataset_builder.build_all_datasets(dataset_kind=dataset_kind)

    def get_training_data(self):
        return self.dataset_builder.get_training_data()

    def get_val_labels(self):
        return self.dataset_builder.get_val_labels()

    def get_test_labels(self):
        return self.dataset_builder.get_test_labels()

    def get_labels(self, indices: list[int]):
        return self.dataset_builder.get_labels(indices)

    def get_statistics(self) -> dict[str, Any]:
        return self.dataset_builder.get_statistics()

    def save_state(self, filepath: str) -> None:
        self.dataset_builder.save_state(filepath)

    @property
    def train_size(self) -> int:
        """Rollouts spent. In suffix mode every pick is one rollout (D22a)."""
        if self.candidate_mode == "suffix":
            return self.dataset_builder._suffix.n_picks
        return len(self.dataset_builder.train_split)

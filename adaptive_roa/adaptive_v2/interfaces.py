"""Public interfaces for adaptive v2 components."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from adaptive_roa.adaptive_v2.types import AcquisitionResult, OutcomeProbabilities, ThresholdState


class PredictorTrainer(Protocol):
    def fit(
        self,
        train_file: str,
        val_file: str,
        output_dir: str,
        resume_checkpoint: str | None = None,
    ) -> Any:
        ...


class ProbabilityBackend(Protocol):
    def bind_model(self, model_handle: Any) -> None:
        ...

    def estimate(self, start_states: np.ndarray) -> OutcomeProbabilities:
        ...

    def sample_endpoints(
        self,
        start_states: np.ndarray,
        num_samples: int,
        verbose: bool = True,
    ) -> np.ndarray:
        """Return a raw endpoint cloud [N, K, D] with no classification.

        Optional capability: only endpoint-sampling backends (e.g.
        EndpointMCProbabilityBackend) implement this. The classifier backend
        deliberately does not, since it has no notion of a predicted final
        state to sample. DispersionAcquisitionStrategy checks for this method
        with a runtime hasattr() guard rather than relying on Protocol
        conformance, so declaring it here documents the capability without
        changing that runtime behavior.
        """
        ...


class ThresholdBackend(Protocol):
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> ThresholdState:
        ...

    def calibrate_qhat(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        threshold_state: ThresholdState,
    ) -> float:
        ...


class AcquisitionStrategy(Protocol):
    def select(
        self,
        pool: Any,
        probability_backend: ProbabilityBackend,
        threshold_backend: ThresholdBackend,
        threshold_state: ThresholdState,
        cfg: Any,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        ...


class Evaluator(Protocol):
    def evaluate_epoch(
        self,
        model_handle: Any,
        threshold_state: ThresholdState,
        epoch_context: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class DatasetPool(Protocol):
    def sample_candidates_without_marking(
        self,
        n: int,
        exclude: set[int] | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        ...

    def mark_indices_as_used(self, indices: list[int]) -> None:
        ...

    def add_to_training_balanced(self, indices: list[int]) -> None:
        ...

    def build_all_datasets(self) -> dict[str, str]:
        ...

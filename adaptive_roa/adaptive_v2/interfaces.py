"""Public interfaces for adaptive v2 components."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from adaptive_roa.adaptive_v2.types import AcquisitionResult, OutcomeProbabilities, ThresholdState


class PredictorTrainer(Protocol):
    """Trains one epoch's predictor and returns a ready-to-use model handle.

    ``dataset_files`` keys depend on the predictor family and prediction mode:
    ``{"train", "val"}`` for classification and global endpoint data,
    ``{"train_trajectories", "val_trajectories"}`` for ``prediction_mode: local``.
    """

    def fit(
        self,
        dataset_files: dict,
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


class OutcomeModelHandle(Protocol):
    """Predictor that emits p(success | x) directly.

    Bound to ``ClassifierProbabilityBackend``, which calls the handle ONCE per
    query. Threshold optimization, calibration, and evaluation each call it
    separately and must agree, so a Bayesian handle marginalizes its weight
    posterior INTERNALLY and returns identical logits for repeated calls on the
    same states (use a ``torch.Generator`` seeded at construction).
    """

    def eval(self) -> Any: ...

    def to(self, device: Any) -> Any: ...

    def __call__(self, raw_states: Any) -> Any:
        """Raw (un-normalized) states [B, state_dim] -> logits [B] or [B, 1]."""
        ...


class FinalStateModelHandle(Protocol):
    """Predictor that emits a distribution over the final state.

    Bound to ``EndpointMCProbabilityBackend``, which calls ``predict_endpoint``
    K times on the SAME batch and counts ``system.classify_attractor`` labels.
    The spread across those calls IS the outcome probability, so each call must
    draw a fresh posterior sample. Seeding this handle collapses every arm to
    p in {0, 1}.
    """

    def eval(self) -> Any: ...

    def to(self, device: Any) -> Any: ...

    def predict_endpoint(self, states: Any) -> Any:
        """One posterior draw: [B, state_dim] -> [B, state_dim], raw space."""
        ...

    def get_manifold_component_names(self) -> list:
        """Called UNGUARDED by adaptive/endpoint_evaluation.py:100."""
        ...

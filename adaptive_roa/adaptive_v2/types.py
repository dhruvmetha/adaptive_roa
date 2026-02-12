"""Core v2 adaptive training datatypes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OutcomeProbabilities:
    """Per-state outcome probabilities."""

    p_success: np.ndarray
    p_failure: np.ndarray
    p_invalid: np.ndarray


@dataclass
class ThresholdState:
    """Threshold/calibration state for an epoch."""

    lambda_star: float
    delta_star: float
    q_hat: float | None = None
    q_hat_eval: float | None = None


@dataclass
class AcquisitionResult:
    """Adaptive acquisition outputs for one epoch."""

    d1_indices: list[int] = field(default_factory=list)
    d2_indices: list[int] = field(default_factory=list)
    n_candidates_evaluated: int = 0
    n_certain_discarded: int = 0
    n_invalid_added: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpochArtifacts:
    """Canonical v2 artifact payload for epoch outputs."""

    epoch: int
    train_trajectories: int
    sampling_mode: str
    threshold_state: ThresholdState
    acquisition: AcquisitionResult
    endpoint_error: dict[str, Any]
    eval_metrics: dict[str, Any]
    d1_eval_metrics: dict[str, Any] | None
    conformal_state: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

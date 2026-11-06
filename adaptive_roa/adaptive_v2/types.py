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
    q_hat_success_eval: float | None = None
    q_hat_failure_eval: float | None = None
    # FPR-capped rule (threshold=fpr_cap, adaptive_v2/eval/fpr_cap.py). None on every
    # other backend. When set, lambda_star/delta_star hold the converted values for
    # logging only, and delta_star can be negative.
    tau_s: float | None = None
    tau_f: float | None = None
    tau_inv: float | None = None
    alpha_s: float | None = None
    # det_v2 threshold opt (threshold=j_sel, adaptive_v2/eval/det_v2_rule.py). None on
    # every other backend; lambda_star/delta_star then hold converted values for logging.
    tau_s_star: float | None = None
    tau_f_star: float | None = None
    threshold_objective: str | None = None
    threshold_w: float | None = None
    # tau* regime hysteresis (ablation A9, threshold=j_sel_hysteresis). None elsewhere.
    # The incumbent regime ("inv": tau_s + tau_f >= 1, "out": < 1), each regime's best
    # J_sel on this round's fit rows, and the margin the other regime had to beat.
    threshold_region: str | None = None
    threshold_j_inv: float | None = None
    threshold_j_out: float | None = None
    threshold_margin: float | None = None

    TAU_FIELDS = ("tau_s", "tau_f", "tau_inv", "alpha_s")
    DET_V2_FIELDS = ("tau_s_star", "tau_f_star", "threshold_objective", "threshold_w")
    HYSTERESIS_FIELDS = ("threshold_region", "threshold_j_inv", "threshold_j_out", "threshold_margin")

    @property
    def taus(self) -> tuple[float, float, float] | None:
        """(tau_s, tau_f, tau_inv), or None when the state carries lambda/delta only."""
        if self.tau_s is None:
            return None
        return float(self.tau_s), float(self.tau_f), float(self.tau_inv)


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

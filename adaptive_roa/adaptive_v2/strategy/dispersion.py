"""Dispersion acquisition strategy (D2 only).

Scores candidates by the spread of their predicted final states rather than by
label counts, so selection never consults the success criteria. See
docs/superpowers/specs/2026-07-26-final-state-dispersion-acquisition-design.md
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    idempotence_defect,
    mean_pairwise_dispersion,
    mode_separation,
    select_greedy,
    select_greedy_diverse,
    select_proportional,
)
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState

_SELECTION_RULES = ("greedy", "greedy_diverse", "proportional")
# All three are criteria-free: they read only the endpoint cloud's geometry.
#   dispersion -- mean pairwise distance; conflates within-mode spread with
#                 between-mode separation, which is how it failed on quad2d
#   mode_sep   -- gap across a 2-way split; between-mode structure only
#   gated      -- mode_sep * exp(-idempotence defect), suppressing clouds that
#                 are diffuse because the dynamics diverge rather than because
#                 two outcomes compete. Costs a second endpoint pass.
_SCORES = ("dispersion", "mode_sep", "gated")


class DispersionAcquisitionStrategy:
    mode = "dispersion"

    def __init__(self, cfg: Any):
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_dispersion_candidates = int(cfg.n_dispersion_candidates)
        self.num_mc_samples_dispersion = int(cfg.num_mc_samples_dispersion)
        self.selection_rule = str(cfg.selection_rule)
        self.diversity_pool_multiplier = int(cfg.diversity_pool_multiplier)
        self.temperature = float(cfg.temperature)
        self.seed = None if cfg.seed is None else int(cfg.seed)
        self.chunk_size = int(cfg.chunk_size)
        self.log_score_correlation = bool(cfg.log_score_correlation)
        self.score = str(cfg.get("score", "dispersion"))
        self.verbose = bool(cfg.verbose)

        if self.score not in _SCORES:
            raise ValueError(f"score must be one of {_SCORES}, got {self.score!r}")

        if self.selection_rule not in _SELECTION_RULES:
            raise ValueError(
                f"selection_rule must be one of {_SELECTION_RULES}, "
                f"got {self.selection_rule!r}"
            )

    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        # threshold_state is accepted for Protocol conformance and never read:
        # selection is independent of the success criteria by design.
        if target_count <= 0:
            return self._skip("target_count_zero")

        if not hasattr(probability_backend, "sample_endpoints"):
            raise RuntimeError(
                "DispersionAcquisitionStrategy requires a probability backend with "
                f"sample_endpoints(); got {type(probability_backend).__name__}"
            )

        system = getattr(probability_backend, "system", None)
        if system is None:
            raise RuntimeError(
                "DispersionAcquisitionStrategy requires probability_backend.system "
                "to compute the normalized state metric"
            )

        scales = system.get_normalization_scales().cpu().numpy().astype(np.float64)
        circular_mask = np.zeros(len(scales), dtype=bool)
        circular_indices = system.get_circular_indices()
        if circular_indices:
            circular_mask[np.asarray(circular_indices, dtype=int)] = True

        states, indices = pool.sample_candidates_without_marking(
            self.n_dispersion_candidates, exclude=exclude
        )
        n_actual = len(indices)
        if n_actual == 0:
            if self.verbose:
                print("    [Dispersion] Pool exhausted. No candidates available.")
            return self._skip("pool_exhausted")

        if self.verbose:
            print(
                f"    [Dispersion] Evaluating {n_actual} candidates with "
                f"K={self.num_mc_samples_dispersion}, rule={self.selection_rule}, "
                f"selecting {target_count}"
            )

        endpoints = probability_backend.sample_endpoints(
            states,
            num_samples=self.num_mc_samples_dispersion,
            verbose=self.verbose,
        )
        scores = self._compute_score(
            endpoints, scales, circular_mask, probability_backend
        )

        # Resolve the effective seed before selection so a `proportional` run
        # is replayable: with seed=None we would otherwise draw OS entropy
        # inside select_proportional and never record what was drawn. Drawing
        # from the already-seeded global numpy RNG (engine.py calls
        # pl.seed_everything before any acquisition runs) means this value
        # inherits the run's determinism instead of being independent entropy.
        resolved_seed = None
        if self.selection_rule == "proportional":
            resolved_seed = (
                int(np.random.randint(0, 2**31 - 1))
                if self.seed is None
                else self.seed
            )

        positions = self._apply_selection_rule(
            scores, np.asarray(states), scales, circular_mask, target_count, resolved_seed
        )
        selected_indices = [indices[int(p)] for p in positions]

        diagnostics = self._build_diagnostics(scores, positions, n_actual)
        diagnostics["dispersion_selection_seed"] = resolved_seed
        # Computed after selection is decided: diagnostic only, never an input.
        diagnostics["dispersion_label_uncertainty_spearman"] = (
            self._label_uncertainty_correlation(
                endpoints, scores, system, probability_backend
            )
            if self.log_score_correlation
            else None
        )

        if self.verbose:
            print(
                f"    Selected {len(selected_indices)}/{n_actual}; "
                f"score min={diagnostics['dispersion_score_min']}, "
                f"max={diagnostics['dispersion_score_max']}, "
                f"threshold={diagnostics['dispersion_score_threshold']}"
            )

        return AcquisitionResult(
            d1_indices=[],
            d2_indices=selected_indices,
            n_candidates_evaluated=n_actual,
            n_certain_discarded=n_actual - len(selected_indices),
            n_invalid_added=0,
            diagnostics=diagnostics,
        )

    def _compute_score(self, endpoints, scales, circular_mask, probability_backend):
        device = getattr(probability_backend, "device", "cpu")
        if self.score == "dispersion":
            return mean_pairwise_dispersion(
                endpoints, scales, circular_mask, chunk_size=self.chunk_size, device=device
            )

        sep = mode_separation(
            endpoints, scales, circular_mask, chunk_size=self.chunk_size, device=device
        )
        if self.score == "mode_sep":
            return sep

        # gated: suppress clouds the learned map does not treat as settled. An
        # attractor is a fixed point of the endpoint map, so re-running the model
        # on each predicted endpoint separates "two competing outcomes" from
        # "trajectory still diverging", which raw gap structure cannot.
        M, K, D = endpoints.shape
        remapped = probability_backend.sample_endpoints(
            endpoints.reshape(-1, D), num_samples=1, verbose=False
        ).reshape(M, K, D)
        defect = idempotence_defect(endpoints, remapped, scales, circular_mask)
        return sep * np.exp(-defect)

    def _apply_selection_rule(
        self,
        scores: np.ndarray,
        states: np.ndarray,
        scales: np.ndarray,
        circular_mask: np.ndarray,
        target_count: int,
        resolved_seed: int | None,
    ) -> np.ndarray:
        if self.selection_rule == "greedy":
            return select_greedy(scores, target_count)
        if self.selection_rule == "greedy_diverse":
            return select_greedy_diverse(
                scores,
                states,
                scales,
                circular_mask,
                target_count,
                pool_multiplier=self.diversity_pool_multiplier,
            )
        return select_proportional(
            scores, target_count, temperature=self.temperature, seed=resolved_seed
        )

    def _label_uncertainty_correlation(
        self,
        endpoints: np.ndarray,
        scores: np.ndarray,
        system: Any,
        probability_backend: Any,
    ) -> float | None:
        """Spearman correlation between dispersion and label-based uncertainty.

        Costs no extra forward passes: the endpoint cloud is already in hand, so
        p_success comes from classifying those same endpoints. Label-based
        uncertainty is u = -|p_success - 0.5|, which peaks at p_success = 0.5.
        Both quantities rise with uncertainty, so +1.0 means the dispersion score
        and the label counts are redundant.

        This is diagnostic only. It runs after selection is decided and never
        influences which candidates are chosen.

        Returns:
            float | None: Spearman rho, or None when it is undefined (fewer than
                two finite candidates, or either series is constant)
        """
        radius = getattr(probability_backend, "attractor_radius", None)
        if radius is None:
            return None

        M, K, D = endpoints.shape
        success_counts = np.zeros(M, dtype=np.float64)

        for start in range(0, M, self.chunk_size):
            stop = min(start + self.chunk_size, M)
            flat = torch.as_tensor(
                endpoints[start:stop].reshape(-1, D), dtype=torch.float32
            )
            labels = system.classify_attractor(flat, radius=radius)
            labels = labels.reshape(stop - start, K)
            success_counts[start:stop] = (labels == 1).sum(dim=1).cpu().numpy()

        p_success = success_counts / K
        u = -np.abs(p_success - 0.5)

        finite = np.isfinite(scores)
        if finite.sum() < 2:
            return None
        s, u = scores[finite], u[finite]
        if np.ptp(s) == 0 or np.ptp(u) == 0:
            return None

        rho = spearmanr(s, u).statistic
        return None if not np.isfinite(rho) else float(rho)

    def _build_diagnostics(
        self,
        scores: np.ndarray,
        positions: np.ndarray,
        n_actual: int,
    ) -> dict[str, Any]:
        finite = scores[np.isfinite(scores)]
        selected = scores[positions] if len(positions) else np.array([])
        return {
            "dispersion_score_threshold": float(selected.min()) if len(selected) else None,
            "dispersion_score_min": float(finite.min()) if len(finite) else None,
            "dispersion_score_max": float(finite.max()) if len(finite) else None,
            "dispersion_score_mean": float(finite.mean()) if len(finite) else None,
            "dispersion_score_median": float(np.median(finite)) if len(finite) else None,
            "n_dispersion_candidates_evaluated": int(n_actual),
            "n_nonfinite_excluded": int((~np.isfinite(scores)).sum()),
            "selection_rule": self.selection_rule,
            "score": self.score,
            "num_mc_samples_dispersion": self.num_mc_samples_dispersion,
        }

    @staticmethod
    def _skip(reason: str) -> AcquisitionResult:
        return AcquisitionResult(
            d1_indices=[],
            d2_indices=[],
            n_candidates_evaluated=0,
            n_certain_discarded=0,
            n_invalid_added=0,
            diagnostics={"skipped_reason": reason},
        )

"""Threshold-free entropy acquisition strategy (D2 only).

Uses the success criterion to label sampled endpoints, but fits NO decision
threshold: no lambda*, no delta*, no q_hat, no alpha, no decision rule. The
score is the binary entropy of p(success), which peaks at p = 0.5 -- a universal
constant rather than a per-system fitted quantity, which is the property the
thresholded strategies lack.

    H(p) = -p log p - (1-p) log(1-p),   p = #(label == success) / K

`threshold_state` is accepted for Protocol conformance and ignored, so the
training loop never depends on a fitted decision boundary.

Optional tie-breaking: with K MC samples p takes only K+1 distinct values, so
many candidates tie at maximum entropy and the top-N is an arbitrary subset of
the maximally-ambiguous set. `tie_breaker: mode_sep` orders within exact ties by
the endpoint cloud's mode separation, which is continuous and does not tie.
The ordering is lexicographic, so the tie-breaker can never override entropy and
introduces no weighting parameter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    mode_separation,
    select_greedy,
    select_greedy_diverse,
)
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState

_TIE_BREAKERS = ("none", "mode_sep")
_SELECTION_RULES = ("greedy", "greedy_diverse")


class EntropyAcquisitionStrategy:
    mode = "entropy"

    def __init__(self, cfg: Any):
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_candidates = int(cfg.n_candidates)
        self.num_mc_samples = int(cfg.num_mc_samples)
        self.tie_breaker = str(cfg.tie_breaker)
        self.selection_rule = str(cfg.selection_rule)
        self.diversity_pool_multiplier = int(cfg.diversity_pool_multiplier)
        self.chunk_size = int(cfg.chunk_size)
        self.verbose = bool(cfg.verbose)

        if self.tie_breaker not in _TIE_BREAKERS:
            raise ValueError(f"tie_breaker must be one of {_TIE_BREAKERS}, got {self.tie_breaker!r}")
        if self.selection_rule not in _SELECTION_RULES:
            raise ValueError(
                f"selection_rule must be one of {_SELECTION_RULES}, got {self.selection_rule!r}"
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
        # threshold_state is deliberately unread: this strategy exists so the
        # training loop never depends on a fitted decision boundary.
        if target_count <= 0:
            return self._skip("target_count_zero")

        system = getattr(probability_backend, "system", None)
        radius = getattr(probability_backend, "attractor_radius", None)
        if system is None or radius is None or not hasattr(probability_backend, "sample_endpoints"):
            raise RuntimeError(
                "EntropyAcquisitionStrategy needs a probability backend exposing "
                "system, attractor_radius and sample_endpoints(); got "
                f"{type(probability_backend).__name__}"
            )

        states, indices = pool.sample_candidates_without_marking(
            self.n_candidates, exclude=exclude
        )
        n_actual = len(indices)
        if n_actual == 0:
            if self.verbose:
                print("    [Entropy] Pool exhausted. No candidates available.")
            return self._skip("pool_exhausted")

        if self.verbose:
            print(
                f"    [Entropy] Evaluating {n_actual} candidates with "
                f"K={self.num_mc_samples}, tie_breaker={self.tie_breaker}, "
                f"rule={self.selection_rule}, selecting {target_count}"
            )

        cloud = probability_backend.sample_endpoints(
            states, num_samples=self.num_mc_samples, verbose=self.verbose
        )
        M, K, D = cloud.shape

        labels = system.classify_attractor(
            torch.as_tensor(cloud.reshape(-1, D), dtype=torch.float32), radius=radius
        ).reshape(M, K)
        p_success = (labels == 1).float().mean(dim=1).cpu().numpy().astype(np.float64)

        eps = 1e-12
        pc = np.clip(p_success, eps, 1.0 - eps)
        entropy = -(pc * np.log(pc) + (1.0 - pc) * np.log(1.0 - pc))
        finite_cloud = np.isfinite(cloud).all(axis=(1, 2))
        entropy = np.where(finite_cloud, entropy, np.nan)

        tie = None
        if self.tie_breaker == "mode_sep":
            tie = mode_separation(
                cloud,
                system.get_normalization_scales().cpu().numpy().astype(np.float64),
                self._circular_mask(system),
                chunk_size=self.chunk_size,
                device=getattr(probability_backend, "device", "cpu"),
            )

        score = self._ranking_score(entropy, tie)
        positions = self._apply_selection_rule(score, np.asarray(states), system, target_count)
        selected = [indices[int(p)] for p in positions]

        diagnostics = self._diagnostics(entropy, p_success, positions, n_actual, tie)
        if self.verbose:
            print(
                f"    Selected {len(selected)}/{n_actual}; entropy "
                f"max={diagnostics['entropy_max']}, mean={diagnostics['entropy_mean']}, "
                f"n_at_max={diagnostics['n_tied_at_max_entropy']}"
            )

        return AcquisitionResult(
            d1_indices=[],
            d2_indices=selected,
            n_candidates_evaluated=n_actual,
            n_certain_discarded=n_actual - len(selected),
            n_invalid_added=0,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _circular_mask(system) -> np.ndarray:
        n = int(system.state_dim)
        mask = np.zeros(n, dtype=bool)
        idx = system.get_circular_indices()
        if idx:
            mask[np.asarray(idx, dtype=int)] = True
        return mask

    @staticmethod
    def _ranking_score(entropy: np.ndarray, tie: np.ndarray | None) -> np.ndarray:
        """A descending score whose order is entropy, then the tie-breaker.

        Ranking is turned back into a score because the selection rules consume
        scores. Lexicographic ordering keeps the tie-breaker strictly subordinate:
        it can reorder candidates of equal entropy and nothing else, so no
        relative-weight parameter is introduced.
        """
        valid = np.flatnonzero(np.isfinite(entropy))
        if tie is None:
            order = valid[np.argsort(-entropy[valid], kind="stable")]
        else:
            t = np.where(np.isfinite(tie), tie, -np.inf)[valid]
            order = valid[np.lexsort((-t, -entropy[valid]))]
        score = np.full(len(entropy), np.nan)
        score[order] = np.arange(len(order), 0, -1, dtype=np.float64)
        return score

    def _apply_selection_rule(self, score, states, system, target_count) -> np.ndarray:
        if self.selection_rule == "greedy":
            return select_greedy(score, target_count)
        return select_greedy_diverse(
            score,
            states,
            system.get_normalization_scales().cpu().numpy().astype(np.float64),
            self._circular_mask(system),
            target_count,
            pool_multiplier=self.diversity_pool_multiplier,
        )

    def _diagnostics(self, entropy, p_success, positions, n_actual, tie) -> dict[str, Any]:
        finite = entropy[np.isfinite(entropy)]
        n_at_max = int((finite == finite.max()).sum()) if len(finite) else 0
        return {
            "entropy_min": float(finite.min()) if len(finite) else None,
            "entropy_max": float(finite.max()) if len(finite) else None,
            "entropy_mean": float(finite.mean()) if len(finite) else None,
            "entropy_median": float(np.median(finite)) if len(finite) else None,
            # How many candidates share the top entropy value: with K samples p has
            # only K+1 levels, so this is the size of the set the tie-breaker ranks
            # within. If it exceeds the acquisition budget, plain entropy is
            # choosing arbitrarily among them.
            "n_tied_at_max_entropy": n_at_max,
            "mean_p_success": float(np.nanmean(p_success)),
            "mean_p_success_selected": float(np.nanmean(p_success[positions]))
            if len(positions) else None,
            "n_candidates_evaluated": int(n_actual),
            "n_nonfinite_excluded": int((~np.isfinite(entropy)).sum()),
            "tie_breaker": self.tie_breaker,
            "selection_rule": self.selection_rule,
            "num_mc_samples": self.num_mc_samples,
            "mode_sep_mean_selected": float(np.nanmean(tie[positions]))
            if (tie is not None and len(positions)) else None,
        }

    @staticmethod
    def _skip(reason: str) -> AcquisitionResult:
        return AcquisitionResult(
            d1_indices=[], d2_indices=[], n_candidates_evaluated=0,
            n_certain_discarded=0, n_invalid_added=0,
            diagnostics={"skipped_reason": reason},
        )

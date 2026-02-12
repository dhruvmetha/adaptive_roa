"""Shared invalid-endpoint refinement logic.

Extracted from ProbabilityEstimator.estimate() and evaluate_full_roa_fast()
to eliminate ~45 lines of duplicated refinement loop code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RefinementStats:
    """Accumulated statistics from one or more refinement calls."""

    n_initially_invalid: int = 0
    n_resolved_success: int = 0
    n_resolved_failure: int = 0
    per_attempt_resolved: list[int] = field(default_factory=list)

    @property
    def n_resolved(self) -> int:
        return self.n_resolved_success + self.n_resolved_failure

    def accumulate(self, other: RefinementStats) -> None:
        """Merge *other* into this instance (in-place)."""
        self.n_initially_invalid += other.n_initially_invalid
        self.n_resolved_success += other.n_resolved_success
        self.n_resolved_failure += other.n_resolved_failure
        for i, count in enumerate(other.per_attempt_resolved):
            if i < len(self.per_attempt_resolved):
                self.per_attempt_resolved[i] += count


def refine_invalid_endpoints(
    flow_matcher,
    system,
    endpoints: torch.Tensor,
    labels: torch.Tensor,
    start_states: torch.Tensor,
    attractor_radius: float,
    t_range: tuple[float, float],
    num_steps: int,
    max_attempts: int,
) -> RefinementStats:
    """Refine invalid (label==0) endpoints in-place.

    Iteratively calls ``flow_matcher.refine_endpoints`` on endpoints whose
    label is still 0, reclassifying after each attempt.  Both *endpoints*
    and *labels* tensors are **mutated** so callers see the refined values.

    Returns:
        RefinementStats with counts of initially-invalid, resolved-to-success,
        resolved-to-failure, and per-attempt breakdown.
    """
    stats = RefinementStats(per_attempt_resolved=[0] * max_attempts)

    still_invalid = labels == 0
    n_initial = int(still_invalid.sum().item())
    if n_initial == 0:
        return stats

    stats.n_initially_invalid = n_initial

    for attempt in range(max_attempts):
        if not still_invalid.any():
            break

        refined = flow_matcher.refine_endpoints(
            invalid_endpoints=endpoints[still_invalid],
            start_states=start_states[still_invalid],
            t_range=t_range,
            num_steps=num_steps,
        )
        refined_labels = system.classify_attractor(
            refined, radius=attractor_radius
        )

        resolved_success = refined_labels == 1
        resolved_failure = refined_labels == -1

        stats.n_resolved_success += int(resolved_success.sum().item())
        stats.n_resolved_failure += int(resolved_failure.sum().item())
        stats.per_attempt_resolved[attempt] = int(
            (resolved_success | resolved_failure).sum().item()
        )

        # Map back to batch-level indices and update labels
        still_invalid_indices = still_invalid.nonzero(as_tuple=True)[0]
        labels[still_invalid_indices[resolved_success]] = 1
        labels[still_invalid_indices[resolved_failure]] = -1

        # Update endpoints for resolved and still-invalid entries
        endpoints[still_invalid] = refined

        # Narrow to entries that remain invalid
        still_invalid_remaining = refined_labels == 0
        new_still_invalid = torch.zeros_like(still_invalid)
        new_still_invalid[still_invalid_indices[still_invalid_remaining]] = True
        still_invalid = new_still_invalid

    return stats

"""Confidence-based training pair filtering.

After each epoch, filters the training dataset to keep only pairs where
the model is UNCERTAIN (decision == 0), focusing training capacity on
the decision boundary rather than trivially easy mid/late-trajectory pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adaptive_roa.adaptive_v2.types import ThresholdState
from adaptive_roa.conformal.lambda_optimizer import apply_one_sided_rule, apply_two_sided_rule


@dataclass
class FilterDiagnostics:
    """Diagnostics from a confidence filtering pass."""

    total_pairs: int
    n_existing: int
    n_new: int
    kept_pairs: int
    discarded_pairs: int
    n_success: int
    n_failure: int
    n_uncertain: int
    n_misclassified: int
    kept_fraction: float


class ConfidencePairFilter:
    """Filters training pairs to keep only those where the model is uncertain.

    Uses the conformal decision rule (one-sided or two-sided) with the
    current epoch's optimized lambda_star and delta_star to classify each
    training pair's start state. Only UNCERTAIN pairs (decision == 0) are
    retained for the next epoch's training.
    """

    def __init__(
        self,
        probability_backend,
        decision_rule: str = "two_sided",
        batch_size: int = 2048,
        min_pairs_floor: int = 100,
    ):
        self.probability_backend = probability_backend
        self.decision_rule = decision_rule
        self.batch_size = batch_size
        self.min_pairs_floor = min_pairs_floor

    def filter_train_file(
        self,
        train_file: str,
        threshold_state: ThresholdState,
        labels: np.ndarray | None = None,
        n_existing: int = 0,
        output_file: str | None = None,
        verbose: bool = True,
    ) -> tuple[str, FilterDiagnostics]:
        """Filter a training file to keep only uncertain and misclassified pairs.

        Only newly added pairs (rows after ``n_existing``) are subject to
        filtering.  Old pairs (the first ``n_existing`` rows) are always
        kept unchanged.

        Args:
            train_file: Path to the training data file (space-separated,
                each row is [start_state... end_state...]).
            threshold_state: Current epoch's threshold state with
                lambda_star and delta_star.
            labels: Ground truth labels (1=success, -1=failure) for each
                pair. If provided, misclassified confident pairs are also
                kept alongside uncertain ones.
            n_existing: Number of rows at the start of the file that belong
                to previous epochs and should be kept as-is.
            output_file: Where to write filtered data. If None, overwrites
                train_file.
            verbose: Whether to print filtering diagnostics.

        Returns:
            Tuple of (output_path, FilterDiagnostics).
        """
        data = np.loadtxt(train_file)
        n_total = len(data)
        state_dim = data.shape[1] // 2

        # Split into old (kept as-is) and new (subject to filtering)
        n_existing = max(0, min(n_existing, n_total))
        old_data = data[:n_existing]
        new_data = data[n_existing:]
        n_new = len(new_data)

        if n_new == 0:
            if verbose:
                print(f"  [filter] No new pairs to filter ({n_existing} existing kept)")
            diagnostics = FilterDiagnostics(
                total_pairs=n_total,
                n_existing=n_existing,
                n_new=0,
                kept_pairs=n_total,
                discarded_pairs=0,
                n_success=0,
                n_failure=0,
                n_uncertain=0,
                n_misclassified=0,
                kept_fraction=1.0,
            )
            return train_file, diagnostics

        new_start_states = new_data[:, :state_dim]
        new_labels = labels[n_existing:] if labels is not None else None

        # Estimate probabilities for new start states only
        probs = self.probability_backend.estimate(new_start_states)

        # Apply decision rule
        if self.decision_rule == "two_sided":
            decisions = apply_two_sided_rule(
                probs.p_success,
                probs.p_failure,
                threshold_state.lambda_star,
                threshold_state.delta_star,
                p_invalid=probs.p_invalid,
            )
        else:
            decisions = apply_one_sided_rule(
                probs.p_success,
                probs.p_failure,
                threshold_state.lambda_star,
                threshold_state.delta_star,
                p_invalid=probs.p_invalid,
            )

        n_success = int(np.sum(decisions == 1))
        n_failure = int(np.sum(decisions == -1))
        n_uncertain = int(np.sum(decisions == 0))

        # Identify misclassified pairs: model is confident but wrong
        uncertain_mask = decisions == 0
        misclassified_mask = np.zeros(n_new, dtype=bool)
        if new_labels is not None:
            misclassified_mask = (decisions != 0) & (decisions != new_labels)
        n_misclassified = int(np.sum(misclassified_mask))

        keep_mask = uncertain_mask | misclassified_mask
        n_kept_new = int(np.sum(keep_mask))
        n_kept_total = n_existing + n_kept_new

        diagnostics = FilterDiagnostics(
            total_pairs=n_total,
            n_existing=n_existing,
            n_new=n_new,
            kept_pairs=n_kept_total,
            discarded_pairs=n_new - n_kept_new,
            n_success=n_success,
            n_failure=n_failure,
            n_uncertain=n_uncertain,
            n_misclassified=n_misclassified,
            kept_fraction=n_kept_total / n_total if n_total > 0 else 0.0,
        )

        if verbose:
            print(f"  [filter] {n_total} total pairs ({n_existing} existing, {n_new} new)")
            print(f"  [filter] New pairs: "
                  f"{n_success} confident-success, "
                  f"{n_failure} confident-failure, "
                  f"{n_uncertain} uncertain, "
                  f"{n_misclassified} misclassified (kept)")
            print(f"  [filter] Keeping {n_kept_total} pairs "
                  f"({n_existing} old + {n_kept_new} new, {diagnostics.kept_fraction:.1%})")

        # Safety floor: if too few total kept pairs, skip filtering
        if n_kept_total < self.min_pairs_floor:
            if verbose:
                print(f"  [filter] Only {n_kept_total} kept pairs "
                      f"(< floor={self.min_pairs_floor}), skipping filter")
            diagnostics.kept_pairs = n_total
            diagnostics.discarded_pairs = 0
            diagnostics.kept_fraction = 1.0
            return train_file, diagnostics

        # Combine old (all kept) + filtered new
        filtered_new = new_data[keep_mask]
        if n_existing > 0:
            filtered_data = np.vstack([old_data, filtered_new])
        else:
            filtered_data = filtered_new

        out_path = output_file if output_file is not None else train_file
        np.savetxt(out_path, filtered_data, fmt="%.8f")

        if verbose:
            print(f"  [filter] Wrote {len(filtered_data)} pairs to {Path(out_path).name}")

        return out_path, diagnostics

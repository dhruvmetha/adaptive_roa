"""
Uncertain Sampler for Adaptive Training.

Samples candidates and filters for uncertain points using q_hat-based prediction sets.
Points are added to training if:
  - Uncertain: prediction set has multiple labels (e.g., {0, 1}, {-1, 0})
  - Certainly invalid: prediction set is {0} (invalid region)
Certain success/failure points (singleton {1} or {-1}) are discarded back to the pool.
"""

import numpy as np
from typing import List, Tuple, Set
from dataclasses import dataclass

from adaptive_roa.adaptive.dataset_builder import AdaptiveDatasetBuilder


@dataclass
class UncertainSamplingResult:
    """Result from uncertain sampling."""
    uncertain_indices: List[int]    # Uncertain indices to add to training
    n_candidates_evaluated: int     # Total candidates evaluated
    n_batches: int                  # Number of batches sampled
    n_certain_discarded: int        # Certain success/failure points discarded (stay in pool)
    n_invalid_added: int = 0        # Certainly invalid points added (prediction set = {0})


@dataclass
class RankedSamplingResult:
    """Result from non-conformity score ranked sampling."""
    selected_indices: List[int]       # Indices selected (lowest NC scores)
    all_scores: np.ndarray            # NC scores for all candidates evaluated
    n_candidates_evaluated: int       # Total candidates evaluated
    score_threshold: float            # NC score of the last selected point


class UncertainSampler:
    """
    Samples candidates and filters for uncertain points using q_hat.

    Algorithm:
    1. Sample batches of candidates
    2. Classify each using q_hat-based prediction sets
    3. Keep points that are:
       - Uncertain (multi-label prediction set), OR
       - Certainly invalid (prediction set = {0})
    4. Discard certain success/failure (singleton {1} or {-1})
    5. Repeat until target_count points found or pool exhausted

    Attributes:
        dataset_builder: AdaptiveDatasetBuilder for sampling
        target_count: Number of uncertain points to find
        batch_size: Size of batches when searching
        max_candidates: Maximum candidates to evaluate (safety cap)
    """

    def __init__(
        self,
        dataset_builder: AdaptiveDatasetBuilder,
        target_count: int,
        batch_size: int = 50,
        max_candidates: int = 50000,
    ):
        """
        Initialize uncertain sampler.

        Args:
            dataset_builder: AdaptiveDatasetBuilder for sampling
            target_count: Number of uncertain points to find
            batch_size: Size of batches when sampling
            max_candidates: Safety cap on total candidates evaluated
        """
        self.dataset_builder = dataset_builder
        self.target_count = target_count
        self.batch_size = batch_size
        self.max_candidates = max_candidates

    def sample(
        self,
        prob_estimator,
        calibrator,
        lambda_star: float,
        delta_star: float,
        q_hat: float,
        decision_rule: str = "two_sided",
        exclude: Set[int] = None,
        verbose: bool = True
    ) -> UncertainSamplingResult:
        """
        Sample until target_count uncertain points found.

        Args:
            prob_estimator: ProbabilityEstimator to estimate p(success|x)
            calibrator: Calibrator to compute prediction sets using q_hat
            lambda_star: Decision boundary from conformal prediction
            delta_star: Uncertainty half-width from conformal prediction
            q_hat: Calibration threshold from conformal prediction
            decision_rule: "one_sided" (p_s only) or "two_sided" (p_s and p_f)
            exclude: Set of indices to exclude from sampling (e.g., D1 indices)
            verbose: Print progress

        Returns:
            UncertainSamplingResult with uncertain indices
        """
        if exclude is None:
            exclude = set()

        if verbose:
            print(f"    Target: {self.target_count} uncertain points")
            print(f"    Batch size: {self.batch_size}, Max candidates: {self.max_candidates}")

        # Track state
        sampled_this_call = set(exclude)  # Start with excluded indices
        n_candidates_evaluated = 0
        n_batches = 0
        uncertain_indices = []
        n_certain_discarded = 0
        n_invalid_added = 0

        # Keep sampling until we find enough uncertain points
        while len(uncertain_indices) < self.target_count and n_candidates_evaluated < self.max_candidates:
            # Check availability
            # Note: sampled_this_call may include indices already in used_indices (from exclude),
            # so we only count indices that are NOT already used to avoid double-counting
            n_sampled_not_used = len(sampled_this_call - self.dataset_builder.used_indices)
            n_available = self.dataset_builder.get_n_available() - n_sampled_not_used
            if n_available <= 0:
                if verbose:
                    print(f"    Pool exhausted. Found {len(uncertain_indices)}/{self.target_count} uncertain.")
                break

            # Sample a batch
            batch_size = min(self.batch_size, self.max_candidates - n_candidates_evaluated)
            batch_states, batch_indices = self.dataset_builder.sample_candidates_without_marking(
                batch_size,
                exclude=sampled_this_call
            )

            if len(batch_indices) == 0:
                break

            sampled_this_call.update(batch_indices)
            n_candidates_evaluated += len(batch_indices)
            n_batches += 1

            # Classify batch using q_hat-based prediction sets
            uncertain_batch, invalid_batch, certain_batch = self._classify_batch(
                batch_states, batch_indices, prob_estimator, calibrator,
                lambda_star, delta_star, q_hat, decision_rule
            )

            # Add both uncertain and invalid points to training
            uncertain_indices.extend(uncertain_batch)
            uncertain_indices.extend(invalid_batch)
            n_invalid_added += len(invalid_batch)
            n_certain_discarded += len(certain_batch)

            if verbose:
                print(f"    Batch {n_batches}: {len(batch_indices)} evaluated, "
                      f"{len(uncertain_batch)} uncertain, {len(invalid_batch)} invalid, "
                      f"{len(certain_batch)} certain. "
                      f"Progress: {len(uncertain_indices)}/{self.target_count}")

        # Truncate if we overshot
        if len(uncertain_indices) > self.target_count:
            n_extra = len(uncertain_indices) - self.target_count
            uncertain_indices = uncertain_indices[:self.target_count]
            if verbose:
                print(f"    Truncated to {self.target_count} (discarded {n_extra} extra uncertain)")

        if verbose:
            print(f"    RESULT: {len(uncertain_indices)} points to add "
                  f"({len(uncertain_indices) - n_invalid_added} uncertain + {n_invalid_added} invalid), "
                  f"{n_candidates_evaluated} evaluated, {n_certain_discarded} certain discarded")

        return UncertainSamplingResult(
            uncertain_indices=uncertain_indices,
            n_candidates_evaluated=n_candidates_evaluated,
            n_batches=n_batches,
            n_certain_discarded=n_certain_discarded,
            n_invalid_added=n_invalid_added,
        )

    def _classify_batch(
        self,
        states: np.ndarray,
        indices: List[int],
        prob_estimator,
        calibrator,
        lambda_star: float,
        delta_star: float,
        q_hat: float,
        decision_rule: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Classify candidates using q_hat-based prediction sets.

        Classification:
        - Uncertain: prediction set has multiple labels (e.g., {0, 1}, {-1, 0})
        - Invalid: prediction set is {0} (certainly in invalid/unknown region)
        - Certain: prediction set is {1} or {-1} (certainly success or failure)

        Args:
            states: [N, state_dim] candidate states
            indices: Trajectory indices corresponding to states
            prob_estimator: ProbabilityEstimator to estimate p(success|x)
            calibrator: Calibrator to get prediction sets based on q_hat
            lambda_star: Decision boundary
            delta_star: Uncertainty half-width
            q_hat: Calibration threshold for conformal prediction sets
            decision_rule: "one_sided" or "two_sided"

        Returns:
            Tuple of (uncertain_indices, invalid_indices, certain_indices)
        """
        if len(states) == 0:
            return [], [], []

        # Estimate probabilities
        p_success, p_failure, _ = prob_estimator.estimate(states)

        # Get prediction sets based on q_hat
        p_fail = p_failure if decision_rule == "two_sided" else None
        prediction_sets = calibrator.get_prediction_sets_batch(
            p_success, lambda_star, q_hat, delta_star, p_fail
        )

        uncertain_indices = []
        invalid_indices = []
        certain_indices = []

        for i, idx in enumerate(indices):
            pred_set = prediction_sets[i]
            if len(pred_set) > 1:
                # Uncertain: prediction set has multiple labels
                uncertain_indices.append(idx)
            elif pred_set == {0}:
                # Certainly invalid: prediction set is {0}
                invalid_indices.append(idx)
            else:
                # Certain success ({1}) or failure ({-1}): discard
                certain_indices.append(idx)

        return uncertain_indices, invalid_indices, certain_indices

    def sample_direct(
        self,
        prob_estimator,
        lambda_star: float,
        delta_star: float,
        decision_rule: str = "two_sided",
        exclude: Set[int] = None,
        verbose: bool = True
    ) -> UncertainSamplingResult:
        """
        Sample uncertain points using λ*/δ* thresholds directly (no q_hat/conformal).

        This is a simpler approach without conformal prediction guarantees.
        Points are classified as:
        - Uncertain: in the [λ-δ, λ+δ] region (neither confident success nor failure)
        - Certain: outside the uncertain region

        Args:
            prob_estimator: ProbabilityEstimator to estimate p(success|x)
            lambda_star: Decision boundary
            delta_star: Uncertainty half-width
            decision_rule: "one_sided" (p_s only) or "two_sided" (p_s and p_f)
            exclude: Set of indices to exclude from sampling (e.g., D1 indices)
            verbose: Print progress

        Returns:
            UncertainSamplingResult with uncertain indices
        """
        if exclude is None:
            exclude = set()

        if verbose:
            print(f"    [Non-conformal] Target: {self.target_count} uncertain points")
            print(f"    Using λ*={lambda_star:.4f} ± δ*={delta_star:.4f} directly")
            print(f"    Batch size: {self.batch_size}, Max candidates: {self.max_candidates}")

        # Track state
        sampled_this_call = set(exclude)
        n_candidates_evaluated = 0
        n_batches = 0
        uncertain_indices = []
        n_certain_discarded = 0

        # Keep sampling until we find enough uncertain points
        while len(uncertain_indices) < self.target_count and n_candidates_evaluated < self.max_candidates:
            n_sampled_not_used = len(sampled_this_call - self.dataset_builder.used_indices)
            n_available = self.dataset_builder.get_n_available() - n_sampled_not_used
            if n_available <= 0:
                if verbose:
                    print(f"    Pool exhausted. Found {len(uncertain_indices)}/{self.target_count} uncertain.")
                break

            batch_size = min(self.batch_size, self.max_candidates - n_candidates_evaluated)
            batch_states, batch_indices = self.dataset_builder.sample_candidates_without_marking(
                batch_size,
                exclude=sampled_this_call
            )

            if len(batch_indices) == 0:
                break

            sampled_this_call.update(batch_indices)
            n_candidates_evaluated += len(batch_indices)
            n_batches += 1

            # Classify using direct thresholds
            uncertain_batch, certain_batch = self._classify_batch_direct(
                batch_states, batch_indices, prob_estimator,
                lambda_star, delta_star, decision_rule
            )

            uncertain_indices.extend(uncertain_batch)
            n_certain_discarded += len(certain_batch)

            if verbose:
                print(f"    Batch {n_batches}: {len(batch_indices)} evaluated, "
                      f"{len(uncertain_batch)} uncertain, {len(certain_batch)} certain. "
                      f"Progress: {len(uncertain_indices)}/{self.target_count}")

        # Truncate if we overshot
        if len(uncertain_indices) > self.target_count:
            n_extra = len(uncertain_indices) - self.target_count
            uncertain_indices = uncertain_indices[:self.target_count]
            if verbose:
                print(f"    Truncated to {self.target_count} (discarded {n_extra} extra)")

        if verbose:
            print(f"    RESULT: {len(uncertain_indices)} uncertain points, "
                  f"{n_candidates_evaluated} evaluated, {n_certain_discarded} certain discarded")

        return UncertainSamplingResult(
            uncertain_indices=uncertain_indices,
            n_candidates_evaluated=n_candidates_evaluated,
            n_batches=n_batches,
            n_certain_discarded=n_certain_discarded,
            n_invalid_added=0,  # No invalid concept in direct mode
        )

    def _classify_batch_direct(
        self,
        states: np.ndarray,
        indices: List[int],
        prob_estimator,
        lambda_star: float,
        delta_star: float,
        decision_rule: str
    ) -> Tuple[List[int], List[int]]:
        """
        Classify candidates using λ*/δ* thresholds directly (no q_hat).

        One-sided rule:
        - Certain SUCCESS: p_success > λ + δ
        - Certain FAILURE: p_success < λ - δ
        - Uncertain: otherwise

        Two-sided rule:
        - Certain SUCCESS: p_success > λ + δ
        - Certain FAILURE: (1 - p_failure) < λ - δ
        - Uncertain: otherwise

        Args:
            states: [N, state_dim] candidate states
            indices: Trajectory indices corresponding to states
            prob_estimator: ProbabilityEstimator to estimate p(success|x)
            lambda_star: Decision boundary
            delta_star: Uncertainty half-width
            decision_rule: "one_sided" or "two_sided"

        Returns:
            Tuple of (uncertain_indices, certain_indices)
        """
        if len(states) == 0:
            return [], []

        # Estimate probabilities
        p_success, p_failure, _ = prob_estimator.estimate(states)

        success_thresh = lambda_star + delta_star
        failure_thresh = lambda_star - delta_star

        uncertain_indices = []
        certain_indices = []

        for i, idx in enumerate(indices):
            p_s = p_success[i]
            p_f = p_failure[i]

            if decision_rule == "two_sided":
                # Two-sided: certain success if p_s > λ+δ, certain failure if (1-p_f) < λ-δ
                is_certain_success = p_s > success_thresh
                is_certain_failure = (1.0 - p_f) < failure_thresh
                is_certain = is_certain_success or is_certain_failure
            else:
                # One-sided: certain success if p_s > λ+δ, certain failure if p_s < λ-δ
                is_certain = (p_s > success_thresh) or (p_s < failure_thresh)

            if is_certain:
                certain_indices.append(idx)
            else:
                uncertain_indices.append(idx)

        return uncertain_indices, certain_indices

    def sample_ranked(
        self,
        prob_estimator,
        calibrator,
        lambda_star: float,
        delta_star: float,
        decision_rule: str = "two_sided",
        n_candidates: int = 1000,
        n_select: int = 100,
        verbose: bool = True,
    ) -> RankedSamplingResult:
        """
        Rank candidates by non-conformity score (with y=UNKNOWN) and select the lowest-scored.

        Computes the NC score with label fixed to UNKNOWN (y=0) for all candidates.
        Low NC score = the point sits in the uncertain region (neither p_success nor
        p_failure is confidently high), making it most informative for training.

        No D1/D2 split. No q_hat calibration. No true labels needed.

        Args:
            prob_estimator: ProbabilityEstimator to estimate p(success|x)
            calibrator: Calibrator with non_conformity_scores_batch method
            lambda_star: Decision boundary from threshold optimization
            delta_star: Uncertainty half-width
            decision_rule: "one_sided" or "two_sided"
            n_candidates: Number of candidates to evaluate from pool
            n_select: Number of points to select (lowest NC scores)
            verbose: Print progress

        Returns:
            RankedSamplingResult with selected indices and score diagnostics
        """
        if verbose:
            print(f"    [Ranked] Evaluating {n_candidates} candidates, selecting {n_select} lowest NC scores")
            print(f"    Using λ*={lambda_star:.4f}, δ*={delta_star:.4f}, rule={decision_rule}")

        # Step 1: Sample candidates from pool
        candidate_states, candidate_indices = self.dataset_builder.sample_candidates_without_marking(
            n_candidates
        )

        n_actual = len(candidate_indices)
        if n_actual == 0:
            if verbose:
                print(f"    Pool exhausted. No candidates available.")
            return RankedSamplingResult(
                selected_indices=[],
                all_scores=np.array([]),
                n_candidates_evaluated=0,
                score_threshold=float('inf'),
            )

        if verbose and n_actual < n_candidates:
            print(f"    Pool has only {n_actual} available (requested {n_candidates})")

        # Step 2: Estimate probabilities
        p_success, p_failure, _ = prob_estimator.estimate(candidate_states)

        # Step 3: Compute NC scores with label = UNKNOWN (y=0) for all candidates
        y_unknown = np.zeros(n_actual, dtype=int)
        p_fail_for_rule = p_failure if decision_rule == "two_sided" else None
        scores = calibrator.non_conformity_scores_batch(
            p_success, y_unknown, lambda_star, delta_star, p_fail_for_rule
        )

        # Step 4: Sort ascending (lowest NC score = most uncertain)
        sorted_order = np.argsort(scores)

        # Step 5: Select top n_select
        n_to_select = min(n_select, len(sorted_order))
        selected_order = sorted_order[:n_to_select]
        selected_indices = [candidate_indices[i] for i in selected_order]

        score_threshold = float(scores[sorted_order[n_to_select - 1]]) if n_to_select > 0 else float('inf')

        if verbose:
            print(f"    Selected {len(selected_indices)}/{n_actual} candidates")
            print(f"    Score stats: min={scores.min():.4f}, max={scores.max():.4f}, "
                  f"mean={scores.mean():.4f}, median={np.median(scores):.4f}")
            print(f"    Selection threshold (max score of selected): {score_threshold:.4f}")

        return RankedSamplingResult(
            selected_indices=selected_indices,
            all_scores=scores,
            n_candidates_evaluated=n_actual,
            score_threshold=score_threshold,
        )

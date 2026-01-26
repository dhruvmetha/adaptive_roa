"""
Uncertain Sampler for Adaptive Training.

Samples candidates and filters for uncertain points using q_hat-based prediction sets.
Uncertain points (prediction set has multiple labels) are kept for training.
Certain points (singleton prediction set) are discarded back to the pool.
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
    n_certain_discarded: int        # Certain points discarded (stay in pool)


class UncertainSampler:
    """
    Samples candidates and filters for uncertain points using q_hat.

    Algorithm:
    1. Sample batches of candidates
    2. Classify each using q_hat-based prediction sets
    3. Keep uncertain (multi-label prediction set), discard certain (singleton)
    4. Repeat until target_count uncertain points found or pool exhausted

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
            uncertain_batch, certain_batch = self._classify_batch(
                batch_states, batch_indices, prob_estimator, calibrator,
                lambda_star, delta_star, q_hat, decision_rule
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
                print(f"    Truncated to {self.target_count} (discarded {n_extra} extra uncertain)")

        if verbose:
            print(f"    RESULT: {len(uncertain_indices)} uncertain found, "
                  f"{n_candidates_evaluated} evaluated, {n_certain_discarded} certain discarded")

        return UncertainSamplingResult(
            uncertain_indices=uncertain_indices,
            n_candidates_evaluated=n_candidates_evaluated,
            n_batches=n_batches,
            n_certain_discarded=n_certain_discarded,
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
    ) -> Tuple[List[int], List[int]]:
        """
        Classify candidates as uncertain or certain using q_hat-based prediction sets.

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
            Tuple of (uncertain_indices, certain_indices)
        """
        if len(states) == 0:
            return [], []

        # Estimate probabilities
        p_success, p_failure, _ = prob_estimator.estimate(states)

        # Get prediction sets based on q_hat
        p_fail = p_failure if decision_rule == "two_sided" else None
        prediction_sets = calibrator.get_prediction_sets_batch(
            p_success, lambda_star, q_hat, delta_star, p_fail
        )

        uncertain_indices = []
        certain_indices = []

        for i, idx in enumerate(indices):
            if len(prediction_sets[i]) > 1:
                # Uncertain: prediction set has multiple labels
                uncertain_indices.append(idx)
            else:
                # Certain: prediction set is singleton
                certain_indices.append(idx)

        return uncertain_indices, certain_indices

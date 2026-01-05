"""
Balanced Uncertain Sampler for Adaptive Training.

Samples until we find equal numbers of certain (calibration) and uncertain points.
Uncertain points are identified using the trained flow matcher + conformal predictor.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.adaptive.dataset_builder import AdaptiveDatasetBuilder


@dataclass
class BalancedSamplingResult:
    """Result from balanced sampling."""
    d1_indices: List[int]           # Calibration indices (unclassified, always add)
    uncertain_indices: List[int]    # Uncertain indices (matched to |D1|)
    n_total_sampled: int            # Total candidates evaluated
    n_batches: int                  # Number of batches sampled
    n_discarded_certain: int        # Certain points discarded (stay in pool)
    n_truncated_uncertain: int = 0  # Uncertain points truncated (if overshot target)


class BalancedUncertainSampler:
    """
    Samples until |uncertain| == |D1| (calibration set size).

    Algorithm:
    1. Sample initial batch, split into D1 (calibration) and D2 (selection)
    2. D1 is always added to training
    3. From D2, identify uncertain points using flow matcher
    4. If |uncertain| < |D1|, sample more batches
    5. Keep only uncertain from additional batches
    6. Stop when |uncertain| == |D1| or max_samples reached

    Attributes:
        dataset_builder: AdaptiveDatasetBuilder for sampling
        initial_batch_size: Size of first batch (split into D1 + D2)
        d1_ratio: Fraction of initial batch for D1 (calibration)
        additional_batch_size: Size of subsequent batches
        max_samples: Maximum total samples per epoch (safety cap)
    """

    def __init__(
        self,
        dataset_builder: AdaptiveDatasetBuilder,
        initial_batch_size: int = 50,
        d1_ratio: float = 0.5,
        additional_batch_size: int = 50,
        max_samples: int = 500,
    ):
        """
        Initialize balanced sampler.

        Args:
            dataset_builder: AdaptiveDatasetBuilder for sampling
            initial_batch_size: Size of first batch
            d1_ratio: Fraction for D1 (calibration)
            additional_batch_size: Size of subsequent batches
            max_samples: Safety cap on total samples per epoch
        """
        self.dataset_builder = dataset_builder
        self.initial_batch_size = initial_batch_size
        self.d1_ratio = d1_ratio
        self.additional_batch_size = additional_batch_size
        self.max_samples = max_samples

    def sample_epoch(
        self,
        prob_estimator,
        lambda_star: float,
        delta_star: float,
        verbose: bool = True
    ) -> BalancedSamplingResult:
        """
        Sample candidates for one epoch using balanced uncertain strategy.

        Args:
            prob_estimator: ProbabilityEstimator to estimate p(success|x)
            lambda_star: Decision boundary from conformal prediction
            delta_star: Uncertainty half-width from conformal prediction
            verbose: Print progress

        Returns:
            BalancedSamplingResult with indices to add to training
        """
        # Track indices sampled within this epoch to avoid re-sampling
        sampled_this_epoch = set()
        
        # Step 1: Sample initial batch
        states, indices = self.dataset_builder.sample_candidates_without_marking(
            self.initial_batch_size
        )
        sampled_this_epoch.update(indices)

        if len(indices) == 0:
            if verbose:
                print("No more trajectories available!")
            return BalancedSamplingResult(
                d1_indices=[],
                uncertain_indices=[],
                n_total_sampled=0,
                n_batches=0,
                n_discarded_certain=0
            )

        # Step 2: Split into D1 (calibration) and D2 (selection pool)
        n_d1 = int(len(indices) * self.d1_ratio)
        d1_indices = indices[:n_d1]
        d2_indices = indices[n_d1:]
        d2_states = states[n_d1:]

        if verbose:
            print(f"    Initial batch: {len(indices)} candidates")
            print(f"    D1 (calibration): {len(d1_indices)}")
            print(f"    D2 (selection pool): {len(d2_indices)}")

        # Step 3: Identify uncertain points in D2
        uncertain_indices = []
        certain_discarded = 0
        n_total_sampled = len(indices)
        n_batches = 1

        if len(d2_indices) > 0:
            uncertain_from_d2, certain_from_d2 = self._classify_candidates(
                d2_states, d2_indices, prob_estimator, lambda_star, delta_star
            )
            uncertain_indices.extend(uncertain_from_d2)
            certain_discarded += len(certain_from_d2)

            if verbose:
                print(f"    D2 classification: {len(uncertain_from_d2)} uncertain, {len(certain_from_d2)} certain (discarded)")

        # Step 4: Keep sampling until |uncertain| == |D1| or max reached
        target_uncertain = len(d1_indices)

        while len(uncertain_indices) < target_uncertain and n_total_sampled < self.max_samples:
            # Check if more candidates available (excluding already sampled this epoch)
            n_available = self.dataset_builder.get_n_available() - len(sampled_this_epoch)
            if n_available <= 0:
                if verbose:
                    print(f"    No more candidates available. Stopping with {len(uncertain_indices)} uncertain.")
                break

            # Sample additional batch (excluding already sampled this epoch)
            batch_size = min(self.additional_batch_size, self.max_samples - n_total_sampled)
            add_states, add_indices = self.dataset_builder.sample_candidates_without_marking(
                batch_size, 
                exclude=sampled_this_epoch
            )

            if len(add_indices) == 0:
                break

            # Track these indices as sampled this epoch
            sampled_this_epoch.update(add_indices)
            n_total_sampled += len(add_indices)
            n_batches += 1

            # Classify additional candidates
            uncertain_from_batch, certain_from_batch = self._classify_candidates(
                add_states, add_indices, prob_estimator, lambda_star, delta_star
            )

            # Only keep uncertain (certain are discarded, stay in pool)
            uncertain_indices.extend(uncertain_from_batch)
            certain_discarded += len(certain_from_batch)

            if verbose:
                print(f"    Batch {n_batches}: {len(add_indices)} sampled, "
                      f"{len(uncertain_from_batch)} uncertain, {len(certain_from_batch)} discarded. "
                      f"Total uncertain: {len(uncertain_indices)}/{target_uncertain}")

        # Step 5: Truncate if we overshot
        n_truncated = 0
        if len(uncertain_indices) > target_uncertain:
            n_truncated = len(uncertain_indices) - target_uncertain
            uncertain_indices = uncertain_indices[:target_uncertain]
            if verbose:
                print(f"    Truncated to {target_uncertain} uncertain (discarded {n_truncated} extra)")

        if verbose:
            print(f"    FINAL: D1={len(d1_indices)}, Uncertain={len(uncertain_indices)}, "
                  f"Sampled={n_total_sampled}, Batches={n_batches}")

        return BalancedSamplingResult(
            d1_indices=d1_indices,
            uncertain_indices=uncertain_indices,
            n_total_sampled=n_total_sampled,
            n_batches=n_batches,
            n_discarded_certain=certain_discarded,
            n_truncated_uncertain=n_truncated
        )

    def _classify_candidates(
        self,
        states: np.ndarray,
        indices: List[int],
        prob_estimator,
        lambda_star: float,
        delta_star: float
    ) -> Tuple[List[int], List[int]]:
        """
        Classify candidates as uncertain or certain using flow matcher.

        Args:
            states: [N, state_dim] candidate states
            indices: Trajectory indices corresponding to states
            prob_estimator: ProbabilityEstimator to estimate p(success|x)
            lambda_star: Decision boundary
            delta_star: Uncertainty half-width

        Returns:
            Tuple of (uncertain_indices, certain_indices)
        """
        if len(states) == 0:
            return [], []

        # Estimate p(success|x) for all candidates
        p_success, _, _ = prob_estimator.estimate(states)

        # Classify based on decision boundaries
        lower = lambda_star - delta_star
        upper = lambda_star + delta_star

        uncertain_indices = []
        certain_indices = []

        for i, idx in enumerate(indices):
            if lower <= p_success[i] <= upper:
                uncertain_indices.append(idx)
            else:
                certain_indices.append(idx)

        return uncertain_indices, certain_indices

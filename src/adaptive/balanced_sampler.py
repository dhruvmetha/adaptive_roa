"""
Balanced Uncertain Sampler for Adaptive Training.

Samples a fixed total per epoch, split between D1 (random/calibration) and D2 (uncertainty-filtered).
D2 ratio controls what fraction goes through uncertainty filtering.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.adaptive.dataset_builder import AdaptiveDatasetBuilder


@dataclass
class BalancedSamplingResult:
    """Result from balanced sampling."""
    d1_indices: List[int]           # Calibration indices (random, always add)
    d2_indices: List[int]           # Uncertain indices from D2 selection
    n_total_sampled: int            # Total candidates evaluated
    n_batches: int                  # Number of batches sampled
    n_discarded_certain: int        # Certain points discarded (stay in pool)
    n_truncated: int = 0            # Points truncated (if overshot target)


class BalancedUncertainSampler:
    """
    Samples a fixed total per epoch with configurable D1/D2 split.

    Algorithm:
    1. Compute target sizes: D1 = total * (1 - d2_ratio), D2 = total * d2_ratio
    2. Sample D1 candidates randomly (always added to training)
    3. Sample and filter D2 candidates until we find enough uncertain ones
    4. Total added = D1 + D2 = adaptive_data_max

    Attributes:
        dataset_builder: AdaptiveDatasetBuilder for sampling
        adaptive_data_max: Total samples to add per epoch (D1 + D2)
        d2_ratio: Fraction of total that goes through uncertainty filtering
        batch_size: Size of batches when searching for uncertain points
        max_samples: Maximum candidates to evaluate per epoch (safety cap)
    """

    def __init__(
        self,
        dataset_builder: AdaptiveDatasetBuilder,
        adaptive_data_max: int = 50,
        d2_ratio: float = 0.5,
        batch_size: int = 50,
        max_samples: int = 50000,
    ):
        """
        Initialize balanced sampler.

        Args:
            dataset_builder: AdaptiveDatasetBuilder for sampling
            adaptive_data_max: Total samples to add per epoch (D1 + D2)
            d2_ratio: Fraction for D2 (uncertainty-filtered). 0 = all random, 1 = all filtered.
            batch_size: Size of batches when sampling for D2
            max_samples: Safety cap on total candidates evaluated per epoch
        """
        self.dataset_builder = dataset_builder
        self.adaptive_data_max = adaptive_data_max
        self.d2_ratio = d2_ratio
        self.batch_size = batch_size
        self.max_samples = max_samples

    def sample_epoch(
        self,
        prob_estimator,
        lambda_star: float,
        delta_star: float,
        verbose: bool = True
    ) -> BalancedSamplingResult:
        """
        Sample candidates for one epoch.

        Args:
            prob_estimator: ProbabilityEstimator to estimate p(success|x)
            lambda_star: Decision boundary from conformal prediction
            delta_star: Uncertainty half-width from conformal prediction
            verbose: Print progress

        Returns:
            BalancedSamplingResult with indices to add to training
        """
        # Compute target sizes
        n_d1_target = int(self.adaptive_data_max * (1 - self.d2_ratio))
        n_d2_target = self.adaptive_data_max - n_d1_target  # Ensures exact total

        if verbose:
            print(f"    Target: D1={n_d1_target} (random), D2={n_d2_target} (uncertain)")
            print(f"    d2_ratio={self.d2_ratio}, adaptive_data_max={self.adaptive_data_max}")

        # Track indices sampled within this epoch to avoid re-sampling
        sampled_this_epoch = set()
        n_total_sampled = 0
        n_batches = 0

        # Step 1: Sample D1 (random, always added)
        d1_states, d1_indices = self.dataset_builder.sample_candidates_without_marking(n_d1_target)
        sampled_this_epoch.update(d1_indices)
        n_total_sampled += len(d1_indices)
        if len(d1_indices) > 0:
            n_batches = 1

        if verbose:
            print(f"    D1 sampled: {len(d1_indices)} random trajectories")

        if len(d1_indices) == 0 and n_d1_target > 0:
            if verbose:
                print("    No more trajectories available!")
            return BalancedSamplingResult(
                d1_indices=[],
                d2_indices=[],
                n_total_sampled=0,
                n_batches=0,
                n_discarded_certain=0
            )

        # Step 2: Sample D2 (uncertainty-filtered)
        d2_indices = []
        certain_discarded = 0
        n_truncated = 0

        if n_d2_target > 0:
            # Keep sampling until we find enough uncertain points
            while len(d2_indices) < n_d2_target and n_total_sampled < self.max_samples:
                # Check availability
                n_available = self.dataset_builder.get_n_available() - len(sampled_this_epoch)
                if n_available <= 0:
                    if verbose:
                        print(f"    No more candidates available. D2 has {len(d2_indices)}/{n_d2_target}.")
                    break

                # Sample a batch
                batch_size = min(self.batch_size, self.max_samples - n_total_sampled)
                batch_states, batch_indices = self.dataset_builder.sample_candidates_without_marking(
                    batch_size,
                    exclude=sampled_this_epoch
                )

                if len(batch_indices) == 0:
                    break

                sampled_this_epoch.update(batch_indices)
                n_total_sampled += len(batch_indices)
                n_batches += 1

                # Classify batch
                uncertain_from_batch, certain_from_batch = self._classify_candidates(
                    batch_states, batch_indices, prob_estimator, lambda_star, delta_star
                )

                d2_indices.extend(uncertain_from_batch)
                certain_discarded += len(certain_from_batch)

                if verbose:
                    print(f"    Batch {n_batches}: {len(batch_indices)} sampled, "
                          f"{len(uncertain_from_batch)} uncertain, {len(certain_from_batch)} certain. "
                          f"D2 progress: {len(d2_indices)}/{n_d2_target}")

            # Truncate if we overshot
            if len(d2_indices) > n_d2_target:
                n_truncated = len(d2_indices) - n_d2_target
                d2_indices = d2_indices[:n_d2_target]
                if verbose:
                    print(f"    Truncated D2 to {n_d2_target} (discarded {n_truncated} extra uncertain)")

        # Final summary
        total_added = len(d1_indices) + len(d2_indices)
        if verbose:
            print(f"    FINAL: D1={len(d1_indices)}, D2={len(d2_indices)}, Total={total_added}")
            print(f"    Evaluated={n_total_sampled} candidates, Discarded={certain_discarded} certain, Batches={n_batches}")

        return BalancedSamplingResult(
            d1_indices=list(d1_indices),
            d2_indices=d2_indices,
            n_total_sampled=n_total_sampled,
            n_batches=n_batches,
            n_discarded_certain=certain_discarded,
            n_truncated=n_truncated
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

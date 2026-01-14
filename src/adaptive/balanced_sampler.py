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
        decision_rule: str = "one_sided",
        verbose: bool = True
    ) -> BalancedSamplingResult:
        """
        Sample candidates for one epoch.

        Args:
            prob_estimator: ProbabilityEstimator to estimate p(success|x)
            lambda_star: Decision boundary from conformal prediction
            delta_star: Uncertainty half-width from conformal prediction
            decision_rule: "one_sided" (p_s only) or "two_sided" (p_s and p_f)
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
        # Track both uncertain and certain samples from D2 search
        d2_indices = []          # Uncertain samples → added as D2
        certain_indices = []     # Certain samples → used for compensation if D2 falls short
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
                    batch_states, batch_indices, prob_estimator, lambda_star, delta_star, decision_rule
                )

                d2_indices.extend(uncertain_from_batch)
                certain_indices.extend(certain_from_batch)  # Keep certain for potential compensation

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

        # Step 3: Compensate with certain samples if D2 fell short
        d2_shortfall = n_d2_target - len(d2_indices)
        extra_d1_indices = []
        
        if d2_shortfall > 0 and len(certain_indices) > 0:
            # Use certain samples we already evaluated as extra D1
            n_to_add = min(d2_shortfall, len(certain_indices))
            extra_d1_indices = certain_indices[:n_to_add]
            
            if verbose:
                print(f"    D2 shortfall: {d2_shortfall}. Using {len(extra_d1_indices)} certain samples as extra D1.")
        
        n_certain_discarded = len(certain_indices) - len(extra_d1_indices)

        # Combine original D1 + extra D1 (from certain samples)
        all_d1_indices = list(d1_indices) + extra_d1_indices

        # Final summary
        total_added = len(all_d1_indices) + len(d2_indices)
        if verbose:
            print(f"    FINAL: D1={len(all_d1_indices)} (original={len(d1_indices)}, from_certain={len(extra_d1_indices)}), "
                  f"D2={len(d2_indices)} (uncertain), Total={total_added}")
            print(f"    Evaluated={n_total_sampled} candidates, Discarded={n_certain_discarded} certain, Batches={n_batches}")

        return BalancedSamplingResult(
            d1_indices=all_d1_indices,
            d2_indices=d2_indices,
            n_total_sampled=n_total_sampled,
            n_batches=n_batches,
            n_discarded_certain=n_certain_discarded,
            n_truncated=n_truncated
        )

    def _classify_candidates(
        self,
        states: np.ndarray,
        indices: List[int],
        prob_estimator,
        lambda_star: float,
        delta_star: float,
        decision_rule: str = "one_sided"
    ) -> Tuple[List[int], List[int]]:
        """
        Classify candidates as uncertain or certain using flow matcher.

        Args:
            states: [N, state_dim] candidate states
            indices: Trajectory indices corresponding to states
            prob_estimator: ProbabilityEstimator to estimate p(success|x)
            lambda_star: Decision boundary
            delta_star: Uncertainty half-width
            decision_rule: "one_sided" (p_s only) or "two_sided" (p_s and p_f)

        Returns:
            Tuple of (uncertain_indices, certain_indices)
        """
        if len(states) == 0:
            return [], []

        # Estimate probabilities for all candidates
        p_success, p_failure, _ = prob_estimator.estimate(states)

        uncertain_indices = []
        certain_indices = []

        if decision_rule == "two_sided":
            # Two-sided: use both p_s and p_f
            # Success if p_s > λ+δ, Failure if (1-p_f) < λ-δ, else uncertain
            success_thresh = lambda_star + delta_star
            failure_thresh = lambda_star - delta_star  # threshold on (1-p_f)

            for i, idx in enumerate(indices):
                is_success = p_success[i] > success_thresh
                is_failure = (1.0 - p_failure[i]) < failure_thresh
                
                if is_success or is_failure:
                    certain_indices.append(idx)
                else:
                    uncertain_indices.append(idx)
        else:
            # One-sided: only use p_s
            lower = lambda_star - delta_star
            upper = lambda_star + delta_star

            for i, idx in enumerate(indices):
                if lower <= p_success[i] <= upper:
                    uncertain_indices.append(idx)
                else:
                    certain_indices.append(idx)

        return uncertain_indices, certain_indices

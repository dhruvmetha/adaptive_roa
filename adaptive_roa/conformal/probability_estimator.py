"""
Probability Estimator for Conformal Prediction.

Uses Monte Carlo sampling with the flow matcher's latent variable
to estimate p(success|x) for each initial state.
"""
import torch
import numpy as np
from typing import Union, Tuple
from tqdm import tqdm
from adaptive_roa.conformal.config import ConformalConfig


class ProbabilityEstimator:
    """
    Estimate success probability via Monte Carlo sampling.

    For each initial state x, samples K different latent vectors z,
    predicts K endpoints using the flow matcher, classifies each endpoint,
    and computes empirical success rate as p(success|x).

    Uses the K-inner-loop pattern (matching evaluate_full_roa_fast):
    for each batch of B states, run K forward passes with fresh z each time,
    accumulate counts on GPU with vectorized ops, transfer to CPU only at the end.

    Attributes:
        flow_matcher: Trained flow matching model with predict_endpoint() method
        system: Dynamical system with classify_attractor() method
        config: ConformalConfig with num_mc_samples, mc_batch_size, attractor_radius
        device: Device for computation (cuda/cpu)
    """

    def __init__(
        self,
        flow_matcher,
        system,
        config: ConformalConfig,
        device: str = "cuda"
    ):
        self.flow_matcher = flow_matcher
        self.system = system
        self.config = config
        self.device = device

        self.flow_matcher.eval()

    @torch.no_grad()
    def estimate(
        self,
        states: Union[torch.Tensor, np.ndarray],
        verbose: bool = True,
        refine_invalids: bool = False,
        refine_t_range: Tuple[float, float] = (0.7, 0.9),
        refine_num_steps: int = 100,
        refine_max_attempts: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate p(success|x) for a batch of states.

        For each state, runs K forward passes (each with fresh z ~ N(0,I)),
        classifies endpoints, and computes empirical probabilities.

        When refine_invalids=True, endpoints classified as invalid (label=0)
        are refined by re-running the ODE from a random t_start ~ U[t_range]
        to t=1.0, using the invalid endpoint as the warm-start x_init.
        This is repeated up to refine_max_attempts times until the endpoint
        resolves or the budget is exhausted.

        Args:
            states: Initial states [N, state_dim] as torch tensor or numpy array
            verbose: Show progress bar and summary stats
            refine_invalids: If True, refine invalid endpoints via late-time ODE
            refine_t_range: (t_min, t_max) for sampling refinement start time
            refine_num_steps: Number of ODE steps for the refinement interval
            refine_max_attempts: Max refinement iterations per invalid endpoint

        Returns:
            Tuple of:
                p_success: [N] array of p(success|x) = count(label=1) / K
                p_failure: [N] array of p(failure|x) = count(label=-1) / K
                p_invalid: [N] array of p(invalid|x) = count(label=0) / K
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        states = states.to(self.device)
        N = states.shape[0]
        K = self.config.num_mc_samples
        batch_size = self.config.mc_batch_size

        if verbose:
            refine_str = (
                f", refine t~U{list(refine_t_range)} max_attempts={refine_max_attempts}"
                if refine_invalids else ""
            )
            print(f"      [MC] N={N} states, K={K} MC samples, batch_size={batch_size}{refine_str}")

        # GPU-side accumulators
        success_counts = torch.zeros(N, dtype=torch.int32, device=self.device)
        failure_counts = torch.zeros(N, dtype=torch.int32, device=self.device)
        invalid_counts = torch.zeros(N, dtype=torch.int32, device=self.device)
        # Track how many invalids were resolved by refinement (per attempt)
        refined_to_success = torch.zeros(N, dtype=torch.int32, device=self.device)
        refined_to_failure = torch.zeros(N, dtype=torch.int32, device=self.device)
        per_attempt_resolved = [0] * refine_max_attempts if refine_invalids else []

        n_batches = (N + batch_size - 1) // batch_size
        total_steps = n_batches * K

        with tqdm(total=total_steps, desc="MC estimation", disable=not verbose) as pbar:
            for batch_start in range(0, N, batch_size):
                batch_end = min(batch_start + batch_size, N)
                batch_states = states[batch_start:batch_end]

                for _ in range(K):
                    endpoints = self.flow_matcher.predict_endpoint(batch_states)
                    labels = self.system.classify_attractor(
                        endpoints, radius=self.config.attractor_radius
                    )

                    # Iteratively refine invalid endpoints
                    if refine_invalids:
                        # Track which original indices are still invalid
                        still_invalid = (labels == 0)
                        current_endpoints = endpoints.clone()

                        for _attempt in range(refine_max_attempts):
                            if not still_invalid.any():
                                break

                            refined = self.flow_matcher.refine_endpoints(
                                invalid_endpoints=current_endpoints[still_invalid],
                                start_states=batch_states[still_invalid],
                                t_range=refine_t_range,
                                num_steps=refine_num_steps,
                            )
                            refined_labels = self.system.classify_attractor(
                                refined, radius=self.config.attractor_radius
                            )

                            # Identify which ones resolved this attempt
                            resolved_success = (refined_labels == 1)
                            resolved_failure = (refined_labels == -1)
                            resolved = resolved_success | resolved_failure
                            per_attempt_resolved[_attempt] += int(resolved.sum().item())

                            # Map back to batch-level indices
                            still_invalid_indices = still_invalid.nonzero(as_tuple=True)[0]
                            for resolved_mask, counter in [
                                (resolved_success, refined_to_success),
                                (resolved_failure, refined_to_failure),
                            ]:
                                if resolved_mask.any():
                                    counter[batch_start + still_invalid_indices[resolved_mask]] += 1

                            # Update labels for resolved endpoints
                            labels[still_invalid_indices[resolved_success]] = 1
                            labels[still_invalid_indices[resolved_failure]] = -1

                            # Update current_endpoints for next attempt (still-invalid get refined output)
                            current_endpoints[still_invalid] = refined

                            # Narrow still_invalid to only the ones that remain label=0
                            still_invalid_remaining = (refined_labels == 0)
                            new_still_invalid = torch.zeros_like(still_invalid)
                            new_still_invalid[still_invalid_indices[still_invalid_remaining]] = True
                            still_invalid = new_still_invalid

                    success_counts[batch_start:batch_end] += (labels == 1).int()
                    failure_counts[batch_start:batch_end] += (labels == -1).int()
                    invalid_counts[batch_start:batch_end] += (labels == 0).int()

                    pbar.update(1)

        # Transfer to CPU and convert to probabilities
        p_success = success_counts.cpu().numpy().astype(np.float64) / K
        p_failure = failure_counts.cpu().numpy().astype(np.float64) / K
        p_invalid = invalid_counts.cpu().numpy().astype(np.float64) / K

        if refine_invalids and verbose:
            total_refined_success = refined_to_success.sum().item()
            total_refined_failure = refined_to_failure.sum().item()
            total_refined = total_refined_success + total_refined_failure
            total_original_invalid = total_refined + invalid_counts.sum().item()
            if total_original_invalid > 0:
                resolve_rate = total_refined / total_original_invalid * 100
                print(f"      [Refine] {total_refined}/{total_original_invalid} invalid samples resolved ({resolve_rate:.1f}%)")
                print(f"               → success: {total_refined_success}, → failure: {total_refined_failure}")
                attempt_strs = [f"a{i+1}={c}" for i, c in enumerate(per_attempt_resolved) if c > 0]
                if attempt_strs:
                    print(f"               per-attempt: {', '.join(attempt_strs)}")

        return p_success, p_failure, p_invalid

    @torch.no_grad()
    def estimate_single(self, state: Union[torch.Tensor, np.ndarray]) -> Tuple[float, float, float]:
        """
        Estimate probabilities for a single state.

        Args:
            state: Single initial state [state_dim] or [1, state_dim]

        Returns:
            Tuple of (p_success, p_failure, p_invalid) as floats
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        p_success, p_failure, p_invalid = self.estimate(state, verbose=False)
        return float(p_success[0]), float(p_failure[0]), float(p_invalid[0])

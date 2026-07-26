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
from adaptive_roa.conformal.refinement import RefinementStats, refine_invalid_endpoints


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
        refine_invalids: bool | None = None,
        refine_t_range: Tuple[float, float] | None = None,
        refine_num_steps: int | None = None,
        refine_max_attempts: int | None = None,
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
        # Resolve refinement params: explicit overrides > config defaults
        if refine_invalids is None:
            refine_invalids = self.config.refine_invalids
        if refine_t_range is None:
            refine_t_range = (self.config.refine_t_min, self.config.refine_t_max)
        if refine_num_steps is None:
            refine_num_steps = self.config.refine_num_steps
        if refine_max_attempts is None:
            refine_max_attempts = self.config.refine_max_attempts

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
        cumulative_rstats = RefinementStats(
            per_attempt_resolved=[0] * refine_max_attempts
        ) if refine_invalids else None

        n_batches = (N + batch_size - 1) // batch_size
        total_steps = n_batches * K

        # Check if trajectory checking is available and enabled
        use_trajectory_checking = (
            self.config.trajectory_checking
            and hasattr(self.flow_matcher, 'predict_trajectory')
            and hasattr(self.flow_matcher, 'classify_trajectory')
        )
        if use_trajectory_checking and verbose:
            print(f"      [MC] Using trajectory checking (per-timestep reachability)")

        with tqdm(total=total_steps, desc="MC estimation", disable=not verbose) as pbar:
            for batch_start in range(0, N, batch_size):
                batch_end = min(batch_start + batch_size, N)
                batch_states = states[batch_start:batch_end]

                for _ in range(K):
                    if use_trajectory_checking:
                        # Generate full trajectory, classify at every timestep
                        trajectory = self.flow_matcher.predict_trajectory(batch_states)
                        labels = self.flow_matcher.classify_trajectory(
                            trajectory, attractor_radius=self.config.attractor_radius
                        )
                    else:
                        # Endpoint-only: predict endpoint, classify once
                        endpoints = self.flow_matcher.predict_endpoint(batch_states)
                        labels = self.system.classify_attractor(
                            endpoints, radius=self.config.attractor_radius
                        )

                        if refine_invalids:
                            rstats = refine_invalid_endpoints(
                                self.flow_matcher, self.system,
                                endpoints, labels, batch_states,
                                attractor_radius=self.config.attractor_radius,
                                t_range=refine_t_range,
                                num_steps=refine_num_steps,
                                max_attempts=refine_max_attempts,
                            )
                            cumulative_rstats.accumulate(rstats)

                    success_counts[batch_start:batch_end] += (labels == 1).int()
                    failure_counts[batch_start:batch_end] += (labels == -1).int()
                    invalid_counts[batch_start:batch_end] += (labels == 0).int()

                    pbar.update(1)

        # Transfer to CPU and convert to probabilities
        p_success = success_counts.cpu().numpy().astype(np.float64) / K
        p_failure = failure_counts.cpu().numpy().astype(np.float64) / K
        p_invalid = invalid_counts.cpu().numpy().astype(np.float64) / K

        if refine_invalids and verbose and cumulative_rstats.n_initially_invalid > 0:
            total_original = cumulative_rstats.n_initially_invalid
            total_resolved = cumulative_rstats.n_resolved
            resolve_rate = total_resolved / total_original * 100
            print(f"      [Refine] {total_resolved}/{total_original} invalid samples resolved ({resolve_rate:.1f}%)")
            print(f"               → success: {cumulative_rstats.n_resolved_success}, → failure: {cumulative_rstats.n_resolved_failure}")
            attempt_strs = [f"a{i+1}={c}" for i, c in enumerate(cumulative_rstats.per_attempt_resolved) if c > 0]
            if attempt_strs:
                print(f"               per-attempt: {', '.join(attempt_strs)}")

        return p_success, p_failure, p_invalid

    @torch.no_grad()
    def sample_endpoints(
        self,
        states: Union[torch.Tensor, np.ndarray],
        num_samples: int,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Sample raw endpoint clouds without classifying them.

        For each state, runs num_samples forward passes (each with a fresh
        z ~ N(0,I)) and returns the predicted endpoints as-is. Unlike
        estimate(), this performs no attractor classification and no
        refinement of invalid endpoints, so it never consults the success
        criteria. It is the input to distribution-based uncertainty scoring.

        Args:
            states: Initial states [N, state_dim] as torch tensor or numpy array
            num_samples: Number of endpoint samples K per state; overrides
                config.num_mc_samples
            verbose: Show a progress bar

        Returns:
            np.ndarray: Endpoint clouds [N, K, endpoint_dim] as float32
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        states = states.to(self.device)
        N = states.shape[0]
        K = int(num_samples)
        if K < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        if N == 0:
            return np.empty((0, K, 0), dtype=np.float32)

        batch_size = self.config.mc_batch_size

        if verbose:
            print(f"      [Cloud] N={N} states, K={K} samples, batch_size={batch_size}")

        n_batches = (N + batch_size - 1) // batch_size
        out: torch.Tensor | None = None

        with tqdm(total=n_batches * K, desc="Endpoint sampling", disable=not verbose) as pbar:
            for batch_start in range(0, N, batch_size):
                batch_end = min(batch_start + batch_size, N)
                batch_states = states[batch_start:batch_end]

                for k in range(K):
                    endpoints = self.flow_matcher.predict_endpoint(batch_states)
                    if out is None:
                        out = torch.empty(
                            (N, K, endpoints.shape[1]), dtype=torch.float32
                        )
                    out[batch_start:batch_end, k, :] = endpoints.detach().float().cpu()
                    pbar.update(1)

        return out.numpy()

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

"""
Probability Estimator for Conformal Prediction.

Uses Monte Carlo sampling with the flow matcher's latent variable
to estimate p(success|x) for each initial state.
"""
import torch
import numpy as np
from typing import Union, Tuple
from src.conformal.config import ConformalConfig


class ProbabilityEstimator:
    """
    Estimate success probability via Monte Carlo sampling.

    For each initial state x, samples K different latent vectors z,
    predicts K endpoints using the flow matcher, classifies each endpoint,
    and computes empirical success rate as p(success|x).

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
        """
        Initialize probability estimator.

        Args:
            flow_matcher: Trained flow matcher (any system). Must have predict_endpoint().
            system: Dynamical system instance. Must have classify_attractor().
            config: ConformalConfig with MC sampling parameters.
            device: Device for computation.
        """
        self.flow_matcher = flow_matcher
        self.system = system
        self.config = config
        self.device = device

        # Put flow matcher in eval mode
        self.flow_matcher.eval()

    @torch.no_grad()
    def estimate(
        self,
        states: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate p(success|x) for a batch of states.

        For each state, samples num_mc_samples latent vectors, predicts endpoints,
        classifies them, and computes empirical probabilities.

        Args:
            states: Initial states [N, state_dim] as torch tensor or numpy array

        Returns:
            Tuple of:
                p_success: [N] array of p(success|x) = count(label=1) / K
                p_failure: [N] array of p(failure|x) = count(label=-1) / K
                p_unknown: [N] array of p(separatrix|x) = count(label=0) / K
        """
        # Convert to torch tensor if needed
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        states = states.to(self.device)
        N = states.shape[0]
        K = self.config.num_mc_samples

        # Accumulators for each class
        success_counts = np.zeros(N)
        failure_counts = np.zeros(N)
        separatrix_counts = np.zeros(N)

        # Process in batches for memory efficiency
        # We expand each state K times, so effective batch is N * K
        # Process N states at a time, each with K samples
        batch_size = max(1, self.config.mc_batch_size // K)

        # Debug: verify MC sampling parameters
        print(f"      [MC Debug] N={N} states, K={K} samples each, batch_size={batch_size}")
        print(f"      [MC Debug] Total forward passes: {N * K} endpoint predictions")

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_states = states[batch_start:batch_end]  # [B, state_dim]
            B = batch_states.shape[0]

            # Expand states: each state repeated K times
            # [B, state_dim] -> [B*K, state_dim]
            expanded_states = batch_states.unsqueeze(1).expand(-1, K, -1)
            expanded_states = expanded_states.reshape(B * K, -1)

            # Debug: verify expansion (only on first batch)
            if batch_start == 0:
                print(f"      [MC Debug] First batch: B={B}, expanded to {expanded_states.shape[0]} inputs")

            # Predict endpoints (flow matcher samples new z internally each call)
            # The flow matcher's predict_endpoint samples z ~ N(0,I) internally
            endpoints = self.flow_matcher.predict_endpoint(expanded_states)

            # Debug: print endpoint stats on first batch (system-agnostic)
            if batch_start == 0:
                state_dim = endpoints.shape[1]
                print(f"      [MC Debug] Endpoint stats (first batch, dim={state_dim}):")
                if state_dim == 2:
                    # Pendulum: (θ, θ̇)
                    print(f"         θ:     min={endpoints[:, 0].min():.4f}, max={endpoints[:, 0].max():.4f}, mean={endpoints[:, 0].mean():.4f}")
                    print(f"         θ̇:     min={endpoints[:, 1].min():.4f}, max={endpoints[:, 1].max():.4f}, mean={endpoints[:, 1].mean():.4f}")
                elif state_dim == 4:
                    # CartPole: (x, θ, ẋ, θ̇)
                    print(f"         x:     min={endpoints[:, 0].min():.4f}, max={endpoints[:, 0].max():.4f}, mean={endpoints[:, 0].mean():.4f}")
                    print(f"         θ:     min={endpoints[:, 1].min():.4f}, max={endpoints[:, 1].max():.4f}, mean={endpoints[:, 1].mean():.4f}")
                    print(f"         ẋ:     min={endpoints[:, 2].min():.4f}, max={endpoints[:, 2].max():.4f}, mean={endpoints[:, 2].mean():.4f}")
                    print(f"         θ̇:     min={endpoints[:, 3].min():.4f}, max={endpoints[:, 3].max():.4f}, mean={endpoints[:, 3].mean():.4f}")
                else:
                    # Generic: print all dims
                    for d in range(state_dim):
                        print(f"         dim{d}: min={endpoints[:, d].min():.4f}, max={endpoints[:, d].max():.4f}, mean={endpoints[:, d].mean():.4f}")
                print(f"      [MC Debug] Attractor radius: {self.config.attractor_radius}")

            # Classify endpoints
            labels = self.system.classify_attractor(
                endpoints,
                radius=self.config.attractor_radius
            )

            # Debug: print classification stats on first batch
            if batch_start == 0:
                n_success = (labels == 1).sum().item()
                n_failure = (labels == -1).sum().item()
                n_separatrix = (labels == 0).sum().item()
                print(f"      [MC Debug] First batch classification: {n_success} success, {n_failure} failure, {n_separatrix} separatrix")

            # Reshape labels back to [B, K]
            labels = labels.reshape(B, K)

            # Count successes, failures, separatrix for each state
            for i in range(B):
                state_labels = labels[i].cpu().numpy()
                success_counts[batch_start + i] = np.sum(state_labels == 1)
                failure_counts[batch_start + i] = np.sum(state_labels == -1)
                separatrix_counts[batch_start + i] = np.sum(state_labels == 0)

        # Convert to probabilities
        p_success = success_counts / K
        p_failure = failure_counts / K
        p_separatrix = separatrix_counts / K

        return p_success, p_failure, p_separatrix

    @torch.no_grad()
    def estimate_single(self, state: Union[torch.Tensor, np.ndarray]) -> Tuple[float, float, float]:
        """
        Estimate probabilities for a single state.

        Args:
            state: Single initial state [state_dim] or [1, state_dim]

        Returns:
            Tuple of (p_success, p_failure, p_separatrix) as floats
        """
        # Convert to torch tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)

        p_success, p_failure, p_separatrix = self.estimate(state)
        return float(p_success[0]), float(p_failure[0]), float(p_separatrix[0])

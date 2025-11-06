"""Mountain Car Inference Wrapper

High-level inference interface for Mountain Car flow matching model.
"""

import torch
import numpy as np
from typing import Optional, Union, List, Tuple
from pathlib import Path

from src.flow_matching.mountain_car.latent_conditional.flow_matcher import (
    MountainCarLatentConditionalFlowMatcher
)


class MountainCarLatentConditionalInference:
    """Inference wrapper for Mountain Car latent conditional flow matching.

    Provides high-level methods for endpoint prediction, sampling, and
    attractor analysis.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        flow_matcher: Optional[MountainCarLatentConditionalFlowMatcher] = None,
        device: Optional[str] = None
    ):
        """Initialize inference wrapper.

        Args:
            checkpoint_path: Path to checkpoint or training directory
            flow_matcher: Pre-loaded flow matcher (alternative to checkpoint_path)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        if flow_matcher is None and checkpoint_path is None:
            raise ValueError("Must provide either checkpoint_path or flow_matcher")

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load or use provided flow matcher
        if flow_matcher is not None:
            self.model = flow_matcher
        else:
            self.model = MountainCarLatentConditionalFlowMatcher.load_from_checkpoint(
                checkpoint_path,
                map_location=self.device
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Cache system reference
        self.system = self.model.system

        print(f"✅ MountainCar inference ready on {self.device}")

    @torch.no_grad()
    def predict_endpoint(
        self,
        start_states: Union[torch.Tensor, np.ndarray],
        num_steps: int = 100,
        method: str = "dopri5"
    ) -> torch.Tensor:
        """Predict endpoint from start states.

        Args:
            start_states: Start states [B, 2] or [2]
            num_steps: Number of integration steps
            method: ODE solver method

        Returns:
            Predicted endpoints [B, 2]

        Note: NO latent variables - deterministic prediction given start state
        """
        # Convert to tensor if needed
        if isinstance(start_states, np.ndarray):
            start_states = torch.from_numpy(start_states).float()

        # Add batch dimension if single state
        if start_states.ndim == 1:
            start_states = start_states.unsqueeze(0)

        # Move to device
        start_states = start_states.to(self.device)

        # Predict endpoint (no latent needed)
        endpoints = self.model.predict_endpoint(
            start_states=start_states,
            num_steps=num_steps,
            method=method
        )

        return endpoints

    @torch.no_grad()
    def sample_endpoints(
        self,
        start_states: Union[torch.Tensor, np.ndarray],
        num_samples: int = 20,
        num_steps: int = 100,
        method: str = "dopri5",
        return_std: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample multiple endpoints.

        Note: Since there are NO latent variables, all samples will be identical.
        This is a deterministic model conditioned on start state only.

        Args:
            start_states: Start states [B, 2]
            num_samples: Number of samples per start state (will be identical)
            num_steps: Integration steps
            method: ODE solver method
            return_std: Whether to return standard deviation (will be zero)

        Returns:
            If return_std=False: Sampled endpoints [B, num_samples, 2] (all identical)
            If return_std=True: (mean [B, 2], std [B, 2] = 0)
        """
        # Convert to tensor if needed
        if isinstance(start_states, np.ndarray):
            start_states = torch.from_numpy(start_states).float()

        if start_states.ndim == 1:
            start_states = start_states.unsqueeze(0)

        start_states = start_states.to(self.device)
        batch_size = start_states.shape[0]

        # Predict endpoint once (deterministic)
        endpoints = self.model.predict_endpoint(
            start_states=start_states,
            num_steps=num_steps,
            method=method
        )  # [B, 2]

        # Repeat to match num_samples
        endpoints_repeated = endpoints.unsqueeze(1).repeat(1, num_samples, 1)  # [B, num_samples, 2]

        if return_std:
            mean = endpoints  # [B, 2]
            std = torch.zeros_like(mean)  # [B, 2] - all zeros (deterministic)
            return mean, std
        else:
            return endpoints_repeated

    @torch.no_grad()
    def predict_attractor_distribution(
        self,
        start_states: Union[torch.Tensor, np.ndarray],
        num_samples: int = 20,
        num_steps: int = 100,
        threshold: Optional[float] = None
    ) -> dict:
        """Predict probability distribution over attractors.

        Args:
            start_states: Start states [B, 2]
            num_samples: Number of samples for probabilistic prediction
            num_steps: Integration steps
            threshold: Distance threshold for attractor (default: system default)

        Returns:
            Dictionary with:
                - 'probabilities': [B, num_attractors] attractor probabilities
                - 'predictions': [B] most likely attractor index
                - 'entropy': [B] prediction entropy (uncertainty)
                - 'endpoints': [B, num_samples, 2] sampled endpoints
        """
        # Sample endpoints
        endpoints_samples = self.sample_endpoints(
            start_states, num_samples, num_steps
        )  # [B, num_samples, 2]

        batch_size = endpoints_samples.shape[0]

        # Check which attractor each sample reached
        # Reshape to [B*num_samples, 2] for batch processing
        endpoints_flat = endpoints_samples.reshape(-1, endpoints_samples.shape[-1])

        # Classify attractors
        attractor_classes = self.system.classify_attractor(
            endpoints_flat, threshold=threshold
        )  # [B*num_samples]

        # Reshape to [B, num_samples]
        attractor_classes = attractor_classes.reshape(batch_size, num_samples)

        # Compute probabilities (only 1 attractor for Mountain Car)
        # Success = attractor 0 (goal), Failure = -1
        success_count = (attractor_classes == 0).sum(dim=1).float()
        prob_success = success_count / num_samples

        probabilities = torch.stack([prob_success, 1 - prob_success], dim=1)  # [B, 2]

        # Most likely attractor
        predictions = (prob_success > 0.5).long()  # 0 if success, else failure (-1 mapped to 1)

        # Entropy (uncertainty measure)
        eps = 1e-10
        entropy = -(probabilities * torch.log(probabilities + eps)).sum(dim=1)

        return {
            'probabilities': probabilities,  # [success, failure]
            'predictions': predictions,
            'entropy': entropy,
            'endpoints': endpoints_samples
        }

    @torch.no_grad()
    def predict_single(
        self,
        state: Union[torch.Tensor, np.ndarray, List[float]],
        num_samples: int = 1
    ) -> dict:
        """Predict endpoint for a single state (convenience method).

        Args:
            state: Single state [2] (position, velocity)
            num_samples: Number of samples

        Returns:
            Dictionary with predictions and statistics
        """
        # Convert to tensor
        if isinstance(state, (list, tuple)):
            state = np.array(state)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        state = state.unsqueeze(0)  # [1, 2]

        if num_samples == 1:
            endpoint = self.predict_endpoint(state)
            return {
                'endpoint': endpoint[0].cpu().numpy(),
                'start_state': state[0].cpu().numpy()
            }
        else:
            endpoints = self.sample_endpoints(state, num_samples)  # [1, num_samples, 2]
            mean = endpoints.mean(dim=1)[0]
            std = endpoints.std(dim=1)[0]

            return {
                'mean_endpoint': mean.cpu().numpy(),
                'std_endpoint': std.cpu().numpy(),
                'samples': endpoints[0].cpu().numpy(),
                'start_state': state[0].cpu().numpy()
            }

    @torch.no_grad()
    def batch_predict(
        self,
        start_states_list: List[Union[torch.Tensor, np.ndarray]],
        return_std: bool = False,
        num_samples: int = 1
    ) -> List[dict]:
        """Batch prediction for list of states.

        Args:
            start_states_list: List of start states
            return_std: Whether to return uncertainty estimates
            num_samples: Number of samples per state (if > 1)

        Returns:
            List of prediction dictionaries
        """
        results = []

        for state in start_states_list:
            if num_samples == 1:
                result = self.predict_single(state, num_samples=1)
            else:
                result = self.predict_single(state, num_samples=num_samples)

            results.append(result)

        return results

    def get_system_info(self) -> dict:
        """Get system configuration information.

        Returns:
            Dictionary with system parameters
        """
        return {
            'system_name': 'MountainCar',
            'state_dim': self.system.state_dim,
            'embedded_dim': self.system.embedded_dim,
            'manifold': 'ℝ²',
            'position_limit': self.system.position_limit,
            'velocity_limit': self.system.velocity_limit,
            'goal_position': self.system.goal_position,
            'goal_tolerance': self.system.goal_tolerance,
            'latent_variables': False,
            'device': str(self.device)
        }

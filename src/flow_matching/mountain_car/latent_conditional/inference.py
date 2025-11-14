"""
Inference wrapper for Mountain Car Latent Conditional Flow Matching (Facebook FM)

Provides interface compatible with evaluation tools and basin analysis.
"""
import torch
import numpy as np
from typing import Optional, Tuple, Union, List
from pathlib import Path


class MountainCarLatentConditionalInference:
    """
    Inference wrapper for Mountain Car Latent Conditional Flow Matching

    Provides:
    - Standard predict_endpoint() interface for RoA analysis
    - Probabilistic methods for uncertainty estimation
    - Multi-sample endpoint prediction
    - Attractor distribution estimation
    """

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 flow_matcher = None,
                 num_integration_steps: int = 100,
                 integration_method: str = "rk4"):
        """
        Initialize inference wrapper

        Args:
            checkpoint_path: Path to trained checkpoint (if loading from file)
            flow_matcher: Pre-loaded flow matcher (if already in memory)
            num_integration_steps: Number of ODE integration steps
            integration_method: Integration method ("euler", "rk4", "midpoint")
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_integration_steps = num_integration_steps
        self.integration_method = integration_method

        if flow_matcher is not None:
            # Use provided flow matcher
            self.model = flow_matcher
            self.model.eval()
            self.model = self.model.to(self.device)
        elif checkpoint_path is not None:
            # Load from checkpoint using the class method
            from src.flow_matching.mountain_car.latent_conditional.flow_matcher import MountainCarLatentConditionalFlowMatcher
            self.model = MountainCarLatentConditionalFlowMatcher.load_from_checkpoint(
                checkpoint_path, device=str(self.device)
            )
        else:
            raise ValueError("Must provide either checkpoint_path or flow_matcher")

        # Extract configuration
        self.latent_dim = self.model.latent_dim
        self.system = self.model.system

    @torch.no_grad()
    def predict_endpoint(self,
                        start_states: Union[torch.Tensor, np.ndarray],
                        num_steps: Optional[int] = None,
                        latent: Optional[torch.Tensor] = None,
                        method: Optional[str] = None) -> torch.Tensor:
        """
        Predict endpoint for given start state(s)

        Compatible with AttractorBasinAnalyzer interface.

        Args:
            start_states: Initial states [batch_size, 2] or [2] for single prediction
                         Format: [position, velocity] in original coordinates
            num_steps: Number of integration steps (uses default if None)
            latent: Optional latent vectors [batch_size, latent_dim].
                    If None, samples random latent.
            method: Integration method (uses default if None)

        Returns:
            predicted_endpoints: Final states [batch_size, 2] or [2]
        """
        # Convert numpy to torch if needed
        if isinstance(start_states, np.ndarray):
            start_states = torch.tensor(start_states, dtype=torch.float32)

        # Move to device
        start_states = start_states.to(self.device)

        # Use defaults if not specified
        if num_steps is None:
            num_steps = self.num_integration_steps
        if method is None:
            method = self.integration_method

        # Call flow matcher's predict_endpoint
        predictions = self.model.predict_endpoint(
            start_states,
            num_steps=num_steps,
            latent=latent,
            method=method
        )

        return predictions

    @torch.no_grad()
    def sample_endpoints(self,
                        start_states: Union[torch.Tensor, np.ndarray],
                        num_samples: int = 10,
                        num_steps: Optional[int] = None,
                        method: Optional[str] = None) -> torch.Tensor:
        """
        Sample multiple endpoints per start state (stochastic due to latent)

        Args:
            start_states: Initial states [batch_size, 2]
            num_samples: Number of samples per start state
            num_steps: Number of integration steps
            method: Integration method

        Returns:
            samples: Sampled endpoints [batch_size, num_samples, 2]
        """
        # Convert numpy to torch if needed
        if isinstance(start_states, np.ndarray):
            start_states = torch.tensor(start_states, dtype=torch.float32)

        start_states = start_states.to(self.device)
        batch_size = start_states.shape[0]

        # Use defaults
        if num_steps is None:
            num_steps = self.num_integration_steps
        if method is None:
            method = self.integration_method

        # Collect samples
        samples = []
        for _ in range(num_samples):
            # Each call samples a new random latent
            endpoints = self.model.predict_endpoint(
                start_states,
                num_steps=num_steps,
                latent=None,
                method=method
            )
            samples.append(endpoints)

        # Stack: [num_samples, batch_size, 2] → [batch_size, num_samples, 2]
        samples = torch.stack(samples, dim=0).permute(1, 0, 2)

        return samples

    def get_system_info(self) -> dict:
        """Get information about the dynamical system"""
        return {
            "system_type": "MountainCar",
            "state_dim": 2,
            "manifold": "ℝ²",
            "position_limit": self.system.position_limit,
            "velocity_limit": self.system.velocity_limit,
            "goal_center": self.system.goal_center
        }

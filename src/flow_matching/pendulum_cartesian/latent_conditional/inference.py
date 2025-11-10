"""
Inference wrapper for Pendulum Cartesian Latent Conditional Flow Matching (Facebook FM)

This wrapper provides an interface compatible with the AttractorBasinAnalyzer
and other evaluation tools while leveraging the flow matcher's inference methods.
"""
import torch
import numpy as np
from typing import Optional, Tuple, Union, List
from pathlib import Path


class PendulumCartesianLatentConditionalInference:
    """
    Inference wrapper for Pendulum Cartesian Latent Conditional Flow Matching

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
            # Load from checkpoint
            self.model = self._load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError("Must provide either checkpoint_path or flow_matcher")

        # Extract configuration
        self.latent_dim = self.model.latent_dim
        self.system = self.model.system

    def _load_from_checkpoint(self, checkpoint_path: str):
        """
        Load flow matcher from checkpoint using the class method

        Args:
            checkpoint_path: Path to checkpoint file or folder

        Returns:
            Loaded flow matcher
        """
        from src.flow_matching.pendulum_cartesian.latent_conditional.flow_matcher import PendulumCartesianLatentConditionalFlowMatcher

        return PendulumCartesianLatentConditionalFlowMatcher.load_from_checkpoint(
            checkpoint_path,
            device=str(self.device)
        )

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
            start_states: Initial states [batch_size, 4] or [4] for single prediction
                         Format: [x, y, ẋ, ẏ] in original coordinates
            num_steps: Number of integration steps (uses default if None)
            latent: Optional latent vectors [batch_size, latent_dim].
                    If None, samples random latent.
            method: Integration method (uses default if None)

        Returns:
            predicted_endpoints: Final states [batch_size, 4] or [4]
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
            start_states: Initial states [batch_size, 4]
            num_samples: Number of samples per start state
            num_steps: Number of integration steps
            method: Integration method

        Returns:
            samples: Sampled endpoints [batch_size, num_samples, 4]
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
                latent=None,  # Sample random latent
                method=method
            )
            samples.append(endpoints)

        # Stack: [num_samples, batch_size, 4] -> [batch_size, num_samples, 4]
        samples = torch.stack(samples, dim=0).transpose(0, 1)

        return samples

    @torch.no_grad()
    def predict_attractor_distribution(self,
                                      start_states: Union[torch.Tensor, np.ndarray],
                                      num_samples: int = 64,
                                      attractor_centers: Optional[torch.Tensor] = None,
                                      attractor_radius: float = 0.15) -> torch.Tensor:
        """
        Estimate probability distribution over attractors for each start state

        This method enables probabilistic RoA analysis by sampling multiple
        endpoints and computing which attractors they converge to.

        Args:
            start_states: Initial states [batch_size, 4]
            num_samples: Number of samples for Monte Carlo estimation
            attractor_centers: Attractor positions [num_attractors, 4]
                              If None, uses system config
            attractor_radius: Radius for attractor membership

        Returns:
            probabilities: Probability distribution [batch_size, num_attractors]
        """
        # Convert numpy to torch if needed
        if isinstance(start_states, np.ndarray):
            start_states = torch.tensor(start_states, dtype=torch.float32)

        start_states = start_states.to(self.device)
        batch_size = start_states.shape[0]

        # Get attractor centers
        if attractor_centers is None:
            # Use system attractors
            attractor_centers = torch.tensor(
                self.system.attractors(),
                dtype=torch.float32,
                device=self.device
            )
        else:
            attractor_centers = attractor_centers.to(self.device)

        num_attractors = attractor_centers.shape[0]

        # Sample endpoints
        samples = self.sample_endpoints(start_states, num_samples=num_samples)  # [B, N, 4]

        # Count how many samples fall into each attractor
        attractor_counts = torch.zeros(
            batch_size, num_attractors,
            device=self.device, dtype=torch.float32
        )

        # For each sample, check which attractor it belongs to
        for sample_idx in range(num_samples):
            endpoints = samples[:, sample_idx, :]  # [B, 4]

            # Compute distances to each attractor (all Euclidean)
            for att_idx in range(num_attractors):
                center = attractor_centers[att_idx]  # [4]

                # Euclidean distance in ℝ⁴
                distances = torch.sqrt(((endpoints - center) ** 2).sum(dim=1))

                # Count if within radius
                in_attractor = (distances < attractor_radius).float()
                attractor_counts[:, att_idx] += in_attractor

        # Normalize to get probabilities
        probabilities = attractor_counts / num_samples

        return probabilities

    def predict_single(self,
                      x: float,
                      y: float,
                      x_dot: float,
                      y_dot: float,
                      num_samples: int = 1) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Convenient method for single prediction

        Args:
            x: Initial x position (float)
            y: Initial y position (float)
            x_dot: Initial x velocity (float)
            y_dot: Initial y velocity (float)
            num_samples: Number of samples (if >1, returns list)

        Returns:
            endpoint(s): Predicted endpoint(s) as numpy array(s)
        """
        start_state = torch.tensor([[x, y, x_dot, y_dot]], dtype=torch.float32)

        if num_samples == 1:
            endpoint = self.predict_endpoint(start_state)
            return endpoint.cpu().numpy()[0]
        else:
            samples = self.sample_endpoints(start_state, num_samples=num_samples)
            return [samples[0, i].cpu().numpy() for i in range(num_samples)]

    def batch_predict(self,
                     start_states_list: List[List[float]],
                     return_std: bool = False,
                     num_samples: int = 10) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict endpoints for a batch of start states with uncertainty

        Args:
            start_states_list: List of [x, y, ẋ, ẏ] lists
            return_std: If True, return standard deviations
            num_samples: Number of samples for uncertainty estimation

        Returns:
            predictions: Mean predicted endpoints [n_samples, 4]
            stds (optional): Standard deviations [n_samples, 4]
        """
        start_tensor = torch.tensor(start_states_list, dtype=torch.float32)

        if return_std:
            samples = self.sample_endpoints(start_tensor, num_samples=num_samples)
            samples_np = samples.cpu().numpy()

            means = samples_np.mean(axis=1)
            stds = samples_np.std(axis=1)

            return means, stds
        else:
            predictions = self.predict_endpoint(start_tensor)
            return predictions.cpu().numpy()

    def get_system_info(self) -> dict:
        """Get system information"""
        return {
            "system_name": "Pendulum Cartesian",
            "state_dim": 4,
            "manifold": "ℝ⁴ (all Euclidean)",
            "latent_dim": self.latent_dim,
            "integration_steps": self.num_integration_steps,
            "integration_method": self.integration_method,
            "device": str(self.device)
        }

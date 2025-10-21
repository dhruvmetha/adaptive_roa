"""
Inference wrapper for CartPole Gaussian-Perturbed Flow Matching

Provides convenient interface for loading trained models and making predictions.
"""
import torch
from pathlib import Path
from typing import Optional, Union

from src.flow_matching.cartpole_gaussian_perturbed.flow_matcher import CartPoleGaussianPerturbedFlowMatcher


class CartPoleGaussianPerturbedInference:
    """
    Inference wrapper for Gaussian-Perturbed Flow Matching

    Simplified inference WITHOUT latent variables or conditioning.
    Stochasticity comes from Gaussian perturbations around start states.

    Usage:
        # Load from training folder
        inferencer = CartPoleGaussianPerturbedInference(
            "outputs/cartpole_gaussian_perturbed_fm/2025-10-17_12-30-45"
        )

        # Single prediction
        start_states = torch.tensor([[0.5, 0.1, 2.0, 1.0]])  # [x, θ, ẋ, θ̇]
        endpoint = inferencer.predict_endpoint(start_states)

        # Multiple samples for uncertainty quantification
        endpoints = inferencer.predict_endpoints_batch(start_states, num_samples=20)
    """

    def __init__(self, checkpoint_path: Union[str, Path], device: Optional[str] = None):
        """
        Initialize inference wrapper

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt) or training folder
            device: Device to run inference on ("cuda", "cpu", or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model from checkpoint
        self.model = CartPoleGaussianPerturbedFlowMatcher.load_from_checkpoint(
            checkpoint_path,
            device=self.device
        )

        self.model.eval()

        # Store system for reference
        self.system = self.model.system

        print(f"\n✅ Inference wrapper initialized")
        print(f"   Device: {self.device}")
        print(f"   Gaussian noise std: {self.model.noise_std}")
        print(f"   Ready for predictions!")

    def predict_endpoint(self,
                        start_states: torch.Tensor,
                        num_steps: int = 100,
                        method: str = "euler") -> torch.Tensor:
        """
        Predict endpoint from start state using Gaussian perturbation

        Args:
            start_states: Start states [B, 4] as (x, θ, ẋ, θ̇) in raw coordinates
            num_steps: Number of ODE integration steps
            method: Integration method ("euler", "rk4", "midpoint")

        Returns:
            Predicted endpoints [B, 4] in raw coordinates
        """
        # Convert to tensor if needed
        if not isinstance(start_states, torch.Tensor):
            start_states = torch.tensor(start_states, dtype=torch.float32)

        # Move to device
        start_states = start_states.to(self.device)

        # Predict using model
        with torch.no_grad():
            endpoints = self.model.predict_endpoint(
                start_states=start_states,
                num_steps=num_steps,
                method=method
            )

        return endpoints

    def predict_endpoints_batch(self,
                                start_states: torch.Tensor,
                                num_samples: int = 10,
                                num_steps: int = 100,
                                method: str = "euler") -> torch.Tensor:
        """
        Predict multiple endpoint samples per start state for uncertainty quantification

        Each sample uses a different Gaussian perturbation around the start state.

        Args:
            start_states: Start states [B, 4] in raw coordinates
            num_samples: Number of samples per start state
            num_steps: Number of ODE integration steps
            method: Integration method

        Returns:
            Predicted endpoints [B*num_samples, 4] in raw coordinates
        """
        # Convert to tensor if needed
        if not isinstance(start_states, torch.Tensor):
            start_states = torch.tensor(start_states, dtype=torch.float32)

        # Move to device
        start_states = start_states.to(self.device)

        # Predict using model
        with torch.no_grad():
            endpoints = self.model.predict_endpoints_batch(
                start_states=start_states,
                num_steps=num_steps,
                num_samples=num_samples
            )

        return endpoints

    def compute_uncertainty(self,
                           start_states: torch.Tensor,
                           num_samples: int = 20,
                           num_steps: int = 100) -> dict:
        """
        Compute uncertainty metrics for predictions

        Args:
            start_states: Start states [B, 4]
            num_samples: Number of samples for uncertainty estimation
            num_steps: Number of integration steps

        Returns:
            Dictionary with uncertainty metrics:
                - mean: Mean predicted endpoint [B, 4]
                - std: Standard deviation [B, 4]
                - samples: All samples [B, num_samples, 4]
        """
        # Get batch predictions
        endpoints = self.predict_endpoints_batch(
            start_states=start_states,
            num_samples=num_samples,
            num_steps=num_steps
        )

        # Reshape to [B, num_samples, 4]
        batch_size = start_states.shape[0]
        endpoints_reshaped = endpoints.reshape(batch_size, num_samples, -1)

        # Compute statistics
        mean = endpoints_reshaped.mean(dim=1)  # [B, 4]
        std = endpoints_reshaped.std(dim=1)    # [B, 4]

        return {
            'mean': mean,
            'std': std,
            'samples': endpoints_reshaped,
            'num_samples': num_samples
        }

    def check_attractor_convergence(self,
                                    start_states: torch.Tensor,
                                    num_samples: int = 20,
                                    attractor_radius: float = 0.3) -> dict:
        """
        Check what proportion of samples converge to attractor

        Useful for probabilistic ROA evaluation.

        Args:
            start_states: Start states [B, 4]
            num_samples: Number of samples per state
            attractor_radius: Radius for attractor membership

        Returns:
            Dictionary with convergence statistics:
                - proportion_success: Proportion reaching attractor [B]
                - all_predictions: Binary predictions for all samples [B, num_samples]
                - consensus_prediction: Majority vote [B]
        """
        # Get batch predictions
        endpoints = self.predict_endpoints_batch(
            start_states=start_states,
            num_samples=num_samples
        )

        # Reshape to [B, num_samples, 4]
        batch_size = start_states.shape[0]
        endpoints_reshaped = endpoints.reshape(batch_size, num_samples, -1)

        # Check attractor membership for all samples
        all_in_attractor = []
        for i in range(num_samples):
            sample_endpoints = endpoints_reshaped[:, i, :]
            in_attractor = self.system.is_in_attractor(sample_endpoints, radius=attractor_radius)
            all_in_attractor.append(in_attractor)

        # Stack to [B, num_samples]
        all_predictions = torch.stack(all_in_attractor, dim=1)

        # Compute proportion that reach attractor
        proportion_success = all_predictions.float().mean(dim=1)  # [B]

        # Consensus prediction (majority vote with 60% threshold)
        consensus = torch.zeros(batch_size, dtype=torch.long)
        consensus[proportion_success >= 0.6] = 1   # Success
        consensus[proportion_success <= 0.4] = 0   # Failure
        # States with 0.4 < p < 0.6 remain uncertain (could mark as -1)

        return {
            'proportion_success': proportion_success,
            'all_predictions': all_predictions,
            'consensus_prediction': consensus,
            'num_samples': num_samples
        }

    def __repr__(self):
        return (f"CartPoleGaussianPerturbedInference("
                f"device={self.device}, "
                f"noise_std={self.model.noise_std})")

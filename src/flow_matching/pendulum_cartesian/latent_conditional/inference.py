"""
Inference wrapper for Pendulum Cartesian Latent Conditional Flow Matching
"""
import torch
from pathlib import Path
from typing import Optional, Union
from src.flow_matching.pendulum_cartesian.latent_conditional.flow_matcher import PendulumCartesianLatentConditionalFlowMatcher


class PendulumCartesianInference:
    """
    Convenience wrapper for Pendulum Cartesian LCFM inference.
    Loads checkpoint and provides simple prediction interface.
    """

    def __init__(self, checkpoint_path: Union[str, Path], device: Optional[str] = None):
        """
        Initialize inference wrapper

        Args:
            checkpoint_path: Path to checkpoint file or training directory
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PendulumCartesianLatentConditionalFlowMatcher.load_from_checkpoint(
            checkpoint_path, device=self.device
        )
        self.model.eval()

    def predict_endpoint(self, start_states: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """
        Predict endpoint from start state

        Args:
            start_states: Start states [B, 4] as (x, y, vx, vy)
            num_steps: Number of ODE integration steps

        Returns:
            Predicted endpoints [B, 4]
        """
        if isinstance(start_states, list):
            start_states = torch.tensor(start_states, dtype=torch.float32)

        start_states = start_states.to(self.device)

        with torch.no_grad():
            endpoints = self.model.predict_endpoint(start_states, num_steps=num_steps)

        return endpoints

    def predict_endpoints_batch(self, start_states: torch.Tensor,
                                num_steps: int = 100,
                                num_samples: int = 10) -> torch.Tensor:
        """
        Predict multiple endpoint samples (for uncertainty estimation)

        Args:
            start_states: Start states [B, 4]
            num_steps: ODE integration steps
            num_samples: Number of samples per start state

        Returns:
            Predicted endpoints [B*num_samples, 4]
        """
        if isinstance(start_states, list):
            start_states = torch.tensor(start_states, dtype=torch.float32)

        start_states = start_states.to(self.device)

        with torch.no_grad():
            endpoints = self.model.predict_endpoints_batch(
                start_states, num_steps=num_steps, num_samples=num_samples
            )

        return endpoints

    def get_system(self):
        """Get the dynamical system"""
        return self.model.system

    def get_model_info(self):
        """Get model architecture info"""
        return self.model.model.get_model_info()

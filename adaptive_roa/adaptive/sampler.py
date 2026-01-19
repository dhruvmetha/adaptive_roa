"""
State Sampler for Adaptive Sampling.

Generates initial states from the state space for evaluation.
"""
import torch
import numpy as np
from typing import Union, Optional, Tuple, List


class StateSampler:
    """
    Sample initial states from the state space.

    Generates states uniformly within the system's bounds for
    candidate evaluation in adaptive sampling.

    Attributes:
        system: Dynamical system with state bounds
        device: Device for tensor output
    """

    def __init__(
        self,
        system,
        device: str = "cuda"
    ):
        """
        Initialize state sampler.

        Args:
            system: Dynamical system with define_state_bounds()
            device: Device for tensor output
        """
        self.system = system
        self.device = device

        # Extract bounds from system
        self._extract_bounds()

    def _extract_bounds(self):
        """Extract sampling bounds from system."""
        bounds_dict = self.system.define_state_bounds()

        # Get state dimension from system
        # Count total dimensions from manifold structure
        components = self.system.define_manifold_structure()
        self.state_dim = sum(c.dim for c in components)

        # Build lower and upper bounds arrays
        # The order should match the manifold structure
        self.lower_bounds = np.zeros(self.state_dim)
        self.upper_bounds = np.zeros(self.state_dim)

        dim_idx = 0
        for component in components:
            comp_name = component.name
            comp_dim = component.dim

            if comp_name in bounds_dict:
                low, high = bounds_dict[comp_name]
                for d in range(comp_dim):
                    self.lower_bounds[dim_idx + d] = low
                    self.upper_bounds[dim_idx + d] = high
            else:
                # Default to [-1, 1] if not specified
                for d in range(comp_dim):
                    self.lower_bounds[dim_idx + d] = -1.0
                    self.upper_bounds[dim_idx + d] = 1.0

            dim_idx += comp_dim

    def sample_uniform(
        self,
        n: int,
        as_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Sample n states uniformly from state bounds.

        Args:
            n: Number of states to sample
            as_tensor: Return as torch tensor (True) or numpy array (False)

        Returns:
            [n, state_dim] sampled states
        """
        # Sample uniformly in [0, 1] then scale to bounds
        samples = np.random.uniform(0, 1, size=(n, self.state_dim))
        samples = samples * (self.upper_bounds - self.lower_bounds) + self.lower_bounds

        if as_tensor:
            return torch.from_numpy(samples).float().to(self.device)
        return samples

    def sample_near(
        self,
        centers: Union[torch.Tensor, np.ndarray],
        n_per_center: int,
        radius: float,
        as_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Sample states near given center points.

        Useful for focused exploration around uncertain regions.

        Args:
            centers: [M, state_dim] center points
            n_per_center: Number of samples per center
            radius: Sampling radius around each center
            as_tensor: Return as torch tensor (True) or numpy array (False)

        Returns:
            [M * n_per_center, state_dim] sampled states
        """
        if isinstance(centers, torch.Tensor):
            centers = centers.cpu().numpy()

        m = len(centers)
        samples = []

        for center in centers:
            # Sample in ball around center
            for _ in range(n_per_center):
                # Sample direction
                direction = np.random.randn(self.state_dim)
                direction = direction / np.linalg.norm(direction)

                # Sample radius (uniform in ball)
                r = radius * np.random.uniform(0, 1) ** (1 / self.state_dim)

                # New sample
                sample = center + r * direction

                # Clip to bounds
                sample = np.clip(sample, self.lower_bounds, self.upper_bounds)
                samples.append(sample)

        samples = np.array(samples)

        if as_tensor:
            return torch.from_numpy(samples).float().to(self.device)
        return samples

    def split_d1_d2(
        self,
        states: Union[torch.Tensor, np.ndarray],
        d1_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split states into D1 (calibration) and D2 (pool) sets.

        D1: Always simulated, used for calibration
        D2: Selective simulation based on uncertainty

        Args:
            states: [N, state_dim] candidate states
            d1_ratio: Fraction for D1 (calibration)

        Returns:
            Tuple of (D1, D2) numpy arrays
        """
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()

        n = len(states)
        n_d1 = int(n * d1_ratio)

        # Shuffle and split
        indices = np.random.permutation(n)
        d1_indices = indices[:n_d1]
        d2_indices = indices[n_d1:]

        return states[d1_indices], states[d2_indices]

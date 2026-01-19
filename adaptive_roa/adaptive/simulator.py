"""
Simulator Interface for Adaptive Sampling.

Provides abstraction for running or looking up trajectory simulations.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Callable, Optional
from abc import ABC, abstractmethod


class Simulator(ABC):
    """
    Abstract base class for simulators.

    Simulators take initial states and return (end_states, labels).
    """

    @abstractmethod
    def simulate(
        self,
        initial_states: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate trajectories from initial states.

        Args:
            initial_states: [N, state_dim] initial states

        Returns:
            Tuple of:
                end_states: [N, state_dim] final states
                labels: [N] labels (-1 for failure, 1 for success)
        """
        pass


class FileBasedSimulator(Simulator):
    """
    Simulator that looks up trajectories from a pre-computed file.

    Loads a trajectory file and matches initial states to find endpoints.
    Useful when simulations have already been run.

    Note: This requires that the requested initial states exist in the file.
    For truly adaptive sampling, use CallbackSimulator.

    Attributes:
        trajectory_file: Path to trajectory file
        system: Dynamical system
        attractor_radius: Radius for classification
        starts: [N, state_dim] all start states in file
        ends: [N, state_dim] all end states in file
    """

    def __init__(
        self,
        trajectory_file: str,
        system,
        attractor_radius: float = 0.2,
        tolerance: float = 1e-6
    ):
        """
        Initialize file-based simulator.

        Args:
            trajectory_file: Path to trajectory file
            system: Dynamical system with classify_attractor()
            attractor_radius: Radius for classification
            tolerance: Tolerance for matching states
        """
        self.trajectory_file = trajectory_file
        self.system = system
        self.attractor_radius = attractor_radius
        self.tolerance = tolerance

        self._load_file()

    def _load_file(self):
        """Load trajectory file."""
        data = np.loadtxt(self.trajectory_file)
        state_dim = data.shape[1] // 2

        self.starts = data[:, :state_dim]
        self.ends = data[:, state_dim:]
        self.state_dim = state_dim

        print(f"Loaded {len(self.starts)} trajectories from {self.trajectory_file}")

    def simulate(
        self,
        initial_states: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Look up trajectories for initial states.

        Args:
            initial_states: [N, state_dim] initial states to look up

        Returns:
            Tuple of (end_states, labels)

        Raises:
            ValueError if any state is not found in the file
        """
        if isinstance(initial_states, torch.Tensor):
            initial_states = initial_states.cpu().numpy()

        n = len(initial_states)
        end_states = np.zeros((n, self.state_dim))

        for i, state in enumerate(initial_states):
            # Find closest match
            distances = np.linalg.norm(self.starts - state, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if min_dist > self.tolerance:
                raise ValueError(
                    f"State {state} not found in trajectory file "
                    f"(closest distance: {min_dist:.6f})"
                )

            end_states[i] = self.ends[min_idx]

        # Compute labels
        ends_tensor = torch.from_numpy(end_states).float()
        labels = self.system.classify_attractor(
            ends_tensor, radius=self.attractor_radius
        ).numpy()

        return end_states, labels


class CallbackSimulator(Simulator):
    """
    Simulator that calls an actual simulation function.

    For truly adaptive sampling where we run new simulations on demand.

    Attributes:
        simulate_fn: Function that takes initial_states and returns end_states
        system: Dynamical system
        attractor_radius: Radius for classification
    """

    def __init__(
        self,
        simulate_fn: Callable[[np.ndarray], np.ndarray],
        system,
        attractor_radius: float = 0.2
    ):
        """
        Initialize callback simulator.

        Args:
            simulate_fn: Function (initial_states: [N, dim]) -> end_states: [N, dim]
            system: Dynamical system with classify_attractor()
            attractor_radius: Radius for classification
        """
        self.simulate_fn = simulate_fn
        self.system = system
        self.attractor_radius = attractor_radius

    def simulate(
        self,
        initial_states: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run actual simulations for initial states.

        Args:
            initial_states: [N, state_dim] initial states

        Returns:
            Tuple of (end_states, labels)
        """
        if isinstance(initial_states, torch.Tensor):
            initial_states = initial_states.cpu().numpy()

        # Run simulation
        end_states = self.simulate_fn(initial_states)

        # Compute labels
        ends_tensor = torch.from_numpy(end_states).float()
        labels = self.system.classify_attractor(
            ends_tensor, radius=self.attractor_radius
        ).numpy()

        return end_states, labels


class PoolSimulator(Simulator):
    """
    Simulator that samples from a pool of pre-computed trajectories.

    Unlike FileBasedSimulator which requires exact matches, this samples
    random trajectories from the pool. Useful for experiments where we
    want to simulate adaptive sampling but don't need exact states.

    Attributes:
        starts: Pool of start states
        ends: Pool of end states
        labels: Pool of labels
        used_indices: Set of already-used trajectory indices
    """

    def __init__(
        self,
        trajectory_file: str,
        system,
        attractor_radius: float = 0.2
    ):
        """
        Initialize pool simulator.

        Args:
            trajectory_file: Path to trajectory file (pool of trajectories)
            system: Dynamical system with classify_attractor()
            attractor_radius: Radius for classification
        """
        self.system = system
        self.attractor_radius = attractor_radius
        self.used_indices = set()

        # Load pool
        data = np.loadtxt(trajectory_file)
        state_dim = data.shape[1] // 2

        self.starts = data[:, :state_dim]
        self.ends = data[:, state_dim:]
        self.state_dim = state_dim

        # Pre-compute labels
        ends_tensor = torch.from_numpy(self.ends).float()
        self.labels = system.classify_attractor(
            ends_tensor, radius=attractor_radius
        ).numpy()

        print(f"Loaded pool of {len(self.starts)} trajectories")

    def simulate(
        self,
        initial_states: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample trajectories from pool.

        Note: The initial_states argument is ignored; we sample randomly
        from unused trajectories in the pool.

        Args:
            initial_states: [N, state_dim] - N trajectories will be sampled

        Returns:
            Tuple of (end_states, labels) from sampled trajectories
        """
        if isinstance(initial_states, torch.Tensor):
            initial_states = initial_states.cpu().numpy()

        n = len(initial_states)

        # Find available indices
        available = [i for i in range(len(self.starts)) if i not in self.used_indices]

        if len(available) < n:
            raise ValueError(
                f"Not enough trajectories in pool: need {n}, have {len(available)}"
            )

        # Sample random indices
        sampled_indices = np.random.choice(available, size=n, replace=False)
        self.used_indices.update(sampled_indices)

        return self.ends[sampled_indices], self.labels[sampled_indices]

    def get_sampled_starts(self, indices: np.ndarray) -> np.ndarray:
        """Get the actual start states for sampled indices."""
        return self.starts[indices]

    def reset(self):
        """Reset used indices to allow resampling."""
        self.used_indices = set()

    def get_available_count(self) -> int:
        """Get number of available trajectories."""
        return len(self.starts) - len(self.used_indices)

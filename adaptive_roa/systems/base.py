"""
Base class for dynamical systems with Lie group structure
"""
from pathlib import Path
import math
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

try:
    import theseus as th
    THESEUS_AVAILABLE = True
except ImportError:
    THESEUS_AVAILABLE = False
    print("Warning: Theseus not available. Some functionality will be limited.")


class ManifoldComponent:
    """Represents a single manifold component in the state space"""

    def __init__(self, manifold_type: str, dim: int, name: str):
        """
        Args:
            manifold_type: Type of manifold ("SO2", "Real")
            dim: Dimension of the component
            name: Human-readable name (e.g., "angle", "position")
        """
        self.manifold_type = manifold_type
        self.dim = dim
        self.name = name


class DynamicalSystem(ABC):
    """
    Abstract base class for dynamical systems with Lie group structure
    
    Each system defines:
    1. Manifold structure of its state space
    2. Embedding/extraction methods for neural networks
    3. State normalization bounds
    """
    
    def __init__(self):
        self._manifold_components = self.define_manifold_structure()
        self._state_bounds = self.define_state_bounds()
        
    @abstractmethod
    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define the manifold structure of the system's state space
        
        Returns:
            List of ManifoldComponent objects describing each state component
        """
        pass
    
    @abstractmethod 
    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define normalization bounds for each state component
        
        Returns:
            Dictionary mapping component names to (min, max) bounds
        """
        pass
    
    @property
    def manifold_components(self) -> List[ManifoldComponent]:
        """Get manifold components"""
        return self._manifold_components
    
    @property
    def state_dim(self) -> int:
        """Total dimension of raw state"""
        return sum(comp.dim for comp in self._manifold_components)

    @property
    def binary_outcomes(self) -> bool:
        """True when the system's ground truth has no 'invalid' outcome.

        Stochastic datasets record ``p_success = successes / trials`` and nothing
        else -- a rollout either reached the goal or it did not. Their labels have
        no third class, so a sampled endpoint that lands near no attractor must be
        counted as a FAILURE rather than becoming an invalid class with nothing to
        be scored against.

        **Inferred from the data, not declared in config, on purpose.** The
        presence of ``eval_success_prob.npz`` beside the dataset IS the definition
        of a binary-outcome dataset: it stores successes/trials and no third
        count. Deriving it here means a stochastic run cannot be misconfigured
        into three-way scoring by forgetting a key -- which is how cartpole
        sigma_020.0 came to score 67.9% of its eval points "invalid" and report
        F1 = 0.035 with a single true positive.

        Deterministic systems have no such file and keep the three-way space,
        where "unresolved" is a genuinely distinct outcome.
        """
        dataset_dir = getattr(self, "dataset_dir", None)
        if not dataset_dir:
            return False
        return (Path(dataset_dir) / "eval_success_prob.npz").exists()


    @property
    def state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get state bounds"""
        return self._state_bounds

    def per_dim_bounds(self) -> List[Tuple[float, float]]:
        """Support box of the raw state space, one (low, high) pair PER RAW DIM.

        ``state_bounds`` is keyed by manifold component, so a component covering
        several raw dimensions carries exactly one entry for all of them. That is
        fine for normalization, where the component's scale is shared by
        construction, but it is wrong for anything that needs the true extent of
        each axis: quadrotor2D's ``velocity`` component spans (x_dot, z_dot,
        theta_dot), whose real ranges are +-1.303, +-1.314 and +-13.365, and
        quadrotor3D's ``position`` spans (x, y, z) where z is [0.07, 3.03] and is
        not symmetric at all.

        Part-X builds its partition root from this box and drops any state
        outside it, so the collapsed version silently excluded 91% of the
        quadrotor2D state space and 43% of quadrotor3D's from acquisition.

        The default below reproduces the component-expansion exactly, so systems
        whose components are all one-dimensional (pendulum, cartpole) are
        unchanged. Systems with multi-dimensional components should override.
        """
        out: List[Tuple[float, float]] = []
        for comp in self.manifold_components:
            dim = int(getattr(comp, "dim", 1))
            if comp.manifold_type == "SO2":
                lo, hi = -math.pi, math.pi
            else:
                b = self.state_bounds[comp.name]
                lo, hi = float(b[0]), float(b[1])
            out.extend([(lo, hi)] * dim)
        return out
    
    def get_circular_indices(self) -> List[int]:
        """
        Get indices of SO2 (circular) components in state vector

        This is useful for operations that need to treat circular coordinates
        specially (e.g., angle wrapping during perturbation).

        Returns:
            List of integer indices where SO2 components appear in the state vector

        Example:
            CartPole state: [x, θ, ẋ, θ̇] → returns [1] (θ at index 1)
            Pendulum state: [θ, θ̇] → returns [0] (θ at index 0)
        """
        indices = []
        idx = 0

        for comp in self._manifold_components:
            if comp.manifold_type == "SO2":
                indices.append(idx)
            idx += comp.dim

        return indices

    def get_loss_weights(self) -> torch.Tensor:
        """
        Get per-dimension loss weights based on normalization limits.

        Returns weights proportional to the normalization limits for each dimension.
        Dimensions with larger physical ranges get larger weights, emphasizing
        their importance in the loss function.

        For circular dimensions (SO2, SO3, Sphere), weight = 1.0 is used since
        they don't have traditional "limits" (angles are in [-π, π], quaternions in [-1, 1]).

        Override in subclasses for system-specific weight computation,
        especially for systems with tangent space different from state space (e.g., SO3).

        Returns:
            torch.Tensor: Per-dimension weights [state_dim] or [tangent_dim]
        """
        weights = []

        for comp in self._manifold_components:
            if comp.manifold_type in ("SO2", "SO3", "Sphere"):
                # Circular/spherical components: use unit weight
                weights.extend([1.0] * comp.dim)
            else:
                # Real components: use the upper bound as weight
                bounds = self._state_bounds.get(comp.name, (-1.0, 1.0))
                limit = max(abs(bounds[0]), abs(bounds[1]))
                weights.extend([limit] * comp.dim)

        return torch.tensor(weights, dtype=torch.float32)

    def get_normalization_scales(self) -> torch.Tensor:
        """
        Get per-dimension distance scales for range-normalized state metrics.

        Dividing a state difference by these scales puts every dimension on
        equal footing: the MAXIMUM POSSIBLE difference in each dimension
        normalizes to 1.0, regardless of dimension type. Without it, the
        widest-range dimension dominates any distance computed over the full
        state vector.

        This is NOT get_loss_weights(): those weights are proportional to each
        dimension's range (amplifying wide dimensions), which is the wrong sign
        for a distance metric.

        Circular (SO2) dimensions use π rather than the full 2π range: angle
        differences are wrapped via atan2(sin, cos) into [-π, π], so the
        largest possible circular disagreement is π, not 2π. SO3/Sphere
        components use 1.0 (their coordinates live in [-1, 1]). Real
        components have no such wrapping, so their largest possible
        disagreement is the FULL declared range (hi - lo), not half of it;
        using the full range is what makes a maximal real disagreement
        normalize to 1.0 to match the circular case. Falls back to 1.0 when
        bounds are missing or the computed range is not positive.

        Returns:
            torch.Tensor: Per-dimension scales [state_dim], all finite and > 0
        """
        scales = []

        for comp in self._manifold_components:
            if comp.manifold_type == "SO2":
                scales.extend([math.pi] * comp.dim)
            elif comp.manifold_type in ("SO3", "Sphere"):
                scales.extend([1.0] * comp.dim)
            else:
                lo, hi = self._state_bounds.get(comp.name, (-1.0, 1.0))
                full_range = hi - lo
                scales.extend([full_range if full_range > 0 else 1.0] * comp.dim)

        return torch.tensor(scales, dtype=torch.float32)

    def embed_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Embed raw state into neural network input space

        Args:
            state: Raw state tensor [..., state_dim]

        Returns:
            embedded: Embedded state tensor [..., embedding_dim]
        """
        embedded_components = []
        start_idx = 0

        for comp in self._manifold_components:
            end_idx = start_idx + comp.dim
            component_state = state[..., start_idx:end_idx]

            if comp.manifold_type == "SO2":
                # θ → (sin θ, cos θ)
                theta = component_state[..., 0]
                embedded = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)

            elif comp.manifold_type == "Real":
                # Pass through unchanged
                embedded = component_state

            else:
                raise NotImplementedError(f"Embedding for {comp.manifold_type} not implemented yet")

            embedded_components.append(embedded)
            start_idx = end_idx

        return torch.cat(embedded_components, dim=-1)

    def __repr__(self) -> str:
        components_str = ", ".join([
            f"{comp.name}({comp.manifold_type})" for comp in self._manifold_components
        ])
        return f"{self.__class__.__name__}(manifolds=[{components_str}])"


# Legacy compatibility
BaseSystem = DynamicalSystem
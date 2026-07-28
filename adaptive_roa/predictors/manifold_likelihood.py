"""Per-manifold-component predictive likelihoods for final-state prediction.

A final-state predictor emits a DISTRIBUTION over x_T, not a point. The
distribution factorizes over the system's manifold components, and each
component type needs its own likelihood: a Euclidean Gaussian will not do for
an angle (pendulum's failure attractors sit at theta = +/-pi, exactly the wrap
seam) nor for a quaternion (where -q is the same rotation as q).

Each likelihood owns four things: how many parameters it consumes, its NLL, how
to sample from it, and its geodesic distance (used for endpoint-error reporting).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List

import torch


def _split_mu_logsigma(params: torch.Tensor, dim: int):
    return params[..., :dim], params[..., dim:2 * dim]


class ComponentLikelihood(ABC):
    """One manifold component's predictive distribution."""

    @abstractmethod
    def n_params(self, dim: int) -> int:
        """How many head outputs this component consumes for a `dim`-wide slice."""

    @abstractmethod
    def nll(self, params: torch.Tensor, target: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
        """Negative log-likelihood per batch row: [B, n_params], [B, dim] -> [B]."""

    @abstractmethod
    def sample(self, params: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """One draw: [B, n_params] -> [B, dim], in RAW state coordinates."""

    @abstractmethod
    def mean(self, params: torch.Tensor) -> torch.Tensor:
        """Distribution mean: [B, n_params] -> [B, dim], RAW coordinates."""

    @abstractmethod
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Geodesic distance per reported sub-component: [B,dim],[B,dim] -> [B, n_dist]."""

    @abstractmethod
    def n_dist(self, dim: int) -> int:
        """How many distance values `distance` returns."""

    @abstractmethod
    def names(self, dim: int, base: str) -> List[str]:
        """Reported names, one per value `distance` returns."""


class RealLikelihood(ComponentLikelihood):
    """Diagonal Gaussian over a Euclidean slice, with optional beta-NLL weighting.

    beta-NLL (Seitzer et al., ICLR 2022) multiplies each dimension's loss by a
    DETACHED sigma^(2*beta). Plain Gaussian NLL shrinks the gradient on
    high-error points as sigma grows there, starving exactly the points that
    need fitting; beta = 0.5 restores most of that signal. beta = 0 is plain NLL.
    """

    LOG_SIGMA_MIN = -7.0
    LOG_SIGMA_MAX = 7.0

    def n_params(self, dim: int) -> int:
        return 2 * int(dim)

    def _sigma(self, params: torch.Tensor, dim: int) -> torch.Tensor:
        _, log_sigma = _split_mu_logsigma(params, dim)
        return log_sigma.clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX).exp()

    def nll(self, params, target, beta: float = 0.0):
        dim = target.shape[-1]
        mu, _ = _split_mu_logsigma(params, dim)
        sigma = self._sigma(params, dim)
        per_dim = 0.5 * torch.log(2 * math.pi * sigma ** 2) + (target - mu) ** 2 / (2 * sigma ** 2)
        if beta > 0.0:
            # Detached: the weight must not create a gradient path of its own.
            per_dim = per_dim * (sigma.detach() ** (2.0 * beta))
        return per_dim.sum(dim=-1)

    def sample(self, params, generator=None):
        dim = params.shape[-1] // 2
        mu, _ = _split_mu_logsigma(params, dim)
        sigma = self._sigma(params, dim)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        return mu + sigma * eps

    def mean(self, params):
        dim = params.shape[-1] // 2
        return _split_mu_logsigma(params, dim)[0]

    def distance(self, a, b):
        return (a - b).abs()

    def n_dist(self, dim: int) -> int:
        return int(dim)

    def names(self, dim: int, base: str) -> List[str]:
        if int(dim) == 1:
            return [base]
        return [f"{base}_{i}" for i in range(int(dim))]


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Wrap to [-pi, pi]. The codebase-wide idiom (systems/pendulum.py:131 etc.)."""
    return torch.atan2(torch.sin(x), torch.cos(x))


class SO2Likelihood(ComponentLikelihood):
    """Wrapped normal over one angle.

    The head emits an UNNORMALIZED (sin, cos) direction plus log sigma; the mean
    angle is atan2(sin, cos), which is well-defined regardless of magnitude and
    has no seam. Sampling adds Gaussian noise in the tangent (angle) space and
    wraps, so mass near +/-pi correctly appears on both sides -- the property a
    Euclidean Gaussian on theta lacks, and the reason this class exists:
    pendulum's FAILURE attractors sit at theta = +/-pi.

    NLL uses the wrapped geodesic residual. This is the wrapped-normal density
    truncated to its principal term, which is accurate for sigma well under pi
    and is what makes the loss seam-aware.
    """

    LOG_SIGMA_MIN = -7.0
    LOG_SIGMA_MAX = math.log(math.pi)

    def n_params(self, dim: int) -> int:
        if int(dim) != 1:
            raise ValueError(f"SO2 component must be 1-dimensional, got {dim}")
        return 3

    def _sigma(self, params: torch.Tensor) -> torch.Tensor:
        return params[..., 2:3].clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX).exp()

    def mean(self, params):
        return torch.atan2(params[..., 0:1], params[..., 1:2])

    def nll(self, params, target, beta: float = 0.0):
        mu = self.mean(params)
        sigma = self._sigma(params)
        resid = wrap_angle(target - mu)
        per_dim = 0.5 * torch.log(2 * math.pi * sigma ** 2) + resid ** 2 / (2 * sigma ** 2)
        if beta > 0.0:
            per_dim = per_dim * (sigma.detach() ** (2.0 * beta))
        return per_dim.sum(dim=-1)

    def sample(self, params, generator=None):
        mu = self.mean(params)
        sigma = self._sigma(params)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        return wrap_angle(mu + sigma * eps)

    def distance(self, a, b):
        return wrap_angle(a - b).abs()

    def n_dist(self, dim: int) -> int:
        return 1

    def names(self, dim: int, base: str) -> List[str]:
        return [f"{base}_geodesic"]

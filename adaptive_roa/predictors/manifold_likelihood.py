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


def _check_real_width(params: torch.Tensor, dim: int) -> None:
    """A Euclidean slice must carry exactly ``2 * dim`` params (mu, log_sigma).

    Silent-wrongness guard: a params/target width mismatch would slice log_sigma
    out of the wrong offset and produce a plausible-looking but wrong loss.
    """
    if params.shape[-1] != 2 * int(dim):
        raise ValueError(
            f"RealLikelihood expects 2*dim={2 * int(dim)} params for a dim={int(dim)} "
            f"slice, got {params.shape[-1]}; the log-sigma slice would be read from "
            f"the wrong offset."
        )


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
        # `nll` derives dim from the TARGET while `sample`/`mean` derive it from
        # the PARAMS width. If a caller ever pairs a params slice with a
        # mismatched target slice, the log-sigma read below silently lands on
        # the wrong half of the vector and the loss is quietly wrong rather than
        # raising. Cross-check the two derivations instead.
        _check_real_width(params, dim)
        mu, _ = _split_mu_logsigma(params, dim)
        sigma = self._sigma(params, dim)
        per_dim = 0.5 * torch.log(2 * math.pi * sigma ** 2) + (target - mu) ** 2 / (2 * sigma ** 2)
        if beta > 0.0:
            # Detached: the weight must not create a gradient path of its own.
            per_dim = per_dim * (sigma.detach() ** (2.0 * beta))
        return per_dim.sum(dim=-1)

    def sample(self, params, generator=None):
        dim = params.shape[-1] // 2
        _check_real_width(params, dim)
        mu, _ = _split_mu_logsigma(params, dim)
        sigma = self._sigma(params, dim)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        return mu + sigma * eps

    def mean(self, params):
        dim = params.shape[-1] // 2
        _check_real_width(params, dim)
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
    # sigma is an angular standard deviation in radians; pi is the largest
    # geodesic distance on the circle, so anything above it is already uniform.
    # SO3Likelihood uses the same ceiling for the same reason.
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


def canonicalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Unit-normalize and force qw >= 0 (mirrors systems/quadrotor3d.py:425-443).

    NOT cosmetic: Quadrotor3DSystem.classify_attractor compares raw 13-vectors by
    L2 against an identity-quaternion goal, so -q -- the same rotation -- lands at
    distance 2 and is misclassified.
    """
    norm = q.norm(dim=-1, keepdim=True)
    # `q / norm.clamp_min(1e-8)` does NOT give a unit quaternion for a degenerate
    # input: at ||q|| = 2e-9 the clamp divides by 1e-8 and leaves ||q|| = 0.2, so
    # a "canonical" quaternion of the wrong magnitude flows on into
    # classify_attractor's raw L2 comparison. Fall back to the identity rotation
    # for inputs too small to carry a direction, and renormalize the rest.
    q = torch.where(norm > 1e-6, q / norm.clamp_min(1e-8), _identity_quaternion(q))
    return torch.where(q[..., 0:1] < 0, -q, q)


def _identity_quaternion(like: torch.Tensor) -> torch.Tensor:
    """(1, 0, 0, 0) broadcast to ``like``'s shape/device/dtype."""
    out = torch.zeros_like(like)
    out[..., 0] = 1.0
    return out


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, (w, x, y, z) convention."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


class SO3Likelihood(ComponentLikelihood):
    """Concentrated Gaussian in the so(3) tangent space at a mean rotation.

    The head emits an unnormalized 4-vector (canonicalized to a unit quaternion
    with qw >= 0) plus three log sigmas for the tangent axes. Sampling draws
    xi ~ N(0, diag(sigma^2)) in so(3) and applies q = q_bar * exp(xi/2), which
    keeps every sample exactly on the unit sphere -- something a Gaussian in R^4
    plus renormalization does not do faithfully.

    NLL uses the tangent-space residual of the target relative to the mean, so it
    is invariant to the target's sign (double cover).
    """

    LOG_SIGMA_MIN = -7.0
    # Matches SO2Likelihood.LOG_SIGMA_MAX, and for the same reason: sigma is an
    # ANGULAR standard deviation in radians, and pi is the largest geodesic
    # distance either manifold admits (on SO(3) every rotation is within pi of
    # every other). A per-axis sigma at that clamp already yields an essentially
    # uniform orientation, so nothing informative lives above it. The previous
    # value of 2.0 (sigma = 7.39 rad, over two full turns) was an unexplained
    # round number that let a diverging fit wander far past uniform while the
    # NLL kept rewarding it; tightening only bites fits that are already
    # degenerate.
    LOG_SIGMA_MAX = math.log(math.pi)

    def n_params(self, dim: int) -> int:
        if int(dim) != 4:
            raise ValueError(f"SO3 component must be 4-dimensional, got {dim}")
        return 7

    def _sigma(self, params: torch.Tensor) -> torch.Tensor:
        return params[..., 4:7].clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX).exp()

    def mean(self, params):
        return canonicalize_quaternion(params[..., 0:4])

    @staticmethod
    def _exp_map(xi: torch.Tensor) -> torch.Tensor:
        """so(3) tangent vector -> unit quaternion. xi is a rotation vector."""
        theta = xi.norm(dim=-1, keepdim=True)
        half = 0.5 * theta
        # sinc-style guard so the theta -> 0 limit is finite and differentiable.
        scale = torch.where(theta > 1e-6, torch.sin(half) / theta.clamp_min(1e-8),
                            torch.full_like(theta, 0.5))
        return torch.cat([torch.cos(half), xi * scale], dim=-1)

    @staticmethod
    def _log_map(q: torch.Tensor) -> torch.Tensor:
        """Unit quaternion -> so(3) rotation vector, sign-canonicalized first."""
        q = canonicalize_quaternion(q)
        w = q[..., 0:1].clamp(-1.0, 1.0)
        v = q[..., 1:4]
        v_norm = v.norm(dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(v_norm, w)
        scale = torch.where(v_norm > 1e-6, angle / v_norm.clamp_min(1e-8),
                            torch.full_like(v_norm, 2.0))
        return v * scale

    def _residual(self, params, target):
        q_bar = self.mean(params)
        q_t = canonicalize_quaternion(target)
        q_bar_inv = q_bar * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q_bar.device,
                                         dtype=q_bar.dtype)
        return self._log_map(quaternion_multiply(q_bar_inv, q_t))

    def nll(self, params, target, beta: float = 0.0):
        sigma = self._sigma(params)
        xi = self._residual(params, target)
        per_dim = 0.5 * torch.log(2 * math.pi * sigma ** 2) + xi ** 2 / (2 * sigma ** 2)
        if beta > 0.0:
            per_dim = per_dim * (sigma.detach() ** (2.0 * beta))
        return per_dim.sum(dim=-1)

    def sample(self, params, generator=None):
        q_bar = self.mean(params)
        sigma = self._sigma(params)
        eps = torch.randn(sigma.shape, generator=generator, device=sigma.device, dtype=sigma.dtype)
        return canonicalize_quaternion(quaternion_multiply(q_bar, self._exp_map(sigma * eps)))

    def distance(self, a, b):
        a = canonicalize_quaternion(a)
        b = canonicalize_quaternion(b)
        dot = (a * b).sum(dim=-1, keepdim=True).abs().clamp(max=1.0)
        return 2.0 * torch.acos(dot)

    def n_dist(self, dim: int) -> int:
        return 1

    def names(self, dim: int, base: str) -> List[str]:
        return [f"{base}_geodesic"]

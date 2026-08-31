"""Output heads for predictor arms.

``FinalStateHead`` turns a flat parameter vector into a predictive distribution
over the final state, factorized across the system's manifold components. It is
built generically from ``system.manifold_components`` so every system is handled
by the same code path.
"""
from __future__ import annotations

from typing import List

import torch

from adaptive_roa.predictors.manifold_likelihood import (
    ComponentLikelihood,
    RealLikelihood,
    SO2Likelihood,
    SO3Likelihood,
)

_LIKELIHOODS = {
    "Real": RealLikelihood,
    "SO2": SO2Likelihood,
    "SO3": SO3Likelihood,
}


class FinalStateHead:
    """Predictive distribution over x_T, factorized by manifold component.

    ``beta`` is the beta-NLL exponent (Seitzer et al., ICLR 2022); 0.5 is the
    recommended default and 0.0 recovers plain Gaussian NLL.

    Coordinate contract
    -------------------
    The distribution's PARAMETERS live in NORMALIZED state coordinates; every
    public method speaks RAW ones. ``nll`` normalizes its target, ``sample`` and
    ``mean`` denormalize their output, and ``distance_per_component`` is a pure
    raw-in/raw-out geometric helper. Callers therefore never handle normalized
    states, which is what keeps this head's contract identical to the flow
    matcher's ``predict_endpoint``.

    Learning in normalized coordinates is not cosmetic. The loss sums a Gaussian
    NLL across dims, and beta-NLL then multiplies each dim by ``sigma^(2*beta)``,
    so in RAW coordinates a wide-range dim dominates the total. Measured on
    quadrotor3d (position +/-1.8 m against angular velocity +/-39 rad/s) at an
    untrained initialization, the three angular-velocity dims carried 97.9% of
    the loss and the per-dim spread was 287x; in normalized coordinates the
    spread is 1.2x. Because ``gradient_clip_val`` is a GLOBAL-norm clip, the
    dominant dims also dominated the clipped update direction, and at the
    ``LOG_SIGMA_MIN`` floor a collapsed dim's sigma gradient is exactly zero and
    can never recover. This also puts the arm on the same footing as every other
    predictor in the codebase, all of which learn normalized.
    """

    def __init__(self, system, beta: float = 0.5):
        self.system = system
        self.beta = float(beta)
        self._parts: List[tuple[ComponentLikelihood, int, int, int, int]] = []

        param_offset = 0
        state_offset = 0
        names: List[str] = []
        for comp in system.manifold_components:
            lik_cls = _LIKELIHOODS.get(comp.manifold_type)
            if lik_cls is None:
                raise ValueError(
                    f"no likelihood registered for manifold component "
                    f"{comp.manifold_type!r}; expected one of {sorted(_LIKELIHOODS)}"
                )
            lik = lik_cls()
            n_p = lik.n_params(comp.dim)
            self._parts.append((lik, param_offset, n_p, state_offset, comp.dim))
            names.extend(lik.names(comp.dim, comp.name))
            param_offset += n_p
            state_offset += comp.dim

        self.n_params = param_offset
        self.state_dim = state_offset
        self.component_names = names

    def _iter(self, params: torch.Tensor, target: torch.Tensor | None = None):
        for lik, p0, n_p, s0, dim in self._parts:
            p = params[..., p0:p0 + n_p]
            t = None if target is None else target[..., s0:s0 + dim]
            yield lik, p, t, dim

    def nll(self, params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """NLL of a RAW target under the predictive distribution: [B]."""
        # Normalized before scoring so no dim's loss contribution is set by its
        # physical units -- see the class docstring for the measured effect.
        # normalize_state leaves SO2 angles and SO3 quaternions untouched on all
        # four systems, so the circular components' semantics are unaffected.
        target = self.system.normalize_state(target)
        total = None
        for lik, p, t, _dim in self._iter(params, target):
            term = lik.nll(p, t, beta=self.beta)
            total = term if total is None else total + term
        return total

    def sample(self, params: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """One draw in RAW coordinates: [B, n_params] -> [B, state_dim]."""
        normalized = torch.cat(
            [lik.sample(p, generator=generator) for lik, p, _t, _d in self._iter(params)], dim=-1
        )
        return self._to_raw(normalized)

    def mean(self, params: torch.Tensor) -> torch.Tensor:
        """Distribution mean in RAW coordinates: [B, n_params] -> [B, state_dim]."""
        normalized = torch.cat(
            [lik.mean(p) for lik, p, _t, _d in self._iter(params)], dim=-1
        )
        return self._to_raw(normalized)

    def _to_raw(self, normalized: torch.Tensor) -> torch.Tensor:
        """Normalized -> raw, preserving each component's manifold invariants.

        denormalize_state is an identity on SO2 angles (so samples stay wrapped)
        and unit-normalizes the quadrotor3d quaternion (idempotent, since
        SO3Likelihood.sample already canonicalized it), so the raw-output
        contract predict_endpoint advertises still holds after the round trip.
        """
        return self.system.denormalize_state(normalized)

    def distance_per_component(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = [
            lik.distance(a[..., s0:s0 + dim], b[..., s0:s0 + dim])
            for lik, _p0, _n_p, s0, dim in self._parts
        ]
        return torch.cat(out, dim=-1)

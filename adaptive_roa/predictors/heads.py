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
        return self._nll_normalized(params, target, self.beta)

    def _nll_normalized(self, params: torch.Tensor, target: torch.Tensor,
                        beta: float) -> torch.Tensor:
        """NLL of an already-NORMALIZED target: [B]. ``nll`` is this after normalizing."""
        total = None
        for lik, p, t, _dim in self._iter(params, target):
            term = lik.nll(p, t, beta=beta)
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


def _location_and_scale_offsets(lik: ComponentLikelihood, p0: int, n_p: int, dim: int):
    """Split one component likelihood's param slice into location and scale rows."""
    if isinstance(lik, RealLikelihood):
        return list(range(p0, p0 + dim)), list(range(p0 + dim, p0 + 2 * dim))
    if isinstance(lik, SO2Likelihood):
        return [p0, p0 + 1], [p0 + 2]
    if isinstance(lik, SO3Likelihood):
        return list(range(p0, p0 + 4)), list(range(p0 + 4, p0 + 7))
    raise TypeError(f"no location/scale layout for {type(lik).__name__}")


class MixtureFinalStateHead:
    """Mixture of ``n_components`` factorized endpoint distributions.

    Why a mixture: in a deterministic closed loop, failure endpoints lie
    scattered over the state-bound walls where rollouts terminate, so a single
    Gaussian per weight draw puts its mass between them, often back inside the
    box, and the final-state BNN of the GRAPE study scored near chance on the
    deterministic cells (sAUROC 0.58 to 0.77). A few components per draw can
    hold mass on several terminal regions at once.

    Each component is a full ``FinalStateHead`` distribution (Euclidean
    Gaussian, SO2 wrapped normal, SO3 tangent Gaussian, per manifold
    component), so every manifold rule of the single head carries over. The
    coordinate contract is the same: parameters live in NORMALIZED
    coordinates, public methods speak RAW ones.

    Parameter layout: ``[logits (M) | component 0 (P) | ... | component M-1 (P)]``
    with ``P = FinalStateHead.n_params``.

    The loss is the plain mixture NLL. beta-NLL is refused: it reweights each
    dimension's Gaussian term by a detached sigma^(2 beta), and inside the
    mixture's logsumexp there is no per-dimension term left to reweight, so any
    "beta" here would be a different, unnamed objective.
    """

    def __init__(self, system, n_components: int = 5):
        if int(n_components) < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.system = system
        self.n_components = int(n_components)
        self.beta = 0.0
        self.base = FinalStateHead(system, beta=0.0)
        self.n_base = self.base.n_params
        self.n_params = self.n_components + self.n_components * self.n_base
        self.state_dim = self.base.state_dim
        self.component_names = self.base.component_names

    def _split(self, params: torch.Tensor):
        m = self.n_components
        logits = params[..., :m]
        comp = params[..., m:].reshape(*params.shape[:-1], m, self.n_base)
        return logits, comp

    def _component_log_prob(self, comp: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
        """``[B, M]`` log density of a NORMALIZED target under each component."""
        b, m = comp.shape[0], self.n_components
        flat = comp.reshape(b * m, self.n_base)
        t = target_norm.unsqueeze(1).expand(b, m, target_norm.shape[-1]).reshape(b * m, -1)
        return -self.base._nll_normalized(flat, t, beta=0.0).view(b, m)

    def _joint(self, params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, comp = self._split(params)
        target_norm = self.system.normalize_state(target)
        return torch.log_softmax(logits, dim=-1) + self._component_log_prob(comp, target_norm)

    def nll(self, params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mixture NLL of a RAW target: [B]."""
        return -torch.logsumexp(self._joint(params, target), dim=-1)

    def responsibilities(self, params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Posterior component probabilities given a RAW target: [B, M]."""
        return torch.softmax(self._joint(params, target), dim=-1)

    def weights(self, params: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._split(params)[0], dim=-1)

    def sample(self, params: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """One draw in RAW coordinates: pick a component, then sample it."""
        logits, comp = self._split(params)
        probs = torch.softmax(logits.float(), dim=-1)
        idx = torch.multinomial(probs, 1, generator=generator)  # [B, 1]
        chosen = comp.gather(1, idx.unsqueeze(-1).expand(-1, 1, self.n_base)).squeeze(1)
        return self.base.sample(chosen, generator=generator)

    def mean(self, params: torch.Tensor) -> torch.Tensor:
        """Mean of the HEAVIEST component, RAW coordinates.

        Not the mixture mean: averaging endpoints across components on the
        manifolds (and across basins) gives a state no component predicts, which
        is exactly the failure the mixture exists to avoid. Callers that want
        every component use ``component_means``.
        """
        logits, comp = self._split(params)
        idx = logits.argmax(dim=-1, keepdim=True)
        chosen = comp.gather(1, idx.unsqueeze(-1).expand(-1, 1, self.n_base)).squeeze(1)
        return self.base.mean(chosen)

    def component_means(self, params: torch.Tensor) -> torch.Tensor:
        """``[B, M, state_dim]`` RAW means of every component."""
        _, comp = self._split(params)
        b, m = comp.shape[0], self.n_components
        return self.base.mean(comp.reshape(b * m, self.n_base)).view(b, m, -1)

    def distance_per_component(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.base.distance_per_component(a, b)

    def param_layout(self) -> dict:
        """Output rows of each part: logits, and per component its location and scale rows."""
        m = self.n_components
        locations, scales = [], []
        for c in range(m):
            off = m + c * self.n_base
            loc, sc = [], []
            for lik, p0, n_p, _s0, dim in self.base._parts:
                lo, sl = _location_and_scale_offsets(lik, off + p0, n_p, dim)
                loc.extend(lo)
                sc.extend(sl)
            locations.append(loc)
            scales.append(sc)
        return {"logits": list(range(m)), "locations": locations, "scales": scales}

    def laplace_chunk_stats(self, params: torch.Tensor, target: torch.Tensor) -> dict:
        """Per-row curvature weights for the block Laplace, from MAP params.

        ``logit_lam``: mean over components of pi_m (1 - pi_m), the diagonal of
        the softmax Fisher averaged over its M outputs, the analogue of the
        Bernoulli head's p (1 - p). ``resp``: responsibilities r_nm. ``sq_resid``:
        each component mean's mean squared NORMALIZED residual to the target.
        The trainer turns r_nm and sq_resid into each component's location block
        with lam_nm = r_nm / sigma_m^2, sigma_m^2 the responsibility-weighted
        mean of sq_resid: the EM form of the Gaussian GGN, homoscedastic within
        a component like the single head's RMS-residual sigma.
        """
        pi = self.weights(params)
        resp = self.responsibilities(params, target)
        means = self.component_means(params)                     # [B, M, d] raw
        b, m, d = means.shape
        t = target.unsqueeze(1).expand(b, m, d)
        dist = self.base.distance_per_component(
            self.system.normalize_state(means.reshape(b * m, d)),
            self.system.normalize_state(t.reshape(b * m, d)),
        )
        sq_resid = dist.pow(2).mean(dim=-1).view(b, m)
        return {"logit_lam": (pi * (1.0 - pi)).mean(dim=-1), "resp": resp, "sq_resid": sq_resid}


def build_head(system, fs_cfg=None):
    """The endpoint head a ``predictor.final_state`` block asks for.

    No ``head`` key, or ``head.kind: gaussian``, builds the single
    ``FinalStateHead`` exactly as every existing arm does. ``head.kind:
    mixture`` builds ``MixtureFinalStateHead`` and requires ``beta_nll: 0``.
    Trainer and export wrapper both call this, so they cannot build different
    heads from one config.
    """
    fs = fs_cfg if fs_cfg is not None else {}
    beta = float(fs.get("beta_nll", 0.5))
    head_cfg = fs.get("head", None)
    kind = "gaussian" if head_cfg is None else str(head_cfg.get("kind", "gaussian"))
    if kind == "gaussian":
        return FinalStateHead(system, beta=beta)
    if kind == "mixture":
        if beta != 0.0:
            raise ValueError(
                f"head.kind=mixture with beta_nll={beta}: beta-NLL is undefined for a "
                "mixture (see MixtureFinalStateHead). Set predictor.final_state.beta_nll: 0.0.")
        return MixtureFinalStateHead(system, n_components=int(head_cfg.get("n_components", 5)))
    raise ValueError(f"unknown head kind {kind!r}; expected 'gaussian' or 'mixture'")

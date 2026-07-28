"""Weight posteriors for Bayesian MLP arms.

A BNN predicts by marginalizing p(y|x,D) = int p(y|x,w) p(w|D) dw. These classes
supply the p(w|D) half: each exposes ``forward_sample`` (one draw from the
approximate posterior) and ``kl_divergence`` (zero for non-variational members).
The likelihood half lives in the heads.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class VILinear(nn.Module):
    """Linear layer with a mean-field Gaussian posterior over weights.

    Bayes-by-Backprop (Blundell et al., 2015): w = mu + softplus(rho) * eps.
    ``rho_init = -5.0`` gives an initial scale of softplus(-5) ~ 0.0067, small
    enough that early training behaves like a deterministic net and the KL term
    does not dominate before the likelihood has any signal.
    """

    def __init__(self, in_features: int, out_features: int,
                 prior_sigma: float = 1.0, rho_init: float = -5.0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.prior_sigma = float(prior_sigma)

        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), float(rho_init)))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), float(rho_init)))

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        w_sigma = F.softplus(self.weight_rho)
        b_sigma = F.softplus(self.bias_rho)
        w_eps = torch.randn(self.weight_mu.shape, generator=generator,
                            device=self.weight_mu.device, dtype=self.weight_mu.dtype)
        b_eps = torch.randn(self.bias_mu.shape, generator=generator,
                            device=self.bias_mu.device, dtype=self.bias_mu.dtype)
        return F.linear(x, self.weight_mu + w_sigma * w_eps, self.bias_mu + b_sigma * b_eps)

    def kl_divergence(self) -> torch.Tensor:
        """Closed-form KL(N(mu, sigma^2) || N(0, prior_sigma^2)), summed."""
        total = torch.zeros((), device=self.weight_mu.device, dtype=self.weight_mu.dtype)
        for mu, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            sigma = F.softplus(rho)
            total = total + (
                math.log(self.prior_sigma) - torch.log(sigma)
                + (sigma.pow(2) + mu.pow(2)) / (2.0 * self.prior_sigma ** 2)
                - 0.5
            ).sum()
        return total


class Posterior(nn.Module, ABC):
    """Uniform interface over approximate weight posteriors."""

    @abstractmethod
    def forward_sample(self, x: torch.Tensor,
                       generator: torch.Generator | None = None) -> torch.Tensor:
        """One draw from q(w): [B, in] -> [B, out]."""

    def forward(self, x: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """Delegate ``__call__`` to ``forward_sample``.

        Required so a Posterior can be nested inside another Posterior --
        EnsemblePosterior's members are DeterministicPosterior instances and are
        invoked as ``member(x)``.
        """
        return self.forward_sample(x, generator=generator)

    def forward_samples(self, x: torch.Tensor, S: int,
                        generator: torch.Generator | None = None) -> torch.Tensor:
        """S independent draws: [B, in] -> [S, B, out]."""
        return torch.stack([self.forward_sample(x, generator=generator) for _ in range(int(S))], dim=0)

    def predictive_logit_samples(self, x: torch.Tensor, S: int,
                                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Atoms to average over when marginalizing: [B, in] -> [K, B, out].

        The default is Monte Carlo -- ``K = S`` independent draws from q(w) --
        because for a continuous posterior no finite exact enumeration exists.
        Posteriors with a FINITE support override this to enumerate it exactly
        (see ``EnsemblePosterior``), in which case ``S`` is ignored and ``K`` is
        the support size. Callers must therefore not assume ``K == S``.
        """
        return self.forward_samples(x, S, generator=generator)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q||p); zero for non-variational posteriors."""
        return torch.zeros((), device=next(self.parameters()).device)


class DeterministicPosterior(Posterior):
    """Point estimate. Included so the deterministic arm shares one code path."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward_sample(self, x, generator=None):
        return self.net(x)


class MFVIPosterior(Posterior):
    """Mean-field variational inference over a net built from VILinear layers."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        if not any(isinstance(m, VILinear) for m in self.net.modules()):
            raise ValueError("MFVIPosterior requires a net containing VILinear layers")

    def forward_sample(self, x, generator=None):
        h = x
        for module in self.net:
            h = module(h, generator=generator) if isinstance(module, VILinear) else module(h)
        return h

    def kl_divergence(self):
        return sum(m.kl_divergence() for m in self.net.modules() if isinstance(m, VILinear))


class EnsemblePosterior(Posterior):
    """Deep ensemble: M independently trained members, drawn uniformly.

    Formally MAP inference rather than Bayesian inference (D'Angelo & Fortuin,
    2021), but empirically a closer match to the HMC predictive than mean-field
    VI (Izmailov et al., 2021), so it is carried as a baseline.

    ``forward_sample`` returns ONE member, giving an M-atom empirical posterior.
    Callers drawing K samples need K >= 2M to resolve it -- which is why the
    predictive marginal does NOT go through it: see
    ``predictive_logit_samples`` below, which enumerates all M members exactly.
    """

    def __init__(self, nets):
        super().__init__()
        nets = list(nets)
        if len(nets) < 2:
            raise ValueError(f"EnsemblePosterior needs >= 2 members, got {len(nets)}")
        self.members = nn.ModuleList(nets)

    @property
    def n_members(self) -> int:
        return len(self.members)

    def forward_sample(self, x, generator=None):
        device = next(self.parameters()).device
        idx = int(torch.randint(self.n_members, (1,), generator=generator,
                                device=device).item())
        return self.members[idx](x)

    def forward_all_members(self, x: torch.Tensor) -> torch.Tensor:
        """Every member, deterministically: [B, in] -> [M, B, out]."""
        return torch.stack([m(x) for m in self.members], dim=0)

    def predictive_logit_samples(self, x, S, generator=None):
        """Exact enumeration of the M-atom posterior; ``S`` is ignored.

        The ensemble marginal is NOT sampled. Its support is the M members with
        weight 1/M each, so averaging over ``forward_all_members`` computes
        sum_m p_m / M exactly. Sampling M atoms with replacement instead gives a
        multinomial weight vector -- and because the handle seeds its generator,
        that vector is FIXED for the whole run, so the error is a systematic
        bias rather than noise that averages out. At M=5, S=64 the seeded
        weights were [.125, .281, .109, .234, .250] against an exact .2, which
        is enough to flip decisions near lambda*. Enumeration is also ~S/M times
        cheaper.
        """
        return self.forward_all_members(x)


class LastLayerLaplacePosterior(Posterior):
    """Post-hoc Gaussian over last-layer weights via the GGN.

    H = sum_n Lambda_n * phi_n phi_n^T + prior_precision * I, with
    Lambda = p(1-p) for a Bernoulli head and Lambda = 1/sigma^2 for a Gaussian
    head. The last-layer restriction keeps H small and PSD.

    ``sigma`` is the observation noise and is REQUIRED for regression. Letting
    it default to 1.0 (the laplace-torch default) silently scales the entire
    posterior covariance by an arbitrary constant unrelated to the data.
    """

    def __init__(self, body: nn.Module, head_layer: nn.Linear, prior_precision: float = 1.0):
        super().__init__()
        self.body = body
        self.head_layer = head_layer
        self.prior_precision = float(prior_precision)
        self.register_buffer("_cov", torch.empty(0), persistent=False)

    @property
    def posterior_covariance(self) -> torch.Tensor:
        if self._cov.numel() == 0:
            raise RuntimeError("LastLayerLaplacePosterior.fit has not been called")
        return self._cov

    @property
    def is_fitted(self) -> bool:
        return self._cov.numel() > 0

    def fit(self, features: torch.Tensor, targets: torch.Tensor,
            task: str, sigma: float | None = None) -> "LastLayerLaplacePosterior":
        """Fit the GGN over last-layer weights. ``features`` are body outputs."""
        phi = features.detach()
        phi = torch.cat([phi, torch.ones(phi.shape[0], 1, dtype=phi.dtype, device=phi.device)], dim=1)

        if task == "outcome":
            with torch.no_grad():
                p = torch.sigmoid(self.head_layer(features.detach()).view(-1))
            lam = (p * (1.0 - p)).clamp_min(1e-6)
        elif task == "final_state":
            if sigma is None:
                raise ValueError(
                    "final_state Laplace requires an explicit observation noise "
                    "sigma; defaulting it to 1.0 would scale the posterior "
                    "covariance by an arbitrary constant."
                )
            lam = torch.full((phi.shape[0],), 1.0 / float(sigma) ** 2,
                             dtype=phi.dtype, device=phi.device)
        else:
            raise ValueError(f"unknown task {task!r}; expected 'outcome' or 'final_state'")

        H = torch.einsum("n,ni,nj->ij", lam, phi, phi)
        H = H + self.prior_precision * torch.eye(phi.shape[1], dtype=phi.dtype, device=phi.device)
        cov = torch.linalg.inv(H)
        self._cov = 0.5 * (cov + cov.T)  # symmetrize away round-off
        return self

    def forward_sample(self, x, generator=None):
        features = self.body(x)
        if not self.is_fitted:
            return self.head_layer(features)  # MAP fallback before fit
        phi = torch.cat(
            [features, torch.ones(features.shape[0], 1, dtype=features.dtype, device=features.device)],
            dim=1,
        )
        map_w = torch.cat([self.head_layer.weight, self.head_layer.bias.unsqueeze(1)], dim=1)
        # Key off the FEATURE device/dtype like every other posterior does; a
        # dtype-only `.to` leaves the covariance on whichever device it was
        # loaded on (torch.load defaults to CPU) after a `.to("cuda")`.
        L = torch.linalg.cholesky(self._cov.to(device=features.device, dtype=features.dtype))
        eps = torch.randn(map_w.shape[0], L.shape[0], generator=generator,
                          device=features.device, dtype=features.dtype)
        w = map_w + eps @ L.T
        return phi @ w.T

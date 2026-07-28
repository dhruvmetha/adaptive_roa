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

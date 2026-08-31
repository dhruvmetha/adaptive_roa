"""Flat-vector view of a network plus its differentiable log-posterior.

HMC operates on a single flat parameter vector, so this maps between that view
and the network's parameters and supplies log p(theta | D) with gradients.

Two deliberate omissions, both load-bearing for the reference tier:

* No ``pos_weight``. The production outcome arms reweight the BCE by class
  frequency, which tempers the likelihood per class. A reference posterior must
  be the posterior the arms approximate, and it must actually BE a posterior, so
  this tier targets the unweighted likelihood and the arms are configured to
  match.
* ``beta = 0`` on the final-state head. beta-NLL multiplies each dimension by a
  DETACHED sigma^(2*beta); it is an optimization aid, not a log-likelihood, and
  no posterior corresponds to it.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class FlatLogPosterior:
    """log p(theta | D) for one predictor head, as a function of a flat vector."""

    def __init__(self, net, head=None, prior_sigma: float = 1.0):
        self.net = net
        self.head = head
        self.prior_sigma = float(prior_sigma)
        self._params = [p for p in net.parameters() if p.requires_grad]
        self.dim = int(sum(p.numel() for p in self._params))

    def get_flat(self) -> torch.Tensor:
        return parameters_to_vector(self._params).detach().clone()

    def set_flat(self, theta: torch.Tensor) -> None:
        vector_to_parameters(theta.detach().to(self._params[0].dtype), self._params)

    def _log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        s = self.prior_sigma
        return (-0.5 * (theta / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)).sum()

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Scalar log p(theta | D), differentiable w.r.t. ``theta``.

        Uses functional parameter injection rather than writing into the module,
        so autograd reaches ``theta`` instead of the module's own leaf tensors --
        and so the module is never mutated, which matters because HMC evaluates
        thousands of candidate vectors against one shared net.
        """
        replacements = {}
        offset = 0
        for name, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            replacements[name] = theta[offset:offset + n].view_as(p)
            offset += n
        out = torch.func.functional_call(self.net, replacements, (x,))
        if self.head is None:
            # Outcome head: Bernoulli. No pos_weight -- see the module docstring.
            ll = -F.binary_cross_entropy_with_logits(
                out.view(-1), y.view(-1).to(out.dtype), reduction="sum"
            )
        else:
            # Final-state head: the manifold likelihood at beta = 0.
            ll = -self.head.nll(out, y).sum()
        return ll + self._log_prior(theta)

    def grad_log_prob(self, theta: torch.Tensor, x: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
        t = theta.detach().clone().requires_grad_(True)
        self.log_prob(t, x, y).backward()
        return t.grad.detach().clone()

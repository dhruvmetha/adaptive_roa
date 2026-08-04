"""A set of HMC draws, exposed through the shared Posterior interface.

Structurally this is the same object as EnsemblePosterior -- a finite support of
weight vectors -- so it plugs into both shipped model handles unchanged. It
follows the ensemble's precedent on the marginal too: enumerate the support
exactly rather than sampling it with replacement, so the predictive is not a
function of the caller's S.
"""
from __future__ import annotations

import torch

from adaptive_roa.predictors.posteriors import Posterior


class HMCPosterior(Posterior):
    def __init__(self, net, samples: torch.Tensor):
        super().__init__()
        if samples.ndim != 2 or samples.shape[0] < 1:
            raise ValueError(
                f"HMCPosterior needs at least 1 draw shaped [n_draws, dim], "
                f"got {tuple(samples.shape)}"
            )
        self.net = net
        self.register_buffer("samples", samples.detach().clone())

    @property
    def n_draws(self) -> int:
        return int(self.samples.shape[0])

    def _forward_with(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        replacements = {}
        offset = 0
        for name, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            replacements[name] = theta[offset:offset + n].view_as(p)
            offset += n
        return torch.func.functional_call(self.net, replacements, (x,))

    def forward_sample(self, x, generator=None):
        # Draw the index on the PARAMETER device, matching EnsemblePosterior
        # (posteriors.py) and the generator OutcomeModelHandle hands down
        # (handles.py, torch.Generator(device=param_device)). A CPU-only draw
        # here would raise a device-mismatch RuntimeError against a CUDA
        # generator on a CUDA model.
        device = next(self.parameters()).device
        idx = int(torch.randint(self.n_draws, (1,), generator=generator,
                                device=device).item())
        with torch.no_grad():
            return self._forward_with(self.samples[idx], x)

    def predictive_logit_samples(self, x, S, generator=None):
        """Enumerate the full support: [n_draws, B, out]. ``S`` is ignored."""
        with torch.no_grad():
            return torch.stack([self._forward_with(t, x) for t in self.samples], dim=0)

"""Probability backends that expose PER-MEMBER predictions, not just the marginal.

`estimate()` keeps the existing contract and returns the ensemble marginal, so
threshold, calibration and eval code needs no changes. `estimate_members()` is
the new surface the decomposition strategy needs: without per-member values the
aleatoric and epistemic parts cannot be separated at all.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from adaptive_roa.adaptive_v2.types import OutcomeProbabilities


class _EnsembleBackendBase:
    def __init__(self, cfg: Any, system: Any, device: str):
        self.attractor_radius = float(cfg.attractor_radius)
        self.system = system
        self.device = device
        self.model_handle: Any = None
        self.n_members: int = 0

    def bind_model(self, model_handle: Any) -> None:
        self.model_handle = model_handle
        n = int(getattr(self._posterior(), "n_members", 0))
        if n < 2:
            raise ValueError(
                f"ensemble backend needs at least 2 members, got {n}. A 1-member "
                "'ensemble' has no epistemic signal and would silently score 0."
            )
        self.n_members = n

    def _posterior(self) -> Any:
        if self.model_handle is None:
            raise RuntimeError("ensemble backend used before bind_model")
        return getattr(self.model_handle, "posterior", self.model_handle)

    def estimate(self, start_states: np.ndarray) -> OutcomeProbabilities:
        """Ensemble marginal. p_invalid is 0 for both backends' label space."""
        p = self.estimate_members(start_states).mean(axis=0)
        return OutcomeProbabilities(
            p_success=p, p_failure=1.0 - p, p_invalid=np.zeros_like(p),
        )


class EnsembleClassifierProbabilityBackend(_EnsembleBackendBase):
    """Per-member probabilities from one exact forward pass per member.

    No sampling anywhere, so `member_sample_size` is None and BALD is unbiased.
    """

    member_sample_size: int | None = None

    @torch.no_grad()
    def estimate_members(self, start_states: np.ndarray, verbose: bool = False) -> np.ndarray:
        post = self._posterior()
        x = torch.as_tensor(np.asarray(start_states), dtype=torch.float32,
                            device=self.device)
        embedded = self.system.embed_state_for_model(self.system.normalize_state(x))
        logits = post.forward_all_members(embedded)    # [M, N, 1]
        p = torch.sigmoid(logits).squeeze(-1)          # [M, N]
        return p.detach().cpu().numpy().astype(np.float64)


class EnsembleEndpointMCProbabilityBackend(_EnsembleBackendBase):
    """Per-member probabilities from K endpoint samples per member.

    Each p_m is a K-sample binomial estimate, so it carries sampling noise;
    `member_sample_size` reports K so the epistemic score can debias it.
    """

    def __init__(self, cfg: Any, system: Any, device: str):
        super().__init__(cfg, system, device)
        self.num_mc_samples = int(cfg.num_mc_samples)

    @property
    def member_sample_size(self) -> int:
        return self.num_mc_samples

    @torch.no_grad()
    def estimate_members(self, start_states: np.ndarray, verbose: bool = False) -> np.ndarray:
        handle = self.model_handle
        if handle is None:
            raise RuntimeError("ensemble backend used before bind_model")
        x = torch.as_tensor(np.asarray(start_states), dtype=torch.float32,
                            device=self.device)
        out = np.empty((self.n_members, x.shape[0]), dtype=np.float64)
        for m in range(self.n_members):
            hits = torch.zeros(x.shape[0], dtype=torch.float64)
            for _ in range(self.num_mc_samples):
                pred = handle.predict_endpoint_member(m, x)
                lab = self.system.classify_attractor(pred, self.attractor_radius)
                hits += (lab == 1).double().cpu()
            out[m] = (hits / float(self.num_mc_samples)).numpy()
        return out


class SampledPosteriorClassifierProbabilityBackend(_EnsembleBackendBase):
    """Per-atom probabilities from S draws of a CONTINUOUS weight posterior.

    MFVI and Laplace approximate q(w) as a Gaussian, so there is no finite member
    set to enumerate: `n_members` is undefined on those posteriors and
    EnsembleClassifierProbabilityBackend rejects them outright. The Gaussian IS
    the posterior, so the entropies BALD needs are computed by averaging over
    draws from it -- which is what `Posterior.predictive_logit_samples` exists
    for (its docstring: Monte Carlo for a continuous posterior, exact
    enumeration for one with finite support).

    Bias, stated so it is not invisible. BALD = H[E_w p] - E_w H[p]; with S atoms
    the first term carries the usual O(1/S) Monte-Carlo bias, downward, so BALD
    is UNDERESTIMATED at small S and the ranking compresses. S=64 is the default.
    `member_sample_size` stays None on purpose: that field debiases BINOMIAL
    noise WITHIN a member (the endpoint-MC backend's K rollouts), and this
    backend has none -- each atom is an exact probability for its weight draw.

    An EnsemblePosterior passed here still works and ignores S, because it
    overrides predictive_logit_samples to enumerate its members exactly.
    """

    member_sample_size: int | None = None

    def __init__(self, cfg: Any, system: Any, device: str):
        super().__init__(cfg, system, device)
        self.n_samples = int(getattr(cfg, "n_posterior_samples", 64))
        if self.n_samples < 2:
            raise ValueError(
                f"n_posterior_samples={self.n_samples} gives no epistemic signal; "
                "BALD would score 0 everywhere. Use >= 2 (default 64)."
            )

    def bind_model(self, model_handle: Any) -> None:
        # Deliberately does NOT call super(): the base asserts a finite
        # `n_members`, which a Gaussian posterior does not have.
        self.model_handle = model_handle
        post = self._posterior()
        if not hasattr(post, "predictive_logit_samples"):
            raise TypeError(
                f"{type(post).__name__} has no predictive_logit_samples(); a BALD "
                "arm needs a posterior it can draw predictive atoms from."
            )
        self.n_members = self.n_samples

    @torch.no_grad()
    def estimate_members(self, start_states: np.ndarray, verbose: bool = False) -> np.ndarray:
        post = self._posterior()
        x = torch.as_tensor(np.asarray(start_states), dtype=torch.float32,
                            device=self.device)
        embedded = self.system.embed_state_for_model(self.system.normalize_state(x))
        logits = post.predictive_logit_samples(embedded, self.n_samples)  # [K, N, 1]
        p = torch.sigmoid(logits).squeeze(-1)                             # [K, N]
        # K is what the posterior actually returned, which is S for a Gaussian
        # but the member count for one that enumerates. Record the truth.
        self.n_members = int(p.shape[0])
        return p.detach().cpu().numpy().astype(np.float64)

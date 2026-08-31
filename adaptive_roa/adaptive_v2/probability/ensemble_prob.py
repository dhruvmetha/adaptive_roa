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

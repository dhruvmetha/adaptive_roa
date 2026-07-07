"""Pluggable T-step dynamics-model interface used by the verifier.

Two backends implement this: a deterministic regressor (``predict``) and a
generative flow-matching model (``sample``). The verifier consumes either
through this interface. Everything operates in **raw state space** so the
verifier can apply ``system.classify_attractor`` directly to rollout states.
"""
from __future__ import annotations

import torch


class DynamicsModel:
    """Maps a raw state ``x`` to its state ``T`` steps later."""

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic T-step map: ``[B, state_dim] -> [B, state_dim]`` (raw)."""
        raise NotImplementedError

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Stochastic T-step map: ``[B, state_dim] -> [num_samples, B, state_dim]``.

        Default repeats ``predict`` (a deterministic model has no spread); the
        generative backend overrides this with genuine sampling.
        """
        return torch.stack([self.predict(x) for _ in range(num_samples)], dim=0)

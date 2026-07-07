"""Autoregressive rollout / reachability verification.

Roll a T-step dynamics model forward from a query state, checking the absorbing
success/failure sets (``system.classify_attractor``: +1 success, -1 failure,
0 unresolved) after the initial state and each of K steps. Resolve on the first
non-zero label (early stop / absorption); if K steps elapse with only zeros the
query is **unresolved**.

- Deterministic backend: one rollout -> a label in {+1, -1, 0}.
- Generative backend: N stochastic rollouts -> p(success)/p(failure)/p(unresolved).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


def _classify(system, state: torch.Tensor, radius: Optional[float]) -> torch.Tensor:
    if radius is None:
        return system.classify_attractor(state)
    return system.classify_attractor(state, radius=radius)


def _rollout(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    system,
    x0: torch.Tensor,
    K: int,
    radius: Optional[float],
):
    """Roll ``x0`` [M, D] forward via ``step_fn``.

    Returns ``(labels, final_state)`` where ``labels`` [M] is the absorbing
    outcome and ``final_state`` [M, D] is each row's state at resolution
    (frozen thereafter) or at step K if never resolved.
    """
    batch = x0.shape[0]
    labels = torch.zeros(batch, dtype=torch.long, device=x0.device)
    resolved = torch.zeros(batch, dtype=torch.bool, device=x0.device)

    def absorb(state: torch.Tensor) -> None:
        cls = _classify(system, state, radius)
        newly = (~resolved) & (cls != 0)
        labels[newly] = cls[newly]
        resolved[newly] = True

    x = x0
    absorb(x)  # a query already in an absorbing set resolves immediately
    for _ in range(K):
        if bool(resolved.all()):
            break
        x_next = step_fn(x)
        # Freeze already-resolved rows so diverged predictions can't contaminate them.
        x = torch.where(resolved.unsqueeze(1), x, x_next)
        absorb(x)
    return labels, x


def resolve_outcome(model, system, x0: torch.Tensor, K: int, radius: Optional[float] = None) -> torch.Tensor:
    """Deterministic rollout: labels [B] in {+1 success, -1 failure, 0 unresolved}."""
    labels, _ = _rollout(model.predict, system, x0, K, radius)
    return labels


def rollout_final_state(
    model, system, x0: torch.Tensor, K: int, radius: Optional[float] = None
) -> torch.Tensor:
    """Deterministic rollout: the resolved/terminal state [B, D] (frozen on absorption)."""
    _, final = _rollout(model.predict, system, x0, K, radius)
    return final


def resolve_probabilistic(
    model,
    system,
    x0: torch.Tensor,
    K: int,
    num_samples: int,
    radius: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """N stochastic rollouts per query -> per-query resolution probabilities."""
    batch = x0.shape[0]
    x_rep = x0.repeat_interleave(num_samples, dim=0)
    step_fn = lambda x: model.sample(x, 1)[0]  # one stochastic draw per step
    labels, _ = _rollout(step_fn, system, x_rep, K, radius)
    labels = labels.view(batch, num_samples)
    return {
        "p_success": (labels == 1).float().mean(dim=1),
        "p_failure": (labels == -1).float().mean(dim=1),
        "p_unresolved": (labels == 0).float().mean(dim=1),
    }

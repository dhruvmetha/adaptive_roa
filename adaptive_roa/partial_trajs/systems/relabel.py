"""Verifier system subclasses for systems with no failure set.

Pendulum and humanoid datasets have no failure criterion: a non-success
trajectory times out and is *unresolved*, not *failed*. Their base
``classify_attractor`` still returns ``-1`` (pendulum: top equilibria; humanoid:
all non-success), which would make the verifier early-stop on a spurious
absorbing failure. These subclasses carry **no criteria logic** — they only
remap ``-1 -> 0`` so non-success states are unresolved.

Cartpole / quad2D / quad3D have genuine out-of-bounds failure sets and use their
base classes directly.
"""
from __future__ import annotations

import torch

from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.systems.humanoid_standup_reach import HumanoidStandUpReachSystem


class _UnresolvedFailureMixin:
    """Remap the base classifier's failure label (-1) to unresolved (0)."""

    def classify_attractor(self, *args, **kwargs) -> torch.Tensor:
        labels = super().classify_attractor(*args, **kwargs).clone()
        labels[labels == -1] = 0
        return labels


class PartialTrajPendulumSystem(_UnresolvedFailureMixin, PendulumSystem):
    pass


class PartialTrajHumanoidStandUpReachSystem(
    _UnresolvedFailureMixin, HumanoidStandUpReachSystem
):
    pass

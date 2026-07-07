"""Verifier system subclasses + factory for the partial-trajectory pipeline.

Pendulum and humanoid use relabel-subclasses (no failure set -> unresolved);
cartpole / quad2D / quad3D use their base classes (genuine OOB failure sets).
"""
from typing import Optional

from adaptive_roa.systems.cartpole import CartPoleSystem
from adaptive_roa.systems.quadrotor2d import Quadrotor2DSystem
from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem
from adaptive_roa.partial_trajs.systems.relabel import (
    PartialTrajPendulumSystem,
    PartialTrajHumanoidStandUpReachSystem,
)

__all__ = [
    "PartialTrajPendulumSystem",
    "PartialTrajHumanoidStandUpReachSystem",
    "make_verifier_system",
]

_VERIFIER_SYSTEMS = {
    "pendulum": PartialTrajPendulumSystem,
    "cartpole": CartPoleSystem,
    "quadrotor2d": Quadrotor2DSystem,
    "quadrotor3d": Quadrotor3DSystem,
    "humanoid_standup_reach": PartialTrajHumanoidStandUpReachSystem,
}


def make_verifier_system(name: str, dataset_dir: Optional[str] = None):
    """Build the verifier system for ``name`` (relabel-subclass where needed)."""
    if name not in _VERIFIER_SYSTEMS:
        raise ValueError(
            f"unknown system {name!r}; choices={sorted(_VERIFIER_SYSTEMS)}"
        )
    return _VERIFIER_SYSTEMS[name](dataset_dir=dataset_dir)

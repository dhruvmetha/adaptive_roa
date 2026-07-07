"""Tests for the verifier relabel-subclasses (no failure set -> unresolved)."""
from pathlib import Path

import pytest
import torch

from adaptive_roa.partial_trajs.systems import (
    PartialTrajPendulumSystem,
    PartialTrajHumanoidStandUpReachSystem,
)

PENDULUM_DATA = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum_lqr_50k"
)
HUMANOID_DATA = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/humanoid_get_up_medium"
)


@pytest.mark.skipif(not PENDULUM_DATA.exists(), reason="pendulum bounds not available")
def test_pendulum_top_equilibrium_becomes_unresolved():
    sys = PartialTrajPendulumSystem()
    # [0,0] success; [2.1,0] is a base FAILURE (-1) -> must become 0; separatrix stays 0
    states = torch.tensor([[0.0, 0.0], [2.1, 0.0], [1.0, 0.0]])
    labels = sys.classify_attractor(states)
    assert labels[0].item() == 1  # success preserved
    assert labels[1].item() == 0  # former failure -> unresolved
    assert (labels != -1).all()  # no failure label at all


@pytest.mark.skipif(not HUMANOID_DATA.exists(), reason="humanoid bounds not available")
def test_humanoid_nonsuccess_becomes_unresolved():
    sys = PartialTrajHumanoidStandUpReachSystem(dataset_dir=str(HUMANOID_DATA))
    succ = torch.zeros(1, 67)
    succ[0, 21] = 1.25  # head >= 1.2, CoM speed 0 -> success
    fail = torch.zeros(1, 67)
    fail[0, 21] = 0.5  # not success -> base -1 -> must become 0
    labels = sys.classify_attractor(torch.cat([succ, fail], dim=0))
    assert labels[0].item() == 1
    assert labels[1].item() == 0
    assert (labels != -1).all()

"""The partition root must contain every state the acquisition will ever score.

PartitionTree.assign returns -1 for a state outside every leaf, -1 is never in
the unresolved set, and select_pool_indices only considers unresolved leaves. So
any state outside the root box is permanently ineligible for acquisition. When
build_root read component-level state_bounds, that silently excluded 91% of the
quadrotor2D state space and 43% of quadrotor3D's, and the partx arms under-spent
their budget or never acquired at all.

Skipped when the shared dataset tree is not mounted.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

from adaptive_roa.partx.tree import build_root

DATA = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/stochastic")

CASES = [
    ("quadrotor2d", "Quadrotor2DSystem",
     DATA / "quadrotor2D/corridor_sine_ambient/rl/smooth"),
    ("quadrotor3d", "Quadrotor3DSystem",
     DATA / "quadrotor3D/corridor_sine_ambient/lqr/f_0.30"),
]


@pytest.mark.parametrize("module,cls_name,dataset", CASES)
def test_root_contains_entire_eval_grid(module, cls_name, dataset):
    grid = dataset / "eval_success_prob.npz"
    if not grid.exists():
        pytest.skip(f"dataset not mounted: {dataset}")
    mod = __import__(f"adaptive_roa.systems.{module}", fromlist=[cls_name])
    with contextlib.redirect_stdout(io.StringIO()):
        system = getattr(mod, cls_name)(dataset_dir=str(dataset))
        root = build_root(system)
    states = np.load(grid)["starts"].astype(float)
    inside = root.contains(states)
    outside = int((~inside).sum())
    assert outside == 0, (
        f"{cls_name}: {outside}/{len(states)} eval states fall outside the "
        f"partition root and can never be acquired"
    )


def test_pendulum_root_is_unchanged_by_per_dim_bounds():
    # Every pendulum component is one-dimensional, so per_dim_bounds must agree
    # with the legacy component expansion exactly. Pendulum was the only system
    # spending 100% of its acquisition budget; the fix must not perturb it.
    from adaptive_roa.systems.pendulum import PendulumSystem

    with contextlib.redirect_stdout(io.StringIO()):
        system = PendulumSystem()
        root = build_root(system, pad_frac=0.0)
    legacy_low, legacy_high = [], []
    for comp in system.manifold_components:
        dim = int(getattr(comp, "dim", 1))
        if comp.manifold_type == "SO2":
            lo, hi = -np.pi, np.pi
        else:
            b = system.state_bounds[comp.name]
            lo, hi = float(b[0]), float(b[1])
        legacy_low.extend([lo] * dim)
        legacy_high.extend([hi] * dim)
    assert np.allclose(root.low, legacy_low)
    assert np.allclose(root.high, legacy_high)

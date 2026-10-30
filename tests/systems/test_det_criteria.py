"""Success and failure criteria come from config for the det-paper systems.

963f4e3 made CartPoleSystem gate failure at the largest overshoot any stored
trajectory reached. On the deterministic LQR set that reads every true failure
endpoint as invalid. Det configs now pass the collector's own criteria plus the
D12 buffer; with no criteria arguments both classes keep their old behaviour,
which GRAPE's stochastic configs rely on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from adaptive_roa.systems.cartpole import CartPoleSystem
from adaptive_roa.systems.pendulum import PendulumSystem

DET = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic")
CARTPOLE_DIR = DET / "cartpole_pybullet"
PENDULUM_DIR = DET / "pendulum" / "lqr"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2"

needs_cartpole = pytest.mark.skipif(not (CARTPOLE_DIR / "dataset_description.json").exists(),
                                    reason="det cartpole dataset not mounted")
needs_pendulum = pytest.mark.skipif(not (PENDULUM_DIR / "dataset_description.json").exists(),
                                    reason="det pendulum dataset not mounted")

DET_CARTPOLE = dict(
    failure_thresholds={"x": 6.0, "x_dot": 5.0, "theta_dot": 5.0},
    success_tolerances={"x": 0.01, "theta": 0.05, "x_dot": 0.05, "theta_dot": 0.05},
    criteria_buffer=0.1,
)


def _label(system, state, radius):
    return int(system.classify_attractor(torch.tensor([state], dtype=torch.float32), radius=radius)[0])


@needs_cartpole
def test_cartpole_default_still_gates_at_the_achieved_maximum():
    sys = CartPoleSystem(dataset_dir=str(CARTPOLE_DIR))
    assert _label(sys, [0.0, 0.0, 7.0, 0.0], 0.2) == 0     # past 5 but below 7.338
    assert _label(sys, [0.0, 0.0, 7.4, 0.0], 0.2) == -1


@needs_cartpole
def test_cartpole_det_criteria_use_collector_thresholds_minus_buffer():
    sys = CartPoleSystem(dataset_dir=str(CARTPOLE_DIR), **DET_CARTPOLE)
    assert _label(sys, [0.0, 0.0, 4.95, 0.0], 0.2) == -1
    assert _label(sys, [0.0, 0.0, 4.85, 0.0], 0.2) == 0
    assert _label(sys, [5.95, 0.0, 0.0, 0.0], 0.2) == -1


@needs_cartpole
def test_cartpole_det_success_is_the_buffered_box():
    sys = CartPoleSystem(dataset_dir=str(CARTPOLE_DIR), **DET_CARTPOLE)
    assert _label(sys, [0.10, 0.14, 0.14, -0.14], 0.2) == 1
    assert _label(sys, [0.12, 0.0, 0.0, 0.0], 0.2) == 0
    assert _label(sys, [0.0, 2 * np.pi + 0.1, 0.0, 0.0], 0.2) == 1   # theta wraps


@needs_cartpole
def test_cartpole_det_criteria_reproduce_every_stored_test_label():
    import pandas as pd
    t = pd.read_csv(CARTPOLE_DIR / "test_set.txt", header=None).values
    sys = CartPoleSystem(dataset_dir=str(CARTPOLE_DIR), **DET_CARTPOLE)
    pred = sys.classify_attractor(torch.tensor(t[:, 4:8], dtype=torch.float32), radius=0.2).numpy()
    y = t[:, 8].astype(int)
    assert np.all(pred[y == 0] == -1)
    assert np.all(pred[y == 1] == 1)


@needs_cartpole
def test_cartpole_refuses_a_buffer_that_makes_the_sets_meet():
    bad = dict(DET_CARTPOLE, criteria_buffer=4.9)
    with pytest.raises(ValueError, match="intersect"):
        CartPoleSystem(dataset_dir=str(CARTPOLE_DIR), **bad)


@needs_pendulum
def test_pendulum_default_uses_one_radius_for_every_attractor():
    sys = PendulumSystem(dataset_dir=str(PENDULUM_DIR))
    assert _label(sys, [2.1, 0.0], 0.1) == -1
    assert _label(sys, [2.25, 0.0], 0.1) == 0


@needs_pendulum
def test_pendulum_failure_radius_and_centre_from_config():
    sys = PendulumSystem(dataset_dir=str(PENDULUM_DIR), failure_radius=0.15, failure_theta=2.0957)
    assert _label(sys, [2.0957 + 0.12, 0.0], 0.1) == -1
    assert _label(sys, [-2.0957, 0.14], 0.1) == -1
    assert _label(sys, [0.09, 0.0], 0.1) == 1
    assert _label(sys, [0.12, 0.0], 0.1) == 0


@needs_pendulum
def test_pendulum_refuses_balls_that_meet():
    sys = PendulumSystem(dataset_dir=str(PENDULUM_DIR), failure_radius=1.5, failure_theta=2.0957)
    with pytest.raises(ValueError, match="intersect"):
        _label(sys, [0.0, 0.0], 1.0)


def _compose(system):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="default", overrides=[f"system={system}"])


def test_det_cartpole_config_carries_the_criteria():
    cfg = _compose("cartpole_pybullet")
    assert dict(cfg.system.failure_thresholds) == DET_CARTPOLE["failure_thresholds"]
    assert dict(cfg.system.success_tolerances) == DET_CARTPOLE["success_tolerances"]
    assert cfg.system.criteria_buffer == 0.1


def test_grape_cartpole_config_does_not():
    cfg = _compose("cartpole_stoch")
    assert "failure_thresholds" not in cfg.system
    assert "success_tolerances" not in cfg.system


def test_det_pendulum_and_quad_radii():
    pen = _compose("pendulum")
    assert pen.attractor_radius == 0.1
    assert pen.system.failure_radius == 0.15
    assert _compose("quadrotor2d").attractor_radius == 0.3
    assert _compose("quadrotor3d").attractor_radius == 0.15

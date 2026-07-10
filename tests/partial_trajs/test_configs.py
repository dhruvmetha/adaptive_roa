"""Composition tests for the partial_trajs Hydra config groups.

Verify each system/backend group composes into the flat keys ``train.run`` reads,
with values mirroring the adaptive experiments. Pure config — no dataset needed.
"""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

# Importing train registers the ``shared_data_base`` OmegaConf resolver used in
# the system configs' ``dataset_dir`` interpolation.
import adaptive_roa.partial_trajs.train  # noqa: F401

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "partial_trajs")


def _compose(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="train_partial_trajs", overrides=list(overrides))


def test_default_is_pendulum_generative():
    cfg = _compose([])
    assert cfg.system == "pendulum"
    assert cfg.backend == "generative"
    assert list(cfg.hidden_dims) == [256, 512, 256]
    assert cfg.attractor_radius == 0.1
    assert cfg.max_epochs == 1000
    assert cfg.num_integration_steps == 100
    assert cfg.eval_split_file == "test_set.txt"
    assert str(cfg.dataset_dir).endswith("pendulum_lqr_50k_T25")


@pytest.mark.parametrize(
    "system,hidden,radius,folder,evalfile",
    [
        ("cartpole", [256, 512, 1024, 512, 256], 0.2, "cartpole_pybullet_T25", "test_set.txt"),
        ("quadrotor2d", [256, 512, 1024, 512, 256], 0.3, "quadrotor2D_rl_T25", "test_set.txt"),
        ("quadrotor3d", [512, 1024, 1024, 512], 0.3, "quadrotor3D_lqr_T25", "test_set.txt"),
        (
            "humanoid_standup_reach",
            [512, 1024, 1024, 512],
            1.0,
            "humanoid_get_up_medium_T25",
            "test_set_fps.txt",
        ),
    ],
)
def test_per_system_composition(system, hidden, radius, folder, evalfile):
    cfg = _compose([f"system={system}"])
    assert cfg.system == system
    assert list(cfg.hidden_dims) == hidden
    assert cfg.attractor_radius == radius
    assert cfg.eval_split_file == evalfile
    assert str(cfg.dataset_dir).endswith(folder)


def test_quad3d_and_humanoid_generative_lr_is_5e4():
    assert _compose(["system=quadrotor3d"]).lr == 5e-4
    assert _compose(["system=humanoid_standup_reach"]).lr == 5e-4


def test_pendulum_generative_lr_is_1e3():
    assert _compose([]).lr == 1e-3


def test_deterministic_lr_in_between_for_quad3d_humanoid():
    # deterministic q3d/humanoid uses a value between generative (5e-4) and the
    # classifier default (1e-3); NOT the generative 5e-4 leaking through.
    assert _compose(["backend=deterministic", "system=quadrotor3d"]).lr == 7.5e-4
    assert _compose(["backend=deterministic", "system=humanoid_standup_reach"]).lr == 7.5e-4


@pytest.mark.parametrize("system", ["pendulum", "cartpole", "quadrotor2d"])
def test_deterministic_lr_is_1e3_for_small_systems(system):
    assert _compose(["backend=deterministic", f"system={system}"]).lr == 1e-3


@pytest.mark.parametrize("system", ["pendulum", "cartpole", "quadrotor2d"])
def test_generative_lr_is_1e3_for_small_systems(system):
    assert _compose([f"system={system}"]).lr == 1e-3


def test_horizon_T_override_changes_dataset_dir():
    cfg = _compose(["system=cartpole", "horizon_T=50"])
    assert str(cfg.dataset_dir).endswith("cartpole_pybullet_T50")


def test_deterministic_backend_epochs():
    cfg = _compose(["backend=deterministic"])
    assert cfg.backend == "deterministic"
    assert cfg.max_epochs == 200


@pytest.mark.parametrize(
    "system,n",
    [
        ("pendulum", 500),
        ("cartpole", 1000),
        ("quadrotor2d", 12000),
        ("quadrotor3d", 25000),
        ("humanoid_standup_reach", 30000),
    ],
)
def test_max_trajectories_default_per_system(system, n):
    assert _compose([f"system={system}"]).max_trajectories == n

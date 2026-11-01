"""Each det system's success and failure criteria have one source: its system config.

The `system:` block of configs/adaptive_v2/system/<name>.yaml sets every criterion
(goal, success test, failure test) with require_criteria, the top-level
attractor_radius only points at it (null for cartpole's box), and a radius that
disagrees with the configured one raises instead of being used.
"""
import os
from pathlib import Path

import pytest
import torch

DATA = "/common/users/shared/pracsys/genMoPlan/data_trajectories"
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2")
pytestmark = pytest.mark.skipif(not os.path.isdir(f"{DATA}/deterministic"), reason="shared data not mounted")


def _system(name):
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from adaptive_roa.utils.env_config import get_data_dir, get_exp_dir, get_net_id, get_shared_data_base
    for key, fn in [("net_id", get_net_id), ("exp_dir", get_exp_dir), ("data_dir", get_data_dir),
                    ("shared_data_base", get_shared_data_base)]:
        if not OmegaConf.has_resolver(key):
            OmegaConf.register_new_resolver(key, (lambda f: (lambda d="": f() or d))(fn))
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default", overrides=[f"system={name}", "predictor=fm_det",
                                                         "+experiment=det_rerun", "output_dir=/tmp/unused"])
    return cfg, hydra.utils.instantiate(cfg.system)


def test_pendulum():
    cfg, s = _system("pendulum")
    assert cfg.system.require_criteria and cfg.attractor_radius == s.success_radius == 0.1
    assert s.failure_radius == 0.1 and s.failure_theta == pytest.approx(2.0946) and s.goal == [0.0, 0.0]
    pts = torch.tensor([[0.09, 0.0], [0.11, 0.0], [2.0946 + 0.09, 0.0], [-2.0946, 0.11], [1.0, 1.0]])
    assert s.classify_attractor(pts, cfg.attractor_radius).tolist() == [1, 0, -1, 0, 0]
    with pytest.raises(ValueError, match="one source"):
        s.classify_attractor(pts, 0.15)


def test_cartpole():
    cfg, s = _system("cartpole_pybullet")
    assert cfg.system.require_criteria and cfg.attractor_radius is None
    assert s.goal == [0.0, 0.0, 0.0, 0.0] and s.criteria_buffer == 0.1
    pts = torch.tensor([[0.10, 0.14, 0.14, 0.14],   # inside the box (0.11, 0.15, 0.15, 0.15)
                        [0.12, 0.0, 0.0, 0.0],      # x just outside the box: neither
                        [5.95, 0.0, 0.0, 0.0],      # |x| > 6.0 - 0.1: failure
                        [0.0, 3.0, 0.0, 4.95]])     # |theta_dot| > 5.0 - 0.1: failure
    assert s.classify_attractor(pts, cfg.attractor_radius).tolist() == [1, 0, -1, -1]
    with pytest.raises(ValueError, match="one source"):
        s.classify_attractor(pts, 0.2)


def test_cartpole_ppo():
    cfg, s = _system("cartpole_ppo")
    assert cfg.system.require_criteria and cfg.attractor_radius == s.success_radius == 0.15
    assert s.goal == [0.7, 0.0, 0.0, 0.0] and s.criteria_buffer == 0.1
    pts = torch.tensor([[0.80, 0.05, 0.05, 0.05],   # ||(0.1, 0.05, 0.05, 0.05)|| = 0.13 < 0.15: success
                        [0.86, 0.0, 0.0, 0.0],      # 0.16 from the goal, inside the walls: neither
                        [0.0, 0.40, 0.0, 0.0],      # |theta| > 0.48 - 0.1: failure (pole angle wall)
                        [2.95, 0.0, 0.0, 0.0],      # |x| > 3.0 - 0.1: failure
                        [0.0, 0.0, 0.0, 2.95]])     # |theta_dot| > 3.0 - 0.1: failure
    assert s.classify_attractor(pts, cfg.attractor_radius).tolist() == [1, 0, -1, -1, -1]
    with pytest.raises(ValueError, match="one source"):
        s.classify_attractor(pts, 0.2)
    assert cfg.initial_train_size == 500 and cfg.samples_per_epoch == 250 and cfg.horizon_steps == 150


def test_cartpole_refuses_ball_and_box_together():
    from adaptive_roa.systems.cartpole import CartPoleSystem
    with pytest.raises(ValueError, match="configure one"):
        CartPoleSystem(dataset_dir=f"{DATA}/deterministic/cartpole_pybullet", success_radius=0.15,
                       success_tolerances={"x": 0.01, "theta": 0.05, "x_dot": 0.05, "theta_dot": 0.05})


def test_quadrotor2d():
    cfg, s = _system("quadrotor2d")
    assert cfg.system.require_criteria and cfg.attractor_radius == s.success_radius == 0.3
    assert s.failure_bounds == {"x": 0.9, "z_min": 0.2, "z_max": 1.4, "x_dot": 0.9, "z_dot": 0.9, "theta_dot": 7.5}
    pts = torch.tensor([[0.29, 1.0, 0.0, 0.0, 0.0, 0.0], [0.31, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.19, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 7.6]])
    assert s.classify_attractor(pts, cfg.attractor_radius).tolist() == [1, 0, -1, -1]
    with pytest.raises(ValueError, match="one source"):
        s.classify_attractor(pts, 0.2)


def test_quadrotor3d():
    cfg, s = _system("quadrotor3d")
    assert cfg.system.require_criteria and cfg.attractor_radius == s.success_radius == 0.15
    assert s.termination_thresholds == {"x": 1.7, "y": 1.7, "z_min": 0.2, "z_max": 2.9, "vel": 2.9, "rate": 23.5}
    goal = torch.tensor(s.goal)
    pts = torch.stack([goal + torch.eye(13)[0] * 0.14, goal + torch.eye(13)[0] * 0.16,
                       goal + torch.eye(13)[0] * 1.75, goal + torch.eye(13)[12] * 23.6])
    assert s.classify_attractor(pts, cfg.attractor_radius).tolist() == [1, 0, -1, -1]
    with pytest.raises(ValueError, match="one source"):
        s.classify_attractor(pts, 0.3)


def test_quadrotor3d_rl():
    cfg, s = _system("quadrotor3d_rl")
    assert cfg.system.require_criteria and cfg.attractor_radius == s.success_radius == 0.15
    assert s.termination_thresholds == {"x": 1.7, "y": 1.7, "z_min": 0.2, "z_max": 2.9, "vel": 1.9, "rate": 7.5}
    goal = torch.tensor(s.goal)
    pts = torch.stack([goal + torch.eye(13)[0] * 0.14, goal + torch.eye(13)[0] * 0.16,
                       goal + torch.eye(13)[1] * 1.75, goal + torch.eye(13)[8] * 1.95,
                       goal + torch.eye(13)[12] * 7.6])
    assert s.classify_attractor(pts, cfg.attractor_radius).tolist() == [1, 0, -1, -1, -1]
    assert cfg.initial_train_size == 10000 and cfg.samples_per_epoch == 3000 and cfg.horizon_steps == 1000


def test_require_criteria_refuses_class_defaults():
    from adaptive_roa.systems.pendulum import PendulumSystem
    from adaptive_roa.systems.quadrotor2d import Quadrotor2DSystem
    with pytest.raises(ValueError, match="require_criteria"):
        PendulumSystem(dataset_dir=f"{DATA}/deterministic/pendulum/lqr_failset", failure_radius=0.1,
                       failure_theta=2.0946, require_criteria=True)          # no success_radius, goal
    with pytest.raises(ValueError, match="require_criteria"):
        Quadrotor2DSystem(dataset_dir=f"{DATA}/deterministic/quadrotor2D_rl", success_radius=0.3,
                          require_criteria=True)                             # no goal, failure_bounds


def test_unconfigured_callers_keep_their_old_defaults():
    # Checkpoint loaders and older scripts build systems for normalization only.
    from adaptive_roa.systems.quadrotor2d import Quadrotor2DSystem
    s = Quadrotor2DSystem(dataset_dir=f"{DATA}/deterministic/quadrotor2D_rl")
    assert s.success_radius is None
    pt = torch.tensor([[0.25, 1.0, 0.0, 0.0, 0.0, 0.0]])
    assert s.classify_attractor(pt).tolist() == [1] and s.classify_attractor(pt, 0.2).tolist() == [0]

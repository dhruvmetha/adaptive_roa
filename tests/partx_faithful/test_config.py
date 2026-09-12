import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from adaptive_roa.utils.env_config import get_data_dir

_CFGDIR = str(Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2")


def _compose(overrides):
    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
    with initialize_config_dir(config_dir=_CFGDIR, version_base=None):
        return compose(config_name="partx_faithful", overrides=overrides)


def test_config_composes_with_the_engine_overrides():
    cfg = _compose(["system=pendulum_stoch", "noise_level=low", "output_dir=/tmp/unused"])
    a = cfg.partx_faithful.algorithm
    assert a.n0 == "auto" and a.n_bo == 10 and a.n_c == 100 and a.R == 20 and a.M == 500
    assert a.B == 2 and a.delta_c == 0.05 and a.delta_v == 0.001 and a.ei_cand_cap == 20000
    assert not a.compat_ci_var and not a.compat_eq4_times_volume and a.ei_f_star == "plugin"
    g = cfg.partx_faithful.gp
    assert g.n_restarts == 5 and g.max_points == 2000
    assert list(g.lengthscale_bounds) == [0.01, 100.0] and list(g.sigma_f2_bounds) == [0.01, 25.0]
    assert cfg.partx_faithful.falsify == "failure"
    assert cfg.data_source.test_set_file.endswith("pendulum/gaussian_signal/lqr/low/test_set.txt")
    assert str(cfg.data_source.pool_format) == "npz"


@pytest.mark.parametrize("overrides", [
    ["system=cartpole_stoch", "controller=safe_explorer_ppo", "noise_level=med", "output_dir=/tmp/unused"],
    ["system=quadrotor2d_stoch", "noise_family=corridor_sine_ambient", "noise_level=smooth",
     "output_dir=/tmp/unused"],
    ["system=quadrotor3d_ppo_stoch", "noise_family=corridor_sine_ambient", "+controller=ppo_800k",
     "noise_level=f_0.12_a0.03", "output_dir=/tmp/unused"],
    ["system=quadrotor3d_ppo_stoch", "noise_family=corridor_sine_ambient", "+controller=ppo_1500k",
     "noise_level=f_0.12_a0.03", "output_dir=/tmp/unused"],
], ids=["cartpole", "quadrotor2d", "quadrotor3d", "quadrotor3d_1500k"])
def test_config_composes_for_the_other_systems(overrides):
    # Finding 3: the quad3D case needs configs/adaptive_v2/controller/ppo_800k.yaml.
    cfg = _compose(overrides)
    assert os.path.exists(cfg.data_source.test_set_file)

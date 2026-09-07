"""The yield strategies must actually receive the system's geometry.

`yield_mlp` and `yield_knn` price candidates by a learned trajectory length, and
they find neighbours with `embed_states`, which needs two things from the system:
per-channel normalisation scales, so a metre and a rad/s are commensurate, and
the circular indices, so theta = +pi and theta = -pi coincide instead of sitting
the maximum possible distance apart.

Both read it as `getattr(pool, "system", None)`. `TrajectoryPool` never carried a
`system`, so that returned None on every run since the family shipped, and
`_geometry` silently fell back to unit scales with an empty circular mask. The
existing yield tests build a fake pool with `system = None`, which pinned the
broken behaviour rather than catching it.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2")


def _register_resolvers() -> None:
    """Same registration `test_decomposition_smoke.py` needs, for the same reason."""
    from adaptive_roa.utils.env_config import (
        get_data_dir, get_env_config, get_exp_dir, get_net_id, get_shared_data_base,
    )
    if not OmegaConf.has_resolver("net_id"):
        OmegaConf.register_new_resolver("net_id", lambda: get_net_id())
    if not OmegaConf.has_resolver("exp_dir"):
        OmegaConf.register_new_resolver("exp_dir", lambda: get_exp_dir())
    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
    if not OmegaConf.has_resolver("shared_data_base"):
        OmegaConf.register_new_resolver(
            "shared_data_base", lambda default="": get_shared_data_base() or default)
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver(
            "env", lambda key, default="": os.environ.get(key, get_env_config().get(key, default)))


def _engine(tmp_path):
    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine
    _register_resolvers()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default", overrides=[
            "system=pendulum_stoch", "noise_level=high",
            "predictor=clf_ensemble", "acquisition=decomp_yield_mlp",
            f"output_dir={tmp_path}/geom",
        ])
        return AdaptiveEngine(cfg)


def test_engine_hands_its_system_to_the_pool(tmp_path):
    engine = _engine(tmp_path)
    assert engine.pool.system is engine.system, (
        "pool.system must be the engine's system; the yield strategies read it "
        "off the pool and silently fall back to unit scales when it is missing"
    )


def test_yield_geometry_is_the_system_geometry_not_the_fallback(tmp_path):
    from adaptive_roa.adaptive_v2.strategy.yield_mlp import YieldMLPAcquisitionStrategy
    engine = _engine(tmp_path)
    state_dim = int(engine.system.state_dim)
    scales, mask = YieldMLPAcquisitionStrategy._geometry(
        getattr(engine.pool, "system", None), state_dim)

    expected = np.asarray(engine.system.get_normalization_scales(), dtype=np.float64)
    assert np.allclose(scales, expected), f"expected {expected}, got {scales}"
    # Pendulum theta is circular. An empty mask means theta stopped wrapping and
    # +pi/-pi became the two most distant points in the space.
    assert mask.any(), "circular mask is empty; the angle is no longer wrapped"

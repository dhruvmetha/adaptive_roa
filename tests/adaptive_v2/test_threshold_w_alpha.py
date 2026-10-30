"""The threshold objective weight w and alpha come from config, not literals.

The adaptive_v2 decoupling (6f7587d) hardcoded w=0.9 and alpha=0.1 in the
threshold backend, so the w ablation could not be run. The defaults stay equal
to those literals so every existing run composes to the same objective.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.threshold.conformal_threshold import ConformalThresholdBackend

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2"


def _cfg(**extra):
    base = {
        "predictor_type": "classifier",
        "decision_rule": "one_sided",
        "delta": 0.05,
        "attractor_radius": 0.2,
        "num_mc_samples": 10,
        "optimize_mode": "joint",
        "optimize_objective": "loss",
        "target_f1": 0.9,
        "lambda_grid_size": 50,
        "delta_grid_size": 50,
        "delta_min": 0.05,
        "delta_max": 0.45,
        "fixed_lambda_star": 0.5,
        "fixed_delta_star": 0.1,
        "use_p_invalid_veto": True,
        "verbose": False,
    }
    base.update(extra)
    return OmegaConf.create(base)


def test_missing_keys_keep_the_old_literals():
    conf = ConformalThresholdBackend(_cfg(), system=None, device="cpu")._build_conformal_config()
    assert conf.w == 0.9
    assert conf.alpha == 0.1


def test_configured_w_and_alpha_reach_the_optimizer_config():
    backend = ConformalThresholdBackend(_cfg(w=0.75, alpha=0.05), system=None, device="cpu")
    conf = backend._build_conformal_config()
    assert conf.w == 0.75
    assert conf.alpha == 0.05


def test_out_of_range_w_is_refused():
    backend = ConformalThresholdBackend(_cfg(w=0.0), system=None, device="cpu")
    with pytest.raises(AssertionError):
        backend._build_conformal_config()


def test_composed_default_matches_the_old_literals():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default", overrides=["system=pendulum"])
    assert cfg.threshold.w == 0.9
    assert cfg.threshold.alpha == 0.1


def test_w_override_reaches_the_composed_threshold_block():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default", overrides=["system=pendulum", "threshold.w=0.95"])
    assert cfg.threshold.w == 0.95

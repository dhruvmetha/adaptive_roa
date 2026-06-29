# tests/test_humanoid_standup_reach_adaptive_config.py
from pathlib import Path
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import pytest
from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_shared_data_base

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs/adaptive_v2")


def _resolvers():
    import os
    for name, fn in [("net_id", get_net_id), ("exp_dir", get_exp_dir), ("shared_data_base", get_shared_data_base)]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, (lambda f: (lambda default="": f() or default))(fn))
    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver("data_dir", lambda default="": os.environ.get("DATA_DIR", default) or default)


def test_adaptive_config_composes():
    _resolvers()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="default", overrides=["system=humanoid_standup_reach"])
    assert cfg.adaptive_v2.system_name == "humanoid_standup_reach"
    assert cfg.system._target_ == "adaptive_roa.systems.humanoid_standup_reach.HumanoidStandUpReachSystem"
    assert cfg.flow_matcher._target_.endswith("HumanoidStandUpReachLatentConditionalFlowMatcher")
    # cal/test point at the pre-shuffled FPS copies so the max_eval_rows front-slice is representative
    assert str(cfg.data_source.test_set_file).endswith("test_set_fps_shuffled.txt")
    assert str(cfg.data_source.cal_set_file).endswith("cal_set_fps_shuffled.txt")
    assert cfg.get("candidate_mode") == "intermediate"
    assert int(cfg.model_dims.output_dim) == 67


def test_humanoid_adaptive_config_scale_hardening():
    """Verify scale-hardening config additions: test_ratio=0, 150k train pool,
    and max_eval_rows cap for FPS cal/test files."""
    import os
    _resolvers()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="default", overrides=["system=humanoid_standup_reach"])

    # test_ratio must be 0 (no leakage through local test-set build)
    assert int(cfg.get("test_ratio", 0)) == 0

    # Acquisition pool must point at the 150k TRAIN split (not all_shuffled_*)
    # Use basename check so the assertion is RED if the old all_* path is used
    assert os.path.basename(str(cfg.data_source.shuffled_indices_file)) == "shuffled_indices.txt", (
        "Acquisition pool must use the 150k train split (shuffled_indices.txt), "
        f"got: {cfg.data_source.shuffled_indices_file}"
    )
    assert os.path.basename(str(cfg.data_source.shuffled_labels_file)) == "shuffled_labels.txt", (
        "Acquisition pool must use the 150k train split (shuffled_labels.txt), "
        f"got: {cfg.data_source.shuffled_labels_file}"
    )

    # FPS eval row cap must exist in eval section
    assert cfg.eval.max_eval_rows is not None, (
        "eval.max_eval_rows must be set to cap 1.3GB FPS test/cal file loads"
    )

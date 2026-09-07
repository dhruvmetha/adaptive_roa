"""One tiny adaptive epoch per score mode, on the real engine.

Guards the failure that cost hours last campaign: an acquisition strategy that
raises on a predictor family is only discovered when every arm has burned GPU
time. This runs the real AdaptiveEngine on a 200-trajectory pool.

Five arms: the four decomposition score modes, plus the non-adaptive control
(`direct` acquisition with `predictor=clf_ensemble`) at `d2_ratio=0` -- an
untested combination that the real campaign also launches, and the one most
likely to fail silently if left unexercised.
"""
import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2")
ARMS = ["decomp_total", "decomp_epi_var", "decomp_epi_bald", "decomp_aleat", "direct"]


def _register_resolvers() -> None:
    """Register the Hydra/OmegaConf resolvers used by the adaptive_v2 configs.

    Mirrors scripts/run_adaptive.py -- the real entrypoint registers these at
    import time, but a bare `hydra.compose()` in a test does not, so `${data_dir:}`
    etc. are unresolved without this.
    """
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


@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_one_epoch_runs_for_each_score_mode(arm, tmp_path):
    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine
    _register_resolvers()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        overrides = [
            "system=pendulum_stoch", "noise_level=high",
            f"acquisition={arm}", "predictor=clf_ensemble",
            "n_epochs=1", "initial_train_size=200", "samples_per_epoch=200",
            "predictor.bnn.n_members=2", "predictor.bnn.max_epochs=2",
            "eval.max_eval_rows=200", "eval.num_mc_samples_eval=4",
            f"output_dir={tmp_path}/{arm}",
        ]
        if arm == "direct":
            # Non-adaptive control: acquires randomly (no D2 uncertainty
            # acquisition), so it never touches the decomposition strategy.
            # direct.yaml has no n_candidates field, unlike the decomp_* arms.
            overrides.append("acquisition.d2_ratio=0")
        else:
            overrides.append("acquisition.n_candidates=500")
        cfg = compose(config_name="default", overrides=overrides)
        AdaptiveEngine(cfg).run()
    assert (tmp_path / arm / "epoch_000" / "full_roa_per_point.npz").exists()

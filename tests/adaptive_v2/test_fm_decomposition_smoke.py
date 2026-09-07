"""The flow-matching half of the decomposition smoke test.

`test_decomposition_smoke.py` covers `predictor=clf_ensemble` only, so the entire
FM acquisition path -- `EnsembleFlowMatchingTrainer` spawning members,
`_load_member` reloading them in the parent, and
`EnsembleEndpointMCProbabilityBackend.estimate_members` scoring against real flow
matchers -- was exercised only by fakes. The design requires all five arms on
*both* predictors before committing GPUs (design doc, "Pre-launch smoke tests"),
and the real campaign reaches that code roughly a day into training, on all five
arms simultaneously.

Two adaptive epochs, not one: the acquisition that consumes `estimate_members`
runs *between* epochs, and `_load_member` only exercises the reload path once a
member checkpoint already exists on disk.

Requires a GPU (the trainer uses torch.multiprocessing.spawn onto CUDA devices),
so it is marked `slow` and skipped when none is visible.
"""
import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2")
ARMS = ["decomp_total", "decomp_epi_var", "decomp_epi_bald", "decomp_aleat", "direct"]


def _register_resolvers() -> None:
    """Register the resolvers scripts/run_adaptive.py registers at import time.

    A bare hydra.compose() in a test does not run that entrypoint, so `${data_dir:}`
    and friends would stay unresolved.
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
def test_two_epochs_run_for_each_score_mode_on_flow_matching(arm, tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("EnsembleFlowMatchingTrainer spawns members onto CUDA devices")

    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine
    _register_resolvers()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        overrides = [
            "system=pendulum_stoch", "noise_level=high",
            f"acquisition={arm}", "predictor=fm_ensemble",
            "n_epochs=2", "initial_train_size=200", "samples_per_epoch=200",
            "predictor.ensemble.n_members=2",
            "predictor.lightning_trainer.max_epochs=2",
            "eval.max_eval_rows=200", "eval.num_mc_samples_eval=4",
            f"output_dir={tmp_path}/{arm}",
        ]
        if arm == "direct":
            # Non-adaptive control: never touches the decomposition strategy, and
            # direct.yaml has no n_candidates field unlike the decomp_* arms.
            overrides.append("acquisition.d2_ratio=0")
        else:
            overrides.append("acquisition.n_candidates=200")
        cfg = compose(config_name="default", overrides=overrides)
        AdaptiveEngine(cfg).run()

    # epoch_001 is the one that proves acquisition and the member reload both ran
    assert (tmp_path / arm / "epoch_001" / "full_roa_per_point.npz").exists()
    # every member must have produced a checkpoint the parent could reload
    members = sorted((tmp_path / arm / "epoch_000").glob("member_*"))
    assert len(members) == 2, f"expected 2 member dirs, got {[m.name for m in members]}"
    for m in members:
        assert list(m.glob("checkpoints/*.ckpt")), f"{m.name} wrote no checkpoint"

"""The three BALD arms added 2026-09-13 compose, and have not drifted.

Two of them copy a `final_state` block from an existing arm. A copy that drifts
silently makes a comparison measure something other than what it claims, so the
equality is asserted rather than trusted.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2"
PRED = CONFIG_DIR / "predictor"

# hidden_dims differs ON PURPOSE: the base arms carry the flat [256, 512, 256],
# which is the pendulum entry, and the BALD arms go per-system.
EXPECTED_DIFFS = {"hidden_dims"}

BALD_ARMS = {
    "bnn_mfvi_reg_bald": "bnn_mfvi_reg",
    "bnn_ens_reg_bald": "bnn_ensemble_reg",
}


def _raw(name: str):
    return OmegaConf.load(PRED / f"{name}.yaml")


@pytest.mark.parametrize("bald,base", sorted(BALD_ARMS.items()))
def test_final_state_block_matches_its_base_arm(bald, base):
    a = _raw(bald).predictor.final_state
    b = _raw(base).predictor.final_state
    assert set(a) == set(b), f"{bald} and {base} define different keys"
    drifted = [k for k in a if k not in EXPECTED_DIFFS and a[k] != b[k]]
    assert not drifted, (
        f"{bald} drifted from {base} on {drifted}; the arms would no longer differ "
        f"only in acquisition")


@pytest.mark.parametrize("bald,base", sorted(BALD_ARMS.items()))
def test_predictor_name_still_keys_the_base_arms_export_class(bald, base):
    # `name` keys the export registry and must name the MODEL, not the arm.
    assert _raw(bald).predictor.name == _raw(base).predictor.name


def _unresolved(node, *path):
    """Read a key WITHOUT resolving interpolations (they need a composed cfg)."""
    for key in path:
        node = node._get_node(key)
    return str(node)


@pytest.mark.parametrize("bald,base", sorted(BALD_ARMS.items()))
def test_width_is_per_system_not_the_pendulum_default(bald, base):
    assert "${model_dims.mlp_hidden_dims}" in _unresolved(
        _raw(bald), "predictor", "final_state", "hidden_dims")
    assert _raw(base).predictor.final_state.hidden_dims == [256, 512, 256]


def test_outcome_ensemble_takes_a_distinct_predictor_name():
    # A different model class, so a shared name would load the wrong checkpoint
    # class on export.
    assert _raw("fm_outcome_bald").predictor.name == "fm_outcome_ens"
    assert _raw("fm_outcome").predictor.name == "fm_outcome"


def test_outcome_ensemble_width_is_per_system():
    assert "${model_dims.mlp_hidden_dims}" in _unresolved(
        _raw("fm_outcome_bald"), "predictor", "outcome_fm", "hidden_dims")
    assert _raw("fm_outcome").predictor.outcome_fm.hidden_dims == [256, 512, 256]


# Every system this campaign runs. The interpolation resolving for cartpole is
# no evidence it resolves for the others, and an unresolvable width fails only
# when the trainer builds the net, an epoch into a queued job.
CAMPAIGN_SYSTEMS = ["cartpole_stoch", "quadrotor2d_stoch", "quadrotor3d_ppo_stoch"]


@pytest.mark.parametrize("system", CAMPAIGN_SYSTEMS)
@pytest.mark.parametrize("arm", ["bnn_mfvi_reg_bald", "bnn_ens_reg_bald", "fm_outcome_bald"])
def test_per_system_width_resolves_on_every_campaign_system(arm, system):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default",
                      overrides=[f"predictor={arm}", f"system={system}"])
    block = cfg.predictor.get("final_state") or cfg.predictor.get("outcome_fm")
    dims = list(block.hidden_dims)
    assert dims and all(int(d) > 0 for d in dims), (arm, system, dims)


@pytest.mark.parametrize("arm", ["bnn_mfvi_reg_bald", "bnn_ens_reg_bald", "fm_outcome_bald"])
def test_arm_composes_with_bald_acquisition(arm):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="default",
            overrides=[f"predictor={arm}", "acquisition=decomp_epi_bald",
                       "system=cartpole_stoch"],
        )
    assert cfg.acquisition.score == "epistemic_bald"
    assert "estimate_members" not in str(cfg)      # sanity: it is code, not config
    target = cfg.probability._target_
    assert target.endswith(("PosteriorEndpointMCProbabilityBackend",
                            "EnsembleOutcomeFMProbabilityBackend")), target


def test_outcome_ensemble_readout_matches_between_model_and_backend():
    # bind_model refuses a mismatch at runtime; catching it in config is cheaper
    # than catching it an epoch in.
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="default",
            overrides=["predictor=fm_outcome_bald", "system=cartpole_stoch"],
        )
    assert cfg.probability.readout == cfg.predictor.outcome_fm.forward_readout


@pytest.mark.parametrize("arm", ["bnn_mfvi_reg_bald", "bnn_ens_reg_bald"])
def test_num_mc_samples_can_resolve_the_ensemble(arm):
    # FinalStateTrainer raises unless num_mc_samples >= 2*n_members.
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default",
                      overrides=[f"predictor={arm}", "system=cartpole_stoch"])
    n_members = cfg.predictor.final_state.get("n_members", 1)
    assert cfg.probability.num_mc_samples >= 2 * n_members

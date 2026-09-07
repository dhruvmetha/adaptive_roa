from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2")
MODES = {"decomp_total": "total", "decomp_epi_var": "epistemic_var",
         "decomp_epi_bald": "epistemic_bald", "decomp_aleat": "aleatoric"}


@pytest.mark.parametrize("arm,mode", sorted(MODES.items()))
def test_each_arm_composes_and_selects_its_score(arm, mode):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default",
                      overrides=["system=pendulum_stoch", "noise_level=high",
                                 f"acquisition={arm}", "predictor=clf_ensemble"])
    assert cfg.acquisition.score == mode
    assert cfg.acquisition._target_.endswith("decomposition.DecompositionAcquisitionStrategy")
    assert cfg.sampling_mode == arm


def test_clf_ensemble_uses_the_ensemble_probability_backend():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default",
                      overrides=["system=pendulum_stoch", "noise_level=high",
                                 "acquisition=decomp_epi_var", "predictor=clf_ensemble"])
    assert cfg.probability._target_.endswith(
        "ensemble_prob.EnsembleClassifierProbabilityBackend")
    assert cfg.predictor.bnn.posterior == "ensemble"
    assert cfg.predictor.bnn.n_members == 5


def test_strategy_instantiates_through_the_engines_real_path():
    """Build the strategy the way the engine does, not the way Hydra would.

    engine._instantiate resolves _target_ and passes the config node positionally;
    every strategy in this repo takes a single cfg object. Using
    hydra.utils.instantiate here would expand the node into kwargs and test a
    convention the engine never uses -- and would also miss a typo'd _target_.
    """
    from adaptive_roa.adaptive_v2.engine import _instantiate

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default",
                      overrides=["system=pendulum_stoch", "noise_level=high",
                                 "acquisition=decomp_epi_var", "predictor=clf_ensemble"])
    strat = _instantiate(cfg.acquisition)
    assert strat.score_mode == "epistemic_var"
    assert strat.d2_ratio == 1.0
    assert strat.mode == "decomposition"

"""The deterministic-paper 2 x 2 arm configs (tracker C13), and a guard that
the arms already in the repo still compose to the values they had before the
C12/C19 changes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "adaptive_v2"
FIXTURE = Path(__file__).parent / "fixtures" / "existing_predictor_configs.json"

DET_ARMS = ["det_mlp_clf", "det_bnn_clf", "det_mlp_reg", "det_bnn_reg"]
DET_SYSTEMS = ["pendulum", "cartpole_pybullet", "quadrotor2d", "quadrotor3d"]


def _compose(arm, system, extra=()):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="default",
                       overrides=[f"predictor={arm}", f"system={system}", *extra])


def _snapshot(cfg):
    blk = {k: OmegaConf.to_container(cfg[k], resolve=True)
           for k in ["predictor", "probability"] if k in cfg}
    blk["threshold.decision_rule"] = cfg.threshold.get("decision_rule")
    blk["calibration.decision_rule"] = cfg.calibration.get("decision_rule")
    return json.loads(json.dumps(blk, sort_keys=True))


_EXISTING = json.loads(FIXTURE.read_text())


@pytest.mark.parametrize("key", sorted(_EXISTING))
def test_existing_arm_composes_to_its_recorded_values(key):
    """Recorded from the branch point (fc9fa10) before any edit here. A diff
    means an existing GRAPE arm would now train differently."""
    system, arm = key.split("/")
    live, recorded = _snapshot(_compose(arm, system)), json.loads(json.dumps(_EXISTING[key]))
    # The success radius is a system value, not a predictor value: the det-paper
    # rerun moved quadrotor3d from 0.3 to 0.15 on purpose (tracker D12), and that
    # is asserted in tests/systems/test_det_criteria.py.
    for snap in (live, recorded):
        snap.get("probability", {}).pop("attractor_radius", None)
    assert live == recorded


@pytest.mark.parametrize("system", DET_SYSTEMS)
@pytest.mark.parametrize("arm", DET_ARMS)
def test_det_arm_composes_with_per_system_width(arm, system):
    cfg = _compose(arm, system)
    block = cfg.predictor.get("bnn") or cfg.predictor.get("final_state")
    dims = list(block.hidden_dims)
    assert dims == list(cfg.model_dims.mlp_hidden_dims), (arm, system, dims)
    assert cfg.predictor.name == arm


def _raw(arm):
    return OmegaConf.load(CONFIG_DIR / "predictor" / f"{arm}.yaml")


@pytest.mark.parametrize("point,bayes,block,allowed", [
    ("det_mlp_clf", "det_bnn_clf", "bnn",
     {"posterior", "prior_precision", "n_marginal_samples"}),
    ("det_mlp_reg", "det_bnn_reg", "final_state", {"posterior", "prior_precision"}),
])
def test_twins_differ_only_in_the_posterior(point, bayes, block, allowed):
    a = OmegaConf.to_container(_raw(point).predictor[block], resolve=False)
    b = OmegaConf.to_container(_raw(bayes).predictor[block], resolve=False)
    differ = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differ <= allowed, differ
    for top in ("type", "trainer_target", "batch_size", "val_batch_size", "lightning_trainer"):
        assert _raw(point).predictor[top] == _raw(bayes).predictor[top], top
    assert _raw(point).defaults == _raw(bayes).defaults


@pytest.mark.parametrize("arm", ["det_mlp_clf", "det_bnn_clf"])
def test_outcome_arms_pin_pos_weight_and_expose_an_estimator(arm):
    cfg = _compose(arm, "quadrotor3d")
    assert float(cfg.predictor.bnn.pos_weight) == 1.0
    assert cfg.predictor.trainer_target.endswith("BayesianMLPTrainer")
    # direct/ranked read probability_backend.estimator, which this backend has.
    assert cfg.probability._target_.endswith("ClassifierProbabilityBackend")
    assert cfg.threshold.decision_rule == "one_sided"


@pytest.mark.parametrize("arm", ["det_mlp_reg", "det_bnn_reg"])
def test_endpoint_arms_use_the_mixture_head(arm):
    cfg = _compose(arm, "quadrotor3d")
    fs = cfg.predictor.final_state
    assert fs.head.kind == "mixture" and int(fs.head.n_components) == 5
    assert float(fs.beta_nll) == 0.0
    assert cfg.probability._target_.endswith("EndpointMCProbabilityBackend")


def test_bayesian_arms_tune_the_prior():
    assert _raw("det_bnn_clf").predictor.bnn.prior_precision == "auto"
    assert _raw("det_bnn_reg").predictor.final_state.prior_precision == "auto"


def test_final_state_mlp_may_run_adaptively_but_mlp_det_may_not():
    from adaptive_roa.adaptive_v2.trainers.final_state_trainer import FinalStateTrainer
    from adaptive_roa.systems.cartpole import CartPoleSystem

    cfg = _compose("det_mlp_reg", "cartpole_pybullet", ["acquisition.d2_ratio=0.5"])
    FinalStateTrainer(cfg, CartPoleSystem(), "cartpole_pybullet")  # must not raise
    old = _compose("mlp_det", "cartpole_pybullet", ["acquisition.d2_ratio=0.5"])
    with pytest.raises(ValueError, match="FIXED-DATASET"):
        FinalStateTrainer(old, CartPoleSystem(), "cartpole_pybullet")


@pytest.mark.parametrize("arm,family", [("det_mlp_clf", "classifier"), ("det_bnn_clf", "classifier"),
                                        ("det_mlp_reg", "generative"), ("det_bnn_reg", "generative")])
def test_det_arm_is_registered_for_export(arm, family):
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class(arm)
    assert cls.predictor_name == arm and cls.predictor_type == family


def test_det_arms_do_not_steal_the_family_aliases():
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.classifier import ClassifierProbabilisticClassifier
    from adaptive_roa.probabilistic_classifier.flow_matching import FMProbabilisticClassifier

    assert get_probabilistic_classifier_class("classifier") is ClassifierProbabilisticClassifier
    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier


def test_prior_precision_is_refused_on_non_laplace_posteriors():
    from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp

    with pytest.raises(ValueError, match="only 'laplace'"):
        build_bayesian_mlp(4, [8], 1, "mfvi", prior_precision="auto")
    with pytest.raises(ValueError, match="number or 'auto'"):
        build_bayesian_mlp(4, [8], 1, "laplace", prior_precision="evidence")

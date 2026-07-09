from hydra import initialize, compose


def test_partx_experiment_composes():
    with initialize(version_base=None, config_path="../../configs/adaptive_v2"):
        cfg = compose(
            config_name="default",
            overrides=[
                "system=pendulum",
                "predictor=gp",
                "acquisition=partx",
                "eval=partx",
            ],
        )
    assert cfg.acquisition.mode == "partx"
    assert cfg.predictor.trainer_target.endswith("GPPredictorTrainer")
    assert cfg.probability._target_.endswith("GPProbabilityBackend")
    assert cfg.threshold.decision_rule == "one_sided"
    assert cfg.calibration.decision_rule == "one_sided"


def test_partx_gp_forces_one_sided_over_two_sided_system():
    # quadrotor2d defaults decision_rule to two_sided at the system level.
    # The gp predictor config must still force one_sided, since
    # GPProbabilityBackend.estimate() always returns p_invalid=0 and has no
    # separatrix/invalid signal to support a two_sided decision rule.
    with initialize(version_base=None, config_path="../../configs/adaptive_v2"):
        cfg = compose(
            config_name="default",
            overrides=[
                "system=quadrotor2d",
                "predictor=gp",
                "acquisition=partx",
                "eval=partx",
            ],
        )
    assert cfg.decision_rule == "two_sided"
    assert cfg.threshold.decision_rule == "one_sided"
    assert cfg.calibration.decision_rule == "one_sided"

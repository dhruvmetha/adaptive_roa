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
    assert cfg.threshold.optimize_objective == "fixed"
    assert float(cfg.threshold.fixed_lambda_star) == 0.5


def test_partx_gp_optdelta_optimizes_delta_with_pinned_lambda():
    # gp_optdelta is an A/B variant of gp: it keeps lambda*=0.5 pinned
    # (optimize_delta() hardcodes lam=0.5) but grid-searches delta instead
    # of using a fixed delta*=0.1 like the baseline gp.yaml.
    with initialize(version_base=None, config_path="../../configs/adaptive_v2"):
        cfg = compose(
            config_name="default",
            overrides=[
                "system=pendulum",
                "predictor=gp_optdelta",
                "acquisition=partx",
                "eval=partx",
            ],
        )
    assert cfg.acquisition.mode == "partx"
    assert cfg.predictor.trainer_target.endswith("GPPredictorTrainer")
    assert cfg.threshold.optimize_mode == "delta"
    assert cfg.threshold.optimize_objective == "loss"
    assert cfg.threshold.decision_rule == "one_sided"

    # Confirm the baseline gp config is undisturbed by the new variant.
    with initialize(version_base=None, config_path="../../configs/adaptive_v2"):
        baseline_cfg = compose(
            config_name="default",
            overrides=[
                "system=pendulum",
                "predictor=gp",
                "acquisition=partx",
                "eval=partx",
            ],
        )
    assert baseline_cfg.threshold.optimize_objective == "fixed"


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


def test_partx_pendulum_experiment_schedule():
    # Total training-pool size grows 50 -> 500 in +50 steps over 10 epochs
    # (init=50, incr=50, n_epochs=10). Fully adaptive (d2_ratio=1.0); conformal
    # coverage comes only from the held-out cal_set at eval time, never on D1.
    with initialize(version_base=None, config_path="../../configs/adaptive_v2"):
        cfg = compose(config_name="default", overrides=["+experiment=partx_pendulum"])
    assert cfg.initial_train_size == 50
    assert cfg.samples_per_epoch == 50
    assert cfg.n_epochs == 10
    assert cfg.acquisition.mode == "partx"
    assert cfg.acquisition.d2_ratio == 1.0
    assert cfg.predictor.trainer_target.endswith("GPPredictorTrainer")
    assert cfg.threshold.decision_rule == "one_sided"


def test_partx_cartpole_experiment_schedule():
    # Total grows 300 -> 1000 in +50 steps over 15 epochs
    # (init=300, incr=50, n_epochs=15). Fully adaptive.
    with initialize(version_base=None, config_path="../../configs/adaptive_v2"):
        cfg = compose(config_name="default", overrides=["+experiment=partx_cartpole"])
    assert cfg.initial_train_size == 300
    assert cfg.samples_per_epoch == 50
    assert cfg.n_epochs == 15
    assert cfg.acquisition.mode == "partx"
    assert cfg.acquisition.d2_ratio == 1.0
    assert cfg.predictor.trainer_target.endswith("GPPredictorTrainer")
    assert cfg.threshold.decision_rule == "one_sided"

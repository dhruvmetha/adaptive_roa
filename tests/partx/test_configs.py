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

import pytest


@pytest.mark.parametrize("arm", ["bnn_mfvi", "bnn_ensemble", "bnn_laplace"])
def test_each_arm_is_registered_for_export(arm):
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class(arm)
    assert cls.predictor_name == arm
    assert cls.predictor_type == "classifier"
    assert cls.native_probs == ("p_success",)


def test_arms_do_not_steal_the_legacy_classifier_alias():
    """A BNN run must never be loaded as a plain ClassifierMLP."""
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.classifier import (
        ClassifierProbabilisticClassifier,
    )

    assert get_probabilistic_classifier_class("classifier") is ClassifierProbabilisticClassifier


def test_laplace_export_round_trips_its_covariance(tmp_path):
    """A Laplace arm that loses its GGN covariance is a MAP model wearing a
    Bayesian label -- the failure is silent, so it needs an explicit test."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer
    from adaptive_roa.probabilistic_classifier.bayesian import LaplaceProbabilisticClassifier
    from adaptive_roa.systems.cartpole import CartPoleSystem

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        X = rng.uniform(-1.0, 1.0, size=(256, 4))
        np.savetxt(tmp_path / f"{split}.txt",
                   np.column_stack([X, (np.abs(X[:, 1]) < 0.5).astype(int)]))

    cfg = OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "classifier", "name": "bnn_laplace", "batch_size": 64,
                      "bnn": {"posterior": "laplace", "hidden_dims": [16, 16],
                              "max_epochs": 2, "n_marginal_samples": 8}},
    })
    run_dir = tmp_path / "run"
    epoch_dir = run_dir / "epoch_000"
    trainer = BayesianMLPTrainer(cfg, CartPoleSystem(), "cartpole")
    fitted = trainer.fit(
        {"train": str(tmp_path / "train.txt"), "val": str(tmp_path / "val.txt")},
        str(epoch_dir),
    )
    loaded = LaplaceProbabilisticClassifier.load_from_run(
        str(run_dir), 0, cfg, CartPoleSystem(), device="cpu"
    )
    assert loaded.handle.posterior.is_fitted
    torch.testing.assert_close(
        loaded.handle.posterior.posterior_covariance,
        fitted.posterior.posterior_covariance,
    )

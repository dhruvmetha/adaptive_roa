import numpy as np
import pytest
import torch


@pytest.mark.parametrize("arm", ["gp", "gp_optdelta", "gp_reg"])
def test_each_gp_arm_is_registered_for_export(arm):
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class(arm)
    assert cls.predictor_name == arm


def test_outcome_gp_arms_declare_the_classifier_family():
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    for arm in ("gp", "gp_optdelta"):
        cls = get_probabilistic_classifier_class(arm)
        assert cls.predictor_type == "classifier"
        assert cls.native_probs == ("p_success",)


def test_gp_reg_declares_the_generative_family():
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class("gp_reg")
    assert cls.predictor_type == "generative"
    assert cls.native_probs == ("p_success", "p_failure", "p_invalid")


def test_gp_arms_do_not_steal_the_legacy_family_aliases():
    """Runs written before arm names existed must still resolve."""
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.classifier import (
        ClassifierProbabilisticClassifier,
    )
    from adaptive_roa.probabilistic_classifier.flow_matching import FMProbabilisticClassifier

    assert get_probabilistic_classifier_class("classifier") is ClassifierProbabilisticClassifier
    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier


def test_partx_trainer_writes_a_warm_start_checkpoint(tmp_path):
    """engine.py:140 globs checkpoints/best*.ckpt. Writing gp.pt meant the GP arm
    silently never warm-started."""
    from omegaconf import OmegaConf
    from adaptive_roa.partx.trainer import GPPredictorTrainer
    from adaptive_roa.systems.cartpole import CartPoleSystem

    system = CartPoleSystem()
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(120, 4))
    y = (np.abs(X[:, 1]) < 0.5).astype(int)
    path = tmp_path / "train.txt"
    np.savetxt(path, np.column_stack([X, y]))

    cfg = OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "classifier",
                      "gp": {"n_inducing": 8, "kernel": "matern52", "n_iters": 3, "lr": 0.1}},
    })
    out = tmp_path / "out"
    GPPredictorTrainer(cfg, system, "cartpole_pybullet").fit(
        {"train": str(path)}, str(out)
    )
    assert list((out / "checkpoints").glob("best*.ckpt")), "engine warm start needs best*.ckpt"


def test_gp_classifier_warm_start_keeps_the_loaded_inducing_points(tmp_path):
    """Direct, deterministic check that GPClassifier.fit's rebuild guard works.

    With n_iters=0 no optimization step runs, so a warm-started fit must leave
    the loaded inducing points byte-for-byte unchanged, while a cold fit (no
    load_state_dict) rebuilds from a fresh random subset and must differ. This
    is the GPClassifier-level counterpart of the guard added at
    gp_classifier.py:56-58; without it, fit's unconditional rebuild discards
    anything loaded before it, and a resume_checkpoint silently becomes a no-op.
    """
    from adaptive_roa.systems.pendulum import PendulumSystem
    from adaptive_roa.partx.gp_classifier import GPClassifier

    system = PendulumSystem()
    rng = np.random.default_rng(0)
    X = np.column_stack([
        rng.uniform(-3.0, 3.0, 200),
        rng.uniform(-8.0, 8.0, 200),
    ])
    y = ((X[:, 0] / 1.5) ** 2 + (X[:, 1] / 4.0) ** 2 < 1.0).astype(np.int64)

    first = GPClassifier(system, n_inducing=16, n_iters=50).fit(X, y)
    sd = first.state_dict()
    loaded_inducing = sd["model"]["variational_strategy.inducing_points"].clone()

    warm = GPClassifier(system, n_inducing=16, n_iters=0)
    warm.load_state_dict(sd)
    warm.fit(X, y)
    warm_inducing = warm.state_dict()["model"]["variational_strategy.inducing_points"]
    assert torch.equal(warm_inducing, loaded_inducing), (
        "warm-started fit must not discard the loaded inducing points"
    )

    cold = GPClassifier(system, n_inducing=16, n_iters=0).fit(X, y)
    cold_inducing = cold.state_dict()["model"]["variational_strategy.inducing_points"]
    assert not torch.equal(cold_inducing, loaded_inducing), (
        "a cold fit must rebuild from a fresh random inducing subset, or this "
        "test cannot distinguish warm start working from it being a no-op"
    )


def test_gp_export_loader_accepts_the_legacy_checkpoint_name(tmp_path):
    """Runs already on disk carry checkpoints/gp.pt; they must still export."""
    from adaptive_roa.probabilistic_classifier.gaussian_process import _find_gp_checkpoint

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    legacy = ckpt_dir / "gp.pt"
    legacy.write_bytes(b"x")
    assert _find_gp_checkpoint(ckpt_dir) == legacy

    modern = ckpt_dir / "best-gp.ckpt"
    modern.write_bytes(b"x")
    assert _find_gp_checkpoint(ckpt_dir) == modern  # prefer the new name


def test_missing_gp_checkpoint_raises(tmp_path):
    from adaptive_roa.probabilistic_classifier.gaussian_process import _find_gp_checkpoint

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _find_gp_checkpoint(ckpt_dir)

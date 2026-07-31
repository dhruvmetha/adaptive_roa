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


def _trained_gp_reg_classifier(tmp_path, num_mc_samples=4):
    from omegaconf import OmegaConf
    from adaptive_roa.adaptive_v2.trainers.gp_regressor_trainer import GPRegressorTrainer
    from adaptive_roa.probabilistic_classifier.gaussian_process import (
        GPRegProbabilisticClassifier,
    )
    from adaptive_roa.systems.cartpole import CartPoleSystem

    def endpoints(path, n, seed):
        rng = np.random.default_rng(seed)
        s = rng.uniform(-0.5, 0.5, size=(n, 4))
        np.savetxt(path, np.column_stack([s, s * 0.5]))
        return str(path)

    files = {"train": endpoints(tmp_path / "train.txt", 200, 0),
             "val": endpoints(tmp_path / "val.txt", 100, 1)}
    run_dir = tmp_path / "run"
    cfg = OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "generative", "name": "gp_reg", "batch_size": 64,
                      "val_batch_size": 64,
                      "gp": {"n_inducing": 16, "n_iters": 20, "lr": 0.05, "batch_size": 128}},
    })
    GPRegressorTrainer(cfg, CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(run_dir / "epoch_000")
    )
    export_cfg = OmegaConf.create({
        "predictor": {"type": "generative", "name": "gp_reg"},
        "probability": {"attractor_radius": 0.4, "num_mc_samples": num_mc_samples},
    })
    return GPRegProbabilisticClassifier.load_from_run(
        str(run_dir), 0, export_cfg, CartPoleSystem(), device="cpu"
    )


def test_gp_reg_export_shares_the_final_state_mc_loop():
    """I3. The GP wrapper duplicated FinalStateProbabilisticClassifier.predict
    and dropped its _BATCH_SIZE chunking and its n == 0 guard. The loop now lives
    in one place so the 1/-1/0 label mapping is written once."""
    import inspect

    from adaptive_roa.probabilistic_classifier import bayesian_final_state, gaussian_process
    from adaptive_roa.probabilistic_classifier.endpoint_mc import endpoint_mc_probabilities

    for module in (bayesian_final_state, gaussian_process):
        assert module.endpoint_mc_probabilities is endpoint_mc_probabilities
    for cls in (bayesian_final_state.FinalStateProbabilisticClassifier,
                gaussian_process.GPRegProbabilisticClassifier):
        src = inspect.getsource(cls.predict)
        assert "endpoint_mc_probabilities" in src
        assert "classify_attractor" not in src, "the label mapping is duplicated again"


def test_gp_reg_export_handles_an_empty_input(tmp_path):
    """The n == 0 guard the GP wrapper was missing: an empty split must return
    empty arrays, not crash inside the MC loop."""
    clf = _trained_gp_reg_classifier(tmp_path)
    probs = clf.predict(np.zeros((0, 4), dtype=np.float32))
    assert probs.p_success.shape == (0,)
    assert probs.p_failure.shape == (0,)
    assert probs.p_invalid.shape == (0,)


def test_gp_reg_export_probabilities_do_not_depend_on_the_export_batch_size(tmp_path):
    """C1 on the export path -- the path that produces the per-point
    probabilities used in analysis. Before the fix p_success was a function of
    how the caller batched rather than of the query state (the strictly-uncertain
    fraction swung 0.018 -> 0.20 on identical inputs and model)."""
    from adaptive_roa.probabilistic_classifier.endpoint_mc import endpoint_mc_probabilities

    clf = _trained_gp_reg_classifier(tmp_path, num_mc_samples=200)
    states = np.random.default_rng(0).uniform(-0.4, 0.4, size=(1024, 4)).astype(np.float32)

    whole = endpoint_mc_probabilities(clf.handle, clf.system, states,
                                      clf.attractor_radius, clf.num_mc_samples)
    chunked = endpoint_mc_probabilities(clf.handle, clf.system, states,
                                        clf.attractor_radius, clf.num_mc_samples,
                                        batch_size=64)
    # MC noise only: with K=200 the per-point SD is at most 0.035, so a mean
    # absolute gap this small cannot hide a collapsed predictive spread.
    assert np.abs(whole.p_success - chunked.p_success).mean() < 0.03
    for p in (whole, chunked):
        np.testing.assert_allclose(p.p_success + p.p_failure + p.p_invalid, 1.0, atol=1e-9)

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.trainers.gp_regressor_trainer import GPRegressorTrainer
from adaptive_roa.systems.cartpole import CartPoleSystem


def _write_endpoints(path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    start = rng.uniform(-0.5, 0.5, size=(n, 4))
    np.savetxt(path, np.column_stack([start, start * 0.5]))
    return str(path)


def _cfg(**overrides):
    gp = {"n_inducing": 16, "kernel": "matern52", "n_iters": 30, "lr": 0.05, "batch_size": 128}
    gp.update(overrides)
    return OmegaConf.create({
        "device": "cpu",
        "predictor": {"type": "generative", "name": "gp_reg",
                      "batch_size": 64, "val_batch_size": 64, "gp": gp},
    })


@pytest.fixture
def files(tmp_path):
    return {"train": _write_endpoints(tmp_path / "train.txt"),
            "val": _write_endpoints(tmp_path / "val.txt", n=100, seed=1)}


def test_trainer_returns_a_working_handle(files, tmp_path):
    handle = GPRegressorTrainer(_cfg(), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(tmp_path / "out")
    )
    x = torch.randn(16, 4) * 0.3
    out = handle.predict_endpoint(x)
    assert out.shape == (16, 4)
    assert torch.isfinite(out).all()
    assert not torch.allclose(handle.predict_endpoint(x), handle.predict_endpoint(x))


def test_trainer_writes_a_warm_start_checkpoint(files, tmp_path):
    """The engine globs checkpoints/best*.ckpt (engine.py:140). partx/trainer.py
    writes gp.pt instead, which is why the GP arm has never warm-started."""
    out = tmp_path / "out"
    GPRegressorTrainer(_cfg(), CartPoleSystem(), "cartpole_pybullet").fit(files, str(out))
    assert list((out / "checkpoints").glob("best*.ckpt"))


def test_warm_start_reloads_a_previous_checkpoint(files, tmp_path):
    trainer = GPRegressorTrainer(_cfg(), CartPoleSystem(), "cartpole_pybullet")
    first_out = tmp_path / "e0"
    trainer.fit(files, str(first_out))
    ckpt = list((first_out / "checkpoints").glob("best*.ckpt"))[0]
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved_inducing = saved["model"][
        "variational_strategy.base_variational_strategy.inducing_points"
    ]
    saved_lengthscale = saved["model"]["covar_module.base_kernel.raw_lengthscale"]

    # n_iters=0 -- no optimization steps run, so the resumed state must be
    # EXACTLY the checkpoint's, byte for byte, if the warm start actually
    # skipped the rebuild rather than silently discarding it.
    resumed = GPRegressorTrainer(_cfg(n_iters=0), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(tmp_path / "e1"), resume_checkpoint=str(ckpt)
    )
    resumed_sd = resumed.gp.state_dict()["model"]
    assert torch.equal(
        resumed_sd["variational_strategy.base_variational_strategy.inducing_points"],
        saved_inducing,
    )
    assert torch.equal(resumed_sd["covar_module.base_kernel.raw_lengthscale"], saved_lengthscale)
    x = torch.randn(8, 4) * 0.3
    assert torch.isfinite(resumed.predict_endpoint(x)).all()

    # A cold start (no resume_checkpoint) with the same n_iters=0 rebuilds from
    # a fresh random inducing-point draw, so it must NOT match the checkpoint --
    # this is what makes the assertions above capable of catching a no-op warm
    # start rather than trivially passing regardless of whether state persisted.
    cold = GPRegressorTrainer(_cfg(n_iters=0), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(tmp_path / "e2")
    )
    cold_sd = cold.gp.state_dict()["model"]
    assert not torch.equal(
        cold_sd["variational_strategy.base_variational_strategy.inducing_points"],
        saved_inducing,
    )


def test_the_arm_actually_learns_the_contraction(files, tmp_path):
    """Guards against shipping an untrained GP.

    The margin used to be 0.5x, which an ARM THAT BARELY TRAINED still clears:
    at n_iters=30 the ratio is already 0.31, and the review measured an untrained
    GP at 2.0x -- so 0.5x only rules out "totally broken", not "under-trained".
    Measured ratios for this fixture at n_iters=300 across five torch seeds:
    0.0013, 0.0021, 0.0029, 0.0076, 0.0154. The seed is pinned below and the
    bound set at 0.02, ~7x above the seeded value and 25x tighter than before.
    """
    torch.manual_seed(0)
    handle = GPRegressorTrainer(_cfg(n_iters=300), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(tmp_path / "out")
    )
    raw = np.loadtxt(files["val"])
    x = torch.as_tensor(raw[:, :4], dtype=torch.float32)
    y = torch.as_tensor(raw[:, 4:], dtype=torch.float32)
    feats = handle.system.embed_state_for_model(handle.system.normalize_state(x))
    pred = handle.decoder.decode(handle.gp.mean(feats))
    ratio = (pred - y).pow(2).mean().item() / (x - y).pow(2).mean().item()
    assert ratio < 0.02, f"fitted/identity MSE ratio {ratio:.5g} is too loose to be trained"


def test_the_trainer_selects_on_the_validation_split(files, tmp_path):
    """I2 regression: the trainer used to load `validation_file` and then never
    read it -- a fixed 300-iteration budget whose terminal state was saved. The
    BNN final-state arms early-stop and checkpoint on val_nll, so gp_reg was the
    one arm not selected under the shared criterion."""
    trainer = GPRegressorTrainer(_cfg(n_iters=40, eval_every=5), CartPoleSystem(),
                                 "cartpole_pybullet")
    handle = trainer.fit(files, str(tmp_path / "out"))
    assert handle.gp.best_val_nll is not None, "no validation score was ever computed"

    # The state that was kept (and written to best-gp.ckpt) must be the one that
    # scored best, not merely the last one the loop produced.
    dm = trainer._create_datamodule(files)
    feats, targets = trainer._features_and_targets(dm.val_dataset)
    assert handle.gp.val_nll(feats, targets) == pytest.approx(
        handle.gp.best_val_nll, rel=1e-5
    )


def test_config_composes_with_the_endpoint_mc_backend():
    import os
    from hydra import compose, initialize_config_dir

    d = os.path.abspath("configs/adaptive_v2")
    with initialize_config_dir(config_dir=d, version_base=None):
        cfg = compose(config_name="default", overrides=["predictor=gp_reg"])
    assert cfg.predictor.name == "gp_reg"
    assert cfg.predictor.type == "generative"
    assert "EndpointMCProbabilityBackend" in cfg.probability._target_


def test_gp_reg_export_round_trips_a_trained_arm(tmp_path):
    """The export path is what produces the per-point probabilities used in
    analysis; nothing else exercises it."""
    from omegaconf import OmegaConf
    from adaptive_roa.probabilistic_classifier.gaussian_process import (
        GPRegProbabilisticClassifier,
    )

    files = {"train": _write_endpoints(tmp_path / "train.txt"),
             "val": _write_endpoints(tmp_path / "val.txt", n=100, seed=1)}
    run_dir = tmp_path / "run"
    GPRegressorTrainer(_cfg(), CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(run_dir / "epoch_000")
    )

    cfg = OmegaConf.create({
        "predictor": {"type": "generative", "name": "gp_reg",
                      "gp": {"n_inducing": 16, "kernel": "matern52"}},
        "probability": {"attractor_radius": 0.37, "num_mc_samples": 7},
    })
    clf = GPRegProbabilisticClassifier.load_from_run(
        str(run_dir), 0, cfg, CartPoleSystem(), device="cpu"
    )
    assert clf.attractor_radius == 0.37
    assert clf.num_mc_samples == 7

    probs = clf.predict(np.random.default_rng(0).uniform(-0.3, 0.3, size=(24, 4)).astype("float32"))
    total = probs.p_success + probs.p_failure + probs.p_invalid
    np.testing.assert_allclose(total, np.ones(24), atol=1e-6)

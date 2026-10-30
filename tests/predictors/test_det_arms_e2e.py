"""CPU smoke of the four deterministic-paper arms: train two epochs on a tiny
synthetic set, reload through the export wrapper, predict. Configs are the
shipped YAML predictor blocks with only the width and epoch count shrunk, so
the head, posterior and prior settings under test are the real ones."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2" / "predictor"
_RADIUS, _K = 0.37, 7


def _shrunk(arm, block):
    pred = OmegaConf.load(CONFIG_DIR / f"{arm}.yaml").predictor
    pred[block].hidden_dims = [16, 16]
    pred[block].max_epochs = 2
    pred.batch_size = 64
    pred.val_batch_size = 64
    return OmegaConf.create({
        "device": "cpu",
        "probability": {"attractor_radius": _RADIUS, "num_mc_samples": _K},
        "predictor": OmegaConf.to_container(pred, resolve=False),
    })


def _endpoint_file(path, n=256, seed=0):
    from adaptive_roa.systems.cartpole import CartPoleSystem

    rng = np.random.default_rng(seed)
    normalized = rng.uniform(-1.0, 1.0, size=(n, 4))
    normalized[:, 1] *= np.pi
    start = CartPoleSystem().denormalize_state(
        torch.as_tensor(normalized, dtype=torch.float32)).numpy()
    # Two terminal regions, so the mixture has something to separate.
    end = np.where(start[:, :1] > 0, start * 0.1, start * 0.0 + np.array([5.0, 0.0, 4.0, 4.0]))
    np.savetxt(path, np.column_stack([start, end]))
    return str(path)


@pytest.mark.parametrize("arm", ["det_mlp_reg", "det_bnn_reg"])
def test_endpoint_arm_trains_exports_and_predicts(arm, tmp_path):
    from adaptive_roa.adaptive_v2.trainers.final_state_trainer import FinalStateTrainer
    from adaptive_roa.predictors.heads import MixtureFinalStateHead
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.systems.cartpole import CartPoleSystem

    cfg = _shrunk(arm, "final_state")
    files = {"train": _endpoint_file(tmp_path / "train.txt"),
             "val": _endpoint_file(tmp_path / "val.txt", n=128, seed=1)}
    run_dir = tmp_path / "run"
    fitted = FinalStateTrainer(cfg, CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(run_dir / "epoch_000"))
    assert isinstance(fitted.head, MixtureFinalStateHead)

    loaded = get_probabilistic_classifier_class(arm).load_from_run(
        str(run_dir), 0, cfg, CartPoleSystem(), device="cpu")
    states = np.loadtxt(tmp_path / "val.txt")[:64, :4]
    probs = loaded.predict(states)
    total = probs.p_success + probs.p_failure + probs.p_invalid
    np.testing.assert_allclose(total, np.ones(len(states)), atol=1e-12)
    np.testing.assert_allclose(probs.p_success * _K, np.round(probs.p_success * _K), atol=1e-9)

    ckpt = run_dir / "epoch_000" / "checkpoints"
    if arm == "det_bnn_reg":
        record = json.loads((ckpt / "laplace_fit.json").read_text())
        assert record["head"] == "mixture"
        names = [b["block"] for b in record["blocks"]]
        assert names == ["logits"] + [f"location_{c}" for c in range(5)]
        assert all(b["tuned"] and b["prior_precision"] > 0 for b in record["blocks"])
        assert (ckpt / "laplace_blocks.pt").exists()
        # The reloaded posterior samples exactly as the fitted one.
        x = torch.as_tensor(states[:8], dtype=torch.float32)
        emb = CartPoleSystem().embed_state_for_model(CartPoleSystem().normalize_state(x))
        a = fitted.posterior.forward_sample(emb, generator=torch.Generator().manual_seed(2))
        b = loaded.handle.posterior.forward_sample(emb, generator=torch.Generator().manual_seed(2))
        torch.testing.assert_close(a, b)
        # Scale rows are not in any block: they never move across draws.
        lay = fitted.head.param_layout()
        d1 = fitted.posterior.forward_sample(emb, generator=torch.Generator().manual_seed(3))
        d2 = fitted.posterior.forward_sample(emb, generator=torch.Generator().manual_seed(4))
        scale_rows = sum(lay["scales"], [])
        torch.testing.assert_close(d1[:, scale_rows], d2[:, scale_rows])
        assert not torch.allclose(d1[:, lay["logits"]], d2[:, lay["logits"]])
    else:
        assert not (ckpt / "laplace_fit.json").exists()


def _classification_files(tmp_path):
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        X = rng.uniform(-1.0, 1.0, size=(256, 4))
        np.savetxt(tmp_path / f"{split}.txt",
                   np.column_stack([X, (np.abs(X[:, 1]) < 0.5).astype(int)]))
    return {"train": str(tmp_path / "train.txt"), "val": str(tmp_path / "val.txt")}


@pytest.mark.parametrize("arm", ["det_mlp_clf", "det_bnn_clf"])
def test_outcome_arm_trains_exports_and_predicts(arm, tmp_path):
    from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.systems.cartpole import CartPoleSystem

    cfg = _shrunk(arm, "bnn")
    run_dir = tmp_path / "run"
    fitted = BayesianMLPTrainer(cfg, CartPoleSystem(), "cartpole").fit(
        _classification_files(tmp_path), str(run_dir / "epoch_000"))
    loaded = get_probabilistic_classifier_class(arm).load_from_run(
        str(run_dir), 0, cfg, CartPoleSystem(), device="cpu")
    states = np.random.default_rng(1).uniform(-1, 1, size=(32, 4))
    p = loaded.predict(states).p_success
    assert p.shape == (32,) and ((p >= 0) & (p <= 1)).all()

    ckpt = run_dir / "epoch_000" / "checkpoints"
    if arm == "det_bnn_clf":
        record = json.loads((ckpt / "laplace_fit.json").read_text())
        (block,) = record["blocks"]
        assert block["tuned"] and block["prior_precision"] > 0
        assert fitted.posterior.prior_precision == pytest.approx(block["prior_precision"])
        torch.testing.assert_close(loaded.handle.posterior.posterior_covariance,
                                   fitted.posterior.posterior_covariance)
        assert loaded.handle.n_marginal_samples == 64
    else:
        assert loaded.handle.n_marginal_samples == 1
        assert not (ckpt / "laplace_cov.pt").exists()

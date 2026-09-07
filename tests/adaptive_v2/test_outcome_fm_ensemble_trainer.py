"""The outcome-FM ensemble trainer, end to end on a tiny pool.

Mirrors what `BayesianMLPTrainer`'s ensemble branch does
(bayesian_mlp_trainer.py:198-217): one seed, optimizer and shuffle per member,
each member's best-val weights reloaded before assembly, and a top-level
`best-ensemble.ckpt` because the engine warm-starts by globbing
`checkpoints/best*.ckpt`.
"""
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


def _dataset(tmp_path, n=240, seed=0):
    """(state, label) rows in the layout AdaptiveClassificationDataModule reads."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 4))
    # A learnable rule so members have something to disagree about early.
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.float64)
    rows = np.column_stack([x, y])
    train = tmp_path / "train.txt"; val = tmp_path / "val.txt"
    np.savetxt(train, rows[: int(0.8 * n)])
    np.savetxt(val, rows[int(0.8 * n) :])
    return {"train": str(train), "val": str(val)}


def _system(tmp_path):
    from adaptive_roa.systems.cartpole import CartPoleSystem
    bounds = {k: {"min": -3.0, "max": 3.0} for k in ("x", "theta", "x_dot", "theta_dot")}
    (tmp_path / "dataset_description.json").write_text(json.dumps({
        "achieved_bounds": bounds,
        "state_space": {"state_order": ["x", "theta", "x_dot", "theta_dot"]},
    }))
    return CartPoleSystem(dataset_dir=str(tmp_path))


def _cfg(n_members=3):
    return OmegaConf.create({
        "seed": 7,
        "device": "cpu",
        "predictor": {
            "type": "classifier",
            "name": "fm_outcome_ensemble",
            "batch_size": 64,
            "outcome_fm": {
                "n_members": n_members,
                "hidden_dims": [16, 16],
                "max_epochs": 2,
                "patience": 2,
                "num_ode_steps": 8,
                "forward_readout": "exact",
            },
            "lightning_trainer": {"gradient_clip_val": 1.0, "log_every_n_steps": 5},
        },
    })


@pytest.mark.slow
def test_trainer_returns_a_usable_ensemble(tmp_path):
    from adaptive_roa.adaptive_v2.trainers.outcome_fm_ensemble_trainer import (
        EnsembleOutcomeFMTrainer,
    )
    system = _system(tmp_path)
    files = _dataset(tmp_path)
    out = tmp_path / "run"
    handle = EnsembleOutcomeFMTrainer(_cfg(3), system, "cartpole").fit(files, str(out))

    post = getattr(handle, "posterior", handle)
    assert post.n_members == 3, f"expected 3 members, got {post.n_members}"

    # The engine warm-starts by globbing checkpoints/best*.ckpt, so one must sit
    # at the top level rather than only inside the per-member directories.
    assert list((out / "checkpoints").glob("best*.ckpt")), \
        "no top-level best*.ckpt; the engine's warm start would find nothing"

    # The handle must answer the classifier contract: raw states -> logits.
    raw = torch.tensor([[0.5, 0.1, 0.0, 0.0], [-0.5, -0.1, 0.0, 0.0]])
    logits = handle(raw)
    assert logits.shape[0] == 2 and torch.isfinite(logits).all(), logits

    # Independent seeds must produce members that actually differ, or BALD and
    # epistemic variance are identically zero and the whole arm is pointless.
    from adaptive_roa.adaptive_v2.probability.ensemble_prob import (
        EnsembleClassifierProbabilityBackend,
    )
    backend = EnsembleClassifierProbabilityBackend(
        OmegaConf.create({"attractor_radius": 0.1}), system, "cpu")
    backend.bind_model(handle)
    p = backend.estimate_members(raw.numpy())
    assert p.shape == (3, 2), f"expected [3, 2], got {p.shape}"
    assert p.std(axis=0).max() > 1e-6, "members agree exactly; no epistemic signal"

    # Exact readout does no sampling, so BALD needs no finite-sample correction.
    assert backend.member_sample_size is None

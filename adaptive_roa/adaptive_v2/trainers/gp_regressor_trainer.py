"""GP regressor trainer for the gp_reg final-state arm.

Contract matches every sibling: __init__(cfg, system, system_name) and
fit(dataset_files, output_dir, resume_checkpoint=None) -> model handle. There is
no Lightning loop -- GPRegressor.fit owns its own minibatched ELBO optimization.

MODEL SELECTION: like the BNN final-state arms, this arm selects on the held-out
validation set rather than keeping the terminal state. The BNN arms early-stop
and checkpoint on ``val_nll`` (final_state_trainer.py's ``_MONITOR``); this one
evaluates the same quantity -- the mean per-point marginal predictive NLL -- every
``gp.eval_every`` ELBO iterations and restores the best-scoring state before
returning, so every final-state arm is selected under one criterion. Running a
fixed iteration budget and saving the last state, as this trainer previously did,
would have made ``gp_reg``'s reported numbers incomparable with its siblings'.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from adaptive_roa.adaptive_v2.trainers.final_state_trainer import (
    _DATAMODULES,
    _full_dataset_tensors,
)
from adaptive_roa.predictors.embedding import EmbeddedStateDecoder
from adaptive_roa.predictors.gp_final_state_handle import GPFinalStateHandle
from adaptive_roa.predictors.gp_regressor import GPRegressor


class GPRegressorTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name

    @property
    def _predictor_cfg(self):
        pred = self.cfg.get("predictor")
        return pred if pred is not None else self.cfg

    def _create_datamodule(self, dataset_files):
        dm_cls = _DATAMODULES.get(self.system_name)
        if dm_cls is None:
            raise ValueError(
                f"no endpoint datamodule for system {self.system_name!r}; "
                f"expected one of {sorted(_DATAMODULES)}"
            )
        dm = dm_cls(
            data_file=dataset_files["train"],
            validation_file=dataset_files["val"],
            test_file=dataset_files["val"],
            batch_size=self._predictor_cfg.get("batch_size", 1024),
            val_batch_size=self._predictor_cfg.get("val_batch_size", 2048),
            num_workers=0,
        )
        dm.setup()
        return dm

    def _features_and_targets(self, dataset):
        """Both features and targets live in EMBEDDED, NORMALIZED space: an angle
        regressed directly would carry the +/-pi seam, and a quaternion would
        carry the double cover."""
        starts, ends = _full_dataset_tensors(dataset)
        return (
            self.system.embed_state_for_model(self.system.normalize_state(starts)),
            self.system.embed_state_for_model(self.system.normalize_state(ends)),
        )

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        gp_cfg = self._predictor_cfg.get("gp", {})
        # Same resolution and default as every other adaptive_v2 trainer
        # (final_state_trainer.py, bayesian_mlp_trainer.py, classifier_trainer.py,
        # partx/trainer.py).
        device = str(self._predictor_cfg.get("device", self.cfg.get("device", "cuda:0")))
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        decoder = EmbeddedStateDecoder(self.system)
        data_module = self._create_datamodule(dataset_files)
        feats, targets = self._features_and_targets(data_module.train_dataset)
        val_feats, val_targets = self._features_and_targets(data_module.val_dataset)

        gp = GPRegressor(
            num_tasks=decoder.embed_dim,
            input_dim=int(feats.shape[-1]),
            n_inducing=int(gp_cfg.get("n_inducing", 128)),
            kernel=str(gp_cfg.get("kernel", "matern52")),
            n_iters=int(gp_cfg.get("n_iters", 300)),
            lr=float(gp_cfg.get("lr", 0.01)),
            batch_size=int(gp_cfg.get("batch_size", 1024)),
            eval_every=int(gp_cfg.get("eval_every", 10)),
            device=device,
        )

        if resume_checkpoint and Path(resume_checkpoint).exists():
            print(f"Warm start: loading GP regressor state from {resume_checkpoint}")
            gp.load_state_dict(torch.load(resume_checkpoint, map_location="cpu",
                                          weights_only=False))
            gp.to(device)

        gp.fit(feats, targets, val_feats, val_targets)

        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # MUST match the engine's glob, checkpoints/best*.ckpt (engine.py:140).
        # The state written here is the best-val_nll one, not the terminal one,
        # matching what the sibling arms' best*.ckpt carries.
        torch.save(gp.state_dict(), ckpt_dir / "best-gp.ckpt")

        return GPFinalStateHandle(gp, self.system, device=device).eval().to(device)

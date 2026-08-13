"""Scalar-outcome flow-matching trainer adapter for adaptive v2.

Mirrors ``ClassifierTrainer``'s contract exactly:
    __init__(cfg, system, system_name)
    fit(dataset_files, output_dir, resume_checkpoint=None) -> model_handle

and consumes the SAME ``AdaptiveClassificationDataModule`` over ``(state, label)``
rows. That is deliberate: the outcome-FM arm and the classifier arm then see
byte-identical training data at every adaptive epoch, so the target x machinery
contrast is clean by construction rather than by careful matching after the fact.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from adaptive_roa.data.adaptive_classification_data import AdaptiveClassificationDataModule
from adaptive_roa.model.outcome_flow_matcher import OutcomeFlowMatcher, OutcomeVelocityMLP


class OutcomeFMTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name

    @property
    def _predictor_cfg(self):
        pred = self.cfg.get("predictor")
        return pred if pred is not None else self.cfg

    def _embedded_dim(self) -> int:
        dummy = torch.zeros(1, int(self.system.state_dim))
        embedded = self.system.embed_state_for_model(self.system.normalize_state(dummy))
        return int(embedded.shape[-1])

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        fm_cfg = self._predictor_cfg.get("outcome_fm", {})

        data_module = AdaptiveClassificationDataModule(
            train_file=dataset_files["train"],
            val_file=dataset_files["val"],
            batch_size=self._predictor_cfg.get("batch_size", 1024),
            # Matches ClassifierTrainer: data is already in-memory tensors, and
            # workers break on NFS (rmtree of `.nfs*` temp dirs => Errno 16).
            num_workers=0,
        )
        data_module.setup()

        velocity_net = OutcomeVelocityMLP(
            condition_dim=self._embedded_dim(),
            # Default mirrors ClassifierMLP's default so the two arms have
            # matched capacity unless a config deliberately breaks the match.
            hidden_dims=list(fm_cfg.get("hidden_dims", [256, 512, 256])),
            dropout=float(fm_cfg.get("dropout", 0.0)),
            activation=str(fm_cfg.get("activation", "relu")),
            num_time_freqs=int(fm_cfg.get("num_time_freqs", 4)),
        )
        module = OutcomeFlowMatcher(
            velocity_net=velocity_net,
            system=self.system,
            lr=float(fm_cfg.get("lr", 1e-3)),
            weight_decay=float(fm_cfg.get("weight_decay", 1e-5)),
            num_ode_steps=int(fm_cfg.get("num_ode_steps", 100)),
            # Must agree with probability.readout; OutcomeFMProbabilityBackend
            # rejects a mismatch at bind time rather than letting calibration and
            # evaluation quietly use different estimators.
            forward_readout=str(fm_cfg.get("forward_readout", "exact")),
            forward_num_samples=int(fm_cfg.get("forward_num_samples", 100)),
        )

        if resume_checkpoint and Path(resume_checkpoint).exists():
            print(f"Warm start: loading outcome-FM weights from {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            module.load_state_dict(ckpt["state_dict"], strict=False)

        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer_cfg = self._predictor_cfg.get("lightning_trainer", {})

        callbacks = [
            ModelCheckpoint(
                dirpath=str(checkpoint_dir), monitor="val_loss", mode="min",
                save_top_k=1, save_last=True, filename="best-{epoch:02d}-{val_loss:.4f}",
            ),
            EarlyStopping(monitor="val_loss", mode="min", patience=int(fm_cfg.get("patience", 20))),
        ]
        logger = CSVLogger(save_dir=output_dir, name="outcome_fm_logs")

        device = str(self._predictor_cfg.get("device", self.cfg.get("device", "cuda:0")))
        use_gpu = device.startswith("cuda") and torch.cuda.is_available()
        trainer = pl.Trainer(
            max_epochs=int(fm_cfg.get("max_epochs", 200)),
            accelerator="gpu" if use_gpu else "cpu",
            devices=1,
            gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
            log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
            check_val_every_n_epoch=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=callbacks,
            logger=logger,
        )

        trainer.fit(module, data_module)

        # Load best-val weights in place (keeps `system` attached; the constructor
        # args cannot be reconstructed by load_from_checkpoint).
        best = glob.glob(str(checkpoint_dir / "best*.ckpt"))
        if best:
            ckpt = torch.load(best[0], map_location="cpu", weights_only=False)
            module.load_state_dict(ckpt["state_dict"], strict=False)

        module.eval()
        if use_gpu:
            module.to(device)
        return module

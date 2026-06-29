"""Discriminative-classifier trainer adapter for adaptive v2.

Mirrors ``FlowMatchingTrainer``'s contract:
    __init__(cfg, system, system_name)
    fit(dataset_files, output_dir, resume_checkpoint=None) -> model_handle

``dataset_files`` is ``{"train": <file>, "val": <file>}`` of ``(state, label)``
rows. Returns a trained ``ClassifierModule`` (with ``system`` attached) loaded
with the best-val-loss weights, ready for the probability estimator.
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
from adaptive_roa.model.classifier_mlp import ClassifierMLP, ClassifierModule


class ClassifierTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name

    def _embedded_dim(self) -> int:
        dummy = torch.zeros(1, int(self.system.state_dim))
        embedded = self.system.embed_state_for_model(self.system.normalize_state(dummy))
        return int(embedded.shape[-1])

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        cls_cfg = self.cfg.get("classifier", {})

        data_module = AdaptiveClassificationDataModule(
            train_file=dataset_files["train"],
            val_file=dataset_files["val"],
            batch_size=self.cfg.get("batch_size", 1024),
            # Data is pre-loaded into in-memory tensors; worker processes add no
            # value and break on NFS (rmtree of `.nfs*` temp dirs => Errno 16).
            num_workers=0,
        )
        data_module.setup()  # compute pos_weight before building the module

        mlp = ClassifierMLP(
            input_dim=self._embedded_dim(),
            hidden_dims=list(cls_cfg.get("hidden_dims", [256, 512, 256])),
            output_dim=1,
            dropout=float(cls_cfg.get("dropout", 0.0)),
        )
        module = ClassifierModule(
            mlp=mlp,
            system=self.system,
            pos_weight=data_module.pos_weight,
            lr=float(cls_cfg.get("lr", 1e-3)),
            weight_decay=float(cls_cfg.get("weight_decay", 1e-5)),
        )

        if resume_checkpoint and Path(resume_checkpoint).exists():
            print(f"Warm start: loading classifier weights from {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            module.load_state_dict(ckpt["state_dict"], strict=False)

        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer_cfg = self.cfg.get("lightning_trainer", {})

        callbacks = [
            ModelCheckpoint(
                dirpath=str(checkpoint_dir), monitor="val_loss", mode="min",
                save_top_k=1, save_last=True, filename="best-{epoch:02d}-{val_loss:.4f}",
            ),
            EarlyStopping(monitor="val_loss", mode="min", patience=int(cls_cfg.get("patience", 20))),
        ]
        logger = CSVLogger(save_dir=output_dir, name="classifier_logs")

        device = str(self.cfg.get("device", "cuda:0"))
        use_gpu = device.startswith("cuda") and torch.cuda.is_available()
        trainer = pl.Trainer(
            max_epochs=int(cls_cfg.get("max_epochs", 200)),
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

        # Load best-val weights into the in-memory module (keeps `system` attached;
        # load_from_checkpoint cannot reconstruct the mlp/system constructor args).
        best = glob.glob(str(checkpoint_dir / "best*.ckpt"))
        if best:
            ckpt = torch.load(best[0], map_location="cpu", weights_only=False)
            module.load_state_dict(ckpt["state_dict"], strict=False)

        module.eval()
        if use_gpu:
            module.to(device)
        return module

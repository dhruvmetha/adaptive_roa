"""Trainer for Bayesian MLP arms with an outcome head.

Mirrors ``ClassifierTrainer``'s contract:
    __init__(cfg, system, system_name)
    fit(dataset_files, output_dir, resume_checkpoint=None) -> OutcomeModelHandle
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from adaptive_roa.data.adaptive_classification_data import AdaptiveClassificationDataModule
from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp
from adaptive_roa.predictors.handles import OutcomeModelHandle
from adaptive_roa.predictors.posteriors import EnsemblePosterior


class _OutcomeModule(pl.LightningModule):
    """One posterior's ELBO: Bernoulli likelihood + KL(q||p)/N."""

    def __init__(self, posterior, system, pos_weight, lr, weight_decay, kl_weight, n_train):
        super().__init__()
        self.posterior = posterior
        self.system = system  # plain attr; methods are device-agnostic
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.kl_weight = float(kl_weight)
        self.n_train = max(int(n_train), 1)
        self.register_buffer("pos_weight", torch.as_tensor(float(pos_weight)))

    def forward(self, raw_states):
        embedded = self.system.embed_state_for_model(self.system.normalize_state(raw_states))
        return self.posterior.forward_sample(embedded)

    def _step(self, batch, stage: str):
        y = batch["label"].float().view(-1)
        logits = self(batch["inputs"]).view(-1)
        nll = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        # The ELBO's KL is a per-DATASET term while nll is a per-batch mean, so
        # scale by 1/N to put them on the same footing.
        kl = self.posterior.kl_divergence() / self.n_train
        loss = nll + self.kl_weight * kl
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_nll", nll, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


class BayesianMLPTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name
        bnn = self._predictor_cfg.get("bnn", {})
        self.posterior_kind = str(bnn.get("posterior", "mfvi"))
        # beta: 1.0 is untempered. Any other value makes the run a TEMPERED
        # result and must be labelled as such in the writeup.
        self.kl_weight = float(bnn.get("kl_weight", 1.0))

    @property
    def _predictor_cfg(self):
        pred = self.cfg.get("predictor")
        return pred if pred is not None else self.cfg

    def _embedded_dim(self) -> int:
        dummy = torch.zeros(1, int(self.system.state_dim))
        return int(self.system.embed_state_for_model(
            self.system.normalize_state(dummy)).shape[-1])

    def _build(self, bnn, posterior_kind):
        return build_bayesian_mlp(
            input_dim=self._embedded_dim(),
            hidden_dims=list(bnn.get("hidden_dims", [256, 512, 256])),
            output_dim=1,
            posterior=posterior_kind,
            prior_sigma=float(bnn.get("prior_sigma", 1.0)),
            n_members=int(bnn.get("n_members", 5)),
            dropout=float(bnn.get("dropout", 0.0)),
            activation=str(bnn.get("activation", "relu")),
        )

    def _run_lightning(self, module, data_module, bnn, ckpt_dir, tag):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer_cfg = self._predictor_cfg.get("lightning_trainer", {})
        device = str(self._predictor_cfg.get("device", self.cfg.get("device", "cuda:0")))
        use_gpu = device.startswith("cuda") and torch.cuda.is_available()
        trainer = pl.Trainer(
            max_epochs=int(bnn.get("max_epochs", 200)),
            accelerator="gpu" if use_gpu else "cpu",
            devices=1,
            gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
            log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
            check_val_every_n_epoch=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                ModelCheckpoint(
                    dirpath=str(ckpt_dir), monitor="val_loss", mode="min",
                    save_top_k=1, filename=f"{tag}-{{epoch:02d}}-{{val_loss:.4f}}",
                ),
                EarlyStopping(monitor="val_loss", mode="min",
                              patience=int(bnn.get("patience", 20))),
            ],
            logger=CSVLogger(save_dir=str(ckpt_dir.parent), name=f"bnn_logs_{tag}"),
        )
        trainer.fit(module, data_module)
        return use_gpu, device

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        bnn = self._predictor_cfg.get("bnn", {})
        ckpt_dir = Path(output_dir) / "checkpoints"

        data_module = AdaptiveClassificationDataModule(
            train_file=dataset_files["train"],
            val_file=dataset_files["val"],
            batch_size=self._predictor_cfg.get("batch_size", 1024),
            num_workers=0,  # in-memory tensors; workers break on NFS
        )
        data_module.setup()
        # The datamodule keeps its datasets on the private `_train` / `_val`
        # attrs (adaptive_classification_data.py:58-59); each dataset exposes
        # `.states` and `.labels` as full in-memory tensors.
        n_train = len(data_module._train)

        common = dict(
            system=self.system, pos_weight=data_module.pos_weight,
            lr=float(bnn.get("lr", 1e-3)),
            weight_decay=float(bnn.get("weight_decay", 1e-5)),
            n_train=n_train,
        )

        if self.posterior_kind == "ensemble":
            # Members must be independent: own seed, own optimizer, own shuffling.
            members, use_gpu, device = [], False, "cpu"
            for m in range(int(bnn.get("n_members", 5))):
                torch.manual_seed(int(bnn.get("seed", 0)) + m)
                member = self._build(bnn, "deterministic")
                module = _OutcomeModule(posterior=member, kl_weight=0.0, **common)
                use_gpu, device = self._run_lightning(
                    module, data_module, bnn, ckpt_dir / f"member_{m}", f"best_member{m}"
                )
                members.append(member)
            posterior = EnsemblePosterior(members)
            # The engine's warm start globs checkpoints/best*.ckpt (engine.py:140-142),
            # so the assembled ensemble needs one at the top level.
            torch.save({"state_dict": posterior.state_dict()}, ckpt_dir / "best-ensemble.ckpt")
        else:
            # "laplace" builds a body+head split; build_bayesian_mlp handles it.
            posterior = self._build(bnn, self.posterior_kind)
            kl_weight = self.kl_weight if self.posterior_kind == "mfvi" else 0.0
            module = _OutcomeModule(posterior=posterior, kl_weight=kl_weight, **common)
            if resume_checkpoint and Path(resume_checkpoint).exists():
                ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
                module.load_state_dict(ckpt["state_dict"], strict=False)
            use_gpu, device = self._run_lightning(module, data_module, bnn, ckpt_dir, "best")

            if self.posterior_kind == "laplace":
                # Post-hoc GGN over the full training set. Without this the arm
                # stays at the MAP point estimate and reports zero uncertainty.
                posterior.eval()
                with torch.no_grad():
                    x = data_module._train.states.float()
                    y = data_module._train.labels.float().view(-1)
                    embedded = self.system.embed_state_for_model(self.system.normalize_state(x))
                    posterior.fit(posterior.body(embedded), y, task="outcome")
                # `_cov` is a non-persistent buffer (its shape is unknown until
                # fit), so it will NOT round-trip through the Lightning
                # checkpoint. Save it explicitly or the exported arm silently
                # falls back to its MAP point estimate and reports zero spread.
                torch.save(posterior.posterior_covariance, ckpt_dir / "laplace_cov.pt")

        handle = OutcomeModelHandle(
            posterior, self.system,
            n_marginal_samples=int(bnn.get("n_marginal_samples", 64)),
            seed=int(bnn.get("seed", 0)),
        ).eval()
        return handle.to(device) if use_gpu else handle

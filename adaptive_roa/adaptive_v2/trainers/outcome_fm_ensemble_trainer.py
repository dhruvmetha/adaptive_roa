"""Deep ensemble of scalar-outcome flow matchers.

`OutcomeFMTrainer` builds one flow; this builds M of them and assembles an
`EnsemblePosterior`, which is what unlocks every epistemic and yield acquisition
strategy for this arm. Those strategies price candidates from PER-MEMBER
probabilities (`yield_mlp.py:213-216` refuses a backend without
`estimate_members`), and BALD is identically zero with a single model, so the
single-flow arm cannot run them at all.

Structure copied from `BayesianMLPTrainer`'s ensemble branch
(bayesian_mlp_trainer.py:198-217), because the parts that are easy to get wrong
are already solved there:

  * one seed, optimizer and shuffle per member, so members are independent and
    the spread means something,
  * each member's best-val weights reloaded before assembly, so the ensemble is
    best-selected member-wise rather than last-epoch,
  * a top-level `best-ensemble.ckpt`, because the engine warm-starts by globbing
    `checkpoints/best*.ckpt` (engine.py:140-142) and would otherwise find only
    the per-member files.

Members are wrapped in `EmbeddedOutcomeFM` so they take already-embedded states,
which is the contract `EnsemblePosterior` and `OutcomeModelHandle` use. That is
what lets this arm reuse `EnsembleClassifierProbabilityBackend` unchanged
instead of needing a probability backend of its own.

Members train SEQUENTIALLY in one process, like `clf_ensemble` and unlike
`fm_ensemble` which spawns a process per member. The velocity net is the same
small MLP the classifier uses, so one GPU per arm is enough and a five-arm
campaign fits on one node rather than needing twenty-five cards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from adaptive_roa.adaptive_v2.trainers._seeding import resolve_seed_base
from adaptive_roa.data.adaptive_classification_data import AdaptiveClassificationDataModule
from adaptive_roa.model.outcome_flow_matcher import (
    EmbeddedOutcomeFM, OutcomeFlowMatcher, OutcomeVelocityMLP,
)
from adaptive_roa.predictors.bayesian_mlp import outcome_handle_from_cfg
from adaptive_roa.predictors.posteriors import EnsemblePosterior


class EnsembleOutcomeFMTrainer:
    """Same contract as `OutcomeFMTrainer`: fit(files, out, resume) -> handle."""

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

    def _build_member(self, fm_cfg) -> OutcomeFlowMatcher:
        velocity_net = OutcomeVelocityMLP(
            condition_dim=self._embedded_dim(),
            # Defaults mirror ClassifierMLP so the two arms stay at matched
            # capacity unless a config deliberately breaks the match.
            hidden_dims=list(fm_cfg.get("hidden_dims", [256, 512, 256])),
            dropout=float(fm_cfg.get("dropout", 0.0)),
            activation=str(fm_cfg.get("activation", "relu")),
            num_time_freqs=int(fm_cfg.get("num_time_freqs", 4)),
        )
        return OutcomeFlowMatcher(
            velocity_net=velocity_net,
            system=self.system,
            lr=float(fm_cfg.get("lr", 1e-3)),
            weight_decay=float(fm_cfg.get("weight_decay", 1e-5)),
            num_ode_steps=int(fm_cfg.get("num_ode_steps", 100)),
            # `exact` does no sampling, so member_sample_size is None and BALD
            # needs no finite-sample correction. Changing it to `mc` would make
            # EnsembleClassifierProbabilityBackend's hardcoded None wrong.
            forward_readout=str(fm_cfg.get("forward_readout", "exact")),
            forward_num_samples=int(fm_cfg.get("forward_num_samples", 100)),
        )

    def _train_member(self, module, data_module, fm_cfg, ckpt_dir: Path, tag: str):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer_cfg = self._predictor_cfg.get("lightning_trainer", {})
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
            callbacks=[
                ModelCheckpoint(
                    dirpath=str(ckpt_dir), monitor="val_loss", mode="min",
                    save_top_k=1, filename=f"{tag}-{{epoch:02d}}-{{val_loss:.4f}}",
                ),
                EarlyStopping(monitor="val_loss", mode="min",
                              patience=int(fm_cfg.get("patience", 20))),
            ],
            logger=CSVLogger(save_dir=str(ckpt_dir.parent), name=f"outcome_fm_logs_{tag}"),
        )
        trainer.fit(module, data_module)

        # Reload best-val weights in place. load_from_checkpoint cannot rebuild
        # the constructor args (`system` is a live object), so load the state
        # dict into the module we already hold, exactly as OutcomeFMTrainer does.
        best = sorted(ckpt_dir.glob(f"{tag}-*.ckpt"))
        if best:
            ckpt = torch.load(str(best[0]), map_location="cpu", weights_only=False)
            module.load_state_dict(ckpt["state_dict"], strict=False)
        return use_gpu, device

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        fm_cfg = self._predictor_cfg.get("outcome_fm", {})
        n_members = int(fm_cfg.get("n_members", 5))
        if n_members < 2:
            raise ValueError(
                f"an outcome-FM ensemble needs >= 2 members, got {n_members}. "
                "With one member BALD and epistemic variance are identically "
                "zero, so every acquisition this arm exists for is inert."
            )

        data_module = AdaptiveClassificationDataModule(
            train_file=dataset_files["train"],
            val_file=dataset_files["val"],
            batch_size=self._predictor_cfg.get("batch_size", 1024),
            num_workers=0,  # in-memory tensors; workers break on NFS
        )
        data_module.setup()

        ckpt_root = Path(output_dir) / "checkpoints"
        ckpt_root.mkdir(parents=True, exist_ok=True)
        seed_base = resolve_seed_base(self.cfg, fm_cfg)

        members, use_gpu, device = [], False, "cpu"
        for m in range(n_members):
            # Own seed => own init, own shuffling, own optimizer trajectory.
            # Without this the members are bit-identical and the spread is 0.
            torch.manual_seed(seed_base + m)
            member = self._build_member(fm_cfg)
            use_gpu, device = self._train_member(
                member, data_module, fm_cfg, ckpt_root / f"member_{m}", f"best_member{m}"
            )
            members.append(EmbeddedOutcomeFM(member.eval()))

        posterior = EnsemblePosterior(members)
        torch.save({"state_dict": posterior.state_dict()}, ckpt_root / "best-ensemble.ckpt")

        # OutcomeModelHandle marginalises in probability space computed in log
        # space; reused rather than reimplemented because the naive form
        # saturates in float32 and silently returns +inf logits.
        handle = outcome_handle_from_cfg(posterior, self.system, fm_cfg).eval()
        return handle.to(device) if use_gpu else handle

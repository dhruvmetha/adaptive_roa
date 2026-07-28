"""Trainer for Bayesian MLP arms with an outcome head.

Mirrors ``ClassifierTrainer``'s contract:
    __init__(cfg, system, system_name)
    fit(dataset_files, output_dir, resume_checkpoint=None) -> OutcomeModelHandle
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from adaptive_roa.data.adaptive_classification_data import AdaptiveClassificationDataModule
from adaptive_roa.predictors.bayesian_mlp import build_from_cfg, outcome_handle_from_cfg
from adaptive_roa.predictors.posteriors import EnsemblePosterior

# Model selection monitors the plain validation NLL, NOT the ELBO. The ELBO adds
# kl_weight * KL/N, which is data-INDEPENDENT and shrinks monotonically as the
# variational scales contract; for MFVI it dominates val_nll by 2-3 orders of
# magnitude, so an ELBO-monitored EarlyStopping never fires and an ELBO-monitored
# ModelCheckpoint always keeps the LAST epoch. Selecting on val_nll also puts all
# three BNN arms on the same criterion as the plain-MLP baseline (val BCE), which
# is required for the arms to be comparable at all. val_loss is still logged as
# an ELBO diagnostic.
_MONITOR = "val_nll"


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
        # val_loss is the ELBO -- a training objective and a diagnostic, but NOT
        # a model-selection criterion (see _MONITOR). val_nll is what the
        # callbacks watch, so it must be an epoch-level metric.
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_nll", nll, prog_bar=True, on_epoch=True, on_step=False)
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
        # Same reasoning as _MONITOR: a monotonically shrinking ELBO never looks
        # like a plateau, so an ELBO-monitored scheduler never drops the LR.
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": _MONITOR}}


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

    def _build(self, bnn, posterior_kind):
        # Shared with the export wrapper so the two can never build different
        # architectures from the same config (see bayesian_mlp.build_from_cfg).
        return build_from_cfg(bnn, self.system, posterior_kind)

    @staticmethod
    def _load_best(module, ckpt_dir: Path, tag: str) -> None:
        """Reload the best-``val_nll`` checkpoint into the in-memory module.

        ``ClassifierTrainer`` does exactly this (classifier_trainer.py:106-109)
        and the export loader reads ``best*.ckpt``. Without this step the
        pipeline would run on LAST-epoch weights while ``load_from_run``
        reproduces BEST-epoch weights, so the exported probabilities would not
        match the run's own artifacts_v2.json -- and the MLP baseline would be
        best-val-selected while the BNN arms were not.

        For the Laplace arm it is also a correctness requirement: the GGN must
        be fitted at the SAME parameters the exported weights carry, so this
        reload has to happen BEFORE ``posterior.fit``.
        """
        best = sorted(glob.glob(str(ckpt_dir / f"{tag}*.ckpt")))
        if not best:
            raise RuntimeError(
                f"no {tag}*.ckpt written in {ckpt_dir}; the model would silently "
                f"keep last-epoch weights while the export loads a best checkpoint"
            )
        print(f"Loading best-{_MONITOR} weights from {best[0]}")
        ckpt = torch.load(best[0], map_location="cpu", weights_only=False)
        module.load_state_dict(ckpt["state_dict"], strict=True)

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
                    dirpath=str(ckpt_dir), monitor=_MONITOR, mode="min",
                    save_top_k=1, filename=f"{tag}-{{epoch:02d}}-{{{_MONITOR}:.4f}}",
                ),
                EarlyStopping(monitor=_MONITOR, mode="min",
                              patience=int(bnn.get("patience", 20))),
            ],
            logger=CSVLogger(save_dir=str(ckpt_dir.parent), name=f"bnn_logs_{tag}"),
        )
        trainer.fit(module, data_module)
        self._load_best(module, ckpt_dir, tag)
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
            n_members = int(bnn.get("n_members", 5))
            member_states = self._ensemble_warm_start_states(resume_checkpoint, n_members)
            # Members must be independent: own seed, own optimizer, own shuffling.
            members, use_gpu, device = [], False, "cpu"
            for m in range(n_members):
                torch.manual_seed(int(bnn.get("seed", 0)) + m)
                member = self._build(bnn, "deterministic")
                if member_states is not None:
                    _load_warm_start(member, member_states[m],
                                     f"{resume_checkpoint} [members.{m}.*]")
                module = _OutcomeModule(posterior=member, kl_weight=0.0, **common)
                use_gpu, device = self._run_lightning(
                    module, data_module, bnn, ckpt_dir / f"member_{m}", f"best_member{m}"
                )
                members.append(member)
            # _run_lightning has already reloaded each member's own best-val
            # weights, so the assembled ensemble is best-selected member-wise.
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
                print(f"Warm start: loading {self.posterior_kind} weights from {resume_checkpoint}")
                ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
                _load_warm_start(module, ckpt.get("state_dict", ckpt), resume_checkpoint)
            # Reloads the best-val_nll checkpoint into `posterior` before returning.
            use_gpu, device = self._run_lightning(module, data_module, bnn, ckpt_dir, "best")

            if self.posterior_kind == "laplace":
                # Post-hoc GGN over the full training set, fitted AFTER the best
                # checkpoint is back in memory: H^-1 is a curvature at a specific
                # parameter vector, so pairing a last-epoch covariance with
                # best-epoch weights would centre the Gaussian at one point and
                # take its curvature from another. Without the fit at all the arm
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

        handle = outcome_handle_from_cfg(posterior, self.system, bnn).eval()
        return handle.to(device) if use_gpu else handle

    @staticmethod
    def _ensemble_warm_start_states(resume_checkpoint, n_members):
        """Split a saved ensemble checkpoint into one state dict per member.

        Without this the ensemble branch silently ignored ``resume_checkpoint``
        and retrained every member from scratch each adaptive epoch, while the
        mfvi/laplace arms warm-started -- an arm asymmetry that changes results,
        not just runtime.
        """
        if not resume_checkpoint or not Path(resume_checkpoint).exists():
            return None
        ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        prefix = "members."
        per_member: dict[int, dict] = {}
        for key, value in state.items():
            k = key[len("posterior."):] if key.startswith("posterior.") else key
            if not k.startswith(prefix):
                continue
            idx, _, rest = k[len(prefix):].partition(".")
            per_member.setdefault(int(idx), {})[rest] = value
        if not per_member:
            raise RuntimeError(
                f"{resume_checkpoint} has no 'members.*' keys; it is not an "
                f"ensemble checkpoint and warm-starting from it would leave "
                f"every member at random init"
            )
        found = len(per_member)
        if found != n_members or sorted(per_member) != list(range(n_members)):
            # Partially warm-starting would leave some members at random init
            # with no visible symptom, so refuse instead.
            raise RuntimeError(
                f"{resume_checkpoint} holds {found} ensemble members "
                f"({sorted(per_member)}) but n_members={n_members}; refusing to "
                f"partially warm-start"
            )
        print(f"Warm start: loading {n_members} ensemble members from {resume_checkpoint}")
        return per_member


def _load_warm_start(module, state, source: str) -> None:
    """Warm-start ``module`` from ``state``, loudly reporting any key mismatch.

    ``strict=False`` on its own turns an architecture or arm mismatch into a
    silent cold start (``ClassifierTrainer`` prints what it loads for the same
    reason). Warm start is an optimization, so a mismatch must not abort the
    run -- but it must be visible in the log rather than inferred later from
    an unexplained jump in the loss curve.
    """
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"WARNING: warm start from {source} did not match the model: "
            f"{len(missing)} missing key(s) {list(missing)[:3]}, "
            f"{len(unexpected)} unexpected key(s) {list(unexpected)[:3]}. "
            f"Those parameters stay at their random initialization."
        )

"""Trainer for final-state predictor arms (a distribution over the endpoint).

Mirrors ``BayesianMLPTrainer``'s contract:
    __init__(cfg, system, system_name)
    fit(dataset_files, output_dir, resume_checkpoint=None) -> FinalStateModelHandle

The likelihood lives in the HEAD (``FinalStateHead``), not in this module: even
the deterministic arm reports a distribution over x_T, because the weights are
a point estimate but the head's Gaussian/wrapped-normal/tangent-space output
never collapses to one. What varies across arms is only the WEIGHT posterior.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from adaptive_roa.data.cartpole_endpoint_data import CartPoleEndpointDataModule
from adaptive_roa.data.pendulum_endpoint_data import PendulumEndpointDataModule
from adaptive_roa.data.quadrotor2d_endpoint_data import Quadrotor2DEndpointDataModule
from adaptive_roa.data.quadrotor3d_endpoint_data import Quadrotor3DEndpointDataModule
from adaptive_roa.predictors.bayesian_mlp import build_from_cfg
from adaptive_roa.predictors.final_state_handle import FinalStateModelHandle
from adaptive_roa.predictors.heads import FinalStateHead
from adaptive_roa.predictors.posteriors import EnsemblePosterior

# Global prediction mode: endpoint data modules (start -> end pairs). Same map
# FlowMatchingTrainer uses (flow_matching_trainer.py:25-31); kept as a separate
# copy here rather than imported so this trainer has no import-time dependency
# on the flow-matching code path.
_DATAMODULES = {
    "pendulum": PendulumEndpointDataModule,
    "cartpole_pybullet": CartPoleEndpointDataModule,
    "quadrotor2d": Quadrotor2DEndpointDataModule,
    "quadrotor3d": Quadrotor3DEndpointDataModule,
}

# Model selection monitors the plain validation NLL, NOT the ELBO. The ELBO adds
# kl_weight * KL/N, which is data-INDEPENDENT and shrinks monotonically as the
# variational scales contract; for MFVI it dominates val_nll by orders of
# magnitude, so an ELBO-monitored EarlyStopping never fires and an ELBO-monitored
# ModelCheckpoint always keeps the LAST epoch. Selecting on val_nll also puts all
# four arms on the same criterion. val_loss is still logged as an ELBO diagnostic.
_MONITOR = "val_nll"


class _FinalStateModule(pl.LightningModule):
    """Head NLL over endpoint pairs, plus KL/N for variational posteriors."""

    def __init__(self, posterior, head, system, lr, weight_decay, kl_weight, n_train):
        super().__init__()
        self.posterior = posterior
        self.head = head
        self.system = system  # plain attr; methods are device-agnostic
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.kl_weight = float(kl_weight)
        self.n_train = max(int(n_train), 1)

    def forward(self, raw_states):
        embedded = self.system.embed_state_for_model(self.system.normalize_state(raw_states))
        return self.posterior.forward_sample(embedded)

    def _step(self, batch, stage: str):
        params = self(batch["start_state"])
        nll = self.head.nll(params, batch["end_state"]).mean()
        # KL is a per-DATASET term while nll is a per-batch mean, so scale by 1/N.
        kl = self.posterior.kl_divergence() / self.n_train
        loss = nll + self.kl_weight * kl
        # Callbacks monitor val_nll, NOT val_loss: the KL term is data-independent
        # and monotone, so monitoring the total makes EarlyStopping inert and
        # always keeps the last epoch.
        self.log(f"{stage}_nll", nll, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False)
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
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": _MONITOR}}


class FinalStateTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name
        fs = self._predictor_cfg.get("final_state", {})
        self.posterior_kind = str(fs.get("posterior", "mfvi"))
        # beta: 1.0 is untempered. Any other value makes the run a TEMPERED
        # result and must be labelled as such in the writeup.
        self.kl_weight = float(fs.get("kl_weight", 1.0))
        # beta-NLL exponent (Seitzer et al. 2022); 0.5 is the head's default.
        self.beta_nll = float(fs.get("beta_nll", 0.5))

        if self.posterior_kind == "ensemble":
            n_members = int(fs.get("n_members", 5))
            k = self._resolve_num_mc_samples()
            if k < 2 * n_members:
                raise ValueError(
                    f"num_mc_samples={k} cannot resolve an ensemble of {n_members} members; "
                    f"forward_sample draws ONE member per call, so an M-atom empirical "
                    f"posterior needs num_mc_samples >= 2*M = {2 * n_members}. Raise "
                    f"num_mc_samples or lower n_members."
                )

    @property
    def _predictor_cfg(self):
        pred = self.cfg.get("predictor")
        return pred if pred is not None else self.cfg

    def _resolve_num_mc_samples(self) -> int:
        """``predictor.num_mc_samples``, else ``probability.num_mc_samples``,
        else the ``endpoint_mc.yaml`` default of 10."""
        val = self._predictor_cfg.get("num_mc_samples", None)
        if val is None:
            prob = self.cfg.get("probability", None)
            val = prob.get("num_mc_samples", None) if prob is not None else None
        return int(val) if val is not None else 10

    def _build(self, fs, posterior_kind, output_dim):
        # Shared with the export wrapper so the two can never build different
        # architectures from the same config (see bayesian_mlp.build_from_cfg).
        return build_from_cfg(fs, self.system, posterior_kind, output_dim=output_dim)

    def _create_datamodule(self, dataset_files: dict):
        dm_cls = _DATAMODULES.get(self.system_name)
        if dm_cls is None:
            raise ValueError(
                f"no endpoint datamodule for system {self.system_name!r}; "
                f"expected one of {sorted(_DATAMODULES)}"
            )
        dm = dm_cls(
            data_file=dataset_files["train"],
            validation_file=dataset_files["val"],
            test_file=dataset_files["val"],  # matches FlowMatchingTrainer:87-89
            batch_size=self._predictor_cfg.get("batch_size", 1024),
            val_batch_size=self._predictor_cfg.get("val_batch_size", 2048),
            num_workers=0,  # in-memory tensors; workers break on NFS
        )
        dm.setup()
        return dm

    @staticmethod
    def _load_best(module, ckpt_dir: Path, tag: str) -> None:
        """Reload the best-``val_nll`` checkpoint into the in-memory module.

        Without this step the pipeline would run on LAST-epoch weights while
        ``load_from_run`` reproduces BEST-epoch weights, so the exported
        predictions would not match the run's own artifacts_v2.json.

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

    def _run_lightning(self, module, data_module, fs, ckpt_dir, tag):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer_cfg = self._predictor_cfg.get("lightning_trainer", {})
        device = str(self._predictor_cfg.get("device", self.cfg.get("device", "cuda:0")))
        use_gpu = device.startswith("cuda") and torch.cuda.is_available()
        trainer = pl.Trainer(
            max_epochs=int(fs.get("max_epochs", 200)),
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
                              patience=int(fs.get("patience", 20))),
            ],
            logger=CSVLogger(save_dir=str(ckpt_dir.parent), name=f"fs_logs_{tag}"),
        )
        trainer.fit(module, data_module)
        self._load_best(module, ckpt_dir, tag)
        return use_gpu, device

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        fs = self._predictor_cfg.get("final_state", {})
        ckpt_dir = Path(output_dir) / "checkpoints"

        data_module = self._create_datamodule(dataset_files)
        n_train = len(data_module.train_dataset)

        head = FinalStateHead(self.system, beta=self.beta_nll)

        common = dict(
            system=self.system, head=head,
            lr=float(fs.get("lr", 1e-3)),
            weight_decay=float(fs.get("weight_decay", 1e-5)),
            n_train=n_train,
        )

        if self.posterior_kind == "ensemble":
            n_members = int(fs.get("n_members", 5))
            member_states = self._ensemble_warm_start_states(resume_checkpoint, n_members)
            # Members must be independent: own seed, own optimizer, own shuffling.
            members, use_gpu, device = [], False, "cpu"
            for m in range(n_members):
                torch.manual_seed(int(fs.get("seed", 0)) + m)
                member = self._build(fs, "deterministic", head.n_params)
                if member_states is not None:
                    _load_warm_start(member, member_states[m],
                                     f"{resume_checkpoint} [members.{m}.*]")
                module = _FinalStateModule(posterior=member, kl_weight=0.0, **common)
                use_gpu, device = self._run_lightning(
                    module, data_module, fs, ckpt_dir / f"member_{m}", f"best_member{m}"
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
            posterior = self._build(fs, self.posterior_kind, head.n_params)
            kl_weight = self.kl_weight if self.posterior_kind == "mfvi" else 0.0
            module = _FinalStateModule(posterior=posterior, kl_weight=kl_weight, **common)
            if resume_checkpoint and Path(resume_checkpoint).exists():
                print(f"Warm start: loading {self.posterior_kind} weights from {resume_checkpoint}")
                ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
                _load_warm_start(module, ckpt.get("state_dict", ckpt), resume_checkpoint)
            # Reloads the best-val_nll checkpoint into `posterior` before returning.
            use_gpu, device = self._run_lightning(module, data_module, fs, ckpt_dir, "best")

            if self.posterior_kind == "laplace":
                # Post-hoc GGN over the full training set, fitted AFTER the best
                # checkpoint is back in memory: H^-1 is a curvature at a specific
                # parameter vector, so pairing a last-epoch covariance with
                # best-epoch weights would centre the Gaussian at one point and
                # take its curvature from another. Without the fit at all the arm
                # stays at the MAP point estimate and reports zero uncertainty.
                posterior.eval()
                with torch.no_grad():
                    starts, ends = _full_training_tensors(data_module)
                    embedded = self.system.embed_state_for_model(
                        self.system.normalize_state(starts)
                    )
                    params = posterior.forward_sample(embedded)
                    # Observation noise = RMS geodesic residual of the fitted mean.
                    # task="final_state" REQUIRES this explicitly: a default of
                    # 1.0 would silently rescale the whole posterior covariance.
                    resid = head.distance_per_component(head.mean(params), ends)
                    sigma = float(resid.pow(2).mean().sqrt().clamp_min(1e-3))
                    posterior.fit(posterior.body(embedded), ends, task="final_state", sigma=sigma)
                # `_cov` is a non-persistent buffer (its shape is unknown until
                # fit), so it will NOT round-trip through the Lightning
                # checkpoint. Save it explicitly or the exported arm silently
                # falls back to its MAP point estimate and reports zero spread.
                torch.save(posterior.posterior_covariance, ckpt_dir / "laplace_cov.pt")

        handle = FinalStateModelHandle(posterior, head, self.system).eval()
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
    silent cold start. Warm start is an optimization, so a mismatch must not
    abort the run -- but it must be visible in the log rather than inferred
    later from an unexplained jump in the loss curve.
    """
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"WARNING: warm start from {source} did not match the model: "
            f"{len(missing)} missing key(s) {list(missing)[:3]}, "
            f"{len(unexpected)} unexpected key(s) {list(unexpected)[:3]}. "
            f"Those parameters stay at their random initialization."
        )


def _full_training_tensors(data_module):
    """Stack the whole endpoint training set as ``(starts, ends)`` raw tensors.

    The endpoint datamodules differ from the classification one in two ways that
    matter here. The dataset lives on the PUBLIC ``train_dataset`` attribute (the
    classification module uses a private ``_train``), and it stores
    ``self.data`` as a list of ``(start, end)`` numpy tuples rather than stacked
    tensors -- angle wrapping is applied in ``__getitem__``, not at load time
    (``data/cartpole_endpoint_data.py:91-103``). Collating through ``__getitem__``
    is therefore the only way to get correctly wrapped angles.
    """
    from torch.utils.data import default_collate

    ds = data_module.train_dataset
    batch = default_collate([ds[i] for i in range(len(ds))])
    return batch["start_state"].float(), batch["end_state"].float()

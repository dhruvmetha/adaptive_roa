"""Deep ensemble of scalar-outcome flow matchers.

`fm_outcome` is a single deterministic model, so it has no epistemic axis and
cannot acquire on BALD. This trains M independent members and puts them behind
one handle that presents the ensemble predictive.

Members train SEQUENTIALLY, not one-per-GPU. `EnsembleFlowMatchingTrainer`
exists because an endpoint flow-matching member costs ~50 h, which makes the
mp.spawn machinery worth its hazards (children reload modules from disk each
epoch; a user-site numpy shadow hangs spawn jobs at 0% GPU). An outcome-FM
member is an MLP over a 1-D flow, and `clf_ensemble` already trains an MLP
ensemble sequentially at this same budget in this same campaign.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, List

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from adaptive_roa.adaptive_v2.trainers._seeding import resolve_seed_base
from adaptive_roa.data.adaptive_classification_data import AdaptiveClassificationDataModule
from adaptive_roa.model.outcome_flow_matcher import OutcomeFlowMatcher, OutcomeVelocityMLP

_LOGIT_CLAMP = 1e-6


class EnsembleOutcomeFMHandle(nn.Module):
    """M outcome flow matchers presenting one ensemble predictive.

    ``forward`` returns ``logit(mean_m p_m)``, NOT the mean of the members'
    logits. Averaging logits is a geometric mean in odds space and is not the
    ensemble predictive; it would also break agreement with
    ``EnsembleOutcomeFMProbabilityBackend.estimate``, which averages
    probabilities. That agreement is load-bearing: calibration and evaluation
    reach p through ``model(x) -> logits`` (``full_roa.py:1301``) while
    acquisition reaches it through the backend, so a divergence would have
    calibration optimising a different quantity than evaluation reports,
    silently, with both numbers looking reasonable.

    ``p_success_mc`` and ``p_success_exact`` return the ensemble mean under the
    member method of the same name, which is what lets
    ``OutcomeFMProbabilityBackend.estimate`` and ``estimate_both`` be inherited
    unchanged rather than reimplemented.
    """

    def __init__(self, members: List[Any], forward_readout: str = "exact"):
        super().__init__()
        members = list(members)
        if len(members) < 2:
            raise ValueError(
                f"EnsembleOutcomeFMHandle needs at least 2 members, got {len(members)}. "
                "A 1-member 'ensemble' has no epistemic signal and would score BALD = 0."
            )
        self.members = nn.ModuleList(members)
        readout = str(forward_readout).lower()
        if readout not in {"mc", "exact"}:
            raise ValueError(f"forward_readout must be 'mc' or 'exact', got {readout!r}")
        self.forward_readout = readout

    @property
    def n_members(self) -> int:
        return len(self.members)

    # ------------------------------------------------------------ predictive

    @torch.no_grad()
    def member_p_success(self, raw_states: torch.Tensor) -> torch.Tensor:
        """Per-member success probability: [B] states -> [M, B]."""
        return torch.stack([m.predict_p_success(raw_states) for m in self.members], dim=0)

    @torch.no_grad()
    def predict_p_success(self, raw_states: torch.Tensor) -> torch.Tensor:
        return self.member_p_success(raw_states).mean(dim=0)

    @torch.no_grad()
    def p_success_mc(self, raw_states: torch.Tensor, **kw) -> torch.Tensor:
        return torch.stack([m.p_success_mc(raw_states, **kw) for m in self.members],
                           dim=0).mean(dim=0)

    @torch.no_grad()
    def p_success_exact(self, raw_states: torch.Tensor, **kw):
        """Ensemble mean, and monotone only where EVERY member was monotone.

        ANDing the flags is the conservative reading: the diagnostic exists so a
        silent fallback to quadrature cannot be mistaken for a clean bisection,
        and one member falling back taints the mean it contributes to.
        """
        ps, monos = [], []
        for m in self.members:
            p, mono = m.p_success_exact(raw_states, **kw)
            ps.append(p)
            monos.append(mono)
        stacked = torch.stack(monos, dim=0)
        return torch.stack(ps, dim=0).mean(dim=0), stacked.all(dim=0)

    def forward(self, raw_states: torch.Tensor) -> torch.Tensor:
        """RAW states -> logit(mean_m p_m), matching ClassifierModule's contract."""
        p = self.predict_p_success(raw_states)
        # logit is unbounded at 0/1 and the exact readout genuinely returns
        # values within 1e-7 of the tails, so clamp before the transform.
        p = p.clamp(_LOGIT_CLAMP, 1.0 - _LOGIT_CLAMP)
        return torch.log(p / (1.0 - p)).to(torch.float32).view(-1, 1)


class EnsembleOutcomeFMTrainer:
    """Same contract as ``OutcomeFMTrainer``; trains M members instead of one."""

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
            forward_readout=str(fm_cfg.get("forward_readout", "exact")),
            forward_num_samples=int(fm_cfg.get("forward_num_samples", 100)),
        )

    @staticmethod
    def _member_warm_start_states(resume_checkpoint, n_members):
        """Split a saved ensemble checkpoint into one state dict per member.

        Without this the arm would silently retrain every member from scratch at
        every adaptive epoch while the single-model arms warm-start, which is an
        arm asymmetry that changes results rather than only runtime.
        """
        if not resume_checkpoint or not Path(resume_checkpoint).exists():
            return None
        ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        per_member: dict[int, dict] = {}
        for key, value in state.items():
            if not key.startswith("members."):
                continue
            idx, _, rest = key[len("members."):].partition(".")
            per_member.setdefault(int(idx), {})[rest] = value
        if not per_member:
            raise RuntimeError(
                f"{resume_checkpoint} has no 'members.*' keys; it is not an ensemble "
                f"checkpoint and warm-starting from it would leave every member at "
                f"random init"
            )
        if len(per_member) != n_members or sorted(per_member) != list(range(n_members)):
            # Partially warm-starting leaves some members at random init with no
            # visible symptom, so refuse instead.
            raise RuntimeError(
                f"{resume_checkpoint} holds {len(per_member)} members "
                f"({sorted(per_member)}) but n_members={n_members}; refusing to "
                f"partially warm-start"
            )
        print(f"Warm start: loading {n_members} outcome-FM members from {resume_checkpoint}")
        return per_member

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        fm_cfg = self._predictor_cfg.get("outcome_fm", {})
        ens_cfg = self._predictor_cfg.get("ensemble", {})
        n_members = int(ens_cfg.get("n_members", 5))
        if n_members < 2:
            raise ValueError(f"ensemble.n_members must be >= 2, got {n_members}")

        data_module = AdaptiveClassificationDataModule(
            train_file=dataset_files["train"],
            val_file=dataset_files["val"],
            batch_size=self._predictor_cfg.get("batch_size", 1024),
            # Matches ClassifierTrainer: data is already in-memory tensors, and
            # workers break on NFS (rmtree of `.nfs*` temp dirs => Errno 16).
            num_workers=0,
        )
        data_module.setup()

        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer_cfg = self._predictor_cfg.get("lightning_trainer", {})
        device = str(self._predictor_cfg.get("device", self.cfg.get("device", "cuda:0")))
        use_gpu = device.startswith("cuda") and torch.cuda.is_available()

        warm = self._member_warm_start_states(resume_checkpoint, n_members)
        # Reads the RUN seed, not a per-arm default of 0. Seeding members 0..M-1
        # regardless of cfg.seed makes nominally independent replicates
        # bit-identical and collapses the run-to-run floor to zero.
        seed_base = resolve_seed_base(self.cfg, fm_cfg)

        members = []
        for m in range(n_members):
            pl.seed_everything(seed_base + m, workers=True)
            module = self._build_member(fm_cfg)
            if warm is not None:
                module.load_state_dict(warm[m], strict=False)

            # Member checkpoints go in their OWN subdirectory. The engine globs
            # `checkpoints/best*.ckpt` (engine.py:218) and hands the result back
            # as resume_checkpoint, so a member checkpoint written beside the
            # ensemble one would be picked up instead of it, at random.
            member_dir = checkpoint_dir / f"member_{m}"
            member_dir.mkdir(parents=True, exist_ok=True)

            trainer = pl.Trainer(
                max_epochs=int(fm_cfg.get("max_epochs", 200)),
                accelerator="gpu" if use_gpu else "cpu",
                devices=1,
                gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
                log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
                check_val_every_n_epoch=1,
                enable_progress_bar=bool(trainer_cfg.get("enable_progress_bar", False)),
                enable_model_summary=False,
                callbacks=[
                    ModelCheckpoint(
                        dirpath=str(member_dir), monitor="val_loss", mode="min",
                        save_top_k=1, save_last=False,
                        filename="best-{epoch:02d}-{val_loss:.4f}",
                    ),
                    EarlyStopping(monitor="val_loss", mode="min",
                                  patience=int(fm_cfg.get("patience", 20))),
                ],
                logger=CSVLogger(save_dir=output_dir, name=f"outcome_fm_member_{m}"),
            )
            trainer.fit(module, data_module)

            # Lightning leaves LAST-epoch weights in the module; export and eval
            # must see the best-val ones or the arm is selected differently from
            # every other arm it is compared against.
            best = sorted(glob.glob(str(member_dir / "best*.ckpt")))
            if best:
                ckpt = torch.load(best[0], map_location="cpu", weights_only=False)
                module.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                print(f"WARNING: member {m} wrote no checkpoint; keeping last-epoch weights")

            module.eval()
            members.append(module)

        handle = EnsembleOutcomeFMHandle(
            members, forward_readout=str(fm_cfg.get("forward_readout", "exact"))
        ).eval()
        torch.save({"state_dict": handle.state_dict()}, checkpoint_dir / "best-ensemble.ckpt")
        return handle.to(device) if use_gpu else handle

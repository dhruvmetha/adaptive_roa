"""Trainer for Bayesian MLP arms with an outcome head.

Mirrors ``ClassifierTrainer``'s contract:
    __init__(cfg, system, system_name)
    fit(dataset_files, output_dir, resume_checkpoint=None) -> OutcomeModelHandle
"""
from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from adaptive_roa.adaptive_v2.trainers._seeding import resolve_seed_base
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

    def __init__(self, posterior, system, pos_weight, lr, weight_decay, kl_weight, n_train,
                 alpha=None, alpha_n_samples: int = 10):
        super().__init__()
        self.posterior = posterior
        self.system = system  # plain attr; methods are device-agnostic
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.kl_weight = float(kl_weight)
        self.n_train = max(int(n_train), 1)
        # alpha=None keeps the single-sample ELBO every already-scored arm was
        # trained under. Opting in is what makes those runs reproducible from
        # this file rather than silently retrained under a new objective.
        self.alpha = None if alpha is None else float(alpha)
        self.alpha_n_samples = int(alpha_n_samples)
        if self.alpha is not None and self.alpha_n_samples < 2:
            raise ValueError(
                f"BB-alpha needs at least 2 weight draws; got {self.alpha_n_samples}. "
                "At K=1 logsumexp collapses and the objective is the plain "
                "likelihood at EVERY alpha, so the arm would be mislabelled."
            )
        self.register_buffer("pos_weight", torch.as_tensor(float(pos_weight)))

    def forward(self, raw_states):
        embedded = self.system.embed_state_for_model(self.system.normalize_state(raw_states))
        return self.posterior.forward_sample(embedded)

    def _bb_alpha_nll(self, raw_states, y):
        """Black-box alpha-divergence likelihood term (Hernandez-Lobato et al. 2016).

            -(1/a) * mean_n [ logsumexp_k( a * log p(y_n|x_n,w_k) ) - log K ]

        with w_1..w_K drawn from q. Two limits pin what this is: as a -> 0 the
        bracket tends to mean_k log p, recovering the ELBO's expected
        log-likelihood; at a = 1 it is log mean_k p, the log of the AVERAGED
        likelihood. That second form is why Depeweg et al. (2018) use a = 1 --
        averaging likelihoods before taking the log rewards a q that covers
        every region assigning mass to the data, where averaging log-likelihoods
        rewards a q concentrated on the single best region.

        This is the tied-site reparameterization (Li & Gal 2017) of the BB-alpha
        energy: the per-site cavity factors (p/q)^(alpha/N) collect into the
        analytic KL(q||p) carried separately in `_step`, leaving the alpha-softened
        likelihood here. It agrees with the exact cavity form to O(alpha/N) per
        site, and N runs from 300 to 6.6M across our systems. The same family at
        alpha -> 0 is the negative ELBO, which is how Depeweg et al.'s supplement
        produces its own "variational Bayes" arm (alpha = 1e-6).

        logsumexp, never a raw exp: a saturated net drives log p to a few
        hundred negative, which underflows to exactly zero and silently makes
        the loss +inf.
        """
        return self._alpha_terms(raw_states, y)[0]

    def _alpha_terms(self, raw_states, y):
        """``(objective, gibbs_nll, predictive_nll)`` from ONE set of K draws.

        Three quantities, three jobs:

        * ``objective`` -- the BB-alpha energy's likelihood term, what we minimize.
        * ``gibbs_nll`` = mean_k -log p, the quantity the legacy single-draw arms
          log. Kept for audit so the two generations stay comparable.
        * ``predictive_nll`` = -log mean_k p, what ``_MONITOR`` selects on for an
          alpha arm.

        Selecting on the predictive rather than the Gibbs value is deliberate and
        is the one place this arm departs from its siblings. Gibbs exceeds
        predictive by exactly the Jensen gap, which IS the posterior spread the
        alpha=1 objective exists to preserve; with ``rho_init=-5`` q starts almost
        deterministic and widens as it trains, so a Gibbs-monitored run can stop
        precisely when the arm begins working and hand "best" to the narrowest q.
        The predictive value is also what the campaign actually scores
        (``log_score`` in stoch_prob_metrics is a predictive NLL), so this aligns
        model selection with the reported metric. At alpha=1 the objective and the
        predictive NLL coincide by construction.
        """
        embedded = self.system.embed_state_for_model(self.system.normalize_state(raw_states))
        k = self.alpha_n_samples
        logits = self.posterior.forward_samples(embedded, k).reshape(k, y.shape[0])
        log_lik = -F.binary_cross_entropy_with_logits(
            logits, y.expand(k, -1), pos_weight=self.pos_weight, reduction="none")
        a = self.alpha
        log_k = math.log(k)
        bb_alpha = -((torch.logsumexp(a * log_lik, dim=0) - log_k) / a).mean()
        predictive = -(torch.logsumexp(log_lik, dim=0) - log_k).mean()
        return bb_alpha, -log_lik.mean(), predictive

    def _step(self, batch, stage: str):
        y = batch["label"].float().view(-1)
        if self.alpha is None:
            logits = self(batch["inputs"]).view(-1)
            nll = selection_nll = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=self.pos_weight)
        else:
            nll, gibbs_nll, selection_nll = self._alpha_terms(batch["inputs"], y)
            self.log(f"{stage}_gibbs_nll", gibbs_nll, on_epoch=True, on_step=False)
        # The ELBO's KL is a per-DATASET term while nll is a per-batch mean, so
        # scale by 1/N to put them on the same footing.
        kl = self.posterior.kl_divergence() / self.n_train
        loss = nll + self.kl_weight * kl
        # val_loss is the ELBO -- a training objective and a diagnostic, but NOT
        # a model-selection criterion (see _MONITOR). val_nll is what the
        # callbacks watch, so it must be an epoch-level metric.
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        # Never the BB-alpha energy itself (it carries the KL and the alpha
        # softening): the plain BCE for legacy arms, the predictive NLL for alpha
        # arms. See _alpha_terms for why the alpha arm departs here.
        self.log(f"{stage}_nll", selection_nll, prog_bar=True, on_epoch=True, on_step=False)
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

    def _member_seed_base(self) -> int:
        """Seed base for the ensemble members, from the RUN seed.

        See ``_seeding.resolve_seed_base`` for the bug this replaced and why a
        bit-identical seed replicate is worse than a missing feature.
        `predictor.bnn.seed` still wins if explicitly set.
        """
        return resolve_seed_base(self.cfg, self._predictor_cfg.get("bnn", {}))

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

        # The reference tier pins pos_weight = 1 so every arm targets ONE shared
        # posterior. Class reweighting tempers the likelihood per class, so an arm
        # trained under it is not approximating the posterior HMC samples. An
        # explicit config value therefore overrides the data-derived default.
        cfg_pos_weight = bnn.get("pos_weight", None)
        pos_weight = (float(cfg_pos_weight) if cfg_pos_weight is not None
                      else data_module.pos_weight)

        # BB-alpha: alpha=1.0 is what Depeweg et al. (2018) fit their BNN with.
        # Left unset the module keeps the single-sample ELBO (the alpha -> 0
        # limit), so arms scored before this existed still train identically.
        # It is refused on the ensemble below rather than ignored: each member
        # there is deterministic, so there is no q(w) to draw K samples from and
        # a configured alpha would silently do nothing.
        alpha = bnn.get("alpha", None)
        common = dict(
            system=self.system, pos_weight=pos_weight,
            lr=float(bnn.get("lr", 1e-3)),
            weight_decay=float(bnn.get("weight_decay", 1e-5)),
            n_train=n_train,
            alpha=None if alpha is None else float(alpha),
            alpha_n_samples=int(bnn.get("alpha_n_samples", 10)),
        )

        if common["alpha"] is not None and self.posterior_kind != "mfvi":
            # Only MFVI has a q(w) that forward_samples actually draws from.
            # Ensemble members are deterministic; a Laplace posterior returns its
            # MAP head until `fit` runs, so all K training draws are IDENTICAL and
            # logsumexp collapses to the plain BCE at every alpha -- the K=1
            # failure with K=10, and silent. Refuse rather than mislabel the arm.
            raise ValueError(
                f"predictor.bnn.alpha is set on a '{self.posterior_kind}' arm, but "
                "BB-alpha needs K distinct draws from a weight posterior and only "
                "'mfvi' provides them. The setting would be inert and the arm "
                "would be reported as alpha-fitted when it is not."
            )
        if common["alpha"] is not None and abs(common["pos_weight"] - 1.0) > 1e-9:
            # Same reasoning hmc_trainer.py uses to refuse pos_weight outright: a
            # BB-alpha fit of a class-tempered likelihood targets a posterior
            # nothing else in the comparison approximates.
            raise ValueError(
                f"predictor.bnn.alpha is set together with pos_weight="
                f"{common['pos_weight']:.4f}. BB-alpha would then fit a TEMPERED "
                "likelihood, which is not the posterior the paper's arms or the "
                "HMC reference target. Pin predictor.bnn.pos_weight: 1.0."
            )

        if self.posterior_kind == "ensemble":
            n_members = int(bnn.get("n_members", 5))
            member_states = self._ensemble_warm_start_states(resume_checkpoint, n_members)
            # Members must be independent: own seed, own optimizer, own shuffling.
            members, use_gpu, device = [], False, "cpu"
            for m in range(n_members):
                torch.manual_seed(self._member_seed_base() + m)
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
                # `_cov` and `_prec_chol` are non-persistent buffers (their shape
                # is unknown until fit), so they will NOT round-trip through the
                # Lightning checkpoint. Save them explicitly or the exported arm
                # silently falls back to its MAP point estimate and reports zero
                # spread. The precision factor is what sampling uses; the
                # covariance is kept because older tooling reads it.
                torch.save(posterior.posterior_covariance, ckpt_dir / "laplace_cov.pt")
                torch.save(posterior.precision_cholesky, ckpt_dir / "laplace_prec_chol.pt")

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

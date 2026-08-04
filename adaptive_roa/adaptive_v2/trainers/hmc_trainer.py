"""Trainer for the HMC reference arm, serving both predictor heads.

Mirrors ``BayesianMLPTrainer`` / ``FinalStateTrainer``'s contract:
    __init__(cfg, system, system_name)
    fit(dataset_files, output_dir, resume_checkpoint=None) -> model_handle

HMC is a REFERENCE arm, not an approximation, which makes this trainer
different from its siblings in two structural ways:

* It does not warm-start. Every epoch re-runs ``n_chains`` fresh chains from
  scratch, seeded from the prior, so ``resume_checkpoint`` is refused outright
  rather than silently ignored (see ``fit`` below).
* It must target the SAME posterior every other arm approximates, or its
  agreement numbers are meaningless. Two things in the production config break
  that: ``pos_weight`` (a data-derived BCE reweighting -- a tempered
  likelihood, not the true one) and ``beta_nll`` (a detached-sigma reweighting
  that is not a likelihood at all). This trainer therefore never reads either
  from config: ``pos_weight`` is pinned to 1.0 and the ``FinalStateHead`` is
  always built with ``beta=0.0``, both asserted below rather than trusted to
  config discipline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from adaptive_roa.adaptive_v2.trainers.final_state_trainer import (
    _DATAMODULES,
    _full_dataset_tensors,
)
from adaptive_roa.data.adaptive_classification_data import AdaptiveClassificationDataModule
from adaptive_roa.predictors.bayesian_mlp import build_bayesian_mlp, embedded_dim
from adaptive_roa.predictors.final_state_handle import FinalStateModelHandle
from adaptive_roa.predictors.handles import OutcomeModelHandle
from adaptive_roa.predictors.heads import FinalStateHead
from adaptive_roa.predictors.hmc.diagnostics import function_space_rhat, hmc_vs_hmc_ceiling
from adaptive_roa.predictors.hmc.log_posterior import FlatLogPosterior
from adaptive_roa.predictors.hmc.posterior import HMCPosterior
from adaptive_roa.predictors.hmc.sampler import hmc_chain

_HEAD_KINDS = ("outcome", "final_state")


class HMCTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name

    @property
    def _predictor_cfg(self):
        pred = self.cfg.get("predictor")
        return pred if pred is not None else self.cfg

    def _load_split(self, dataset_files: dict, head_kind: str):
        """Raw ``(x_train, y_train, x_val, y_val)`` tensors for the given head.

        Reuses the SAME datamodules the production trainers use, so an HMC run
        sees exactly the data an approximate arm would -- the only thing that
        may differ is the likelihood/prior wired up around it.
        """
        if head_kind == "outcome":
            dm = AdaptiveClassificationDataModule(
                train_file=dataset_files["train"],
                val_file=dataset_files["val"],
                batch_size=self._predictor_cfg.get("batch_size", 1024),
                num_workers=0,  # in-memory tensors; workers break on NFS
            )
            dm.setup()
            # Private `_train` / `_val` attrs, same as bayesian_mlp_trainer.py.
            return dm._train.states, dm._train.labels, dm._val.states, dm._val.labels

        dm_cls = _DATAMODULES.get(self.system_name)
        if dm_cls is None:
            raise ValueError(
                f"no endpoint datamodule for system {self.system_name!r}; "
                f"expected one of {sorted(_DATAMODULES)}"
            )
        dm = dm_cls(
            data_file=dataset_files["train"],
            validation_file=dataset_files["val"],
            test_file=dataset_files["val"],  # matches FinalStateTrainer/GPRegressorTrainer
            batch_size=self._predictor_cfg.get("batch_size", 1024),
            val_batch_size=self._predictor_cfg.get("val_batch_size", 2048),
            num_workers=0,  # in-memory tensors; workers break on NFS
        )
        dm.setup()
        # Collating through __getitem__ (not reading storage directly) applies
        # each dataset's own per-sample manifold fix-up -- see
        # _full_dataset_tensors's docstring in final_state_trainer.py.
        x_train, y_train = _full_dataset_tensors(dm.train_dataset)
        x_val, y_val = _full_dataset_tensors(dm.val_dataset)
        return x_train, y_train, x_val, y_val

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.embed_state_for_model(self.system.normalize_state(x))

    def _chain_predictions(self, net, samples: torch.Tensor, x_val: torch.Tensor) -> torch.Tensor:
        """Per-draw, per-point scalar summary of one chain's predictive: [draws, N].

        Function-space R-hat needs exactly one scalar per query point. For the
        outcome head that scalar is the natural one: p(success) = sigmoid(logit)
        under each draw. The final-state head has no single equally natural
        scalar (its predictive is multi-dimensional and manifold-valued), so
        this uses the FIRST raw network output -- the location parameter of the
        system's first manifold component -- as a proxy. That choice is what
        "converged" means for the final-state diagnostics: agreement of the
        chains on that one coordinate, not on the full predictive.

        Reuses HMCPosterior.predictive_logit_samples (exact enumeration over the
        chain's own draws, one net, no functional-call duplication) rather than
        re-deriving the flat-vector-to-net injection here a third time.
        """
        x_embedded = self._embed(x_val)
        # [draws, N, out_dim]; S is ignored by HMCPosterior (full enumeration).
        outs = HMCPosterior(net, samples).predictive_logit_samples(
            x_embedded, S=samples.shape[0]
        )
        first = outs[..., 0]
        return torch.sigmoid(first) if self.head is None else first

    def fit(self, dataset_files: dict, output_dir: str, resume_checkpoint: str | None = None):
        if resume_checkpoint:
            raise ValueError(
                "HMC is a reference arm and does not resume: each epoch re-runs "
                "its chains from scratch. Accepting and ignoring a "
                "resume_checkpoint would make warm start appear to work."
            )

        hmc_cfg = self._predictor_cfg.get("hmc", {})
        head_kind = str(self._predictor_cfg.get("head", "outcome"))
        if head_kind not in _HEAD_KINDS:
            raise ValueError(
                f"unknown predictor.head {head_kind!r}; expected one of {_HEAD_KINDS}"
            )
        device = "cpu"  # chains are tiny; the gradient is full-batch over ~3k params

        x_train, y_train, x_val, y_val = self._load_split(dataset_files, head_kind)

        # Reference-tier invariants, asserted rather than trusted to config
        # discipline: pos_weight tempers the likelihood and beta-NLL is not a
        # likelihood at all, so with either active HMC would reference a
        # target no arm is approximating. Neither is ever read from hmc_cfg.
        self.pos_weight = 1.0
        self.head = None
        if head_kind == "final_state":
            self.head = FinalStateHead(self.system, beta=0.0)
        assert self.pos_weight == 1.0, "HMC must target the untempered likelihood"
        assert self.head is None or self.head.beta == 0.0, (
            "HMC must target a real log-likelihood; beta-NLL under beta != 0 is not one"
        )
        out_dim = 1 if self.head is None else self.head.n_params

        input_dim = embedded_dim(self.system)
        net = build_bayesian_mlp(
            input_dim=input_dim,
            hidden_dims=list(hmc_cfg.get("hidden_dims", [50, 50])),
            output_dim=out_dim,
            posterior="deterministic",
            activation=str(hmc_cfg.get("activation", "tanh")),
        ).to(device)

        prior_sigma = float(hmc_cfg.get("prior_sigma", 1.0))
        lp = FlatLogPosterior(net, head=self.head, prior_sigma=prior_sigma)

        # The net owns normalize+embed for its inputs, so feed it raw states;
        # embed ONCE here rather than inside the closures HMC calls thousands
        # of times per chain.
        x_train_embedded = self._embed(x_train)

        def log_prob(theta):
            return lp.log_prob(theta, x_train_embedded, y_train)

        def grad_log_prob(theta):
            return lp.grad_log_prob(theta, x_train_embedded, y_train)

        base_seed = int(hmc_cfg.get("seed", 0))
        n_chains = int(hmc_cfg.get("n_chains", 3))
        results, per_chain_preds = [], []
        for c in range(n_chains):
            # Each chain gets its own seed (base_seed + c) and starts from an
            # independent prior draw, so chains are not accidentally coupled
            # through a shared generator or a shared starting point.
            gen = torch.Generator().manual_seed(base_seed + c)
            theta0 = torch.randn(lp.dim, generator=gen) * prior_sigma
            res = hmc_chain(
                log_prob, grad_log_prob, theta0,
                n_samples=int(hmc_cfg.get("n_samples", 200)),
                n_warmup=int(hmc_cfg.get("n_warmup", 200)),
                n_leapfrog=int(hmc_cfg.get("n_leapfrog", 20)),
                seed=base_seed + c,
            )
            results.append(res)
            per_chain_preds.append(self._chain_predictions(net, res.samples, x_val))

        posterior = HMCPosterior(net, torch.cat([r.samples for r in results], dim=0))

        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # MUST match the engine's glob, checkpoints/best*.ckpt (engine.py:140).
        # Metadata is enough to rebuild the net WITHOUT consulting a config, so
        # a later config edit can never desync from what these weights are.
        torch.save(
            {
                "samples": posterior.samples,
                "hidden_dims": list(hmc_cfg.get("hidden_dims", [50, 50])),
                "activation": str(hmc_cfg.get("activation", "tanh")),
                "input_dim": input_dim,
                "output_dim": out_dim,
                "head": head_kind,
            },
            ckpt_dir / "best-hmc.ckpt",
        )

        # A reference arm whose own convergence cannot be audited is not a
        # reference: write per-chain sampler diagnostics, the function-space
        # R-hat, and the HMC-vs-HMC agreement ceiling next to the checkpoint.
        preds = torch.stack(per_chain_preds, dim=0)  # [chains, draws, N]
        rhat = function_space_rhat(preds)
        (ckpt_dir / "hmc_diagnostics.json").write_text(json.dumps({
            "chains": [
                {
                    "accept_rate": r.accept_rate,
                    "step_size": r.step_size,
                    "divergences": r.divergences,
                }
                for r in results
            ],
            "rhat_max": float(rhat.max()),
            "rhat_mean": float(rhat.mean()),
            "ceiling": hmc_vs_hmc_ceiling(preds),
        }, indent=2))

        if self.head is None:
            return OutcomeModelHandle(posterior, self.system).eval().to(device)
        return FinalStateModelHandle(posterior, self.head, self.system).eval().to(device)

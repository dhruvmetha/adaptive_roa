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
  from config: the likelihood is always unweighted and the ``FinalStateHead`` is
  always built with ``beta=0.0``. Setting either key under ``predictor.hmc``
  RAISES (``_reject_tempering_keys``) rather than being silently ignored --
  silently ignoring them is what would let a caller believe the reference had
  been tempered to match an arm when it had not.
* Its own convergence is gated (``_convergence_warning``). Chains that have not
  mixed produce a posterior whose every downstream agreement number is
  meaningless, and the failure is invisible: the handle looks normal.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import torch

from adaptive_roa.adaptive_v2.eval.system_hooks import resolve_system_hook
from adaptive_roa.adaptive_v2.trainers._seeding import resolve_seed_base
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

# Keys that would temper the target away from the one every other arm
# approximates. Present under `predictor.hmc` => hard error, never ignored.
_TEMPERING_KEYS = ("pos_weight", "beta_nll", "beta")

# Gelman-Rubin convention. 1.1 is the standard published cutoff; it is NOT tuned
# to what this arm happens to produce. Measured at the production budget:
# `hmc` 1.19, `hmc_reg` 91.4 -- both trip it, which is the point.
RHAT_THRESHOLD = 1.1


class HMCConvergenceError(RuntimeError):
    """Raised when a reference run's chains have not mixed and strict mode is on."""


def _convergence_warning(rhat_max: float, chains: list, threshold: float) -> str | None:
    """The non-convergence message, or None if the run passed the gate.

    Split out from ``fit`` so the gate can be tested at the observed values
    (``rhat_max`` 91.4 / 1.19) without paying for a sampling run.
    """
    if rhat_max <= threshold:
        return None
    worst = max(chains, key=lambda c: c["divergences"]) if chains else {}
    return (
        f"HMC REFERENCE DID NOT CONVERGE: function-space rhat_max={rhat_max:.4g} "
        f"exceeds the Gelman-Rubin threshold {threshold}. The chains are in "
        f"disjoint regions, so pooling them produces a 'posterior' whose support "
        f"is partly garbage and EVERY agreement number computed against it is "
        f"invalid. Worst chain: accept_rate="
        f"{worst.get('accept_rate', float('nan')):.3g}, "
        f"step_size={worst.get('step_size', float('nan')):.3g}, "
        f"divergences={worst.get('divergences', 'n/a')}. "
        f"Do not quote this run's fidelity numbers. Fixes, in order: a diagonal "
        f"mass matrix estimated from warmup, or reparameterizing the head's "
        f"log_sigma -- NOT raising n_samples, and NOT raising this threshold."
    )


class HMCTrainer:
    def __init__(self, cfg: Any, system: Any, system_name: str):
        self.cfg = cfg
        self.system = system
        self.system_name = system_name

    @property
    def _predictor_cfg(self):
        pred = self.cfg.get("predictor")
        return pred if pred is not None else self.cfg

    def _chain_seed_base(self) -> int:
        """Seed base for the chains, from the RUN seed.

        This read ``predictor.hmc.seed``, which both arm configs pinned to 0, so
        run seeds 42/43/44 produced BIT-IDENTICAL draws. See
        ``_seeding.resolve_seed_base``: for a reference arm that is the worst
        case, because the reference's own sampling variability is exactly the
        scale against which an approximation's gap must be judged, and here it
        was 0 by construction. ``predictor.hmc.seed`` still wins if set.
        """
        return resolve_seed_base(self.cfg, self._predictor_cfg.get("hmc", {}))

    @staticmethod
    def _reject_tempering_keys(hmc_cfg) -> None:
        """Refuse a config that tries to temper the reference target.

        ``pos_weight`` tempers the likelihood per class and beta-NLL multiplies
        each dim by a detached ``sigma^(2*beta)`` -- no posterior corresponds to
        it. Neither is ever read here, so accepting the key and ignoring it
        would let a caller believe the reference had been moved to match an arm
        when it had not. Raising is the only behaviour that cannot mislead.
        """
        present = [k for k in _TEMPERING_KEYS if hmc_cfg.get(k) is not None]
        if present:
            raise ValueError(
                f"predictor.hmc.{{{', '.join(present)}}} set, but HMC is the "
                f"REFERENCE arm and always targets the untempered likelihood "
                f"(pos_weight=1, beta=0). These keys are never read; setting "
                f"them would silently mean something other than what it says. "
                f"Remove them, or temper the approximations instead."
            )

    def _attractor_radius(self) -> float:
        """Radius for the final-state head's success classification.

        ``probability.attractor_radius`` is what the arm is actually scored on
        (endpoint_mc.yaml resolves it from the top-level ``attractor_radius``);
        the system hook's default is the fallback when this trainer is driven
        without a probability block, as the tests do.
        """
        prob = self.cfg.get("probability") or {}
        for source in (prob, self.cfg):
            value = source.get("attractor_radius")
            if value is not None:
                return float(value)
        return float(resolve_system_hook(self.system).attractor_radius_default)

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

        NOT bounded for the final-state head, deliberately: R-hat is scale-free
        in the sense that matters here (it is a variance ratio), and squashing
        the proxy through a sigmoid was measured to make the statistic WORSE,
        not better (26 -> 143 at the truncated budget), so a bounded transform
        here would hide nothing and cost interpretability. The AGREEMENT
        ceiling, which does need a probability, uses ``_chain_success_probs``.
        """
        outs = self._chain_outputs(net, samples, x_val)
        first = outs[..., 0]
        return torch.sigmoid(first) if self.head is None else first

    def _chain_outputs(self, net, samples: torch.Tensor, x_val: torch.Tensor) -> torch.Tensor:
        """Raw per-draw network outputs for one chain: [draws, N, out_dim]."""
        # S is ignored by HMCPosterior (full enumeration over the chain's draws).
        return HMCPosterior(net, samples).predictive_logit_samples(
            self._embed(x_val), S=samples.shape[0]
        )

    def _chain_success_probs(self, net, samples: torch.Tensor,
                             x_val: torch.Tensor) -> torch.Tensor:
        """Per-draw p(success) indicator for one chain: [draws, N], in [0, 1].

        ``hmc_vs_hmc_ceiling`` thresholds at 0.5 and reports a total-variation
        distance, both of which are defined only for a PROBABILITY. The outcome
        head supplies one directly. The final-state head does not: its natural
        per-draw scalar is a raw network output (a location parameter in
        normalized coordinates), and feeding that in produced
        ``total_variation = 3.89`` -- impossible for a distance in [0, 1] -- and
        an ``agreement`` computed by thresholding a normalized cart position at
        0.5, which means nothing.

        So the final-state head is summarized by the quantity it is actually
        SCORED on: whether the draw's predicted endpoint lands in the attractor.
        The head's ``mean`` is used rather than ``sample`` so the diagnostic
        carries no RNG of its own and is reproducible from the checkpoint.
        """
        if self.head is None:
            return self._chain_predictions(net, samples, x_val)
        params = self._chain_outputs(net, samples, x_val)      # [draws, N, P]
        n_draws, n_points, _ = params.shape
        endpoints = self.head.mean(params.reshape(n_draws * n_points, -1))
        labels = self.system.classify_attractor(endpoints, self._attractor_radius())
        return (labels.reshape(n_draws, n_points) == 1).to(params.dtype)

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

        # Reference-tier invariant, enforced against the CONFIG rather than
        # re-read off a literal this function just assigned: pos_weight tempers
        # the likelihood and beta-NLL is not a likelihood at all, so with either
        # active HMC would reference a target no arm is approximating. Neither
        # is ever read from hmc_cfg -- setting one is therefore an error, not a
        # no-op. The head below is built at beta=0.0 unconditionally, and it is
        # the object the log-posterior and the returned handle both use, so
        # `handle.head.beta` is what a test should read to pin this.
        self._reject_tempering_keys(hmc_cfg)
        self.head = None
        if head_kind == "final_state":
            self.head = FinalStateHead(self.system, beta=0.0)
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

        base_seed = self._chain_seed_base()
        n_chains = int(hmc_cfg.get("n_chains", 3))
        results, per_chain_preds, per_chain_probs = [], [], []
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
            per_chain_probs.append(self._chain_success_probs(net, res.samples, x_val))

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
        #
        # ess/acf1 are recorded, not just accept_rate/divergences, because the
        # failure the sampler's trajectory jitter exists to prevent -- a
        # resonance that turns the chain antithetic and destroys its second
        # moments -- shows FULL acceptance and ZERO divergences while it
        # happens. Those two fields cannot see it; a collapsed ESS and an
        # acf1 near 1 (measured on theta**2, see sampler.py) can.
        preds = torch.stack(per_chain_preds, dim=0)  # [chains, draws, N]
        probs = torch.stack(per_chain_probs, dim=0)  # [chains, draws, N] in [0,1]
        rhat = function_space_rhat(preds)
        rhat_max = float(rhat.max())
        chains = [
            {
                "accept_rate": r.accept_rate,
                "step_size": r.step_size,
                "divergences": r.divergences,
                "warmup_divergences": r.warmup_divergences,
                "ess": r.ess,
                "acf1": r.acf1,
            }
            for r in results
        ]

        # THE GATE. Without it a fully diverged run wrote rhat_max = 91.4 to
        # this file, returned a normal-looking handle, and flowed into the
        # benchmark: nothing anywhere read the artifact.
        threshold = float(hmc_cfg.get("rhat_threshold", RHAT_THRESHOLD))
        message = _convergence_warning(rhat_max, chains, threshold)
        (ckpt_dir / "hmc_diagnostics.json").write_text(json.dumps({
            "chains": chains,
            "rhat_max": rhat_max,
            "rhat_mean": float(rhat.mean()),
            "rhat_threshold": threshold,
            "converged": message is None,
            "ceiling": hmc_vs_hmc_ceiling(probs),
        }, indent=2))
        if message is not None:
            # Both channels on purpose: warnings can be filtered to nothing in a
            # long campaign, and a bare print can be lost in Lightning's output.
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            print(f"\n{'=' * 78}\n{message}\n{'=' * 78}\n", file=sys.stderr, flush=True)
            if bool(hmc_cfg.get("strict_convergence", False)):
                raise HMCConvergenceError(message)

        if self.head is None:
            return OutcomeModelHandle(posterior, self.system).eval().to(device)
        return FinalStateModelHandle(posterior, self.head, self.system).eval().to(device)

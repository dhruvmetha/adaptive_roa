"""Declarative campaign specification.

A campaign is data, not code: the full cross-product is ~450 runs, and which
subset to run is a scope decision that should never require editing Python.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

# Arms that may run adaptively, and the tier each is valid in.
OUTCOME_ARMS = ("mlp", "gp", "gp_optdelta", "bnn_mfvi", "bnn_ensemble", "bnn_laplace")
FINAL_STATE_ARMS = ("fm", "mlp_det", "gp_reg", "bnn_mfvi_reg",
                    "bnn_ensemble_reg", "bnn_laplace_reg")
REFERENCE_ONLY_ARMS = ("hmc", "hmc_reg")
ALL_ARMS = OUTCOME_ARMS + FINAL_STATE_ARMS + REFERENCE_ONLY_ARMS

# Fixed-dataset baselines: no ranking signal, so never expanded adaptively.
NON_ADAPTIVE_ARMS = ("mlp_det",)

# Arms whose posterior is over MLP weights, so an HMC reference applies to them.
MLP_BACKBONE_ARMS = ("bnn_mfvi", "bnn_ensemble", "bnn_laplace",
                     "bnn_mfvi_reg", "bnn_ensemble_reg", "bnn_laplace_reg")

TIERS = ("production", "reference")

# Arm name is the identity (keys run_id, the export registry, provenance).
# The Hydra config GROUP is the filename, which differs for the two baselines:
# classifier.yaml declares name: mlp, generative.yaml declares name: fm.
ARM_CONFIG_GROUP = {"mlp": "classifier", "fm": "generative"}


@dataclass(frozen=True)
class RunSpec:
    arm: str
    system: str
    tier: str
    acquisition: str
    seed: int
    n_epochs: int
    overrides: tuple[str, ...] = ()

    @property
    def run_id(self) -> str:
        """Stable identity for this experiment.

        Deterministic across processes: built from a sorted, explicit tuple and
        hashed with sha1 rather than builtin hash(), which is randomized per
        process. n_epochs is deliberately EXCLUDED -- extending a run from 10 to
        20 epochs resumes the same experiment rather than starting a new one.
        """
        parts = (self.arm, self.system, self.tier, self.acquisition,
                 str(self.seed)) + tuple(sorted(self.overrides))
        digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]
        return f"{self.system}_{self.arm}_{self.tier}_{self.acquisition}_s{self.seed}_{digest}"

    def hydra_overrides(self) -> list[str]:
        config_group = ARM_CONFIG_GROUP.get(self.arm, self.arm)
        ov = [f"system={self.system}", f"predictor={config_group}",
              f"seed={self.seed}", f"n_epochs={self.n_epochs}"]
        if self.tier == "reference":
            ov.append("+experiment=reference_tier")
        if self.arm in NON_ADAPTIVE_ARMS:
            # MUST be the experiment config: a d2_ratio written in the predictor
            # group is silently discarded (defaults list orders predictor first).
            ov.append("+experiment=mlp_det_baseline")
        else:
            ov.append(f"acquisition={self.acquisition}")
        ov.extend(self.overrides)
        return ov


def expand_manifest(cfg: dict[str, Any]) -> list[RunSpec]:
    tier = str(cfg["tier"])
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")

    arms = list(cfg["arms"])
    for arm in arms:
        if arm not in ALL_ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ALL_ARMS}")
        if arm in REFERENCE_ONLY_ARMS and tier != "reference":
            raise ValueError(
                f"arm {arm!r} is valid in the reference tier only: HMC is a "
                f"reference for the SAME architecture the approximations use, "
                f"so an HMC run at [50,50] says nothing about a posterior at "
                f"[256,512,256]."
            )
        if arm in NON_ADAPTIVE_ARMS and tier == "reference":
            raise ValueError(
                f"arm {arm!r} is not valid in the reference tier: the reference "
                f"tier exists to carry posterior-fidelity claims at the shared "
                f"[50,50] backbone against the HMC reference, but {arm!r} is a "
                f"deterministic point-estimate baseline with no posterior at "
                f"all -- there is nothing for HMC to be a reference FOR. The "
                f"combination is not merely awkward to encode: it also emits "
                f"two conflicting +experiment= overrides (reference_tier and "
                f"mlp_det_baseline), which Hydra rejects at composition time."
            )

    specs: list[RunSpec] = []
    for arm in arms:
        modes = ["random"] if arm in NON_ADAPTIVE_ARMS else list(cfg["acquisition"])
        for system in cfg["systems"]:
            for mode in modes:
                for seed in cfg["seeds"]:
                    specs.append(RunSpec(
                        arm=arm, system=system, tier=tier, acquisition=mode,
                        seed=int(seed), n_epochs=int(cfg["n_epochs"]),
                        overrides=tuple(cfg.get("overrides", ())),
                    ))
    return specs

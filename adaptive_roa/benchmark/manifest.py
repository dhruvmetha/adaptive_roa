"""Declarative campaign specification.

A campaign is data, not code: the full cross-product is ~450 runs, and which
subset to run is a scope decision that should never require editing Python.

Because the manifest IS the interface, it ships validated: ``expand_manifest``
resolves every config-group override a ``RunSpec`` emits (``system=``,
``predictor=``, ``acquisition=``, ``+experiment=``, and any extra
``overrides`` naming a real group) against the Hydra config tree on disk and
RAISES on a value with no ``<group>/<value>.yaml``. The shipped pilot manifest
previously named two groups that do not exist -- ``acquisition=random`` (the
group is ``direct``) and ``system=cartpole`` (the group is
``cartpole_pybullet``) -- so 125 of its 126 runs would have queued, started,
and died inside SLURM on a ``MissingConfigException``, ~90 separate times,
after submission. Validating the arm against ``ALL_ARMS`` alone was not
enough: an arm is one of four groups a spec emits, and the other three were
unchecked. A campaign must fail at EXPANSION time, in one message, naming
what does exist.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Arms that may run adaptively, and the tier each is valid in.
OUTCOME_ARMS = ("mlp", "gp", "gp_optdelta", "bnn_mfvi", "bnn_ensemble", "bnn_laplace")
FINAL_STATE_ARMS = ("fm", "mlp_det", "gp_reg", "bnn_mfvi_reg",
                    "bnn_ensemble_reg", "bnn_laplace_reg")
REFERENCE_ONLY_ARMS = ("hmc", "hmc_reg")
ALL_ARMS = OUTCOME_ARMS + FINAL_STATE_ARMS + REFERENCE_ONLY_ARMS

# Fixed-dataset baselines: no ranking signal, so never expanded adaptively.
NON_ADAPTIVE_ARMS = ("mlp_det",)

# The acquisition mode a non-adaptive arm is recorded under. MUST be the group
# such an arm actually composes under, which is `direct` -- default.yaml's
# `acquisition: direct` -- because these arms deliberately emit no
# `acquisition=` override at all (they emit `+experiment=mlp_det_baseline`
# instead, which is the only place a d2_ratio survives Hydra's defaults
# order). Labelling them "random" in run_id, as an earlier draft did, made
# run_id disagree with the `acquisition` column aggregate.collect_runs reads
# back out of the artifact (`sampling_mode: direct`), so the same run had two
# names depending on which end you looked from.
NON_ADAPTIVE_ACQUISITION = "direct"

# Arms whose posterior is over MLP weights, so an HMC reference applies to them.
MLP_BACKBONE_ARMS = ("bnn_mfvi", "bnn_ensemble", "bnn_laplace",
                     "bnn_mfvi_reg", "bnn_ensemble_reg", "bnn_laplace_reg")

TIERS = ("production", "reference")

# Arm name is the identity (keys run_id, the export registry, provenance).
# The Hydra config GROUP is the filename, which differs for the two baselines:
# classifier.yaml declares name: mlp, generative.yaml declares name: fm.
ARM_CONFIG_GROUP = {"mlp": "classifier", "fm": "generative"}

# The Hydra config tree every emitted override is resolved against.
CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2"


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


def acquisition_modes_for(arm: str, modes) -> list[str]:
    """The acquisition modes a campaign runs ``arm`` under, given its mode list.

    SINGLE SOURCE OF TRUTH for "which acquisition cells is this arm expected
    to cover". ``expand_manifest`` builds the campaign from it, and
    ``guards.assert_matched_coverage`` judges a frame's completeness against
    it, so the expander and the guard cannot drift apart. A global
    cross-product would (and did) refuse the shipped pilot at 100%
    completion: ``mlp_det`` has no ``ranked`` cell BY DESIGN -- its outcome
    probability collapses to {0, 1}, so there is no ranking signal to
    acquire on -- and a rule that demanded one made the pilot table
    unproducible with no flag able to fix it, because the missing axis was
    ``acquisition`` rather than ``system``.
    """
    if arm in NON_ADAPTIVE_ARMS:
        return [NON_ADAPTIVE_ACQUISITION]
    return list(modes)


def _parse_group_override(override: str) -> tuple[str, str] | None:
    """(group, value) for an override that SELECTS a config group, else None.

    Hydra distinguishes selecting a config group (``system=pendulum``,
    ``+experiment=reference_tier``) from setting a config VALUE
    (``seed=42``, ``predictor.hmc.num_samples=100``). Only the first kind
    names a file that has to exist. A dotted key is always the second kind,
    and a leading ``+``/``~`` is Hydra's append/delete marker on an
    otherwise ordinary group selection, so it is stripped before the lookup.
    """
    key, sep, value = override.partition("=")
    if not sep:
        return None
    key = key.lstrip("+~")
    if "." in key or not key:
        return None
    return key, value


def assert_config_groups_exist(spec: "RunSpec", config_root=None) -> None:
    """Refuse a spec that emits a Hydra override no config file can satisfy.

    Checked for EVERY group a spec emits -- ``system=``, ``predictor=``,
    ``acquisition=``, ``+experiment=``, plus any manifest-supplied extra
    override whose key names a real group directory -- not just the
    predictor. ``ALL_ARMS`` covers arm identity and
    ``test_every_arm_config_group_exists_on_disk`` covers the arm ->
    predictor-group mapping, but nothing covered the other three, which is
    how a manifest naming ``acquisition=random`` and ``system=cartpole``
    shipped: both are plausible English, neither is a file.

    An override whose key is not a group DIRECTORY under ``config_root``
    (``seed=42``, ``n_epochs=10``, any dotted path) is not this function's
    business -- it sets a value inside the composed config, and Hydra's own
    struct-mode checking is what judges it.
    """
    root = Path(config_root) if config_root is not None else CONFIG_ROOT
    for override in spec.hydra_overrides():
        parsed = _parse_group_override(override)
        if parsed is None:
            continue
        group, value = parsed
        group_dir = root / group
        if not group_dir.is_dir():
            continue
        if (group_dir / f"{value}.yaml").is_file():
            continue
        available = sorted(
            p.relative_to(group_dir).with_suffix("").as_posix()
            for p in group_dir.rglob("*.yaml")
        )
        raise ValueError(
            f"run {spec.run_id!r} emits the Hydra override {override!r}, but "
            f"there is no {group}/{value}.yaml under {root}. Hydra selects a "
            f"config group by FILENAME, so this run would queue, start, and "
            f"die on a MissingConfigException inside SLURM -- once per "
            f"affected run -- rather than here. Available {group} groups: "
            f"{available or '(none)'}"
        )


def expand_manifest(cfg: dict[str, Any], *, config_root=None) -> list[RunSpec]:
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
        modes = acquisition_modes_for(arm, cfg["acquisition"])
        for system in cfg["systems"]:
            for mode in modes:
                for seed in cfg["seeds"]:
                    spec = RunSpec(
                        arm=arm, system=system, tier=tier, acquisition=mode,
                        seed=int(seed), n_epochs=int(cfg["n_epochs"]),
                        overrides=tuple(cfg.get("overrides", ())),
                    )
                    assert_config_groups_exist(spec, config_root=config_root)
                    specs.append(spec)
    return specs

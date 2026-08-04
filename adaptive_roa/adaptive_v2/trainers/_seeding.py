"""Seed resolution shared by every trainer that seeds sub-models itself.

Three trainers (`BayesianMLPTrainer`, `FinalStateTrainer`, `HMCTrainer`) each
re-seed the global/per-object RNG per ensemble member or per chain. Each one
originally did it with `int(<arm_cfg>.get("seed", 0)) + i`, reading a key that
exists in NO predictor config -- so it silently resolved to 0 in every run and
members/chains were always seeded `0..n-1` regardless of `cfg.seed`.

The engine's `pl.seed_everything(cfg.seed)` does not compensate: these
per-member `torch.manual_seed` / per-chain `torch.Generator` calls immediately
override it.

The consequence is worse than a missing feature: it silently invalidates the
run-to-run floor. Seed replicates then vary only by cross-cluster float
nondeterminism, so replicates on one cluster come out identical to the last bit,
which reads as a floor of exactly zero and makes every arm gap look significant.
For a REFERENCE arm it is worse still -- the reference's own sampling
variability is precisely the scale against which an approximation's gap must be
judged, and a bit-identical replicate sets that scale to 0 by construction.

An explicit `<arm_cfg>.seed` still wins, so a config can pin sub-model seeds
independently of the run seed.
"""
from __future__ import annotations

from typing import Any

DEFAULT_RUN_SEED = 42


def resolve_seed_base(run_cfg: Any, arm_cfg: Any) -> int:
    """Seed base for a trainer's sub-models: explicit arm seed, else the RUN seed.

    ``run_cfg`` is the full run config (the one carrying ``seed``); ``arm_cfg``
    is the per-arm sub-config (``predictor.bnn``, ``predictor.final_state``,
    ``predictor.hmc``) that may pin ``seed`` explicitly.
    """
    explicit = arm_cfg.get("seed") if arm_cfg is not None else None
    if explicit is not None:
        return int(explicit)
    return int(run_cfg.get("seed", DEFAULT_RUN_SEED))

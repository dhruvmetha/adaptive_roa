"""Every trainer that seeds its own sub-models must follow the run seed.

`torch.manual_seed(<arm>.get("seed", 0) + i)` read a key that exists in no
predictor config, so it resolved to 0 in every run and `seed=43`/`seed=44`
trained bit-identical ensembles. Seed replicates then measured nothing but
cross-cluster float nondeterminism, which made the run-to-run floor collapse to
~0 and every arm gap look significant.

The fix landed for `BayesianMLPTrainer` first; `FinalStateTrainer` carried the
identical line untouched, so `bnn_ensemble_reg` kept the bug on the very arm
family the HMC final-state reference is supposed to be measured against. Both
are parameterized here so a future refactor cannot half-apply it again.
"""
import re
from pathlib import Path

import pytest

from adaptive_roa.adaptive_v2.trainers._seeding import resolve_seed_base
from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer
from adaptive_roa.adaptive_v2.trainers.final_state_trainer import FinalStateTrainer

# (trainer class, the sub-config key it reads its own seed from)
_TRAINERS = [(BayesianMLPTrainer, "bnn"), (FinalStateTrainer, "final_state")]


def _trainer(cls, cfg: dict):
    t = cls.__new__(cls)   # skip __init__ side effects
    t.cfg = cfg
    return t


@pytest.mark.parametrize("cls,key", _TRAINERS)
def test_member_seed_follows_the_run_seed(cls, key):
    for seed in (42, 43, 44):
        cfg = {"seed": seed, "predictor": {key: {"n_members": 5}}}
        assert _trainer(cls, cfg)._member_seed_base() == seed


@pytest.mark.parametrize("cls,key", _TRAINERS)
def test_different_run_seeds_give_different_member_seeds(cls, key):
    a = _trainer(cls, {"seed": 43, "predictor": {key: {}}})._member_seed_base()
    b = _trainer(cls, {"seed": 44, "predictor": {key: {}}})._member_seed_base()
    assert a != b, "seed replicates would train identical ensembles"


@pytest.mark.parametrize("cls,key", _TRAINERS)
def test_explicit_arm_seed_still_wins(cls, key):
    cfg = {"seed": 43, "predictor": {key: {"seed": 7}}}
    assert _trainer(cls, cfg)._member_seed_base() == 7


@pytest.mark.parametrize("cls,key", _TRAINERS)
def test_defaults_to_42_when_no_seed_given(cls, key):
    assert _trainer(cls, {"predictor": {key: {}}})._member_seed_base() == 42


def test_seed_zero_is_honoured_rather_than_treated_as_unset():
    """`seed: 0` is a legitimate explicit pin. A truthiness check (`or`) instead
    of an `is not None` check would silently promote it to the run seed."""
    assert resolve_seed_base({"seed": 43}, {"seed": 0}) == 0


def test_no_predictor_config_pins_a_sub_model_seed():
    """The original bug was a key that existed nowhere; the HMC arms then made it
    worse by pinning it to 0, which no explicit-wins fallback can rescue. If a
    config starts setting one again, that arm's replicates stop following the run
    seed and this test is the place that says so."""
    offenders = [
        path.name
        for path in sorted(Path("configs/adaptive_v2/predictor").glob("*.yaml"))
        # `seed:` indented under bnn/final_state/hmc, not a top-level run seed.
        if re.search(r"^\s{4,}seed:", path.read_text(), flags=re.MULTILINE)
    ]
    assert not offenders, f"predictor configs pin a sub-model seed: {offenders}"

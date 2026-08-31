"""The ensemble member seed must follow the run seed.

`torch.manual_seed(bnn.get("seed", 0) + m)` read a key that exists in no predictor
config, so it resolved to 0 in every run and `seed=43`/`seed=44` trained
bit-identical ensembles. Seed replicates then measured nothing but cross-cluster
float nondeterminism, which made the run-to-run floor collapse to ~0 and every
arm gap look significant.
"""
from types import SimpleNamespace

import pytest

from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer


def _trainer(cfg: dict):
    t = BayesianMLPTrainer.__new__(BayesianMLPTrainer)   # skip __init__ side effects
    t.cfg = cfg
    return t


def test_member_seed_follows_the_run_seed():
    for seed in (42, 43, 44):
        cfg = {"seed": seed, "predictor": {"bnn": {"n_members": 5}}}
        assert _trainer(cfg)._member_seed_base() == seed


def test_different_run_seeds_give_different_member_seeds():
    a = _trainer({"seed": 43, "predictor": {"bnn": {}}})._member_seed_base()
    b = _trainer({"seed": 44, "predictor": {"bnn": {}}})._member_seed_base()
    assert a != b, "seed replicates would train identical ensembles"


def test_explicit_bnn_seed_still_wins():
    cfg = {"seed": 43, "predictor": {"bnn": {"seed": 7}}}
    assert _trainer(cfg)._member_seed_base() == 7


def test_defaults_to_42_when_no_seed_given():
    assert _trainer({"predictor": {"bnn": {}}})._member_seed_base() == 42

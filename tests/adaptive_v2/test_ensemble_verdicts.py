"""Tests for the campaign's verdict rules.

`pooled_floor` decides every conclusion in the ensemble-epistemic campaign, and both
of its non-obvious behaviours were adopted only after getting them wrong:

  * excluding epoch 0 — all seeds hold identical data pre-acquisition, so its spread
    is structurally ~0. Including it drags the pooled floor toward zero and makes
    every gap look significant.
  * pooling across epochs — with three seeds a single epoch's SD has two degrees of
    freedom. The measured per-epoch floor varied 15.6x, which flipped a verdict
    depending on which epoch was used.

A silent regression in either would not crash anything; it would just quietly change
what the campaign concludes.
"""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ensemble_verdicts",
    Path(__file__).resolve().parents[2] / "scripts" / "ensemble_verdicts.py",
)
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)


def test_pooled_floor_matches_hand_calculation():
    # seeds (0,1,2) at epoch 1 -> sample variance 1.0
    # seeds (0,2,4) at epoch 2 -> sample variance 4.0
    # mean variance 2.5 -> floor = 2*sqrt(2.5)
    s1 = {1: 0.0, 2: 0.0}
    s2 = {1: 1.0, 2: 2.0}
    s3 = {1: 2.0, 2: 4.0}
    floor, n, shared = ev.pooled_floor([s1, s2, s3])
    assert n == 2 and shared == [1, 2]
    assert floor == pytest.approx(2 * (2.5 ** 0.5))


def test_epoch_zero_is_excluded():
    """Epoch 0 has zero spread by construction and must not enter the pool."""
    identical = 0.5
    s1 = {0: identical, 1: 0.0, 2: 0.0}
    s2 = {0: identical, 1: 1.0, 2: 2.0}
    s3 = {0: identical, 1: 2.0, 2: 4.0}
    with_zero_excluded, n, shared = ev.pooled_floor([s1, s2, s3])
    assert 0 not in shared, "epoch 0 must be excluded"
    # including it would average in a 0 variance and shrink the floor
    including_zero, _, _ = ev.pooled_floor([s1, s2, s3], min_epoch=0)
    assert including_zero < with_zero_excluded


def test_pooling_is_not_the_same_as_any_single_epoch():
    """The whole point: one epoch is a noisy estimate, the pool is not."""
    s1 = {1: 0.0, 2: 0.0, 3: 0.0}
    s2 = {1: 0.01, 2: 5.0, 3: 0.01}      # epoch 2 is a huge outlier
    s3 = {1: 0.02, 2: 10.0, 3: 0.02}
    pooled, _, _ = ev.pooled_floor([s1, s2, s3])
    ep1_only, _, _ = ev.pooled_floor([{1: s[1], 2: s[1]} for s in (s1, s2, s3)])
    assert pooled > ep1_only * 10, "an outlier epoch must lift the pooled floor"


def test_too_few_shared_epochs_returns_none_rather_than_guessing():
    a = {1: 0.0}
    b = {1: 1.0}
    c = {1: 2.0}
    floor, n, _ = ev.pooled_floor([a, b, c])
    assert floor is None and n == 0, "one epoch cannot support a pooled floor"


def _mk(level, arm_vals):
    """Build the (predictor, level, arm) -> {epoch: value} structure verdicts() wants."""
    return {("clf", level, arm): vals for arm, vals in arm_vals.items()}


def test_verdict_labels_distinguish_null_from_unstable():
    """A null (both epochs inside the floor) must not read as 'not stable'."""
    floor_seeds = {e: 0.0 for e in (1, 2, 3)}
    data = _mk("high", {
        "dir00":     {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
        "dir00_s43": {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.0, 5: 0.0},
        "dir00_s44": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.0, 5: 0.0},
        "epi_bald":  {4: 0.01, 5: 0.01},    # tiny -> inside the floor -> null
        "total":     {4: 5.0, 5: 6.0},      # large, same sign -> distinguishable
        "aleat":     {4: -5.0, 5: 6.0},     # large, sign flips -> not stable
    })
    res, why = ev.verdicts(data, "clf", "high")
    assert res is not None, why
    labels = {arm: label for arm, _, _, _, label in res[2]}
    assert labels["epi_bald"] == "within noise (null)"
    assert labels["total"] == "DISTINGUISHABLE"
    assert labels["aleat"] == "not stable"


def test_missing_floor_seeds_reports_why_instead_of_skipping():
    data = _mk("xhigh", {"dir00": {1: 0.0, 2: 0.0}, "total": {1: 1.0, 2: 1.0}})
    res, why = ev.verdicts(data, "clf", "xhigh")
    assert res is None and "floor incomplete" in why

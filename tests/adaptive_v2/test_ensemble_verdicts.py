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
    # DISTINGUISHABLE now carries a trajectory annotation; the verdict is the prefix.
    assert labels["total"].startswith("DISTINGUISHABLE")
    assert "sign stable" in labels["total"]
    assert labels["aleat"] == "not stable"


def test_missing_floor_seeds_reports_why_instead_of_skipping():
    data = _mk("xhigh", {"dir00": {1: 0.0, 2: 0.0}, "total": {1: 1.0, 2: 1.0}})
    res, why = ev.verdicts(data, "clf", "xhigh")
    assert res is None and "floor incomplete" in why


def test_sign_flip_over_the_full_trajectory_is_flagged():
    """Two adjacent epochs can agree while the quantity oscillates underneath.

    At [CLF] high the least-harmful arm rotated almost every epoch and all four arms
    held that slot, so a two-epoch window -- even with two independent metrics
    agreeing -- reported a stable effect that did not exist. The verdict must carry
    the full-trajectory sign check, not just the window.
    """
    data = _mk("high", {
        "dir00":     {e: 0.0 for e in range(1, 8)},
        "dir00_s43": {e: 0.1 for e in range(1, 8)},
        "dir00_s44": {e: 0.2 for e in range(1, 8)},
        # large and same-signed in the last two epochs, but flips earlier
        "total":     {1: -5.0, 2: 5.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 5.0, 7: 6.0},
        # large and same-signed throughout
        "aleat":     {e: 5.0 + e for e in range(1, 8)},
    })
    res, why = ev.verdicts(data, "clf", "high")
    assert res is not None, why
    labels = {arm: label for arm, _, _, _, label in res[2]}
    assert "[!]" in labels["total"], "an early sign flip must be flagged"
    assert "[!]" not in labels["aleat"] and "sign stable" in labels["aleat"]


def test_restart_in_place_contamination_is_detected(tmp_path):
    """A requeued job overwrites its own epoch dirs from 0, stitching two runs.

    Nothing about the result looks wrong -- every epoch still carries a valid
    full_roa_per_point.npz, so the scorer consumes it happily. The only signature is
    that epoch mtimes stop increasing with epoch number.
    """
    import os

    clean = tmp_path / "clf_high_dir00"
    for e in range(5):
        d = clean / f"epoch_{e:03d}"
        d.mkdir(parents=True)
        os.utime(d, (1000 + e * 10, 1000 + e * 10))     # written in order

    stitched = tmp_path / "clf_high_total"
    for e in range(5):
        d = stitched / f"epoch_{e:03d}"
        d.mkdir(parents=True)
        # epochs 0-1 rewritten by the restart, so they are NEWER than 2-4
        t = 9000 + e if e < 2 else 1000 + e
        os.utime(d, (t, t))

    # a preserved backup must never be flagged -- it is the rescue copy
    backup = tmp_path / "_preserved_clf_high_total_preempt_1931"
    (backup / "epoch_000").mkdir(parents=True)

    bad = ev.contaminated_runs(tmp_path)
    assert "clf_high_total" in bad, "an out-of-order rewrite must be caught"
    assert "clf_high_dir00" not in bad, "a clean run must not be flagged"
    assert not any(k.startswith("_preserved") for k in bad), "backups are not runs"


def test_load_det_reads_the_requested_predictor(tmp_path):
    """load_det was hardcoded to clf_det_*, so FM det runs scored as nothing.

    The failure was silent: `ensemble_verdicts.py --levels det --predictors fm`
    printed no FM row at all rather than erroring, so a whole level looked
    un-launched when it was actually running.
    """
    import json

    def write(run, arm, epochs):
        for e, v in epochs.items():
            d = tmp_path / f"{run}_{arm}" / f"epoch_{e:03d}"
            d.mkdir(parents=True)
            (d / "artifacts_v2.json").write_text(
                json.dumps({"eval_metrics": {"threshold_free": {"brier": v}}}))

    write("clf_det", "dir00", {0: 0.10, 1: 0.11})
    write("fm_det", "dir00", {0: 0.20, 1: 0.21})
    write("fm_det", "total", {0: 0.30, 1: 0.31})

    clf = ev.load_det(tmp_path, "brier", "clf")
    assert set(clf) == {("clf", "det", "dir00")}
    assert clf[("clf", "det", "dir00")][1] == 0.11

    fm = ev.load_det(tmp_path, "brier", "fm")
    assert set(fm) == {("fm", "det", "dir00"), ("fm", "det", "total")}
    assert fm[("fm", "det", "total")][1] == 0.31
    # the clf runs must not leak into the fm view
    assert all(k[0] == "fm" for k in fm)

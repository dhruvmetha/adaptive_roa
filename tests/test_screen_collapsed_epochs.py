"""Tests for the collapsed-epoch screen.

The screen exists because a chance-level epoch corrupts results in two
directions: inside a floor pool it manufactures false nulls, inside an arm it
manufactures a false catastrophe. Both happened in the ensemble campaign before
the screen existed, so these tests pin the exact cases that were missed.
"""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "screen_collapsed_epochs",
    Path(__file__).resolve().parents[1] / "scripts" / "screen_collapsed_epochs.py",
)
screen_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(screen_mod)
screen = screen_mod.screen


def _row(predictor="clf", level="low", arm="total", epoch=10, sauroc=0.99, res=0.19):
    return {"predictor": predictor, "level": level, "arm": arm,
            "epoch": str(epoch), "sAUROC": str(sauroc), "RES": str(res)}


def test_flags_the_real_collapse_that_was_reported_as_a_finding():
    """clf_low_total ep10 was written up as a +896x acquisition failure.

    It is a collapsed model: sAUROC 0.5002, RES 0.00001.
    """
    hits, _, _ = screen([_row(sauroc=0.5002, res=0.00001)])
    assert len(hits) == 1
    assert hits[0]["arm"] == "total" and hits[0]["epoch"] == 10
    assert hits[0]["is_floor_seed"] is False


def test_flags_floor_seeds_and_marks_them_as_such():
    """A collapse in a floor seed is the dangerous case -- it inflates the floor."""
    hits, _, _ = screen([_row(arm="dir00_s44", epoch=14, sauroc=0.5038, res=0.00062)])
    assert len(hits) == 1 and hits[0]["is_floor_seed"] is True


def test_healthy_epochs_are_not_flagged():
    rows = [_row(sauroc=0.9813, res=0.18956), _row(sauroc=0.8977, res=0.13241)]
    hits, _, _ = screen(rows)
    assert hits == []


def test_requires_both_conditions_not_either():
    """Low sAUROC alone, or low RES alone, must not trip the screen.

    A genuinely hard level can have modest sAUROC while still being a working
    model, and RES varies with the base rate. Only the conjunction is the
    collapse signature.
    """
    assert screen([_row(sauroc=0.55, res=0.19)])[0] == []   # chance-ish AUC, healthy resolution
    assert screen([_row(sauroc=0.99, res=0.001)])[0] == []  # tiny resolution, healthy ranking


def test_counts_rate_per_predictor():
    rows = [_row(predictor="clf", sauroc=0.50, res=0.0),
            _row(predictor="clf"), _row(predictor="clf"), _row(predictor="clf"),
            _row(predictor="fm"), _row(predictor="fm")]
    _, totals, collapsed = screen(rows)
    assert totals == {"clf": 4, "fm": 2}
    assert collapsed == {"clf": 1}          # fm absent, not zero -- caller uses .get


def test_malformed_rows_are_skipped_not_fatal():
    """A partially written epoch must not crash the screen that guards it."""
    rows = [{"predictor": "clf", "level": "low", "arm": "total",
             "epoch": "not-an-int", "sAUROC": "0.5", "RES": "0.0"},
            {"predictor": "clf"},                       # missing columns entirely
            _row(sauroc=0.5002, res=0.00001)]
    hits, totals, _ = screen(rows)
    assert len(hits) == 1
    assert totals == {"clf": 1}


def test_resolve_accepts_directory_or_file():
    """stoch_prob_metrics --out takes a DIRECTORY and writes metrics.csv inside.

    Passing that directory straight to the screen must work, since getting this
    wrong once already cost a failed scoring run (IsADirectoryError).
    """
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / "metrics.csv"
        csv_path.write_text("predictor,level,arm,epoch,sAUROC,RES\n")
        assert screen_mod._resolve(Path(d)) == csv_path
        assert screen_mod._resolve(csv_path) == csv_path

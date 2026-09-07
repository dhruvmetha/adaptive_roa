from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "resume_adaptive_until.py"
SPEC = importlib.util.spec_from_file_location("resume_adaptive_until", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _result(epoch: int, size: int, f1: float) -> dict:
    return {
        "epoch": epoch,
        "train_trajectories": size,
        "full_roa": {"conservative_lambda_delta": {"f1": f1}},
    }


def test_improvement_state_uses_min_delta_and_consecutive_patience():
    results = [
        _result(0, 50, 0.50),
        _result(1, 100, 0.504),
        _result(2, 150, 0.506),
        _result(3, 200, 0.508),
    ]

    best, stale, history = MODULE.improvement_state(
        results, MODULE.DEFAULT_METRIC, min_delta=0.005
    )

    assert best == pytest.approx(0.506)
    assert stale == 1
    assert [row["improved"] for row in history] == [True, False, True, False]


def test_nested_metric_rejects_missing_metric():
    with pytest.raises(KeyError, match="conservative_lambda_delta"):
        MODULE.improvement_state(
            [_result(0, 50, 0.5) | {"full_roa": {}}],
            MODULE.DEFAULT_METRIC,
            min_delta=0.005,
        )

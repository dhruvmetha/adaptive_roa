"""The threshold-free block must survive flattening into metrics_summary.csv."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "compile_adaptive_metrics",
    Path(__file__).resolve().parent.parent / "scripts" / "compile_adaptive_metrics.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
extract_metrics_from_epoch = _MOD.extract_metrics_from_epoch


def test_threshold_free_metrics_are_flattened_into_columns():
    results = {
        "epoch": 3,
        "full_roa": {
            "threshold_free": {
                "auc": 0.94,
                "auprc": 0.88,
                "brier": 0.07,
                "log_score": 0.31,
                "log_score_smoothing": "kt_count",
                "n_saturated": 12,
                "base_rate": 0.38,
            },
        },
    }

    metrics = extract_metrics_from_epoch(results)

    assert metrics["auc"] == 0.94
    assert metrics["auprc"] == 0.88
    assert metrics["brier"] == 0.07
    assert metrics["log_score"] == 0.31
    assert metrics["log_score_smoothing"] == "kt_count"
    assert metrics["n_saturated"] == 12
    assert metrics["base_rate"] == 0.38


def test_missing_threshold_free_block_yields_none_not_zero():
    """Older runs predate the block; 0.0 would read as a real inverted-ranking AUC."""
    metrics = extract_metrics_from_epoch({"epoch": 0, "full_roa": {}})

    assert metrics["auc"] is None
    assert metrics["auprc"] is None
    assert metrics["brier"] is None
    assert metrics["log_score"] is None

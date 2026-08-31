"""The verbose line must not crash the whole eval on degenerate inputs."""
import numpy as np

from adaptive_roa.adaptive_v2.eval.full_roa import (
    _print_threshold_free,
    _threshold_free_metrics,
)


def test_printing_single_class_metrics_does_not_raise(capsys):
    """auc/auprc are None here; formatting them with :.4f would be a TypeError."""
    m = _threshold_free_metrics(np.array([0.2, 0.8]), np.array([1, 1]))

    _print_threshold_free(m, tag="Full ROA")

    assert "n/a" in capsys.readouterr().out


def test_printing_empty_metrics_does_not_raise(capsys):
    m = _threshold_free_metrics(np.array([]), np.array([]))

    _print_threshold_free(m, tag="Full ROA")

    assert "n/a" in capsys.readouterr().out


def test_printing_reports_saturation_share(capsys):
    p = np.array([0.0, 1.0, 0.4, 0.6])
    y = np.array([1, -1, -1, 1])

    _print_threshold_free(_threshold_free_metrics(p, y), tag="Full ROA")

    assert "saturated=50.0%" in capsys.readouterr().out

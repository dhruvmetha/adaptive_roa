"""Tests for the FM cal/test gt_label basis fix in export.py.

FM splits must all derive gt_label from endpoints at the eval radius (so cal/test
share the same radius-dependent basis as train/val and the MC probabilities),
while the classifier keeps its radius-independent binary labels from the file.
"""
import numpy as np
import pytest
import torch

from adaptive_roa.probabilistic_classifier.export import (
    load_split_states,
    _labels_from_endpoints,
)


def _sys():
    try:
        from adaptive_roa.systems.pendulum import PendulumSystem
        return PendulumSystem()
    except FileNotFoundError:
        pytest.skip("pendulum dataset unavailable")


def _write_eval(path, starts, ends, labels):
    data = np.hstack([starts, ends, labels.reshape(-1, 1)])
    np.savetxt(path, data, delimiter=",")


def test_fm_cal_defers_labels_and_returns_endpoints(tmp_path):
    system = _sys()
    sd = int(system.state_dim)
    n = 5
    rng = np.random.default_rng(0)
    starts, ends = rng.normal(size=(n, sd)), rng.normal(size=(n, sd))
    labels = rng.integers(0, 2, size=n)
    cal = tmp_path / "cal.txt"
    _write_eval(cal, starts, ends, labels)
    cfg = {"data_source": {"cal_set_file": str(cal), "test_set_file": str(cal)}}

    states, lab, endpts = load_split_states(str(tmp_path), "cal", "generative", system, cfg)
    # FM defers label derivation; carries endpoints for radius-based classification.
    assert lab is None
    assert endpts is not None and endpts.shape == (n, sd)

    derived = _labels_from_endpoints(endpts, system, 0.2)
    expected = system.classify_attractor(
        torch.as_tensor(endpts, dtype=torch.float32), radius=0.2
    ).numpy()
    assert np.array_equal(derived, expected)
    # radius-dependent basis can include the invalid/separatrix class (0)
    assert set(np.unique(derived)).issubset({-1, 0, 1})


def test_classifier_cal_keeps_file_labels(tmp_path):
    system = _sys()
    sd = int(system.state_dim)
    n = 4
    rng = np.random.default_rng(1)
    starts, ends = rng.normal(size=(n, sd)), rng.normal(size=(n, sd))
    labels = np.array([0, 1, 0, 1])
    cal = tmp_path / "cal.txt"
    _write_eval(cal, starts, ends, labels)
    cfg = {"data_source": {"cal_set_file": str(cal), "test_set_file": str(cal)}}

    states, lab, endpts = load_split_states(str(tmp_path), "cal", "classifier", system, cfg)
    assert endpts is None
    assert lab is not None
    # load_eval_states maps file {0,1} -> {-1,1}; never the radius-only invalid class.
    assert set(np.unique(lab)).issubset({-1, 1})

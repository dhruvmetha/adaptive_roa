"""Tests for the standalone eval driver: metric/per-query assembly + output writing."""
import json

import numpy as np
import torch

from adaptive_roa.partial_trajs.model.base import DynamicsModel
from adaptive_roa.partial_trajs.verifier.evaluate import (
    run_verifier_eval,
    write_eval_outputs,
    eval_filename_for_split,
)


class StubSystem:
    def classify_attractor(self, state, radius=None):
        x = state[:, 0]
        labels = torch.zeros(state.shape[0], dtype=torch.long)
        labels[x > 1.0] = 1
        labels[x < -1.0] = -1
        return labels

    def get_circular_indices(self):
        return ()


class DriftModel(DynamicsModel):
    def __init__(self, step):
        self.step = step

    def predict(self, x):
        out = x.clone()
        out[:, 0] = out[:, 0] + self.step
        return out


def test_run_verifier_eval_assembles_metrics_and_per_query():
    init = torch.zeros(4, 2)
    terminal = torch.full((4, 2), 1.5)
    terminal[:, 1] = 0.0
    label = torch.ones(4, dtype=torch.long)

    metrics, per_query = run_verifier_eval(
        DriftModel(0.5), StubSystem(), init, terminal, label, K=5
    )

    # ROA block (extended roa_scores) present and perfect on this all-success case
    assert metrics["roa"]["f1"] == 1.0
    assert metrics["roa"]["n_total"] == 4
    # stratified metric #2
    strat = metrics["rollout_final_state_error"]
    assert strat["overall"]["n"] == 4
    assert "by_gt" in strat and "by_pred" in strat and "by_gt_x_pred" in strat
    # per-query arrays
    assert set(per_query) >= {"init", "pred_label", "true_label", "final_state", "terminal_class"}
    assert per_query["pred_label"].tolist() == [1, 1, 1, 1]
    assert per_query["terminal_class"].tolist() == [1, 1, 1, 1]  # GT terminals in success set


def test_run_verifier_eval_probabilistic_reports_p_success():
    init = torch.zeros(3, 2)
    terminal = torch.full((3, 2), 1.5)
    label = torch.ones(3, dtype=torch.long)
    metrics, per_query = run_verifier_eval(
        DriftModel(0.5), StubSystem(), init, terminal, label, K=5, num_samples=4
    )
    assert "p_success_mean" in metrics
    assert "p_success" in per_query


def test_write_eval_outputs(tmp_path):
    metrics = {"roa": {"f1": 1.0}, "rollout_final_state_error": {"overall": {"n": 2}}}
    per_query = {
        "init": torch.zeros(2, 2),
        "pred_label": torch.tensor([1, 0]),
        "true_label": torch.tensor([1, 1]),
        "final_state": torch.ones(2, 2),
        "terminal_class": torch.tensor([1, 0]),
    }
    metadata = {"system": "stub", "K": 5}

    path = write_eval_outputs(tmp_path, metrics, per_query, metadata, export_predictions=True)

    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["metadata"]["system"] == "stub"
    assert loaded["roa"]["f1"] == 1.0
    npz = np.load(tmp_path / "predictions.npz")
    assert npz["pred_label"].tolist() == [1, 0]
    assert npz["init"].shape == (2, 2)


def test_write_eval_outputs_metrics_only(tmp_path):
    write_eval_outputs(
        tmp_path, {"roa": {}}, {"init": torch.zeros(1, 2)}, {}, export_predictions=False
    )
    assert (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "predictions.npz").exists()


def test_eval_filename_for_split_standard_and_fps():
    # standard naming
    assert eval_filename_for_split("test_set.txt", "test") == "test_set.txt"
    assert eval_filename_for_split("test_set.txt", "cal") == "cal_set.txt"
    assert eval_filename_for_split("test_set.txt", "eval") == "eval_states.txt"
    # humanoid fps naming
    assert eval_filename_for_split("test_set_fps.txt", "test") == "test_set_fps.txt"
    assert eval_filename_for_split("test_set_fps.txt", "cal") == "cal_set_fps.txt"
    assert eval_filename_for_split("test_set_fps.txt", "eval") == "eval_fps.txt"

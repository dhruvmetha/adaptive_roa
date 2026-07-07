"""Tests for ROA evaluation orchestration (loader + evaluate_roa)."""
import torch

from adaptive_roa.partial_trajs.model.base import DynamicsModel
from adaptive_roa.partial_trajs.verifier.evaluate_roa import (
    load_eval_states,
    evaluate_roa,
)


class StubSystem:
    def classify_attractor(self, state, radius=None):
        x = state[:, 0]
        labels = torch.zeros(state.shape[0], dtype=torch.long)
        labels[x > 1.0] = 1
        labels[x < -1.0] = -1
        return labels


class DriftModel(DynamicsModel):
    def __init__(self, step):
        self.step = step

    def predict(self, x):
        out = x.clone()
        out[:, 0] = out[:, 0] + self.step
        return out


def test_load_eval_states(tmp_path):
    p = tmp_path / "eval.txt"
    p.write_text("0.0,0.0,1.0,0.0,1\n0.5,0.0,-2.0,0.0,0\n")
    init, terminal, label = load_eval_states(p, state_dim=2)
    assert init.shape == (2, 2)
    assert terminal.shape == (2, 2)
    assert label.tolist() == [1, 0]
    assert init[0].tolist() == [0.0, 0.0]
    assert terminal[1].tolist() == [-2.0, 0.0]


def test_evaluate_roa_perfect_success():
    init = torch.zeros(4, 2)
    terminal = torch.full((4, 2), 1.5)  # where DriftModel(0.5) freezes
    terminal[:, 1] = 0.0
    label = torch.ones(4)
    res = evaluate_roa(DriftModel(0.5), StubSystem(), init, terminal, label, K=5)
    assert res["f1"] == 1.0
    assert res["recall"] == 1.0
    assert "rollout_final_state_error" in res
    # model lands exactly on the true terminal -> ~zero error
    assert res["rollout_final_state_error"] < 1e-5


def test_evaluate_roa_probabilistic_path_runs():
    init = torch.zeros(3, 2)
    terminal = torch.full((3, 2), 1.5)
    label = torch.ones(3)
    res = evaluate_roa(
        DriftModel(0.5), StubSystem(), init, terminal, label, K=5, num_samples=4
    )
    assert res["f1"] == 1.0
    assert "p_success_mean" in res

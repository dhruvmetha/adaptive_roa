"""Tests for the verifier rollout (absorbing resolution + probabilistic aggregation)."""
import torch

from adaptive_roa.partial_trajs.model.base import DynamicsModel
from adaptive_roa.partial_trajs.verifier.rollout import (
    resolve_outcome,
    resolve_probabilistic,
    rollout_final_state,
)


class StubSystem:
    """1-D toy: x[:,0] > 1 -> success (+1), < -1 -> failure (-1), else unresolved (0)."""

    def classify_attractor(self, state, radius=None):
        x = state[:, 0]
        labels = torch.zeros(state.shape[0], dtype=torch.long)
        labels[x > 1.0] = 1
        labels[x < -1.0] = -1
        return labels


class DriftModel(DynamicsModel):
    """Deterministic drift by `step` along dim 0 each call."""

    def __init__(self, step):
        self.step = step

    def predict(self, x):
        out = x.clone()
        out[:, 0] = out[:, 0] + self.step
        return out


class NoisyDriftModel(DynamicsModel):
    def __init__(self, step, seed=0):
        self.step = step
        self.g = torch.Generator().manual_seed(seed)

    def predict(self, x):
        out = x.clone()
        out[:, 0] = out[:, 0] + self.step
        return out

    def sample(self, x, num_samples):
        base = torch.stack([self.predict(x) for _ in range(num_samples)], dim=0)
        noise = torch.randn(base.shape, generator=self.g) * 0.5
        noise[..., 1:] = 0.0
        return base + noise


def test_drift_to_goal_resolves_success():
    labels = resolve_outcome(DriftModel(0.5), StubSystem(), torch.zeros(1, 2), K=5)
    assert labels.tolist() == [1]


def test_drift_to_failure_resolves_failure():
    labels = resolve_outcome(DriftModel(-0.5), StubSystem(), torch.zeros(1, 2), K=5)
    assert labels.tolist() == [-1]


def test_no_drift_is_unresolved():
    labels = resolve_outcome(DriftModel(0.0), StubSystem(), torch.zeros(1, 2), K=5)
    assert labels.tolist() == [0]


def test_initial_state_already_at_goal_resolves_immediately():
    x0 = torch.tensor([[2.0, 0.0]])
    labels = resolve_outcome(DriftModel(-0.5), StubSystem(), x0, K=5)
    assert labels.tolist() == [1]


def test_batch_of_mixed_outcomes():
    x0 = torch.zeros(3, 2)
    # different drifts via three separate calls collapsed: use one model per row is awkward,
    # so check three homogeneous batches instead
    up = resolve_outcome(DriftModel(0.5), StubSystem(), x0, K=5)
    down = resolve_outcome(DriftModel(-0.5), StubSystem(), x0, K=5)
    flat = resolve_outcome(DriftModel(0.0), StubSystem(), x0, K=5)
    assert up.tolist() == [1, 1, 1]
    assert down.tolist() == [-1, -1, -1]
    assert flat.tolist() == [0, 0, 0]


def test_probabilistic_deterministic_model_gives_degenerate_probs():
    out = resolve_probabilistic(DriftModel(0.5), StubSystem(), torch.zeros(2, 2), K=5, num_samples=8)
    assert torch.allclose(out["p_success"], torch.ones(2))
    assert torch.allclose(out["p_failure"], torch.zeros(2))


def test_probabilistic_noisy_model_spreads_between_zero_and_one():
    # small drift + noise near the boundary -> mixed success/unresolved across samples
    x0 = torch.full((1, 2), 0.6)
    out = resolve_probabilistic(NoisyDriftModel(0.1), StubSystem(), x0, K=3, num_samples=64)
    p = out["p_success"].item()
    assert 0.0 < p < 1.0
    total = out["p_success"] + out["p_failure"] + out["p_unresolved"]
    assert torch.allclose(total, torch.ones(1))


def test_rollout_final_state_freezes_after_absorption():
    # drift 0.5 from 0: 0.5, 1.0, 1.5 (>1 -> resolved, frozen), stays 1.5
    final = rollout_final_state(DriftModel(0.5), StubSystem(), torch.zeros(1, 2), K=5)
    assert final.shape == (1, 2)
    assert abs(final[0, 0].item() - 1.5) < 1e-6

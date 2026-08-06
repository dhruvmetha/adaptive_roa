import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.strategy.decomposition import DecompositionAcquisitionStrategy
from adaptive_roa.adaptive_v2.types import ThresholdState
from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent


class _FakeSystem(DynamicalSystem):
    def define_manifold_structure(self):
        return [ManifoldComponent("SO2", 1, "angle"),
                ManifoldComponent("Real", 1, "angular_velocity")]

    def define_state_bounds(self):
        return {"angle": (-math.pi, math.pi), "angular_velocity": (-8.0, 8.0)}

    def classify_attractor(self, state, radius=0.1):
        lab = torch.zeros(state.shape[0], dtype=torch.int64)
        lab[state[:, 0].abs() < radius] = 1
        return lab


class _FakePool:
    def __init__(self, states):
        self.states = np.asarray(states, dtype=np.float32)
        self.last_exclude = None

    def sample_candidates_without_marking(self, n, exclude=None):
        self.last_exclude = exclude
        take = min(n, len(self.states))
        return self.states[:take], list(range(100, 100 + take))


class _FakeMemberBackend:
    """p_members fixed per candidate: [M, N]."""

    def __init__(self, p_members, system, k=None):
        self.p_members = np.asarray(p_members, dtype=np.float64)
        self.system, self.device, self.attractor_radius = system, "cpu", 0.2
        self.n_members = self.p_members.shape[0]
        self.member_sample_size = k

    def estimate_members(self, states, verbose=False):
        return self.p_members[:, : len(states)]


def _cfg(**over):
    base = dict(score="epistemic_var", d2_ratio=1.0, n_candidates=50,
                selection_rule="greedy", diversity_pool_multiplier=5, verbose=False)
    base.update(over)
    return OmegaConf.create(base)


def _ts():
    return ThresholdState(lambda_star=0.5, delta_star=0.1)


def test_epistemic_mode_prefers_disagreement_over_ambiguity():
    # candidate 0: all members say 0.5 -> max ALEATORIC, zero epistemic
    # candidate 1: members split 0/1  -> max EPISTEMIC
    p = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 1.0], [0.5, 1.0]])
    states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    sys_ = _FakeSystem()
    strat = DecompositionAcquisitionStrategy(_cfg(score="epistemic_var"))
    res = strat.select(_FakePool(states), _FakeMemberBackend(p, sys_), None, _ts(), 1)
    assert res.d2_indices == [101]          # the disagreement candidate


def test_aleatoric_mode_prefers_the_opposite_candidate():
    p = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 1.0], [0.5, 1.0]])
    states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    strat = DecompositionAcquisitionStrategy(_cfg(score="aleatoric"))
    res = strat.select(_FakePool(states), _FakeMemberBackend(p, _FakeSystem()), None, _ts(), 1)
    assert res.d2_indices == [100]          # the genuinely-ambiguous candidate


def test_total_mode_reproduces_plain_entropy_ranking():
    # marginals 0.5 and 0.5 -> tie on total, though epistemic differs sharply
    p = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 1.0], [0.5, 1.0]])
    strat = DecompositionAcquisitionStrategy(_cfg(score="total"))
    states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    res = strat.select(_FakePool(states), _FakeMemberBackend(p, _FakeSystem()), None, _ts(), 2)
    assert sorted(res.d2_indices) == [100, 101]


def test_diagnostics_report_score_mode_and_members():
    p = np.array([[0.2, 0.6], [0.8, 0.6]])
    strat = DecompositionAcquisitionStrategy(_cfg(score="epistemic_bald"))
    states = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    res = strat.select(_FakePool(states), _FakeMemberBackend(p, _FakeSystem()), None, _ts(), 1)
    assert res.diagnostics["score_mode"] == "epistemic_bald"
    assert res.diagnostics["n_members"] == 2
    assert "epistemic_mean" in res.diagnostics and "aleatoric_mean" in res.diagnostics


def test_backend_without_estimate_members_raises():
    class _NoMembers:
        system, device, attractor_radius = _FakeSystem(), "cpu", 0.2
    strat = DecompositionAcquisitionStrategy(_cfg())
    with pytest.raises(RuntimeError, match="estimate_members"):
        strat.select(_FakePool(np.zeros((2, 2), np.float32)), _NoMembers(), None, _ts(), 1)


def test_unknown_score_mode_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown score mode"):
        DecompositionAcquisitionStrategy(_cfg(score="bogus"))


def test_zero_target_count_skips_cleanly():
    strat = DecompositionAcquisitionStrategy(_cfg())
    res = strat.select(_FakePool(np.zeros((2, 2), np.float32)),
                       _FakeMemberBackend(np.full((2, 2), 0.5), _FakeSystem()),
                       None, _ts(), 0)
    assert res.d2_indices == [] and res.diagnostics["skipped_reason"] == "target_count_zero"

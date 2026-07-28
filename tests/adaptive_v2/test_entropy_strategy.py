import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.strategy.entropy import EntropyAcquisitionStrategy
from adaptive_roa.adaptive_v2.types import ThresholdState
from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent


class _FakeSystem(DynamicalSystem):
    """SO2 angle + Real velocity; success iff |theta| < radius."""

    def define_manifold_structure(self):
        return [ManifoldComponent("SO2", 1, "angle"),
                ManifoldComponent("Real", 1, "angular_velocity")]

    def define_state_bounds(self):
        return {"angle": (-math.pi, math.pi), "angular_velocity": (-8.0, 8.0)}

    def classify_attractor(self, state, radius=0.1):
        theta = state[:, 0]
        lab = torch.zeros(state.shape[0], dtype=torch.int64)
        lab[theta.abs() < radius] = 1
        lab[theta > 1.0] = -1
        return lab


class _FakePool:
    def __init__(self, states):
        self.states = np.asarray(states, dtype=np.float32)
        self.last_exclude = None

    def sample_candidates_without_marking(self, n, exclude=None):
        self.last_exclude = exclude
        take = min(n, len(self.states))
        return self.states[:take], list(range(100, 100 + take))


class _FakeBackend:
    def __init__(self, clouds, system):
        self.clouds = np.asarray(clouds, dtype=np.float32)
        self.system, self.device, self.attractor_radius = system, "cpu", 0.2
        self.last_num_samples = None

    def sample_endpoints(self, start_states, num_samples, verbose=True):
        self.last_num_samples = num_samples
        return self.clouds[: len(start_states)]


def _cfg(**over):
    base = dict(d2_ratio=0.5, n_candidates=50, num_mc_samples=4, tie_breaker="none",
                selection_rule="greedy_diverse", diversity_pool_multiplier=5,
                chunk_size=2048, verbose=False)
    base.update(over)
    return OmegaConf.create(base)


def _ts():
    return ThresholdState(lambda_star=0.5, delta_star=0.1)


def _cloud(n_success, n_total=4):
    """One candidate's cloud: n_success endpoints at theta=0, rest at theta=2 (failure)."""
    pts = [[0.0, 0.0]] * n_success + [[2.0, 0.0]] * (n_total - n_success)
    return np.array(pts, dtype=np.float32)


def test_stores_cfg_and_rejects_bad_tie_breaker():
    s = EntropyAcquisitionStrategy(_cfg())
    assert s.mode == "entropy" and s.d2_ratio == 0.5
    with pytest.raises(ValueError, match="tie_breaker"):
        EntropyAcquisitionStrategy(_cfg(tie_breaker="bogus"))


def test_selects_maximum_entropy_candidates():
    system = _FakeSystem()
    # p_success = 4/4, 2/4, 0/4, 3/4 -> entropy maximal for the 2/4 candidate
    clouds = np.stack([_cloud(4), _cloud(2), _cloud(0), _cloud(3)])
    pool = _FakePool(np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]]))
    backend = _FakeBackend(clouds, system)

    r = EntropyAcquisitionStrategy(_cfg(selection_rule="greedy")).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_ts(), target_count=1)

    assert r.d2_indices == [101]          # the p=0.5 candidate
    assert backend.last_num_samples == 4


def test_ignores_threshold_state_entirely():
    system = _FakeSystem()
    clouds = np.stack([_cloud(4), _cloud(2), _cloud(0)])
    states = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    a = EntropyAcquisitionStrategy(_cfg(selection_rule="greedy")).select(
        pool=_FakePool(states), probability_backend=_FakeBackend(clouds, system),
        threshold_backend=None, threshold_state=ThresholdState(0.1, 0.01), target_count=1)
    b = EntropyAcquisitionStrategy(_cfg(selection_rule="greedy")).select(
        pool=_FakePool(states), probability_backend=_FakeBackend(clouds, system),
        threshold_backend=None, threshold_state=ThresholdState(0.9, 0.4), target_count=1)
    assert a.d2_indices == b.d2_indices


def test_tie_breaker_reorders_only_within_entropy_ties():
    system = _FakeSystem()
    # All three sit at p_success = 2/4 (two endpoints inside the success radius
    # 0.2, two past the failure threshold 1.0), so entropy is identical and the
    # ordering is decided entirely by the tie-breaker. They differ only in
    # within-mode spread, so mode separation prefers the tightest.
    loose = np.array([[0.0, 0.0], [0.19, 0.0], [2.0, 0.0], [3.5, 0.0]], dtype=np.float32)
    mid = np.array([[0.0, 0.0], [0.10, 0.0], [2.0, 0.0], [2.7, 0.0]], dtype=np.float32)
    tight = np.array([[0.0, 0.0], [0.02, 0.0], [2.0, 0.0], [2.05, 0.0]], dtype=np.float32)
    clouds = np.stack([loose, mid, tight])
    pool = _FakePool(np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]))

    plain = EntropyAcquisitionStrategy(_cfg(selection_rule="greedy")).select(
        pool=pool, probability_backend=_FakeBackend(clouds, system), threshold_backend=None,
        threshold_state=_ts(), target_count=1)
    broken = EntropyAcquisitionStrategy(
        _cfg(selection_rule="greedy", tie_breaker="mode_sep")).select(
        pool=pool, probability_backend=_FakeBackend(clouds, system), threshold_backend=None,
        threshold_state=_ts(), target_count=1)

    assert plain.d2_indices == [100]      # stable order: first candidate wins the tie
    assert broken.d2_indices == [102]     # tightest/most separated modes wins
    assert broken.diagnostics["tie_breaker"] == "mode_sep"


def test_diagnostics_and_no_threshold_fields():
    system = _FakeSystem()
    clouds = np.stack([_cloud(4), _cloud(2), _cloud(1)])
    r = EntropyAcquisitionStrategy(_cfg()).select(
        pool=_FakePool(np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])),
        probability_backend=_FakeBackend(clouds, system), threshold_backend=None,
        threshold_state=_ts(), target_count=2)
    d = r.diagnostics
    for k in ("entropy_min", "entropy_max", "entropy_mean", "n_candidates_evaluated",
              "tie_breaker", "selection_rule", "mean_p_success"):
        assert k in d
    assert not any("lambda" in k or "delta" in k or "q_hat" in k for k in d)


def test_target_count_zero_and_empty_pool_skip():
    system = _FakeSystem()
    s = EntropyAcquisitionStrategy(_cfg())
    z = s.select(pool=_FakePool(np.zeros((3, 2))),
                 probability_backend=_FakeBackend(np.zeros((3, 4, 2)), system),
                 threshold_backend=None, threshold_state=_ts(), target_count=0)
    assert z.diagnostics["skipped_reason"] == "target_count_zero"
    e = s.select(pool=_FakePool(np.zeros((0, 2))),
                 probability_backend=_FakeBackend(np.zeros((0, 4, 2)), system),
                 threshold_backend=None, threshold_state=_ts(), target_count=5)
    assert e.diagnostics["skipped_reason"] == "pool_exhausted"


def test_passes_exclude_through():
    system = _FakeSystem()
    pool = _FakePool(np.array([[0.0, 0.0], [1.0, 0.0]]))
    EntropyAcquisitionStrategy(_cfg()).select(
        pool=pool, probability_backend=_FakeBackend(np.stack([_cloud(2), _cloud(1)]), system),
        threshold_backend=None, threshold_state=_ts(), target_count=1, exclude={7})
    assert pool.last_exclude == {7}

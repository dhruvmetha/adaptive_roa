import math

import numpy as np
import pytest
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.strategy.dispersion import DispersionAcquisitionStrategy
from adaptive_roa.adaptive_v2.types import ThresholdState
from adaptive_roa.systems.base import DynamicalSystem, ManifoldComponent


class _FakeSystem(DynamicalSystem):
    """Pendulum-like: SO2 angle + Real velocity in [-8, 8]."""

    def define_manifold_structure(self):
        return [
            ManifoldComponent("SO2", 1, "angle"),
            ManifoldComponent("Real", 1, "angular_velocity"),
        ]

    def define_state_bounds(self):
        return {"angle": (-math.pi, math.pi), "angular_velocity": (-8.0, 8.0)}

    def classify_attractor(self, state, radius=0.1):
        import torch

        # Success when |theta| < radius, failure when theta > 1.0, else invalid.
        theta = state[:, 0]
        labels = torch.zeros(state.shape[0], dtype=torch.int64)
        labels[theta.abs() < radius] = 1
        labels[theta > 1.0] = -1
        return labels


class _FakePool:
    """Serves a fixed candidate array and records what was excluded."""

    def __init__(self, states):
        self.states = np.asarray(states, dtype=np.float32)
        self.last_exclude = None
        self.last_n = None

    def sample_candidates_without_marking(self, n, exclude=None):
        self.last_exclude = exclude
        self.last_n = n
        take = min(n, len(self.states))
        return self.states[:take], list(range(100, 100 + take))


class _FakeBackend:
    """Returns a preset cloud per candidate; records the requested K."""

    def __init__(self, clouds, system):
        self.clouds = np.asarray(clouds, dtype=np.float32)
        self.system = system
        self.device = "cpu"
        self.attractor_radius = 0.2
        self.last_num_samples = None

    def sample_endpoints(self, start_states, num_samples, verbose=True):
        self.last_num_samples = num_samples
        return self.clouds[: len(start_states)]


def _cfg(**overrides):
    base = {
        "d2_ratio": 0.5,
        "n_dispersion_candidates": 50,
        "num_mc_samples_dispersion": 4,
        "selection_rule": "greedy",
        "diversity_pool_multiplier": 5,
        "temperature": 0.1,
        "seed": None,
        "chunk_size": 2048,
        "log_score_correlation": False,
        "verbose": False,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _threshold_state():
    return ThresholdState(lambda_star=0.5, delta_star=0.1)


def _clouds_with_spreads(spreads):
    """Build [M, 2, 2] clouds where candidate i has the given velocity spread."""
    return np.array([[[0.0, 0.0], [0.0, s]] for s in spreads], dtype=np.float32)


def test_stores_cfg():
    s = DispersionAcquisitionStrategy(_cfg())
    assert s.mode == "dispersion"
    assert s.d2_ratio == 0.5
    assert s.num_mc_samples_dispersion == 4
    assert s.selection_rule == "greedy"


def test_rejects_unknown_selection_rule():
    with pytest.raises(ValueError, match="selection_rule"):
        DispersionAcquisitionStrategy(_cfg(selection_rule="nonsense"))


def test_selects_highest_dispersion_candidates():
    system = _FakeSystem()
    states = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]])
    pool = _FakePool(states)
    backend = _FakeBackend(_clouds_with_spreads([0.1, 8.0, 0.2, 4.0]), system)

    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=pool,
        probability_backend=backend,
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=2,
    )

    # Candidates 1 (spread 8.0) and 3 (spread 4.0) -> pool indices 101 and 103.
    assert result.d2_indices == [101, 103]
    assert result.d1_indices == []
    assert result.n_candidates_evaluated == 4
    assert backend.last_num_samples == 4


def test_passes_exclude_through_to_pool():
    system = _FakeSystem()
    pool = _FakePool(np.array([[0.0, 0.0], [1.0, 0.0]]))
    backend = _FakeBackend(_clouds_with_spreads([1.0, 2.0]), system)

    DispersionAcquisitionStrategy(_cfg()).select(
        pool=pool,
        probability_backend=backend,
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=1,
        exclude={7, 9},
    )

    assert pool.last_exclude == {7, 9}
    assert pool.last_n == 50


def test_diagnostics_populated():
    system = _FakeSystem()
    pool = _FakePool(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    backend = _FakeBackend(_clouds_with_spreads([1.0, 8.0, 4.0]), system)

    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=pool,
        probability_backend=backend,
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=2,
    )

    d = result.diagnostics
    for key in (
        "dispersion_score_threshold",
        "dispersion_score_min",
        "dispersion_score_max",
        "dispersion_score_mean",
        "dispersion_score_median",
        "n_dispersion_candidates_evaluated",
        "n_nonfinite_excluded",
        "selection_rule",
    ):
        assert key in d, f"missing diagnostic: {key}"
    assert d["selection_rule"] == "greedy"
    assert d["n_nonfinite_excluded"] == 0
    # threshold is the lowest score among the two selected
    assert d["dispersion_score_threshold"] == pytest.approx(4.0 / 8.0, abs=1e-6)


def test_target_count_zero_skips():
    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=_FakePool(np.zeros((3, 2))),
        probability_backend=_FakeBackend(_clouds_with_spreads([1.0, 2.0, 3.0]), _FakeSystem()),
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=0,
    )
    assert result.d2_indices == []
    assert result.diagnostics["skipped_reason"] == "target_count_zero"


def test_empty_pool_skips():
    system = _FakeSystem()
    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=_FakePool(np.zeros((0, 2))),
        probability_backend=_FakeBackend(np.zeros((0, 2, 2)), system),
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=5,
    )
    assert result.d2_indices == []
    assert result.diagnostics["skipped_reason"] == "pool_exhausted"


def test_selects_all_when_pool_smaller_than_target():
    system = _FakeSystem()
    pool = _FakePool(np.array([[0.0, 0.0], [1.0, 0.0]]))
    backend = _FakeBackend(_clouds_with_spreads([1.0, 2.0]), system)

    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=pool,
        probability_backend=backend,
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=10,
    )

    assert sorted(result.d2_indices) == [100, 101]


def test_non_finite_endpoints_excluded_and_counted():
    system = _FakeSystem()
    pool = _FakePool(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    clouds = _clouds_with_spreads([1.0, 2.0, 3.0])
    clouds[1, 0, 1] = np.nan
    backend = _FakeBackend(clouds, system)

    result = DispersionAcquisitionStrategy(_cfg()).select(
        pool=pool,
        probability_backend=backend,
        threshold_backend=None,
        threshold_state=_threshold_state(),
        target_count=3,
    )

    assert 101 not in result.d2_indices
    assert result.diagnostics["n_nonfinite_excluded"] == 1


def test_rejects_backend_without_sample_endpoints():
    class _NoCloudBackend:
        system = _FakeSystem()
        device = "cpu"

    with pytest.raises(RuntimeError, match="sample_endpoints"):
        DispersionAcquisitionStrategy(_cfg()).select(
            pool=_FakePool(np.zeros((2, 2))),
            probability_backend=_NoCloudBackend(),
            threshold_backend=None,
            threshold_state=_threshold_state(),
            target_count=1,
        )


def test_proportional_rule_runs_and_is_seeded():
    system = _FakeSystem()
    states = np.random.default_rng(0).normal(size=(20, 2))
    pool = _FakePool(states)
    spreads = np.linspace(0.1, 8.0, 20)
    backend = _FakeBackend(_clouds_with_spreads(spreads), system)
    cfg = _cfg(selection_rule="proportional", seed=11)

    a = DispersionAcquisitionStrategy(cfg).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=5,
    )
    b = DispersionAcquisitionStrategy(cfg).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=5,
    )
    assert a.d2_indices == b.d2_indices
    assert len(set(a.d2_indices)) == 5


def test_greedy_diverse_rule_runs():
    system = _FakeSystem()
    states = np.random.default_rng(1).normal(size=(30, 2))
    pool = _FakePool(states)
    spreads = np.linspace(0.1, 8.0, 30)
    backend = _FakeBackend(_clouds_with_spreads(spreads), system)

    result = DispersionAcquisitionStrategy(_cfg(selection_rule="greedy_diverse")).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=6,
    )

    assert len(set(result.d2_indices)) == 6
    assert result.diagnostics["selection_rule"] == "greedy_diverse"


def test_shipped_config_instantiates():
    from hydra.utils import get_class

    cfg = OmegaConf.load("configs/adaptive_v2/acquisition/dispersion.yaml")
    assert cfg.sampling_mode == "dispersion"

    node = cfg.acquisition
    strategy = get_class(node._target_)(node)   # mirrors engine.py:28 _instantiate

    assert strategy.mode == "dispersion"
    assert strategy.num_mc_samples_dispersion == 20
    assert strategy.selection_rule == "greedy"
    assert strategy.log_score_correlation is True
    assert strategy.seed is None


def test_correlation_diagnostic_absent_when_disabled():
    system = _FakeSystem()
    pool = _FakePool(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    backend = _FakeBackend(_clouds_with_spreads([1.0, 8.0, 4.0]), system)

    result = DispersionAcquisitionStrategy(_cfg(log_score_correlation=False)).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=2,
    )

    assert result.diagnostics["dispersion_label_uncertainty_spearman"] is None


def test_correlation_diagnostic_computed_when_enabled():
    system = _FakeSystem()
    rng = np.random.default_rng(4)
    n = 25
    pool = _FakePool(rng.normal(size=(n, 2)))
    # Clouds whose theta values straddle the success/failure boundary by varying
    # amounts, so both dispersion and p_success vary across candidates.
    clouds = np.stack(
        [
            np.stack(
                [
                    np.array([0.0, 0.0]),
                    np.array([float(i) / n * 2.0, 0.0]),
                ]
            )
            for i in range(n)
        ]
    ).astype(np.float32)
    backend = _FakeBackend(clouds, system)

    result = DispersionAcquisitionStrategy(_cfg(log_score_correlation=True)).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=5,
    )

    rho = result.diagnostics["dispersion_label_uncertainty_spearman"]
    assert rho is not None
    assert -1.0 <= rho <= 1.0


def test_correlation_does_not_change_selection():
    system = _FakeSystem()
    states = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    clouds = _clouds_with_spreads([0.1, 8.0, 0.2, 4.0])

    off = DispersionAcquisitionStrategy(_cfg(log_score_correlation=False)).select(
        pool=_FakePool(states), probability_backend=_FakeBackend(clouds, system),
        threshold_backend=None, threshold_state=_threshold_state(), target_count=2,
    )
    on = DispersionAcquisitionStrategy(_cfg(log_score_correlation=True)).select(
        pool=_FakePool(states), probability_backend=_FakeBackend(clouds, system),
        threshold_backend=None, threshold_state=_threshold_state(), target_count=2,
    )

    assert off.d2_indices == on.d2_indices


def test_correlation_is_none_when_p_success_is_constant():
    system = _FakeSystem()
    n = 10
    pool = _FakePool(np.zeros((n, 2)))
    # Every endpoint has theta = 5.0 -> all classified failure -> p_success
    # constant -> Spearman undefined.
    clouds = np.stack(
        [np.array([[5.0, 0.0], [5.0, float(i)]]) for i in range(n)]
    ).astype(np.float32)
    backend = _FakeBackend(clouds, system)

    result = DispersionAcquisitionStrategy(_cfg(log_score_correlation=True)).select(
        pool=pool, probability_backend=backend, threshold_backend=None,
        threshold_state=_threshold_state(), target_count=3,
    )

    assert result.diagnostics["dispersion_label_uncertainty_spearman"] is None

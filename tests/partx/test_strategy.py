import numpy as np
import pytest
from omegaconf import OmegaConf
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.gp_classifier import GPClassifier
from adaptive_roa.partx.model_handle import GPModelHandle
from adaptive_roa.partx.backend import GPProbabilityBackend
from adaptive_roa.partx.strategy import PartXAcquisitionStrategy


class _FakePool:
    def __init__(self, states):
        self._states = states
        self.system = PendulumSystem()
    def sample_candidates_without_marking(self, n, exclude=None):
        idx = [i for i in range(len(self._states)) if not exclude or i not in exclude][:n]
        return self._states[idx], idx


def test_strategy_selects_target_count():
    system = PendulumSystem()
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-3, 3, 400), rng.uniform(-8, 8, 400)])
    y = (X[:, 0] < 0).astype(int)
    gp = GPClassifier(system, n_inducing=64, n_iters=150).fit(X, y)
    backend = GPProbabilityBackend(OmegaConf.create({"attractor_radius": 0.2}), system, "cpu")
    backend.bind_model(GPModelHandle(gp, system))

    cfg = OmegaConf.create({
        "mode": "partx", "d2_ratio": 0.5, "beta": 1.96, "allocation": "global_top",
        "n_candidates": 400, "tree": {"branching_factor": 2, "delta": 0.1,
        "alpha": 0.05, "m_class": 64}, "bounds": {"R": 50, "M": 32}})
    strat = PartXAcquisitionStrategy(cfg)
    pool = _FakePool(X)
    res = strat.select(pool, backend, None, None, target_count=15)
    assert len(res.d2_indices) == 15
    assert "roa_volume" in res.diagnostics


class _EndpointBackend:
    """Stands in for EndpointMCProbabilityBackend: models the FINAL STATE, so it
    has no robustness latent to partition on."""
    system = None

    def estimate(self, X):
        raise NotImplementedError


def test_partx_rejects_a_final_state_predictor():
    # Part-X partitions on the sign of a robustness function. A predictor that
    # models the endpoint has no such latent; pairing them must fail loudly
    # rather than at a bare AttributeError deep in select().
    cfg = OmegaConf.create({"d2_ratio": 1.0, "beta": 1.96,
                            "allocation": "per_region_volume", "n_candidates": 10})
    strat = PartXAcquisitionStrategy(cfg)
    with pytest.raises(TypeError, match="robustness score / label probability"):
        strat.select(pool=None, probability_backend=_EndpointBackend(),
                     threshold_backend=None, threshold_state=None, target_count=5)


def test_partx_rejects_d2_ratio_zero():
    # d2_ratio=0 makes engine.py skip d2 acquisition entirely, so select() is
    # never called and the arm silently becomes a uniform baseline still named
    # Part-X (gaussian_torque/cp_med_partx did exactly this for 12 epochs).
    cfg = OmegaConf.create({"d2_ratio": 0, "beta": 1.96,
                            "allocation": "per_region_volume", "n_candidates": 10})
    with pytest.raises(ValueError, match="never runs"):
        PartXAcquisitionStrategy(cfg)

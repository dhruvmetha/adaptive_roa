import numpy as np
import torch

from adaptive_roa.probabilistic_classifier.flow_matching import (
    FMProbabilisticClassifier,
)
from adaptive_roa.probabilistic_classifier.registry import (
    get_probabilistic_classifier_class,
)
from adaptive_roa.adaptive_v2.eval.mc_cache import MCCache, save_mc_cache


def test_registered_under_generative():
    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier


def test_native_probs_includes_invalid():
    assert FMProbabilisticClassifier.native_probs == ("p_success", "p_failure", "p_invalid")


def _make_pendulum_system():
    """Return a real PendulumSystem, skipping if dataset is unavailable."""
    import pytest
    try:
        from adaptive_roa.systems.pendulum import PendulumSystem
        return PendulumSystem()
    except FileNotFoundError:
        pytest.skip("Pendulum dataset not available")


def test_predict_cached_reads_mc_cache(tmp_path):
    """predict_cached reclassifies endpoints at the eval radius and returns correct probs."""
    system = _make_pendulum_system()
    R = 0.2  # attractor_radius used for the wrapper

    # N=2, K=4 endpoints: use known positions so we can independently verify labels.
    # success_ep: near [0.0, 0.0] → distance 0.0 → label 1
    # failure_ep: near [2.1, 0.0] → distance 0.0 → label -1
    # invalid_ep: [1.0, 1.0] → not within R=0.2 of any attractor → label 0
    success_ep = [0.0, 0.0]
    failure_ep = [2.1, 0.0]
    invalid_ep = [1.0, 1.0]

    # Row 0: success, success, failure, invalid  -> p_s=0.5, p_f=0.25, p_inv=0.25
    # Row 1: failure, failure, success, success  -> p_s=0.5, p_f=0.5, p_inv=0.0
    state_dim = 2
    N, K = 2, 4
    mc_endpoints = np.array([
        [success_ep, success_ep, failure_ep, invalid_ep],
        [failure_ep, failure_ep, success_ep, success_ep],
    ], dtype=np.float32)  # shape [2, 4, 2]

    start = np.zeros((N, state_dim), dtype=np.float32)

    # mc_labels here are intentionally wrong (all zeros) — reclassify must overwrite them.
    junk_labels = np.zeros((N, K), dtype=np.int8)

    cache = MCCache(
        start_states=start,
        mc_endpoints=mc_endpoints,
        mc_labels=junk_labels,
        attractor_radius=0.99,   # deliberately different from eval radius
        num_mc_samples=K,
    )
    cache_dir = tmp_path / "mc_cache"
    cache_dir.mkdir()
    save_mc_cache(cache, str(cache_dir / "epoch_000_test.npz"))

    pc = FMProbabilisticClassifier(
        flow_matcher=None, system=system, device="cpu",
        attractor_radius=R, num_mc_samples=K,
    )
    out = pc.predict_cached(str(tmp_path), 0, "test", start)
    assert out is not None

    # Independently compute expected probs by calling classify_attractor.
    flat = torch.as_tensor(mc_endpoints.reshape(-1, state_dim), dtype=torch.float32)
    expected_labels = system.classify_attractor(flat, radius=R).numpy().reshape(N, K)
    exp_ps = (expected_labels == 1).sum(axis=1) / K
    exp_pf = (expected_labels == -1).sum(axis=1) / K
    exp_pinv = (expected_labels == 0).sum(axis=1) / K

    assert np.allclose(out.p_success, exp_ps), f"p_success mismatch: {out.p_success} vs {exp_ps}"
    assert np.allclose(out.p_failure, exp_pf), f"p_failure mismatch: {out.p_failure} vs {exp_pf}"
    assert np.allclose(out.p_invalid, exp_pinv), f"p_invalid mismatch: {out.p_invalid} vs {exp_pinv}"


def test_predict_cached_missing_returns_none(tmp_path):
    pc = FMProbabilisticClassifier(
        flow_matcher=None, system=None, device="cpu",
        attractor_radius=0.2, num_mc_samples=4,
    )
    assert pc.predict_cached(str(tmp_path), 7, "val", np.zeros((1, 2))) is None


def test_predict_cached_length_mismatch_returns_none(tmp_path):
    """predict_cached returns None when cache row count != len(states)."""
    N = 3
    state_dim = 2
    K = 4
    start = np.zeros((N, state_dim), dtype=np.float32)
    cache = MCCache(
        start_states=start,
        mc_endpoints=np.zeros((N, K, state_dim), dtype=np.float32),
        mc_labels=np.zeros((N, K), dtype=np.int8),
        attractor_radius=0.2,
        num_mc_samples=K,
    )
    cache_dir = tmp_path / "mc_cache"
    cache_dir.mkdir()
    save_mc_cache(cache, str(cache_dir / "epoch_001_val.npz"))

    pc = FMProbabilisticClassifier(
        flow_matcher=None, system=None, device="cpu",
        attractor_radius=0.2, num_mc_samples=K,
    )
    # Pass N+1 states — should not match cache rows
    result = pc.predict_cached(str(tmp_path), 1, "val", np.zeros((N + 1, state_dim)))
    assert result is None


def test_predict_cached_state_value_mismatch_returns_none(tmp_path):
    """Same row count but different state VALUES must fall through (not misalign)."""
    N, state_dim, K = 3, 2, 4
    cache = MCCache(
        start_states=np.zeros((N, state_dim), dtype=np.float32),
        mc_endpoints=np.zeros((N, K, state_dim), dtype=np.float32),
        mc_labels=np.zeros((N, K), dtype=np.int8),
        attractor_radius=0.2,
        num_mc_samples=K,
    )
    cache_dir = tmp_path / "mc_cache"
    cache_dir.mkdir()
    save_mc_cache(cache, str(cache_dir / "epoch_002_val.npz"))

    pc = FMProbabilisticClassifier(
        flow_matcher=None, system=None, device="cpu",
        attractor_radius=0.2, num_mc_samples=K,
    )
    # Query states have the same shape as the cache but different values -> None
    # (system=None, so reaching reclassify would raise; returning None proves the guard).
    result = pc.predict_cached(str(tmp_path), 2, "val", np.ones((N, state_dim), dtype=np.float32))
    assert result is None

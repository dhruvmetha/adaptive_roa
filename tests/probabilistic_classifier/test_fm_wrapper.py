import numpy as np

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


def test_predict_cached_reads_mc_cache(tmp_path):
    # N=2, K=4 MC labels: row0 -> 3 success / 1 invalid; row1 -> 2 fail / 2 success
    start = np.zeros((2, 2), dtype=np.float32)
    endpoints = np.zeros((2, 4, 2), dtype=np.float32)
    labels = np.array([[1, 1, 1, 0], [-1, -1, 1, 1]], dtype=np.int64)
    cache = MCCache(
        start_states=start,
        mc_endpoints=endpoints,
        mc_labels=labels,
        attractor_radius=0.2,
        num_mc_samples=4,
    )
    cache_dir = tmp_path / "mc_cache"
    cache_dir.mkdir()
    save_mc_cache(cache, str(cache_dir / "epoch_000_test.npz"))

    pc = FMProbabilisticClassifier(
        flow_matcher=None, system=None, device="cpu",
        attractor_radius=0.2, num_mc_samples=4,
    )
    out = pc.predict_cached(str(tmp_path), 0, "test", start)
    assert out is not None
    assert np.allclose(out.p_success, [0.75, 0.5])
    assert np.allclose(out.p_failure, [0.0, 0.5])
    assert np.allclose(out.p_invalid, [0.25, 0.0])


def test_predict_cached_missing_returns_none(tmp_path):
    pc = FMProbabilisticClassifier(
        flow_matcher=None, system=None, device="cpu",
        attractor_radius=0.2, num_mc_samples=4,
    )
    assert pc.predict_cached(str(tmp_path), 7, "val", np.zeros((1, 2))) is None

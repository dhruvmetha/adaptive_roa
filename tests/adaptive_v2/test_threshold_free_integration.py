"""The threshold_free block must reach the metrics dict on both eval paths."""
import numpy as np

from adaptive_roa.adaptive_v2.eval.full_roa import (
    evaluate_full_roa_classifier,
    evaluate_full_roa_fast,
)
from adaptive_roa.adaptive_v2.eval.mc_cache import MCCache


class _FakeClassifier:
    """logit = 10 * state[:, 0]  -> p(success) high when s0 > 0."""

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x):
        return (10.0 * x[:, 0]).unsqueeze(-1)


def _write_eval_file(tmp_path, rng, n=200):
    rows = []
    for _ in range(n):
        s0, s1 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        label = 1 if s0 > 0 else 0  # external {0,1}; loader maps to {-1,1}
        rows.append([s0, s1, 0.0, 0.0, label])
    eval_file = tmp_path / "eval.txt"
    np.savetxt(str(eval_file), np.array(rows), delimiter=",", fmt="%.6f")
    return eval_file, np.array(rows)


def test_classifier_path_reports_threshold_free_block(tmp_path):
    rng = np.random.default_rng(0)
    eval_file, _ = _write_eval_file(tmp_path, rng)

    metrics = evaluate_full_roa_classifier(
        _FakeClassifier(), system=None, eval_states_file=str(eval_file),
        lambda_star=0.5, delta=0.1, decision_rule="one_sided",
        device="cpu", output_dir=None, verbose=False,
    )

    tf = metrics["threshold_free"]
    assert tf["n_scored"] == 200
    assert tf["auc"] > 0.99          # separable problem
    assert tf["auprc"] > 0.99
    assert 0.0 <= tf["brier"] <= 1.0
    assert np.isfinite(tf["log_score"])
    # a single forward pass is not a K-sample estimate
    assert tf["log_score_smoothing"] == "clip"


def test_generative_path_reports_threshold_free_block_with_count_smoothing(tmp_path):
    """The MC path must smooth by K, not clip, and must survive saturation."""
    rng = np.random.default_rng(1)
    eval_file, rows = _write_eval_file(tmp_path, rng)
    n, K = len(rows), 100

    # Build a cache whose labels agree with the s0>0 rule, fully saturated
    # (every sample unanimous) so p_success is exactly 0.0 or 1.0.
    true_success = rows[:, 4] == 1
    mc_labels = np.where(true_success[:, None], 1, -1).astype(np.int8)
    mc_labels = np.repeat(mc_labels, K, axis=1)
    # flip one point's samples entirely -> confidently wrong, log score would be -inf
    mc_labels[0, :] = -1 if true_success[0] else 1

    cache = MCCache(
        mc_endpoints=np.zeros((n, K, 2), dtype=np.float32),
        mc_labels=mc_labels,
        start_states=rows[:, :2].astype(np.float32),
        attractor_radius=0.2,
        num_mc_samples=K,
    )

    metrics = evaluate_full_roa_fast(
        flow_matcher=None, system=None, eval_states_file=str(eval_file),
        num_mc_samples=K, lambda_star=0.5, delta=0.1,
        decision_rule="one_sided", device="cpu", output_dir=None,
        verbose=False, mc_cache=cache,
    )

    tf = metrics["threshold_free"]
    assert tf["n_scored"] == n
    assert tf["log_score_smoothing"] == "kt_count"
    assert np.isfinite(tf["log_score"])
    assert tf["n_saturated"] == n     # every point unanimous
    assert tf["auc"] is not None

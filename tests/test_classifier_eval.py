"""Task 8 test: evaluate_full_roa_classifier on a separable problem."""
import numpy as np
import torch

from adaptive_roa.adaptive_v2.eval.full_roa import evaluate_full_roa_classifier


class _FakeClassifier:
    """logit = 10 * state[:, 0]  -> p(success) high when s0 > 0."""

    def eval(self):
        return self

    def __call__(self, x):
        return (10.0 * x[:, 0]).unsqueeze(-1)


def test_classifier_eval_separable(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(200):
        s0, s1 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        label = 1 if s0 > 0 else 0  # external {0,1}; loader maps to {-1,1}
        rows.append([s0, s1, 0.0, 0.0, label])  # [s0,s1, e0,e1, label]
    eval_file = tmp_path / "eval.txt"
    np.savetxt(str(eval_file), np.array(rows), delimiter=",", fmt="%.6f")

    metrics = evaluate_full_roa_classifier(
        _FakeClassifier(), system=None, eval_states_file=str(eval_file),
        lambda_star=0.5, delta=0.1, decision_rule="one_sided",
        device="cpu", output_dir=None, verbose=False,
    )

    assert metrics["n_total"] == 200
    assert metrics["predictor"] == "classifier"
    assert metrics["endpoint_errors"] is None
    assert metrics["qhat_prediction_sets"] is None
    # classifier reproduces the separable rule -> high F1 on the lambda+/-delta band
    assert metrics["lambda_delta"]["f1"] > 0.9
    assert metrics["lambda_delta"]["accuracy"] > 0.9

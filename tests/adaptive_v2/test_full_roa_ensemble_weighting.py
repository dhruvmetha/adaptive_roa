"""K eval samples must be split EVENLY across members.

Sampling members instead gives a fixed skewed weight vector under a seeded
generator, which biases the marginal systematically rather than averaging out
(see EnsemblePosterior.predictive_logit_samples in
adaptive_roa/predictors/posteriors.py).
"""
import numpy as np
import torch

from adaptive_roa.adaptive_v2.eval.full_roa import evaluate_full_roa_fast
from adaptive_roa.adaptive_v2.trainers.ensemble_flow_matching_trainer import (
    EnsembleFlowMatcherHandle,
)


class _CountingMember:
    def __init__(self, idx, counter):
        self.idx, self.counter = idx, counter
    def predict_endpoint(self, x, **kw):
        self.counter[self.idx] += 1
        return torch.zeros((x.shape[0], 2))
    def eval(self):
        return self


def test_each_member_is_used_exactly_k_over_m_times():
    counter = {i: 0 for i in range(5)}
    h = EnsembleFlowMatcherHandle([_CountingMember(i, counter) for i in range(5)])
    for _ in range(100):                       # K = 100
        h.predict_endpoint(torch.zeros((3, 2)))
    assert set(counter.values()) == {20}       # exactly K/M each, no skew


def test_uneven_k_distributes_within_one_call_of_balanced():
    counter = {i: 0 for i in range(3)}
    h = EnsembleFlowMatcherHandle([_CountingMember(i, counter) for i in range(3)])
    for _ in range(10):
        h.predict_endpoint(torch.zeros((1, 2)))
    assert max(counter.values()) - min(counter.values()) <= 1


# ---------------------------------------------------------------------------
# The two tests above only exercise EnsembleFlowMatcherHandle's own
# round-robin cursor (Task 6). They do not prove that evaluate_full_roa_fast's
# MC loop actually pins the member explicitly by `sample_idx % n_members`
# rather than delegating to that cursor. This test drives the REAL MC
# sampling loop in full_roa.py (no mc_cache, so the GPU-loop branch runs)
# against a fake handle whose `predict_endpoint` is wired to raise -- if
# full_roa.py ever fell back to the handle's stateful, cursor-based
# `predict_endpoint` instead of calling `predict_endpoint_member` with an
# explicit pinned index, this fails loudly instead of silently passing with
# correct-looking counts.
# ---------------------------------------------------------------------------
class _TrapEnsembleHandle:
    """n_members + predict_endpoint_member(m, x); predict_endpoint is a trap."""

    def __init__(self, n_members, counter):
        self.n_members = n_members
        self.counter = counter

    def predict_endpoint(self, x, **kw):
        raise AssertionError(
            "full_roa.py must call predict_endpoint_member with an explicit "
            "pinned index for ensembles, not the handle's round-robin "
            "predict_endpoint cursor"
        )

    def predict_endpoint_member(self, m, x, **kw):
        self.counter[m] += 1
        return torch.zeros((x.shape[0], 2))

    def eval(self):
        return self


class _AlwaysSuccessSystem:
    def classify_attractor(self, pred, radius):
        return torch.ones(pred.shape[0], dtype=torch.long)


def _write_eval_file(tmp_path, n=7):
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        s0, s1 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        rows.append([s0, s1, 0.0, 0.0, 1])
    eval_file = tmp_path / "eval.txt"
    np.savetxt(str(eval_file), np.array(rows), delimiter=",", fmt="%.6f")
    return eval_file


def test_evaluate_full_roa_fast_splits_k_evenly_across_members(tmp_path):
    n_members, k = 5, 100
    counter = {i: 0 for i in range(n_members)}
    handle = _TrapEnsembleHandle(n_members, counter)
    eval_file = _write_eval_file(tmp_path)

    evaluate_full_roa_fast(
        handle, system=_AlwaysSuccessSystem(), eval_states_file=str(eval_file),
        num_mc_samples=k, lambda_star=0.5, delta=0.1,
        decision_rule="one_sided", device="cpu", output_dir=None, verbose=False,
    )

    assert set(counter.values()) == {k // n_members}
    assert sum(counter.values()) == k

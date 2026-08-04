import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.ensemble_prob import (
    EnsembleEndpointMCProbabilityBackend,
)
from adaptive_roa.adaptive_v2.trainers.ensemble_flow_matching_trainer import (
    EnsembleFlowMatcherHandle,
    EnsembleFlowMatchingTrainer,
)


class _ConstMember:
    def __init__(self, value):
        self.value = value
    def predict_endpoint(self, x, **kw):
        return torch.full((x.shape[0], 2), float(self.value))
    def eval(self):
        return self
    def to(self, device):
        return self


def test_handle_reports_member_count():
    h = EnsembleFlowMatcherHandle([_ConstMember(i) for i in range(5)])
    assert h.n_members == 5


def test_predict_endpoint_member_selects_that_member():
    h = EnsembleFlowMatcherHandle([_ConstMember(i) for i in range(3)])
    x = torch.zeros((4, 2))
    for m in range(3):
        assert torch.allclose(h.predict_endpoint_member(m, x), torch.full((4, 2), float(m)))


def test_predict_endpoint_round_robins_exactly_not_randomly():
    # Exact enumeration, matching EnsemblePosterior.predictive_logit_samples:
    # sampling members with a seeded generator gives a FIXED skewed weight vector.
    h = EnsembleFlowMatcherHandle([_ConstMember(i) for i in range(5)])
    x = torch.zeros((1, 2))
    seen = [float(h.predict_endpoint(x)[0, 0]) for _ in range(10)]
    assert seen == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


def test_single_member_handle_rejected():
    with pytest.raises(ValueError, match="at least 2 members"):
        EnsembleFlowMatcherHandle([_ConstMember(0)])


def test_out_of_range_member_raises():
    h = EnsembleFlowMatcherHandle([_ConstMember(i) for i in range(2)])
    with pytest.raises(IndexError):
        h.predict_endpoint_member(5, torch.zeros((1, 2)))


# ---------------------------------------------------------------------------
# Requirement A: predict_endpoint_member must accept RAW (un-embedded) states
# and hand them to the member's own predict_endpoint unchanged. Task 2 had
# this exact bug on the classifier path (reached past the module into the raw
# network, skipping system.embed_state_for_model(system.normalize_state(x)));
# EnsembleEndpointMCProbabilityBackend.estimate_members passes RAW x straight
# from the caller's start_states into predict_endpoint_member (ensemble_prob.py
# :88-94), matching how eval/full_roa.py:740 calls the single-model
# flow_matcher.predict_endpoint with raw batch_inputs and lets the module's own
# predict_endpoint (flow_matching/base/flow_matcher.py:746-814) do
# normalization + embedding internally via _prepare_model_inputs. If the
# handle embedded x itself before forwarding, states would be double-embedded
# and crash (or silently corrupt) inside the member's own predict_endpoint.
# ---------------------------------------------------------------------------
class _RecordingRawInputMember:
    """Records exactly what tensor it was handed, to prove pass-through."""
    def __init__(self):
        self.received = None
    def predict_endpoint(self, x, **kw):
        self.received = x.clone()
        return torch.zeros((x.shape[0], 2))
    def eval(self):
        return self
    def to(self, device):
        return self


def test_predict_endpoint_member_forwards_raw_x_unchanged():
    mem0, mem1 = _RecordingRawInputMember(), _RecordingRawInputMember()
    h = EnsembleFlowMatcherHandle([mem0, mem1])
    # Deliberately out-of-[0,1]/out-of-embedded-range raw values: a pendulum
    # angle of 3.0 rad and velocity of -7.5 -- if the handle normalized or
    # embedded before forwarding, this would not survive byte-for-byte.
    raw_x = torch.tensor([[3.0, -7.5], [0.1, 0.2]])
    h.predict_endpoint_member(0, raw_x)
    assert torch.equal(mem0.received, raw_x)
    assert mem0.received.shape == (2, 2)  # NOT widened by an embedding step


# ---------------------------------------------------------------------------
# Requirement B: member m must be held FIXED across all K MC draws inside one
# per-member probability estimate. If the member index varied per draw, every
# p_m would collapse toward the ensemble marginal and epistemic_var/BALD would
# read ~0 -- a null result indistinguishable from "the idea does not work".
# This drives the REAL EnsembleFlowMatcherHandle through the REAL
# EnsembleEndpointMCProbabilityBackend.estimate_members, not a mock of either.
# ---------------------------------------------------------------------------
class _CountingMember:
    """Appends its own index to a shared call log on every predict_endpoint
    call, so an interleaved-member bug is visible directly in call ORDER, not
    just in the total count."""
    def __init__(self, idx, call_log):
        self.idx = idx
        self.call_log = call_log
    def predict_endpoint(self, x, **kw):
        self.call_log.append(self.idx)
        return torch.zeros((x.shape[0], 2))
    def eval(self):
        return self
    def to(self, device):
        return self


class _AlwaysSuccessSystem:
    """Minimal system stub: every predicted endpoint classifies as success.

    estimate_members only needs system.classify_attractor(pred, radius); the
    actual labels are irrelevant to the member-fixing property under test.
    """
    def classify_attractor(self, pred, radius):
        return torch.ones(pred.shape[0], dtype=torch.long)


def test_estimate_members_holds_member_fixed_across_all_k_draws():
    call_log: list[int] = []
    n_members, k = 3, 20
    members = [_CountingMember(i, call_log) for i in range(n_members)]
    handle = EnsembleFlowMatcherHandle(members)

    cfg = OmegaConf.create({"attractor_radius": 0.1, "num_mc_samples": k})
    backend = EnsembleEndpointMCProbabilityBackend(
        cfg, system=_AlwaysSuccessSystem(), device="cpu")
    backend.bind_model(handle)

    backend.estimate_members(np.zeros((4, 2), dtype=np.float32))

    assert len(call_log) == n_members * k
    for m in range(n_members):
        assert call_log.count(m) == k, f"member {m} called {call_log.count(m)} times, expected {k}"
    # Not interleaved: exactly K consecutive calls to member 0, then K to
    # member 1, then K to member 2 -- never member 0, 1, 0, 1, ...
    expected_order = [m for m in range(n_members) for _ in range(k)]
    assert call_log == expected_order


def test_estimate_members_output_shape_matches_members_by_states():
    n_members, k, n_states = 4, 5, 7
    members = [_ConstMember(i) for i in range(n_members)]
    handle = EnsembleFlowMatcherHandle(members)
    cfg = OmegaConf.create({"attractor_radius": 0.1, "num_mc_samples": k})
    backend = EnsembleEndpointMCProbabilityBackend(
        cfg, system=_AlwaysSuccessSystem(), device="cpu")
    backend.bind_model(handle)
    out = backend.estimate_members(np.zeros((n_states, 2), dtype=np.float32))
    assert out.shape == (n_members, n_states)
    # Every draw classifies as success (label 1), so every member's estimate is 1.0.
    np.testing.assert_allclose(out, 1.0)


# ---------------------------------------------------------------------------
# handle.to(device): required by AdaptiveEngine.run, which calls
# model_handle.to(self.device) unconditionally right after model_handle.eval()
# (engine.py:137-138) for every predictor type, not just classifiers.
# ---------------------------------------------------------------------------
class _DeviceTrackingMember(_ConstMember):
    def __init__(self, value):
        super().__init__(value)
        self.device_seen = None
    def to(self, device):
        self.device_seen = device
        return self


def test_handle_to_delegates_to_every_member():
    members = [_DeviceTrackingMember(i) for i in range(3)]
    h = EnsembleFlowMatcherHandle(members)
    ret = h.to("cuda:2")
    assert ret is h
    assert all(m.device_seen == "cuda:2" for m in members)


# ---------------------------------------------------------------------------
# Manifold passthroughs: adaptive/endpoint_evaluation.py calls
# get_manifold_component_names()/compute_manifold_distance_per_component()
# unconditionally on the model handle (no hasattr guard), and full_roa.py
# separately gates its own geodesic-vs-Euclidean choice on
# hasattr(flow_matcher, "distance_manifold"). All members share the same
# system, so delegating to member 0 is exact.
# ---------------------------------------------------------------------------
class _ManifoldMember(_ConstMember):
    def __init__(self, value, tag):
        super().__init__(value)
        self.tag = tag
        self.distance_manifold = f"manifold-{tag}"
    def get_manifold_component_names(self):
        return [f"component-{self.tag}"]
    def compute_manifold_distance_per_component(self, predicted, true):
        return (self.tag, predicted, true)


def test_handle_delegates_manifold_methods_to_member_zero():
    h = EnsembleFlowMatcherHandle([_ManifoldMember(0, "m0"), _ManifoldMember(1, "m1")])
    assert hasattr(h, "distance_manifold")
    assert h.distance_manifold == "manifold-m0"
    assert h.get_manifold_component_names() == ["component-m0"]
    pred, true = torch.zeros((2, 2)), torch.ones((2, 2))
    tag, p, t = h.compute_manifold_distance_per_component(pred, true)
    assert tag == "m0"
    assert torch.equal(p, pred) and torch.equal(t, true)


# ---------------------------------------------------------------------------
# _load_member: FileNotFoundError guard against assembling an ensemble from a
# partially-trained member (real production code path, no GPU/checkpoint
# needed since the guard fires before any instantiation).
# ---------------------------------------------------------------------------
def test_load_member_raises_when_no_checkpoint_present(tmp_path):
    cfg = OmegaConf.create({
        "predictor": {"ensemble": {"n_members": 2}},
        "seed": 0,
    })
    trainer = EnsembleFlowMatchingTrainer(cfg, system=None, system_name="pendulum")
    with pytest.raises(FileNotFoundError, match="did not finish training"):
        trainer._load_member(str(tmp_path), 0)


def test_ensemble_trainer_reads_n_members_and_seed_from_cfg():
    cfg = OmegaConf.create({
        "predictor": {"ensemble": {"n_members": 5}},
        "seed": 123,
    })
    trainer = EnsembleFlowMatchingTrainer(cfg, system=None, system_name="pendulum")
    assert trainer.n_members == 5
    assert trainer.seed_base == 123

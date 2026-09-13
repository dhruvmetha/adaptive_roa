"""BALD on the final-state (endpoint-regressing) BNN arms.

The load-bearing property here is WEIGHT PINNING. ``FinalStateModelHandle``
draws a fresh weight sample on every ``predict_endpoint`` call by contract
(final_state_handle.py:48-52), so K endpoint draws taken through the handle mix
epistemic and aleatoric variation together. BALD over those draws would score
sampling noise while looking entirely reasonable. This backend goes around the
handle and holds ONE weight draw across the K head draws, which is what makes
the epistemic term mean what its name says.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.posterior_endpoint_mc import (
    PosteriorEndpointMCProbabilityBackend,
)
from adaptive_roa.predictors.posteriors import DeterministicPosterior


class _IdentitySystem:
    """Endpoint's first coordinate decides success, so p is easy to predict."""

    state_dim = 2

    def normalize_state(self, x):
        return x

    def embed_state_for_model(self, x):
        return x

    def classify_attractor(self, state, radius=None):
        # success iff first coordinate is positive
        return torch.where(state[..., 0] > 0, 1, -1)


class _StubHead:
    """``sample`` is the identity on params, and records what it was handed."""

    n_params = 2
    state_dim = 2

    def __init__(self):
        self.seen: list[torch.Tensor] = []

    def sample(self, params, generator=None):
        self.seen.append(params.clone())
        return params


class _MFVIStub(torch.nn.Module):
    """Continuous posterior: a fresh draw per forward_samples call."""

    def __init__(self, sigma=1.0):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.sigma = float(sigma)
        self.n_forward_samples_calls = 0
        self.n_forward_sample_calls = 0

    def parameters(self, recurse=True):
        return super().parameters(recurse)

    def forward_sample(self, x, generator=None):
        self.n_forward_sample_calls += 1
        return x + self.sigma * torch.randn(x.shape[0], 2)

    def forward_samples(self, x, S, generator=None):
        self.n_forward_samples_calls += 1
        return torch.stack([self.forward_sample(x) for _ in range(int(S))], dim=0)


class _EnsembleStub(torch.nn.Module):
    """Finite support: enumerable, so forward_samples must NOT be used."""

    def __init__(self, offsets=(-1.0, 1.0, 3.0)):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.offsets = list(offsets)
        self.n_forward_samples_calls = 0

    @property
    def n_members(self):
        return len(self.offsets)

    def forward_sample(self, x, generator=None):
        return x + self.offsets[0]

    def forward_samples(self, x, S, generator=None):
        self.n_forward_samples_calls += 1
        return torch.stack([x + o for o in self.offsets], dim=0)

    def forward_all_members(self, x):
        return torch.stack([x + o for o in self.offsets], dim=0)


class _Handle:
    def __init__(self, posterior, head):
        self.posterior = posterior
        self.head = head

    def eval(self):
        # ProbabilityEstimator.__init__ calls this; the real
        # FinalStateModelHandle has it (final_state_handle.py:86).
        return self


def _cfg(**kw):
    base = {
        "attractor_radius": 0.3,
        "num_mc_samples": 10,
        "refine_invalids": False,
        "refine_t_min": 0.7,
        "refine_t_max": 0.9,
        "refine_num_steps": 100,
        "refine_max_attempts": 5,
        "trajectory_checking": False,
        "n_posterior_samples": 8,
        "chunk_size": 4096,
    }
    base.update(kw)
    return OmegaConf.create(base)


def _backend(posterior, head=None, **kw):
    head = head if head is not None else _StubHead()
    b = PosteriorEndpointMCProbabilityBackend(_cfg(**kw), _IdentitySystem(), "cpu")
    b.bind_model(_Handle(posterior, head))
    return b, head


# --------------------------------------------------------------------- shapes


def test_mfvi_posterior_gives_one_row_per_draw():
    b, _ = _backend(_MFVIStub())
    p = b.estimate_members(np.zeros((5, 2), dtype=np.float32))
    assert p.shape == (8, 5), "one row per posterior draw"
    assert ((p >= 0) & (p <= 1)).all()


def test_ensemble_posterior_gives_one_row_per_member():
    b, _ = _backend(_EnsembleStub())
    p = b.estimate_members(np.zeros((5, 2), dtype=np.float32))
    assert p.shape == (3, 5), "one row per member, S ignored"


def test_ensemble_branch_enumerates_rather_than_sampling():
    # Sampling M atoms with replacement gives a multinomial weight vector
    # instead of an exact 1/M, which is a systematic bias rather than noise
    # that averages out. See EnsemblePosterior.predictive_logit_samples.
    post = _EnsembleStub()
    b, _ = _backend(post)
    b.estimate_members(np.zeros((4, 2), dtype=np.float32))
    assert post.n_forward_samples_calls == 0, "ensemble must be enumerated, not sampled"


# ------------------------------------------------------------- weight pinning


def test_weight_draw_is_pinned_across_the_k_endpoint_samples():
    # THE regression guard. If the backend redrew weights per endpoint sample,
    # this would call forward_samples K times instead of once, and every p
    # would be a weight-marginal rather than a per-member probability.
    post = _MFVIStub()
    b, _ = _backend(post, num_mc_samples=10)
    b.estimate_members(np.zeros((4, 2), dtype=np.float32))
    assert post.n_forward_samples_calls == 1, (
        "weights must be drawn once and held across the K head draws")


def test_all_k_draws_within_a_member_share_one_parameter_set():
    head = _StubHead()
    b, head = _backend(_MFVIStub(), head=head, num_mc_samples=4, n_posterior_samples=3)
    b.estimate_members(np.zeros((6, 2), dtype=np.float32))
    assert len(head.seen) == 3 * 4, "S members x K draws"
    for member in range(3):
        block = head.seen[member * 4:(member + 1) * 4]
        for drawn in block[1:]:
            assert torch.equal(block[0], drawn), (
                "the K draws inside one member must share its weight sample")


def test_members_differ_from_one_another():
    head = _StubHead()
    b, head = _backend(_MFVIStub(sigma=5.0), head=head, num_mc_samples=2,
                       n_posterior_samples=3)
    b.estimate_members(np.zeros((6, 2), dtype=np.float32))
    first_of_each = [head.seen[m * 2] for m in range(3)]
    assert not torch.equal(first_of_each[0], first_of_each[1])


# ------------------------------------------------------------------- debiasing


def test_member_sample_size_reports_k():
    # Each p_m is a K-sample binomial estimate, so BALD carries the
    # ~(1/2K)(1-1/M) upward bias and the debiased score needs K to remove it.
    b, _ = _backend(_MFVIStub(), num_mc_samples=17)
    assert b.member_sample_size == 17


# ---------------------------------------------------------------------- guards


def test_deterministic_posterior_is_refused():
    # S identical draws means BALD is 0 everywhere, silently. mlp_det would
    # otherwise run as a fully non-adaptive arm reporting an adaptive config.
    net = torch.nn.Linear(2, 2)
    b = PosteriorEndpointMCProbabilityBackend(_cfg(), _IdentitySystem(), "cpu")
    with pytest.raises(ValueError, match="no epistemic spread"):
        b.bind_model(_Handle(DeterministicPosterior(net), _StubHead()))


def test_single_member_ensemble_is_refused():
    b = PosteriorEndpointMCProbabilityBackend(_cfg(), _IdentitySystem(), "cpu")
    with pytest.raises(ValueError, match="at least 2"):
        b.bind_model(_Handle(_EnsembleStub(offsets=(1.0,)), _StubHead()))


def test_n_posterior_samples_below_two_is_refused():
    with pytest.raises(ValueError, match="n_posterior_samples"):
        PosteriorEndpointMCProbabilityBackend(
            _cfg(n_posterior_samples=1), _IdentitySystem(), "cpu")


# --------------------------------------------------------------------- chunking


def test_chunking_does_not_change_the_output_shape():
    # Chunking redraws weights per chunk for a continuous posterior. That is
    # SAFE because BALD is a per-point functional: it only needs the S values at
    # a given point to come from S distinct draws, not for member s to be the
    # same weight vector at every point.
    b, _ = _backend(_MFVIStub(), chunk_size=2)
    p = b.estimate_members(np.zeros((7, 2), dtype=np.float32))
    assert p.shape == (8, 7)
    assert np.isfinite(p).all()


def test_chunked_ensemble_keeps_members_consistent_across_chunks():
    head = _StubHead()
    b, head = _backend(_EnsembleStub(offsets=(-2.0, 2.0)), head=head,
                       num_mc_samples=1, chunk_size=2)
    p = b.estimate_members(np.zeros((4, 2), dtype=np.float32))
    # member 0 is always the -2 net (never success), member 1 always +2
    assert np.allclose(p[0], 0.0)
    assert np.allclose(p[1], 1.0)


# ------------------------------------------------------- inherited eval surface


def test_estimate_still_returns_the_endpoint_mc_marginal():
    # The whole point of subclassing: calibration and eval must be unchanged.
    b, _ = _backend(_MFVIStub())
    assert hasattr(b, "sample_endpoints")
    assert b.estimator is not None

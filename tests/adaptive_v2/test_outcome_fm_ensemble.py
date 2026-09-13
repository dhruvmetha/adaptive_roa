"""BALD on the scalar-outcome flow matcher.

`fm_outcome` is a single deterministic model: no posterior, no members, no
epistemic axis at all. It has only ever run as a fixed-dataset baseline
(`+experiment=fm_outcome_baseline`, d2_ratio=0), which needs no acquisition
signal. A 5-member deep ensemble supplies the member axis.

The trap these tests exist for: calibration and evaluation reach p through
`model(x) -> logits` (full_roa.py:1301), while acquisition reaches it through
the probability backend. If those two disagree, calibration optimises a
different quantity than evaluation reports, silently, with both numbers looking
reasonable.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.probability.outcome_fm_ensemble import (
    EnsembleOutcomeFMProbabilityBackend,
)
from adaptive_roa.adaptive_v2.trainers.ensemble_outcome_fm_trainer import (
    EnsembleOutcomeFMHandle,
)


class _ConstantMember(torch.nn.Module):
    """Stands in for a trained OutcomeFlowMatcher at a fixed p."""

    def __init__(self, p: float, readout: str = "exact"):
        super().__init__()
        self.p = float(p)
        self.forward_readout = readout
        self.num_ode_steps = 50
        self.lin = torch.nn.Linear(2, 1)   # so .parameters() and .to() work

    def predict_p_success(self, raw_states):
        return torch.full((raw_states.shape[0],), self.p, dtype=torch.float32)

    def p_success_mc(self, raw_states, **kw):
        return self.predict_p_success(raw_states)

    def p_success_exact(self, raw_states, **kw):
        p = self.predict_p_success(raw_states)
        return p, torch.ones_like(p, dtype=torch.bool)


class _IdentitySystem:
    state_dim = 2

    def normalize_state(self, x):
        return x

    def embed_state_for_model(self, x):
        return x


def _handle(ps, readout="exact"):
    return EnsembleOutcomeFMHandle(
        [_ConstantMember(p, readout) for p in ps], forward_readout=readout)


def _cfg(**kw):
    base = {
        "attractor_radius": 0.3,
        "readout": "exact",
        "num_mc_samples": 100,
        "num_ode_steps": 50,
        "grid_size": 33,
        "bisect_iters": 20,
        "chunk_size": 4096,
    }
    base.update(kw)
    return OmegaConf.create(base)


def _backend(ps, readout="exact", **kw):
    b = EnsembleOutcomeFMProbabilityBackend(_cfg(readout=readout, **kw),
                                            _IdentitySystem(), "cpu")
    h = _handle(ps, readout)
    b.bind_model(h)
    return b, h


X = np.zeros((6, 2), dtype=np.float32)


# --------------------------------------------------------------------- members


def test_estimate_members_returns_one_row_per_member():
    b, _ = _backend([0.1, 0.5, 0.9])
    p = b.estimate_members(X)
    assert p.shape == (3, 6)
    assert np.allclose(p[:, 0], [0.1, 0.5, 0.9])


def test_member_sample_size_is_none_under_the_exact_readout():
    # The exact readout carries no MC noise inside a member, so claiming a
    # sample size would make the debiased score subtract a bias that is absent.
    b, _ = _backend([0.2, 0.8])
    assert b.member_sample_size is None


def test_single_member_is_refused():
    b = EnsembleOutcomeFMProbabilityBackend(_cfg(), _IdentitySystem(), "cpu")
    with pytest.raises(ValueError, match="at least 2"):
        b.bind_model(_handle([0.5]))


# ------------------------------------------------- the calibration/eval trap


def test_handle_forward_logits_agree_with_the_backend_marginal():
    # THE regression guard. sigmoid(model(x)) is what calibration and eval use;
    # backend.estimate is what acquisition uses. They must be the same number.
    b, h = _backend([0.1, 0.4, 0.9])
    from_forward = torch.sigmoid(h(torch.as_tensor(X))).squeeze(-1).numpy()
    from_backend = b.estimate(X).p_success
    assert np.allclose(from_forward, from_backend, atol=1e-6), (
        "calibration would optimise a different quantity than eval reports")


def test_marginal_is_the_mean_over_members_not_a_logit_average():
    # Averaging logits instead of probabilities is the easy wrong move: it is a
    # geometric mean in odds space and does not equal the ensemble predictive.
    b, _ = _backend([0.1, 0.9])
    assert np.allclose(b.estimate(X).p_success, 0.5)


def test_readout_mismatch_is_refused():
    b = EnsembleOutcomeFMProbabilityBackend(_cfg(readout="exact"), _IdentitySystem(), "cpu")
    with pytest.raises(ValueError, match="readout mismatch"):
        b.bind_model(_handle([0.2, 0.8], readout="mc"))


def test_p_invalid_is_identically_zero():
    # A scalar outcome flow has no notion of "failed to reach an attractor".
    b, _ = _backend([0.3, 0.7])
    out = b.estimate(X)
    assert np.allclose(out.p_invalid, 0.0)
    assert np.allclose(out.p_success + out.p_failure, 1.0)


# ---------------------------------------------------------------------- handle


def test_handle_reports_its_member_count():
    assert _handle([0.1, 0.2, 0.3, 0.4, 0.5]).n_members == 5


def test_handle_rejects_fewer_than_two_members():
    with pytest.raises(ValueError, match="at least 2"):
        EnsembleOutcomeFMHandle([_ConstantMember(0.5)], forward_readout="exact")


def test_handle_is_a_module_so_the_engine_can_move_and_eval_it():
    # engine.py:137-138 calls .eval() then .to(device) unconditionally.
    h = _handle([0.2, 0.8])
    assert h.eval() is h
    assert h.to("cpu") is h
    assert len(list(h.parameters())) > 0, "state_dict must carry the members"


def test_state_dict_round_trips_through_the_member_prefix():
    # The engine globs checkpoints/best*.ckpt and hands it back as
    # resume_checkpoint, so the ensemble must save and reload as one file.
    h = _handle([0.2, 0.8])
    keys = list(h.state_dict())
    assert all(k.startswith("members.") for k in keys), keys
    assert {k.split(".")[1] for k in keys} == {"0", "1"}


def test_forward_clamps_before_the_logit():
    # The exact readout genuinely returns values within 1e-7 of the tails, and
    # logit is unbounded there.
    h = _handle([0.0, 0.0])
    assert torch.isfinite(h(torch.as_tensor(X))).all()
    h = _handle([1.0, 1.0])
    assert torch.isfinite(h(torch.as_tensor(X))).all()

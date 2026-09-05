"""An outcome-FM ensemble member must speak the contract EnsemblePosterior uses.

`EnsemblePosterior.forward_all_members` hands every member ALREADY-EMBEDDED
states, because that is what the classifier members take, and
`EnsembleClassifierProbabilityBackend.estimate_members` embeds before calling it.
`OutcomeFlowMatcher.forward` takes RAW states and embeds internally, so dropping
one in unwrapped would embed twice and silently score a different point in state
space than the caller asked about.
"""
import json

import numpy as np
import torch

from adaptive_roa.model.outcome_flow_matcher import OutcomeFlowMatcher, OutcomeVelocityMLP
from adaptive_roa.systems.cartpole import CartPoleSystem


def _system(tmp_path):
    bounds = {n: {"min": -3.0, "max": 3.0} for n in ("x", "theta", "x_dot", "theta_dot")}
    (tmp_path / "dataset_description.json").write_text(json.dumps({
        "achieved_bounds": bounds,
        "state_space": {"state_order": ["x", "theta", "x_dot", "theta_dot"]},
    }))
    return CartPoleSystem(dataset_dir=str(tmp_path))


def _member(system, seed):
    torch.manual_seed(seed)
    dummy = torch.zeros(1, int(system.state_dim))
    cond_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
    net = OutcomeVelocityMLP(condition_dim=cond_dim, hidden_dims=[16, 16])
    return OutcomeFlowMatcher(velocity_net=net, system=system, num_ode_steps=10,
                              forward_readout="exact").eval()


def test_exact_readout_accepts_an_already_embedded_condition(tmp_path):
    """The split that lets a member take processed states without re-embedding."""
    system = _system(tmp_path)
    fm = _member(system, 0)
    raw = torch.tensor([[0.1, 0.2, 0.0, 0.0], [-0.5, 0.3, 0.1, -0.2]])

    p_raw, _ = fm.p_success_exact(raw, num_steps=10)
    embedded = fm.embed(raw)
    p_emb, _ = fm.p_success_exact(embedded, num_steps=10, already_embedded=True)

    assert torch.allclose(p_raw, p_emb, atol=1e-6), (
        f"same states, different answer: raw {p_raw.tolist()} vs embedded {p_emb.tolist()}")


def test_wrapped_members_stack_into_an_ensemble_posterior(tmp_path):
    """EnsemblePosterior + the classifier backend's call shape, end to end."""
    from adaptive_roa.model.outcome_flow_matcher import EmbeddedOutcomeFM
    from adaptive_roa.predictors.posteriors import EnsemblePosterior

    system = _system(tmp_path)
    members = [_member(system, s) for s in (0, 1, 2)]
    post = EnsemblePosterior([EmbeddedOutcomeFM(m) for m in members])
    assert post.n_members == 3

    raw = torch.tensor([[0.1, 0.2, 0.0, 0.0], [-0.5, 0.3, 0.1, -0.2], [1.0, 0.0, 0.0, 0.0]])
    embedded = system.embed_state_for_model(system.normalize_state(raw))
    logits = post.forward_all_members(embedded)
    assert logits.shape == (3, 3, 1), f"expected [M=3, N=3, 1], got {tuple(logits.shape)}"

    # Each row must equal that member's own probability on the RAW states. If the
    # wrapper embedded twice, these would disagree.
    # The readout works in float64 and the wrapper casts the logit to float32,
    # so compare in the wider type rather than tripping over the dtype.
    p_members = torch.sigmoid(logits).squeeze(-1).double()
    for m, member in enumerate(members):
        expected = member.predict_p_success(raw).double()
        assert torch.allclose(p_members[m], expected, atol=1e-5), (
            f"member {m}: ensemble path {p_members[m].tolist()} vs direct {expected.tolist()}")

    # Members are independently initialised, so they must not all agree.
    assert p_members.std(dim=0).max() > 1e-6, "members are identical; no epistemic signal"

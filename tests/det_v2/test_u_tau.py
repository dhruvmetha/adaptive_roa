"""U_tau: U_H with the decision entropy taken at the tau* rule instead of at p_s = p_f (Sumanth, 2026-11-08).

U_tau(x) = p_inv + (1 - p_inv) H(q_S, q_F, q_A) in bits for a final-state model, and
H(q_S, q_F, q_A) for a classifier, whose q come from Beta(K p_s + 1, K (1 - p_s) + 1).
In bits, as in U_H, so a unit of invalid mass scores what an even two-way verdict
split scores (Sumanth, 2026-11-08: the invalid and uncertain parts scale alike).
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval import det_v2_rule as rule
from adaptive_roa.adaptive_v2.strategy import verdict_scores as vs

ROOT = Path(__file__).resolve().parents[2]


def _q(s, f, a):
    return {"q_s": [s], "q_f": [f], "q_a": [a]}


def test_utau_final_state_terms():
    assert vs.u_tau(_q(1 / 3, 1 / 3, 1 / 3), [0.2]) == pytest.approx([0.2 + 0.8 * np.log2(3)])  # most unsure verdict
    assert vs.u_tau(_q(0.0, 0.0, 1.0), [1.0]) == pytest.approx([1.0])           # every draw invalid
    assert vs.u_tau(_q(0.0, 0.0, 1.0), [0.3]) == pytest.approx([0.3])           # sure abstain: invalid term only
    assert vs.u_tau(_q(0.5, 0.0, 0.5), [0.0]) == pytest.approx([1.0])           # even S/A split = all invalid
    assert vs.u_tau(_q(1.0, 0.0, 0.0), [0.0]) == pytest.approx([0.0])           # sure S


def test_utau_classifier_is_the_verdict_entropy_in_bits():
    assert vs.u_tau(_q(0.5, 0.0, 0.5)) == pytest.approx([1.0])
    assert vs.u_tau(_q(0.0, 1.0, 0.0)) == pytest.approx([0.0])


CLF_CASES = [(0.95, 0.9, 0.9), (0.5, 0.9, 0.9), (0.12, 0.9, 0.9), (0.62, 0.6, 0.6),
             (0.45, 0.3, 0.5), (0.99, 0.03, 0.01), (0.02, 0.03, 0.01)]


@pytest.mark.parametrize("p,ts,tf", CLF_CASES)
def test_classifier_beta_verdicts_match_the_rule_under_mc(p, ts, tf):
    k = 100
    q = vs.classifier_beta_verdict_probabilities([p], ts, tf, k=k)
    d = np.random.default_rng(0).beta(k * p + 1, k * (1 - p) + 1, size=1_000_000)
    lab = rule.predict(d, None, ts, tf)
    for key, hit in (("q_s", lab == rule.S), ("q_f", lab == rule.F), ("q_a", lab < 0)):
        mc = hit.mean()
        sigma = np.sqrt(max(mc * (1 - mc), 1e-12) / len(d))
        assert abs(q[key][0] - mc) <= max(5 * sigma, 1e-4), key
    assert q["q_s"][0] + q["q_f"][0] + q["q_a"][0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("ts,tf", [(0.9, 0.9), (0.3, 0.5)])
def test_classifier_utau_peaks_at_the_two_band_edges_not_at_one_half(ts, tf):
    p = np.linspace(0.001, 0.999, 999)
    u = vs.u_tau(vs.classifier_beta_verdict_probabilities(p, ts, tf, k=100))
    lo, hi = sorted((ts, 1 - tf))
    left = p < (lo + hi) / 2
    assert abs(p[left][np.argmax(u[left])] - lo) < 0.02
    assert abs(p[~left][np.argmax(u[~left])] - hi) < 0.02


def test_classifier_utau_is_zero_in_the_middle_of_a_wide_band():
    u = vs.u_tau(vs.classifier_beta_verdict_probabilities([0.5], 0.9, 0.9, k=100))
    assert u[0] < 0.01                                   # U_H's peak; the rule surely abstains here


class _Est:
    def __init__(self, p):
        self.p_success = p


class _ClassifierBackend:
    """What VerdictAcquisitionStrategy reads from a classifier's probability backend."""

    def __init__(self, p):
        self._p, self.model_handle, self.system, self.device = p, object(), None, "cpu"

    def estimate(self, states):
        return _Est(self._p)


def _strategy(score):
    from adaptive_roa.adaptive_v2.strategy.verdict import VerdictAcquisitionStrategy
    return VerdictAcquisitionStrategy(OmegaConf.create({"score": score, "d2_ratio": 1.0, "num_mc_samples": 100}))


def test_strategy_scores_classifier_candidates_with_utau():
    p = np.array([0.05, 0.5, 0.9, 0.99])
    sc = _strategy("u_tau")._candidate_scores(np.zeros((4, 2)), _ClassifierBackend(p), True, (0.9, 0.9), 0)
    want = vs.u_tau(vs.classifier_beta_verdict_probabilities(p, 0.9, 0.9, k=100))
    assert sc["u_tau"] == pytest.approx(want)
    assert int(np.argmax(sc["u_tau"])) == 2          # p_s = 0.9 sits on the S edge; U_H would take 0.5
    assert int(np.argmax(sc["u_h"])) == 1


def test_strategy_needs_tau_for_utau():
    strategy = _strategy("u_tau")
    with pytest.raises(ValueError, match="score=u_tau needs tau"):
        strategy.select(None, None, None, SimpleNamespace(tau_s_star=None), 5)


def test_engine_fits_tau_for_utau():
    from adaptive_roa.adaptive_v2 import engine
    assert {"q_a", "u_h", "u_tau"} <= engine.THRESHOLD_CONSUMERS


def test_u_tau_acquisition_config_is_u_h_but_the_score():
    a = OmegaConf.load(ROOT / "configs/adaptive_v2/acquisition/u_tau.yaml")
    b = OmegaConf.load(ROOT / "configs/adaptive_v2/acquisition/u_h.yaml")
    assert a.sampling_mode == "u_tau" and a.acquisition.score == "u_tau"
    a.sampling_mode, a.acquisition.score = b.sampling_mode, b.acquisition.score
    assert OmegaConf.to_container(a) == OmegaConf.to_container(b)

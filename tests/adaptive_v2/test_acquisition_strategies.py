from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.strategy.direct import DirectAcquisitionStrategy
from adaptive_roa.adaptive_v2.strategy.ranked import RankedAcquisitionStrategy
from adaptive_roa.adaptive_v2.strategy.conformal import ConformalAcquisitionStrategy


def _direct_cfg():
    return OmegaConf.create({
        "d2_ratio": 0.5, "batch_size_sampling": 50, "max_samples_per_epoch": 1000,
        "decision_rule": "two_sided", "verbose": False,
    })


def _ranked_cfg():
    return OmegaConf.create({
        "d2_ratio": 0.5, "batch_size_sampling": 50, "max_samples_per_epoch": 1000,
        "n_ranked_candidates": 500, "decision_rule": "two_sided", "verbose": False,
    })


def _conformal_cfg():
    return OmegaConf.create({
        "d2_ratio": 0.5, "batch_size_sampling": 50, "max_samples_per_epoch": 1000,
        "decision_rule": "two_sided", "verbose": False,
    })


def test_direct_stores_cfg():
    s = DirectAcquisitionStrategy(_direct_cfg())
    assert s.decision_rule == "two_sided"
    assert s.batch_size_sampling == 50


def test_ranked_stores_cfg():
    s = RankedAcquisitionStrategy(_ranked_cfg())
    assert s.n_ranked_candidates == 500


def test_conformal_stores_cfg():
    s = ConformalAcquisitionStrategy(_conformal_cfg())
    assert s.decision_rule == "two_sided"

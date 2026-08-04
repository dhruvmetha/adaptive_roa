import pytest


@pytest.mark.parametrize("arm,family", [("hmc", "classifier"), ("hmc_reg", "generative")])
def test_each_arm_is_registered(arm, family):
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class(arm)
    assert cls.predictor_name == arm
    assert cls.predictor_type == family


def test_hmc_arms_do_not_steal_the_legacy_family_aliases():
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.classifier import (
        ClassifierProbabilisticClassifier,
    )
    from adaptive_roa.probabilistic_classifier.flow_matching import FMProbabilisticClassifier

    assert get_probabilistic_classifier_class("classifier") is ClassifierProbabilisticClassifier
    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier


def test_hmc_reg_declares_all_three_probabilities():
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    assert get_probabilistic_classifier_class("hmc_reg").native_probs == (
        "p_success", "p_failure", "p_invalid"
    )

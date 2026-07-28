import pytest

ARMS = ["mlp_det", "bnn_mfvi_reg", "bnn_ensemble_reg", "bnn_laplace_reg"]


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_is_registered_for_export(arm):
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class(arm)
    assert cls.predictor_name == arm
    assert cls.predictor_type == "generative"
    assert cls.native_probs == ("p_success", "p_failure", "p_invalid")


def test_arms_do_not_steal_the_legacy_generative_alias():
    """Runs written before arm names existed resolve via the family alias."""
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.flow_matching import FMProbabilisticClassifier

    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier

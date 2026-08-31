import numpy as np
import pytest
from adaptive_roa.adaptive_v2.types import OutcomeProbabilities
from adaptive_roa.probabilistic_classifier.base import ProbabilisticClassifier
from adaptive_roa.probabilistic_classifier.registry import (
    register_probabilistic_classifier,
    get_probabilistic_classifier_class,
)


@register_probabilistic_classifier
class _DummyPC(ProbabilisticClassifier):
    predictor_type = "dummy"
    predictor_name = "dummy"
    native_probs = ("p_success",)

    def predict(self, states):
        n = len(states)
        return OutcomeProbabilities(
            p_success=np.full(n, 0.5),
            p_failure=np.full(n, 0.5),
            p_invalid=np.zeros(n),
        )

    @classmethod
    def load_from_run(cls, run_dir, epoch, cfg, system, device="cuda"):
        return cls()


def test_registry_resolves_registered_type():
    assert get_probabilistic_classifier_class("dummy") is _DummyPC


def test_registry_raises_on_unknown_type():
    with pytest.raises(KeyError):
        get_probabilistic_classifier_class("does-not-exist")


def test_base_predict_cached_defaults_to_none():
    pc = _DummyPC()
    assert pc.predict_cached("run", 0, "test", np.zeros((3, 2))) is None


def test_native_probs_declared():
    assert _DummyPC.native_probs == ("p_success",)


def test_registry_keys_on_arm_name_not_family():
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.classifier import (
        ClassifierProbabilisticClassifier,
    )
    assert get_probabilistic_classifier_class("mlp") is ClassifierProbabilisticClassifier


def test_legacy_family_keys_still_resolve():
    """Runs already on disk carry only predictor.type; they must still export."""
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.classifier import (
        ClassifierProbabilisticClassifier,
    )
    from adaptive_roa.probabilistic_classifier.flow_matching import (
        FMProbabilisticClassifier,
    )
    assert get_probabilistic_classifier_class("classifier") is ClassifierProbabilisticClassifier
    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier


def test_resolve_predictor_name_prefers_name_over_type():
    from omegaconf import OmegaConf
    from adaptive_roa.probabilistic_classifier.export import resolve_predictor_name

    named = OmegaConf.create({"predictor": {"type": "classifier", "name": "bnn_mfvi"}})
    legacy = OmegaConf.create({"predictor": {"type": "classifier"}})
    assert resolve_predictor_name(named) == "bnn_mfvi"
    assert resolve_predictor_name(legacy) == "classifier"

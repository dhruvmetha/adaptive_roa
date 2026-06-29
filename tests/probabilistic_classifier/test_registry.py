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

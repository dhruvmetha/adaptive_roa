import numpy as np
import torch

from adaptive_roa.probabilistic_classifier.classifier import (
    ClassifierProbabilisticClassifier,
)
from adaptive_roa.probabilistic_classifier.registry import (
    get_probabilistic_classifier_class,
)
from adaptive_roa.model.classifier_mlp import ClassifierMLP, ClassifierModule
from adaptive_roa.systems.pendulum import PendulumSystem


def _tiny_module(system):
    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(
        system.embed_state_for_model(system.normalize_state(dummy)).shape[-1]
    )
    mlp = ClassifierMLP(input_dim=input_dim, hidden_dims=[8], output_dim=1, dropout=0.0)
    return ClassifierModule(
        mlp=mlp, system=system, pos_weight=torch.tensor(1.0), lr=1e-3, weight_decay=1e-5
    )


def test_registered_under_classifier():
    assert get_probabilistic_classifier_class("classifier") is ClassifierProbabilisticClassifier


def test_native_probs_is_p_success_only():
    assert ClassifierProbabilisticClassifier.native_probs == ("p_success",)


def test_predict_shapes_and_ranges():
    system = PendulumSystem()
    module = _tiny_module(system).eval()
    pc = ClassifierProbabilisticClassifier(module, system, device="cpu")
    states = np.random.randn(5, int(system.state_dim)).astype(np.float32)
    out = pc.predict(states)
    assert out.p_success.shape == (5,)
    assert np.all((out.p_success >= 0.0) & (out.p_success <= 1.0))
    assert np.allclose(out.p_failure, 1.0 - out.p_success)
    assert np.allclose(out.p_invalid, 0.0)


def test_predict_cached_returns_none():
    system = PendulumSystem()
    pc = ClassifierProbabilisticClassifier(_tiny_module(system).eval(), system, device="cpu")
    assert pc.predict_cached("run", 0, "test", np.zeros((2, 2))) is None

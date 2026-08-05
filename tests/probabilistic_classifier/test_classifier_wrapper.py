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


def test_the_export_loader_rebuilds_the_trained_activation(tmp_path):
    """Activations carry no parameters, so a train/export activation mismatch
    loads the state_dict CLEANLY and then computes different logits -- there is
    no error, no missing key, nothing to catch it. `load_clf_module` must read
    the same `classifier.activation` key `ClassifierTrainer` writes with.

    The cfg here is nested under `predictor`, matching what `export_run` (the
    real caller, via `load_from_run`) actually passes: the full run config,
    where the arm block lives at `predictor.classifier` -- the same shape
    every sibling wrapper reads (bayesian.py's `predictor.bnn`,
    bayesian_final_state.py's `predictor.final_state`). A flat
    `{"classifier": {...}}` cfg is not a shape any real caller produces and
    previously hid a size-mismatch bug that made reference-tier classifier
    exports fail silently.
    """
    from omegaconf import OmegaConf

    from adaptive_roa.probabilistic_classifier.classifier import load_clf_module

    system = PendulumSystem()
    cfg = OmegaConf.create(
        {"predictor": {"classifier": {"hidden_dims": [8], "activation": "tanh"}}}
    )

    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
    trained = ClassifierModule(
        mlp=ClassifierMLP(input_dim=input_dim, hidden_dims=[8], output_dim=1,
                          activation="tanh"),
        system=system, pos_weight=torch.tensor(1.0), lr=1e-3, weight_decay=1e-5,
    ).eval()
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    torch.save({"state_dict": trained.state_dict()}, ckpt_dir / "best.ckpt")

    loaded = load_clf_module(str(tmp_path), system, cfg, device="cpu")
    assert [type(m).__name__ for m in loaded.mlp.net] == \
           [type(m).__name__ for m in trained.mlp.net]
    states = torch.randn(6, int(system.state_dim))
    torch.testing.assert_close(loaded(states), trained(states))


def test_the_export_loader_also_accepts_a_legacy_flat_cfg(tmp_path):
    """Pre-`predictor.name` run configs recorded `predictor` as the bare
    string "classifier" (see `resolve_predictor_name`) and kept the arm block
    flat at the top level, as `test_export_integration.py` exercises through
    `export_run`. `load_clf_module` must still honour that shape.
    """
    from omegaconf import OmegaConf

    from adaptive_roa.probabilistic_classifier.classifier import load_clf_module

    system = PendulumSystem()
    cfg = OmegaConf.create(
        {"predictor": "classifier", "classifier": {"hidden_dims": [8], "activation": "tanh"}}
    )

    dummy = torch.zeros(1, int(system.state_dim))
    input_dim = int(system.embed_state_for_model(system.normalize_state(dummy)).shape[-1])
    trained = ClassifierModule(
        mlp=ClassifierMLP(input_dim=input_dim, hidden_dims=[8], output_dim=1,
                          activation="tanh"),
        system=system, pos_weight=torch.tensor(1.0), lr=1e-3, weight_decay=1e-5,
    ).eval()
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    torch.save({"state_dict": trained.state_dict()}, ckpt_dir / "best.ckpt")

    loaded = load_clf_module(str(tmp_path), system, cfg, device="cpu")
    states = torch.randn(6, int(system.state_dim))
    torch.testing.assert_close(loaded(states), trained(states))
